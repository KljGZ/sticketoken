"""Pure metric implementations for all three experiment modes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def exact_pairwise_mean(normalized_embeddings: np.ndarray) -> float:
    """Exact O(Md) mean cosine over all unordered distinct pairs."""
    values = np.asarray(normalized_embeddings, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("At least two embeddings are required")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 1e-12) or not np.isfinite(norms).all():
        raise ValueError("Embeddings must have finite non-zero norms")
    # Some GPU kernels return normalized BF16/FP16 values whose FP64 norms
    # differ from one by a few 1e-4.  Renormalize in FP64 before applying the
    # exact identity so that the reported mean is not precision-dependent.
    values = values / norms[:, None]
    total = values.sum(axis=0, dtype=np.float64)
    count = len(values)
    return float((np.dot(total, total) - count) / (count * (count - 1)))


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(left).rank(method="average").to_numpy(dtype=float)
    right_rank = pd.Series(right).rank(method="average").to_numpy(dtype=float)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def single_sticky_score(
    similarities: np.ndarray,
    baseline: np.ndarray,
    semantic_penalty: np.ndarray,
    *,
    alpha: float,
    beta: float,
    gamma: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Paper-style score for [candidate, insertion_mode, pair] similarities."""
    values = np.asarray(similarities, dtype=float)
    delta = values - np.asarray(baseline, dtype=float)[None, None, :]
    positive_magnitude = np.maximum(delta, 0).sum(axis=2)
    negative_magnitude = np.maximum(-delta, 0).sum(axis=2)
    positive_frequency = (delta > 0).mean(axis=2)
    negative_frequency = (delta < 0).mean(axis=2)
    raw_denominator = (
        negative_magnitude
        + beta * negative_frequency
        + np.asarray(semantic_penalty, dtype=float)[:, None]
        + gamma
    )
    nonpositive = raw_denominator <= 0
    denominator = np.maximum(raw_denominator, gamma)
    per_mode = (positive_magnitude + alpha * positive_frequency) / denominator
    return per_mode.sum(axis=1), {
        "M_positive": positive_magnitude.sum(axis=1),
        "M_negative": negative_magnitude.sum(axis=1),
        "F_positive": positive_frequency.mean(axis=1),
        "F_negative": negative_frequency.mean(axis=1),
        "nonpositive_denominator_count": nonpositive.sum(axis=1),
    }


def booster_metrics(
    similarities: np.ndarray,
    baseline: np.ndarray,
    low_mask: np.ndarray,
    high_mask: np.ndarray,
    constraints: dict[str, float],
) -> list[dict[str, Any]]:
    """Evaluate monotone-booster candidates.

    ``similarities`` has shape [candidate, insertion_mode, pair].  Robust tail
    constraints are evaluated on the pooled insertion-mode/pair observations;
    range and rank preservation use the worst insertion mode.
    """
    values = np.asarray(similarities, dtype=float)
    base = np.asarray(baseline, dtype=float)
    low = np.asarray(low_mask, dtype=bool)
    high = np.asarray(high_mask, dtype=bool)
    if values.ndim != 3 or values.shape[2] != len(base):
        raise ValueError("Expected [candidate, mode, pair] similarities")
    if not low.any() or not high.any():
        raise ValueError("Low and high groups must both be non-empty")
    delta = values - base[None, None, :]
    base_range = float(np.quantile(base, 0.90) - np.quantile(base, 0.10))
    records: list[dict[str, Any]] = []
    for candidate_index in range(len(values)):
        candidate_delta = delta[candidate_index]
        low_delta = candidate_delta[:, low].ravel()
        high_delta = candidate_delta[:, high].ravel()
        all_delta = candidate_delta.ravel()
        range_ratios = [
            (np.quantile(mode_values, 0.90) - np.quantile(mode_values, 0.10))
            / max(base_range, 1e-12)
            for mode_values in values[candidate_index]
        ]
        correlations = [_spearman(base, mode_values) for mode_values in values[candidate_index]]
        low_gain_mean = float(low_delta.mean())
        low_positive_gain = float(np.maximum(low_delta, 0).mean())
        low_coverage = float(np.mean(low_delta >= constraints["low_gain_margin"]))
        high_q05 = float(np.quantile(high_delta, 0.05))
        global_drop_rate = float(np.mean(all_delta < -constraints["global_drop_tolerance"]))
        range_ratio = float(min(range_ratios))
        spearman = float(min(correlations))
        violations = {
            "low_gain": max(0.0, constraints["min_low_gain"] - low_gain_mean) / max(constraints["min_low_gain"], 1e-12),
            "low_coverage": max(0.0, constraints["min_low_coverage"] - low_coverage) / max(constraints["min_low_coverage"], 1e-12),
            "high_tail": max(0.0, -constraints["high_drop_tolerance"] - high_q05) / max(constraints["high_drop_tolerance"], 1e-12),
            "drop_rate": max(0.0, global_drop_rate - constraints["max_global_drop_rate"]) / max(constraints["max_global_drop_rate"], 1e-12),
            "range": max(0.0, constraints["min_range_ratio"] - range_ratio) / max(constraints["min_range_ratio"], 1e-12),
            "rank": max(0.0, constraints["min_spearman"] - spearman) / max(constraints["min_spearman"], 1e-12),
        }
        total_violation = float(sum(violations.values()))
        objective = low_positive_gain + constraints.get("coverage_weight", 0.05) * low_coverage
        records.append(
            {
                "objective": float(objective),
                "constraint_violation": total_violation,
                "feasible": total_violation <= 1e-12,
                "low_gain_mean": low_gain_mean,
                "low_positive_gain": low_positive_gain,
                "low_gain_q10": float(np.quantile(low_delta, 0.10)),
                "low_coverage": low_coverage,
                "high_gain_q05": high_q05,
                "high_gain_mean": float(high_delta.mean()),
                "global_drop_rate": global_drop_rate,
                "range_ratio": range_ratio,
                "spearman": spearman,
                **{f"violation_{name}": float(value) for name, value in violations.items()},
            }
        )
    return records


def prefix_path_metrics(
    similarities: np.ndarray,
    baseline: np.ndarray,
    *,
    drop_tolerance: float,
    max_failure_rate: float,
) -> dict[str, Any]:
    """Certify that adding successive trigger tokens rarely decreases similarity."""
    values = np.asarray(similarities, dtype=float)
    if values.ndim != 3:
        raise ValueError("Expected [prefix, mode, pair] similarities")
    previous = np.concatenate([np.broadcast_to(baseline, (1, values.shape[1], len(baseline))), values[:-1]], axis=0)
    steps = values - previous
    failure_rate = float(np.mean(steps < -drop_tolerance))
    return {
        "path_min_step": float(steps.min()),
        "path_step_q05": float(np.quantile(steps, 0.05)),
        "path_failure_rate": failure_rate,
        "path_pass": failure_rate <= max_failure_rate,
    }


def repulsive_attractor_metrics(
    triggered_first: np.ndarray,
    triggered_second: np.ndarray,
    original_first: np.ndarray,
    original_second: np.ndarray,
    baseline: np.ndarray,
    low_mask: np.ndarray,
    high_mask: np.ndarray,
    constraints: dict[str, float],
) -> list[dict[str, Any]]:
    """Evaluate shared-trigger compactness, self-repulsion and pair preservation."""
    first = np.asarray(triggered_first, dtype=float)
    second = np.asarray(triggered_second, dtype=float)
    if first.shape != second.shape or first.ndim != 4:
        raise ValueError("Expected matching [candidate, mode, pair, dimension] arrays")
    pair_similarity = np.einsum("cmpd,cmpd->cmp", first, second, optimize=True)
    delta = pair_similarity - baseline[None, None, :]
    records: list[dict[str, Any]] = []
    for candidate_index in range(len(first)):
        displacement_q05_by_mode: list[float] = []
        compact_loss_by_mode: list[float] = []
        compact_radius_q95_by_mode: list[float] = []
        pairwise_similarity_by_mode: list[float] = []
        center_norm_by_mode: list[float] = []
        for mode_index in range(first.shape[1]):
            vectors = np.concatenate([first[candidate_index, mode_index], second[candidate_index, mode_index]], axis=0)
            originals = np.concatenate([original_first, original_second], axis=0)
            self_cosine = np.einsum("ij,ij->i", vectors, originals, optimize=True)
            displacement = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * self_cosine))
            center = vectors.mean(axis=0)
            center_norm_sq = float(np.dot(center, center))
            radii = np.linalg.norm(vectors - center[None, :], axis=1)
            count = len(vectors)
            mean_pair = (count * center_norm_sq - 1.0) / max(count - 1, 1)
            displacement_q05_by_mode.append(float(np.quantile(displacement, 0.05)))
            compact_loss_by_mode.append(float(1.0 - center_norm_sq))
            compact_radius_q95_by_mode.append(float(np.quantile(radii, 0.95)))
            pairwise_similarity_by_mode.append(float(mean_pair))
            center_norm_by_mode.append(float(np.sqrt(max(center_norm_sq, 0.0))))
        low_delta = delta[candidate_index, :, low_mask].ravel()
        high_delta = delta[candidate_index, :, high_mask].ravel()
        displacement_q05 = min(displacement_q05_by_mode)
        compact_loss = max(compact_loss_by_mode)
        compact_radius_q95 = max(compact_radius_q95_by_mode)
        high_q05 = float(np.quantile(high_delta, 0.05))
        low_gain = float(low_delta.mean())
        violations = {
            "repulsion": max(0.0, constraints["min_displacement_q05"] - displacement_q05) / max(constraints["min_displacement_q05"], 1e-12),
            "compact_radius": max(0.0, compact_radius_q95 - constraints["max_compact_radius_q95"]) / max(constraints["max_compact_radius_q95"], 1e-12),
            "high_tail": max(0.0, -constraints["high_drop_tolerance"] - high_q05) / max(constraints["high_drop_tolerance"], 1e-12),
            "low_gain": max(0.0, constraints["min_low_gain"] - low_gain) / max(abs(constraints["min_low_gain"]), 1e-12),
        }
        total_violation = float(sum(violations.values()))
        objective = low_gain - constraints.get("compactness_weight", 0.5) * compact_loss
        records.append(
            {
                "objective": float(objective),
                "constraint_violation": total_violation,
                "feasible": total_violation <= 1e-12,
                "low_gain_mean": low_gain,
                "low_gain_q10": float(np.quantile(low_delta, 0.10)),
                "high_gain_q05": high_q05,
                "high_gain_mean": float(high_delta.mean()),
                "displacement_q05": float(displacement_q05),
                "displacement_q05_mean_modes": float(np.mean(displacement_q05_by_mode)),
                "compactness_loss": float(compact_loss),
                "compact_radius_q95": float(compact_radius_q95),
                "triggered_pairwise_similarity": float(min(pairwise_similarity_by_mode)),
                "center_norm": float(min(center_norm_by_mode)),
                "local_uniqueness_lower_bound": float(displacement_q05 - compact_radius_q95),
                **{f"violation_{name}": float(value) for name, value in violations.items()},
            }
        )
    return records


def feasibility_sort_key(record: dict[str, Any], trigger_length: int) -> tuple[float, ...]:
    return (
        0.0 if bool(record.get("feasible", False)) else 1.0,
        float(record.get("constraint_violation", float("inf"))),
        -float(record.get("objective", -float("inf"))),
        float(trigger_length),
    )
