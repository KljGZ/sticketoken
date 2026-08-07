"""Registered V2 metrics and feasibility-first ordering."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(left).rank(method="average").to_numpy(dtype=float)
    right_rank = pd.Series(right).rank(method="average").to_numpy(dtype=float)
    if np.std(left_rank) <= 1e-12 or np.std(right_rank) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _positive_part(value: float) -> float:
    return max(0.0, float(value))


def mode2_metrics(
    similarities: np.ndarray,
    baseline: np.ndarray,
    low_mask: np.ndarray,
    high_mask: np.ndarray,
    constraints: dict[str, float],
    *,
    low_threshold: float,
    high_threshold: float,
) -> list[dict[str, Any]]:
    """Evaluate Minimal Monotone Similarity Booster candidates.

    Tail conditions are pooled across registered insertion modes, while the
    dynamic-range and rank checks use the worst insertion mode.
    """
    values = np.asarray(similarities, dtype=float)
    base = np.asarray(baseline, dtype=float)
    low = np.asarray(low_mask, dtype=bool)
    high = np.asarray(high_mask, dtype=bool)
    if values.ndim != 3 or values.shape[2] != len(base):
        raise ValueError("Expected [candidate, insertion_mode, pair] similarities")
    if not low.any() or not high.any():
        raise ValueError("Low and high groups must both be non-empty")
    delta = values - base[None, None, :]
    base_range = float(np.quantile(base, 0.90) - np.quantile(base, 0.10))
    records: list[dict[str, Any]] = []
    for index in range(len(values)):
        low_delta = delta[index, :, low].ravel()
        high_delta = delta[index, :, high].ravel()
        all_delta = delta[index].ravel()
        high_state = values[index, :, high].ravel()
        low_coverage = float(np.mean(low_delta >= constraints["low_gain_margin"]))
        low_q10 = float(np.quantile(low_delta, 0.10))
        high_q05 = float(np.quantile(high_delta, 0.05))
        high_state_retention = float(
            np.mean(high_state >= high_threshold - constraints["high_state_tolerance"])
        )
        global_drop_rate = float(np.mean(all_delta < -constraints["global_drop_tolerance"]))
        range_ratio = float(
            min(
                (np.quantile(mode_values, 0.90) - np.quantile(mode_values, 0.10))
                / max(base_range, 1e-12)
                for mode_values in values[index]
            )
        )
        spearman = float(min(_spearman(base, mode_values) for mode_values in values[index]))
        violations = {
            "low_coverage": _positive_part(constraints["min_low_coverage"] - low_coverage)
            / max(constraints["min_low_coverage"], 1e-12),
            "high_tail": _positive_part(-constraints["high_drop_tolerance"] - high_q05)
            / max(constraints["high_drop_tolerance"], 1e-12),
            "high_state": _positive_part(constraints["min_high_state_retention"] - high_state_retention)
            / max(constraints["min_high_state_retention"], 1e-12),
            "drop_rate": _positive_part(global_drop_rate - constraints["max_global_drop_rate"])
            / max(constraints["max_global_drop_rate"], 1e-12),
        }
        core_violation = float(sum(violations.values()))
        core_feasible = core_violation <= 1e-12
        structure_violations = {
            "range": _positive_part(constraints["min_range_ratio"] - range_ratio)
            / max(constraints["min_range_ratio"], 1e-12),
            "rank": _positive_part(constraints["min_spearman"] - spearman)
            / max(constraints["min_spearman"], 1e-12),
        }
        structure_feasible = core_feasible and sum(structure_violations.values()) <= 1e-12
        # This objective is descriptive.  Ordering is constraint-first and
        # therefore cannot hide a hard-condition failure behind a weighted sum.
        objective = low_q10 + 0.05 * low_coverage + 0.05 * high_q05
        records.append(
            {
                "objective": float(objective),
                "constraint_violation": core_violation,
                "feasible": core_feasible,
                "core_feasible": core_feasible,
                "structure_feasible": structure_feasible,
                "low_gain_mean": float(low_delta.mean()),
                "low_positive_gain": float(np.maximum(low_delta, 0).mean()),
                "low_gain_q10": low_q10,
                "low_coverage": low_coverage,
                "high_gain_q05": high_q05,
                "high_gain_mean": float(high_delta.mean()),
                "high_state_retention": high_state_retention,
                "global_drop_rate": global_drop_rate,
                "range_ratio": range_ratio,
                "spearman": spearman,
                "low_threshold": float(low_threshold),
                "high_threshold": float(high_threshold),
                **{f"violation_{name}": float(value) for name, value in violations.items()},
                **{f"structure_violation_{name}": float(value) for name, value in structure_violations.items()},
            }
        )
    return records


def _normalized_center(vectors: np.ndarray) -> np.ndarray:
    center = np.asarray(vectors, dtype=float).mean(axis=0)
    norm = float(np.linalg.norm(center))
    if norm <= 1e-12:
        raise ValueError("Triggered center has zero norm")
    return center / norm


def _sample_pairwise(values: np.ndarray, limit: int, seed: int) -> np.ndarray:
    count = len(values)
    total = count * (count - 1) // 2
    if total <= limit:
        gram = values @ values.T
        return gram[np.triu_indices(count, 1)]
    rng = np.random.default_rng(seed)
    left = rng.integers(0, count, size=limit * 2)
    right = rng.integers(0, count, size=limit * 2)
    keep = left != right
    left, right = left[keep][:limit], right[keep][:limit]
    return np.einsum("ij,ij->i", values[left], values[right], optimize=True)


def mode3_metrics(
    triggered: np.ndarray,
    original: np.ndarray,
    source_cluster: np.ndarray,
    cluster_centers: np.ndarray,
    cluster_radii: np.ndarray,
    constraints: dict[str, float],
    *,
    pairwise_sample_size: int = 20000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Evaluate Cluster-Escape Compact Attractor candidates.

    ``triggered`` is [candidate, insertion_mode, sentence, dimension].  Every
    registered mode must satisfy the core constraints, so the record contains
    the worst value across modes.
    """
    values = np.asarray(triggered, dtype=float)
    original_values = np.asarray(original, dtype=float)
    centers = np.asarray(cluster_centers, dtype=float)
    radii = np.asarray(cluster_radii, dtype=float)
    labels = np.asarray(source_cluster, dtype=int)
    if values.ndim != 4 or values.shape[2:] != original_values.shape:
        raise ValueError("Expected [candidate, insertion_mode, sentence, dimension]")
    source_centers = centers[labels]
    source_radii = radii[labels]
    original_source_distance = np.linalg.norm(original_values - source_centers, axis=1)
    records: list[dict[str, Any]] = []
    for candidate_index in range(len(values)):
        by_mode: list[dict[str, float]] = []
        for mode_index, vectors in enumerate(values[candidate_index]):
            source_distance = np.linalg.norm(vectors - source_centers, axis=1)
            absolute_escape = source_distance - source_radii
            relative_outward = source_distance - original_source_distance
            center = _normalized_center(vectors)
            compact_radius = np.linalg.norm(vectors - center[None, :], axis=1)
            all_cluster_distance = np.linalg.norm(vectors[:, None, :] - centers[None, :, :], axis=2)
            all_cluster_clearance = np.min(all_cluster_distance - radii[None, :], axis=1)
            pairwise = _sample_pairwise(vectors, pairwise_sample_size, seed + candidate_index * 97 + mode_index)
            by_mode.append(
                {
                    "absolute_escape_q05": float(np.quantile(absolute_escape, 0.05)),
                    "relative_outward_q05": float(np.quantile(relative_outward, 0.05)),
                    "escape_rate": float(np.mean(absolute_escape >= 0.0)),
                    "compact_radius_q95": float(np.quantile(compact_radius, 0.95)),
                    "triggered_pairwise_mean": float(pairwise.mean()),
                    "triggered_pairwise_q05": float(np.quantile(pairwise, 0.05)),
                    "all_cluster_clearance_q05": float(np.quantile(all_cluster_clearance, 0.05)),
                    "center_norm_pre_normalization": float(np.linalg.norm(vectors.mean(axis=0))),
                }
            )
        absolute_q05 = min(item["absolute_escape_q05"] for item in by_mode)
        relative_q05 = min(item["relative_outward_q05"] for item in by_mode)
        escape_rate = min(item["escape_rate"] for item in by_mode)
        radius_q95 = max(item["compact_radius_q95"] for item in by_mode)
        violations = {
            "absolute_escape": _positive_part(constraints["min_absolute_escape_q05"] - absolute_q05)
            / max(abs(constraints["min_absolute_escape_q05"]), 1e-6),
            "relative_outward": _positive_part(constraints["min_relative_outward_q05"] - relative_q05)
            / max(abs(constraints["min_relative_outward_q05"]), 1e-6),
            "escape_coverage": _positive_part(constraints["min_escape_rate"] - escape_rate)
            / max(constraints["min_escape_rate"], 1e-12),
            "compact_radius": _positive_part(radius_q95 - constraints["max_compact_radius_q95"])
            / max(constraints["max_compact_radius_q95"], 1e-12),
        }
        total_violation = float(sum(violations.values()))
        feasible = total_violation <= 1e-12
        records.append(
            {
                "objective": float(absolute_q05 + relative_q05 - radius_q95),
                "constraint_violation": total_violation,
                "feasible": feasible,
                "core_feasible": feasible,
                "absolute_escape_q05": float(absolute_q05),
                "relative_outward_q05": float(relative_q05),
                "escape_rate": float(escape_rate),
                "compact_radius_q95": float(radius_q95),
                "triggered_pairwise_mean": float(min(item["triggered_pairwise_mean"] for item in by_mode)),
                "triggered_pairwise_q05": float(min(item["triggered_pairwise_q05"] for item in by_mode)),
                "all_cluster_clearance_q05": float(min(item["all_cluster_clearance_q05"] for item in by_mode)),
                "blank_region_cluster_feasible": bool(
                    min(item["all_cluster_clearance_q05"] for item in by_mode) > 0.0
                ),
                "per_mode_metrics": by_mode,
                **{f"violation_{name}": float(value) for name, value in violations.items()},
            }
        )
    return records


def mode2_sort_key(record: dict[str, Any]) -> tuple[float, ...]:
    return (
        0.0 if bool(record.get("core_feasible", record.get("feasible", False))) else 1.0,
        float(record.get("constraint_violation", float("inf"))),
        -float(record.get("low_gain_q10", -float("inf"))),
        -float(record.get("low_coverage", -float("inf"))),
        -float(record.get("high_gain_q05", -float("inf"))),
        float(record.get("component_length", 0)),
    )


def mode3_sort_key(record: dict[str, Any]) -> tuple[float, ...]:
    return (
        0.0 if bool(record.get("core_feasible", record.get("feasible", False))) else 1.0,
        float(record.get("constraint_violation", float("inf"))),
        -float(record.get("absolute_escape_q05", -float("inf"))),
        -float(record.get("relative_outward_q05", -float("inf"))),
        float(record.get("compact_radius_q95", float("inf"))),
        float(record.get("component_length", 0)),
    )


def dose_certification(gaps: np.ndarray, epsilon: float, coverage_target: float) -> dict[str, Any]:
    values = np.asarray(gaps, dtype=float).ravel()
    coverage = float(np.mean(values <= epsilon))
    q95 = float(np.quantile(values, 0.95))
    maximum = float(values.max())
    return {
        "coverage": coverage,
        "GE_q95": q95,
        "GE_max": maximum,
        "coverage_certified": coverage >= coverage_target,
        "q95_certified": q95 <= epsilon,
        "strict_max_certified": maximum <= epsilon,
    }

