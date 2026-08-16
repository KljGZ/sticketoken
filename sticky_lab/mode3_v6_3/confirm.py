"""Pure fixed-cap confirmation; this module performs no fit or selection."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .errors import ManifestMismatch, ShapeMismatch
from .freeze import FreezeArtifact
from .statistics import (
    independent_text_strata,
    migration_bounds,
    radial_occupancy,
    simultaneous_balanced_bounds,
    simultaneous_source_occupancy,
)


def _rank_delta(left: np.ndarray, right: np.ndarray) -> float:
    values = np.concatenate([left, right])
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + 1 + end) / 2.0
        cursor = end
    n, m = len(left), len(right)
    u = float(ranks[:n].sum() - n * (n + 1) / 2.0)
    return float(2.0 * u / (n * m) - 1.0)


def _ks_less(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    support = np.unique(np.concatenate([left, right]))
    left_sorted, right_sorted = np.sort(left), np.sort(right)
    left_cdf = np.searchsorted(left_sorted, support, side="right") / len(left_sorted)
    right_cdf = np.searchsorted(right_sorted, support, side="right") / len(right_sorted)
    # alternative='less': left values tend smaller, hence F_left > F_right.
    statistic = float(np.max(left_cdf - right_cdf))
    effective = len(left) * len(right) / (len(left) + len(right))
    pvalue = min(1.0, math.exp(-2.0 * effective * statistic * statistic))
    return statistic, pvalue


def _wasserstein_1d(left: np.ndarray, right: np.ndarray) -> float:
    support = np.sort(np.unique(np.concatenate([left, right])))
    if len(support) < 2:
        return 0.0
    left_sorted, right_sorted = np.sort(left), np.sort(right)
    cdf_left = np.searchsorted(left_sorted, support[:-1], side="right") / len(left_sorted)
    cdf_right = np.searchsorted(right_sorted, support[:-1], side="right") / len(right_sorted)
    return float(np.sum(np.abs(cdf_left - cdf_right) * np.diff(support)))


def _verify_role_hashes(
    artifact: FreezeArtifact, observed_role_hashes: Mapping[str, str]
) -> None:
    for role, digest in observed_role_hashes.items():
        expected = artifact.confirm_role_hashes.get(str(role))
        if expected is None or expected != str(digest):
            raise ManifestMismatch(f"confirmation role hash mismatch for {role}")
    if set(observed_role_hashes).intersection(artifact.discovery_role_hashes):
        raise ManifestMismatch("confirmation attempted a discovery role")


def confirm_fixed_cap(
    artifact: FreezeArtifact,
    *,
    trigger_rows: Sequence[Mapping[str, Any]],
    triggered_vectors: np.ndarray,
    paired_clean_vectors: np.ndarray,
    benign_rows: Sequence[Mapping[str, Any]],
    benign_vectors: np.ndarray,
    observed_role_hashes: Mapping[str, str],
    freeze_sha256: str,
    radial_multipliers: Sequence[float],
    familywise_alpha: float = 0.05,
) -> dict[str, Any]:
    """Evaluate one immutable cap on independent text-level observations."""
    _verify_role_hashes(artifact, observed_role_hashes)
    cap = artifact.frozen_cap()
    triggered = np.asarray(triggered_vectors)
    clean = np.asarray(paired_clean_vectors)
    benign = np.asarray(benign_vectors)
    if len(trigger_rows) != len(triggered) or triggered.shape != clean.shape:
        raise ShapeMismatch("paired confirmation rows/vectors differ")
    if len(benign_rows) != len(benign):
        raise ShapeMismatch("benign rows/vectors differ")
    text_ids = [str(row["text_id"]) for row in trigger_rows]
    if len(text_ids) != len(set(text_ids)):
        raise ShapeMismatch("formal confirmation has repeated text units")
    triggered_depth = cap.normalized_radius(triggered)
    clean_depth = cap.normalized_radius(clean)
    benign_depth = cap.normalized_radius(benign)
    triggered_inside = independent_text_strata(trigger_rows, triggered_depth <= 1.0 + 1e-12)
    clean_inside = independent_text_strata(trigger_rows, clean_depth <= 1.0 + 1e-12)
    coverage = simultaneous_balanced_bounds(
        triggered_inside, familywise_alpha=float(familywise_alpha)
    )
    migration = migration_bounds(
        clean_inside, triggered_inside, familywise_alpha=float(familywise_alpha)
    )
    benign_membership: dict[str, list[bool]] = {}
    benign_depth_by_source: dict[str, list[float]] = {}
    for row, depth in zip(benign_rows, benign_depth):
        source = str(row["source_id"])
        benign_membership.setdefault(source, []).append(bool(depth <= 1.0 + 1e-12))
        benign_depth_by_source.setdefault(source, []).append(float(depth))
    occupancy = simultaneous_source_occupancy(
        benign_membership, familywise_alpha=float(familywise_alpha)
    )
    radial = radial_occupancy(
        benign_depth, radial_multipliers, confidence=1.0 - float(familywise_alpha)
    )
    thresholds = artifact.certification_thresholds
    core_gates = {
        "balanced_coverage": coverage.balanced_lower > float(thresholds["balanced_coverage_lcb"]),
        "worst_position": coverage.worst_position_lower > float(thresholds["worst_position_lcb"]),
        "worst_source": coverage.worst_source_lower > float(thresholds["worst_source_lcb"]),
        "independent_benign_core": occupancy["balanced_ucb"] < float(thresholds["independent_benign_core_ucb"]),
        "outside_to_inside": migration["outside_to_inside"]["balanced_lower"] >= float(thresholds["outside_to_inside_lcb"]),
        "conditional_outside_origin": migration["conditional_outside_origin"]["balanced_lower"] >= float(thresholds["conditional_outside_origin_lcb"]),
        "radius": cap.radius_degrees <= float(thresholds["maximum_radius_degrees"]),
    }
    uniform = min(bound.lower for bound in coverage.strata.values()) > float(thresholds["balanced_coverage_lcb"])
    moat = next((row for row in radial["curve"] if np.isclose(row["multiplier"], 1.10)), None)
    core = all(core_gates.values())
    triggered_median = float(np.median(triggered_depth))
    benign_median = float(np.median(benign_depth))
    ks_statistic, ks_pvalue = _ks_less(triggered_depth, benign_depth)
    cliffs_delta = _rank_delta(triggered_depth, benign_depth)
    levels = {
        "A_ST_RADIAL_SHIFT": bool(triggered_median < benign_median and ks_pvalue < 0.05),
        "B_ST_FCA_CORE": bool(core),
        "C_ST_FCA_MOAT": bool(core and moat is not None and moat["upper"] < float(thresholds["moat_occupancy_1_10_ucb"])),
        "D_ST_FCA_BASIN": bool(core and radial["lambda_star"] >= float(thresholds["basin_lambda_star"]) and radial["occupancy_auc_1_1_5"] <= float(thresholds["basin_occupancy_auc_1_1_5"])),
        "E_ST_CENTRAL_COLLAPSE": bool(core and triggered_median <= float(thresholds["central_collapse_median_depth"])),
    }
    interval_rows = [
        {"source_position": key, **bound.to_dict()}
        for key, bound in coverage.strata.items()
    ]
    observation_rows = [
        {
            "text_id": str(row["text_id"]), "source_id": str(row["source_id"]),
            "position": str(row["position"]), "triggered_depth": float(tr_depth),
            "clean_depth": float(cl_depth), "triggered_inside": bool(tr_depth <= 1.0 + 1e-12),
            "clean_inside": bool(cl_depth <= 1.0 + 1e-12),
        }
        for row, tr_depth, cl_depth in zip(trigger_rows, triggered_depth, clean_depth)
    ]
    return {
        "schema_version": "mode3-v6-3-confirmation-v1",
        "freeze_sha256": str(freeze_sha256),
        "token_id": artifact.token_id,
        "refit_performed": False,
        "fixed_center": True,
        "fixed_radius": True,
        "independent_text_units": len(trigger_rows),
        "coverage": coverage.to_dict(),
        "occupancy": occupancy,
        "migration": migration,
        "radial_occupancy": radial,
        "core_gates": core_gates,
        "p3_uniform_secondary": bool(uniform),
        "levels": levels,
        "radial_shift": {
            "ks_statistic": ks_statistic, "ks_pvalue": ks_pvalue,
            "wasserstein_distance": _wasserstein_1d(triggered_depth, benign_depth),
            "cliffs_delta": cliffs_delta,
            "triggered_median_depth": triggered_median,
            "benign_median_depth": benign_median,
            "median_shift": triggered_median - benign_median,
        },
        "source_position_intervals": interval_rows,
        "observations": observation_rows,
        "benign_depth_by_source": benign_depth_by_source,
    }


def paired_position_audit(
    artifact: FreezeArtifact,
    rows: Sequence[Mapping[str, Any]],
    vectors_by_position: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Non-IID robustness audit; never contributes three times to Confirm n."""
    cap = artifact.frozen_cap()
    if set(vectors_by_position) != {"prefix", "suffix", "random"}:
        raise ShapeMismatch("paired audit requires all three positions")
    membership = {position: cap.contains(values) for position, values in vectors_by_position.items()}
    if any(len(values) != len(rows) for values in membership.values()):
        raise ShapeMismatch("paired audit position lengths differ")
    all_inside = np.logical_and.reduce(list(membership.values()))
    any_inside = np.logical_or.reduce(list(membership.values()))
    return {
        "schema_version": "mode3-v6-3-paired-position-audit-v1",
        "texts": len(rows), "iid_units_added_to_confirm": 0,
        "per_position_coverage": {position: float(np.mean(values)) for position, values in membership.items()},
        "all_three_inside": float(np.mean(all_inside)),
        "any_position_inside": float(np.mean(any_inside)),
        "random_vectors_averaged": False,
    }
