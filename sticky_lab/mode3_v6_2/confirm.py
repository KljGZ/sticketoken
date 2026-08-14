"""Pure frozen-geometry confirmation. This module contains no fit operation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from .errors import ManifestMismatch, NumericalNonFinite, ShapeMismatch
from .freeze import FreezeArtifact
from .statistics import (
    migration_certificates,
    radial_occupancy_summary,
    simultaneous_balanced_bounds,
    simultaneous_source_occupancy,
)


Stratum = Tuple[str, str]


def _normalize(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not np.all(np.isfinite(x)):
        raise NumericalNonFinite("confirmation vectors must be finite 2-D arrays")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise NumericalNonFinite("confirmation contains a zero-norm vector")
    return x / norms


def _normalized_radius(artifact: FreezeArtifact, values: np.ndarray) -> np.ndarray:
    x = _normalize(values)
    centers = _normalize(artifact.cap.centers)
    angles = np.arccos(np.clip(x @ centers.T, -1.0, 1.0))
    return np.min(angles / artifact.cap.radii[None, :], axis=1)


def _verify_role_hashes(artifact: FreezeArtifact, observed: Mapping[str, str]) -> None:
    for role, digest in observed.items():
        expected = artifact.data_role_hashes.get(role)
        if expected is None or expected != digest:
            raise ManifestMismatch(f"confirmation role hash mismatch for {role}")


def confirm_frozen_cap(
    artifact: FreezeArtifact,
    *,
    triggered: Mapping[Stratum, np.ndarray],
    paired_clean: Mapping[Stratum, np.ndarray],
    independent_benign: Mapping[str, np.ndarray],
    observed_role_hashes: Mapping[str, str],
    radial_multipliers: Sequence[float],
    familywise_alpha: float = 0.05,
    random_robustness: Mapping[tuple[str, int], np.ndarray] | None = None,
) -> dict[str, Any]:
    """Apply an immutable cap to independent confirmation roles only."""
    _verify_role_hashes(artifact, observed_role_hashes)
    if set(triggered) != set(paired_clean):
        raise ShapeMismatch("triggered and paired-clean confirmation strata differ")
    triggered_inside: dict[Stratum, np.ndarray] = {}
    clean_inside: dict[Stratum, np.ndarray] = {}
    triggered_depth: list[np.ndarray] = []
    clean_depth: list[np.ndarray] = []
    for key in sorted(triggered):
        if len(triggered[key]) != len(paired_clean[key]):
            raise ShapeMismatch(f"paired confirmation length mismatch for {key}")
        tr = _normalized_radius(artifact, triggered[key])
        cl = _normalized_radius(artifact, paired_clean[key])
        triggered_inside[key] = tr <= 1.0 + 1e-12
        clean_inside[key] = cl <= 1.0 + 1e-12
        triggered_depth.append(tr); clean_depth.append(cl)
    coverage = simultaneous_balanced_bounds(
        triggered_inside, familywise_alpha=familywise_alpha
    )
    migration = migration_certificates(
        clean_inside, triggered_inside, familywise_alpha=familywise_alpha
    )
    benign_depth = {
        str(source): _normalized_radius(artifact, values)
        for source, values in sorted(independent_benign.items())
    }
    occupancy = simultaneous_source_occupancy(
        {source: values <= 1.0 + 1e-12 for source, values in benign_depth.items()},
        familywise_alpha=familywise_alpha,
    )
    pooled_benign = np.concatenate(list(benign_depth.values()))
    radial = radial_occupancy_summary(
        pooled_benign, radial_multipliers, confidence=1.0 - familywise_alpha
    )
    thresholds = artifact.certification_thresholds
    core_gates = {
        "balanced_coverage": coverage.balanced_lower > thresholds["p3_balanced_coverage_lcb"],
        "worst_position": coverage.worst_position_lower > thresholds["worst_position_coverage_lcb"],
        "worst_source": coverage.worst_source_lower > thresholds["worst_source_coverage_lcb"],
        "benign_core": occupancy["worst_source_ucb"] < thresholds["independent_benign_occupancy_ucb"],
        "outside_to_inside": migration["outside_to_inside"]["balanced_lower"] >= thresholds["outside_to_inside_lcb"],
        "outside_origin": migration["conditional_outside_origin"]["balanced_lower"] >= thresholds["conditional_outside_origin_lcb"],
    }
    moat_row = next((row for row in radial["curve"] if np.isclose(row["multiplier"], 1.10)), None)
    depths = np.concatenate(triggered_depth)
    benign_median = float(np.median(pooled_benign))
    triggered_median = float(np.median(depths))
    levels = {
        "A_ST_RadialShift": triggered_median < benign_median,
        "B_ST_FCA_Core": all(core_gates.values()),
        "C_ST_FCA_Moat": all(core_gates.values()) and moat_row is not None and moat_row["ucb"] < thresholds["moat_occupancy_1_10_ucb"],
        "D_ST_FCA_Basin": all(core_gates.values()) and radial["lambda_star"] >= thresholds["basin_lambda_star"] and radial["occupancy_auc_1_1_5"] <= thresholds["basin_occupancy_auc_1_1_5"],
        "E_ST_Central_Collapse": all(core_gates.values()) and triggered_median <= thresholds["central_collapse_median_depth"],
    }
    uniform_secondary = min(bound.lower for bound in coverage.strata.values()) > thresholds["p3_uniform_secondary_lcb"]
    robustness: dict[str, Any] = {}
    if random_robustness:
        grouped: dict[tuple[str, str], np.ndarray] = {
            (str(source), f"random:{int(replicate)}"): _normalized_radius(artifact, values) <= 1.0 + 1e-12
            for (source, replicate), values in random_robustness.items()
        }
        robustness = simultaneous_balanced_bounds(
            grouped, familywise_alpha=familywise_alpha
        ).to_dict()
    return {
        "schema_version": "mode3-v6-2-confirmation-v1",
        "freeze_sha256": artifact.freeze_sha256,
        "refit_performed": False,
        "coverage": coverage.to_dict(),
        "occupancy": occupancy,
        "migration": migration,
        "radial_benign": radial,
        "triggered_median_normalized_depth": triggered_median,
        "paired_clean_median_normalized_depth": float(np.median(np.concatenate(clean_depth))),
        "benign_median_normalized_depth": benign_median,
        "core_gates": core_gates,
        "uniform_secondary_gate": uniform_secondary,
        "secondary_uniform_certificate": bool(all(core_gates.values()) and uniform_secondary),
        "levels": levels,
        "random_replicate_robustness": robustness,
    }
