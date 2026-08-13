"""Common full-search and certification metrics across candidate sources."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import numpy as np

from .geometry import FrozenCap, angular_distance, conformal_radius, equal_position_center, fit_robust_single_center
from .statistics import clopper_pearson_lower, clopper_pearson_upper, migration_table, radial_profile, radial_depth_summary


@dataclass(frozen=True)
class CandidateMetrics:
    token_id: int
    token_text: str
    source_tracks: tuple[str, ...]
    protocol: str
    cap_count: int
    radius_radians: float
    radius_degrees: float
    triggered_coverage: float
    benign_occupancy: float
    search_margin_m90_1: float
    search_margin_m95_05: float
    outside_to_inside: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _position_balanced(values: Mapping[str, np.ndarray]) -> np.ndarray:
    required = ("prefix", "suffix", "random")
    if set(values) != set(required):
        raise ValueError("P3 requires prefix/suffix/random")
    count = min(len(np.asarray(values[name])) for name in required)
    # Interleave equal counts to make any empirical mean exactly position-balanced.
    return np.stack([np.asarray(values[name])[:count] for name in required], axis=1).reshape(-1, np.asarray(values[required[0]]).shape[-1])


def fit_and_calibrate_single_cap(
    token_id: int,
    token_text: str,
    protocol: str,
    triggered_fit: np.ndarray | Mapping[str, np.ndarray],
    triggered_calibration: np.ndarray | Mapping[str, np.ndarray],
    *,
    coverage: float,
    maximum_radius_degrees: float,
    source_tracks: tuple[str, ...] = (),
) -> FrozenCap:
    if protocol == "P3_shared":
        if not isinstance(triggered_fit, Mapping) or not isinstance(triggered_calibration, Mapping):
            raise ValueError("P3 inputs must be position mappings")
        center = equal_position_center(dict(triggered_fit))
        calibration = _position_balanced(triggered_calibration)
    else:
        if isinstance(triggered_fit, Mapping) or isinstance(triggered_calibration, Mapping):
            raise ValueError("P1 inputs must be arrays")
        center = fit_robust_single_center(triggered_fit, target_coverage=coverage).center
        calibration = np.asarray(triggered_calibration)
    distances = angular_distance(calibration, center).reshape(-1)
    radius = conformal_radius(distances, coverage)
    if math.degrees(radius) > maximum_radius_degrees:
        raise RuntimeError(f"anti-triviality radius exceeded: {math.degrees(radius):.4f} degrees")
    return FrozenCap(token_id, token_text, protocol, center[None, :], np.asarray([radius]), coverage, "cap_fit", "cap_calibration", 1)


def candidate_metrics(
    cap: FrozenCap,
    triggered: np.ndarray,
    paired_clean: np.ndarray,
    independent_benign: np.ndarray,
    source_tracks: tuple[str, ...],
) -> CandidateMetrics:
    triggered_radius = cap.normalized_radius(triggered)
    clean_radius = cap.normalized_radius(paired_clean)
    benign_radius = cap.normalized_radius(independent_benign)
    migration = migration_table(clean_radius <= 1, triggered_radius <= 1)
    center = cap.centers[0]
    triggered_similarity = np.cos(angular_distance(triggered, center).reshape(-1))
    benign_similarity = np.cos(angular_distance(independent_benign, center).reshape(-1))
    return CandidateMetrics(
        cap.token_id, cap.token_text, tuple(sorted(set(source_tracks))), cap.protocol, cap.cap_count,
        float(cap.radii.max()), float(np.degrees(cap.radii.max())), float(np.mean(triggered_radius <= 1)),
        float(np.mean(benign_radius <= 1)),
        float(np.quantile(triggered_similarity, 0.10) - np.quantile(benign_similarity, 0.99)),
        float(np.quantile(triggered_similarity, 0.05) - np.quantile(benign_similarity, 0.995)),
        migration.outside_to_inside / migration.total,
    )


def certify_frozen_cap(
    cap: FrozenCap,
    triggered: np.ndarray,
    paired_clean: np.ndarray,
    independent_benign: np.ndarray,
    *,
    confidence: float,
    coverage_lcb_threshold: float,
    occupancy_ucb_threshold: float,
    outside_to_inside_lcb_threshold: float,
    conditional_outside_origin_lcb_threshold: float,
    radial_multipliers: list[float],
) -> dict[str, object]:
    tr = cap.normalized_radius(triggered)
    cl = cap.normalized_radius(paired_clean)
    be = cap.normalized_radius(independent_benign)
    ti = tr <= 1
    ci = cl <= 1
    table = migration_table(ci, ti)
    coverage_lcb = clopper_pearson_lower(int(ti.sum()), len(ti), confidence)
    occupancy_ucb = clopper_pearson_upper(int((be <= 1).sum()), len(be), confidence)
    oi_lcb = clopper_pearson_lower(table.outside_to_inside, table.total, confidence)
    triggered_inside = table.outside_to_inside + table.inside_to_inside
    conditional_lcb = clopper_pearson_lower(table.outside_to_inside, triggered_inside, confidence) if triggered_inside else 0.0
    gates = {
        "coverage": coverage_lcb > coverage_lcb_threshold,
        "low_core_occupancy": occupancy_ucb < occupancy_ucb_threshold,
        "outside_to_inside": oi_lcb >= outside_to_inside_lcb_threshold,
        "conditional_outside_origin": conditional_lcb >= conditional_outside_origin_lcb_threshold,
    }
    return {
        "refit_performed": False,
        "cap": cap.to_json(),
        "counts": {
            "triggered": len(tr), "triggered_inside": int(ti.sum()),
            "paired_clean": len(cl), "paired_clean_inside": int(ci.sum()),
            "independent_benign": len(be), "independent_benign_inside": int((be <= 1).sum()),
        },
        "migration": dict(asdict(table), **table.proportions()),
        "bounds": {
            "coverage_lcb": coverage_lcb, "benign_occupancy_ucb": occupancy_ucb,
            "outside_to_inside_lcb": oi_lcb, "conditional_outside_origin_lcb": conditional_lcb,
        },
        "gates": gates,
        "certified": all(gates.values()),
        "radial": {
            "triggered": radial_profile(tr, radial_multipliers),
            "paired_clean": radial_profile(cl, radial_multipliers),
            "independent_benign": radial_profile(be, radial_multipliers),
        },
        "raw_normalized_radius": {
            "triggered": tr.tolist(),
            "paired_clean": cl.tolist(),
            "independent_benign": be.tolist(),
        },
        "depth": radial_depth_summary(tr, be),
    }
