"""Registered equal-source/equal-position angular geometry for V6.2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Tuple

import numpy as np

from sticky_lab.mode3_v6.geometry import (
    angular_distance,
    chord_distance_from_angle,
    normalize_rows,
    normalize_vector,
    spherical_mean,
)

from .errors import CandidateRejectedDegenerateCluster, CandidateRejectedRadius, ShapeMismatch


Stratum = Tuple[str, str]
POSITIONS = ("prefix", "suffix", "random")


def _validate_grid(values: Mapping[Stratum, np.ndarray]) -> dict[Stratum, np.ndarray]:
    if not values:
        raise ShapeMismatch("empty source x position vector grid")
    result = {(str(source), str(position)): normalize_rows(vectors) for (source, position), vectors in values.items()}
    sources = sorted({source for source, _ in result})
    missing = [(source, position) for source in sources for position in POSITIONS if (source, position) not in result]
    if missing:
        raise ShapeMismatch(f"incomplete source x position vector grid: {missing}")
    dimensions = {vectors.shape[1] for vectors in result.values()}
    if len(dimensions) != 1:
        raise ShapeMismatch("embedding dimensions differ across strata")
    return result


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(x) == 0 or len(x) != len(w) or np.any(w < 0) or not 0 < quantile < 1:
        raise ValueError("invalid weighted quantile")
    order = np.argsort(x, kind="stable")
    cumulative = np.cumsum(w[order])
    threshold = quantile * cumulative[-1]
    return float(x[order[min(len(order) - 1, int(np.searchsorted(cumulative, threshold, side="left")))]])


@dataclass(frozen=True)
class RobustSharedCenter:
    center: np.ndarray
    objective_worst_q90: float
    objective_average_q90: float
    restart: int
    iterations: int
    inlier_indices: dict[str, np.ndarray]
    restart_summaries: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "center": self.center.tolist(),
            "objective_worst_q90": self.objective_worst_q90,
            "objective_average_q90": self.objective_average_q90,
            "restart": self.restart,
            "iterations": self.iterations,
            "inlier_indices": {key: value.tolist() for key, value in self.inlier_indices.items()},
            "restart_summaries": list(self.restart_summaries),
        }


def fit_equal_strata_robust_center(
    vectors: Mapping[Stratum, np.ndarray],
    *,
    trim_fraction: float = 0.10,
    restarts: int = 20,
    maximum_iterations: int = 100,
    tolerance: float = 1e-6,
    seed: int = 0,
) -> RobustSharedCenter:
    """Fit one shared center with equal total mass per source x position stratum."""
    grid = _validate_grid(vectors)
    if not 0 <= trim_fraction < 0.5 or restarts < 4:
        raise ValueError("invalid robust center request")
    keys = sorted(grid)
    stratum_means = {key: spherical_mean(grid[key]) for key in keys}
    starts: list[np.ndarray] = [spherical_mean(np.stack(list(stratum_means.values())))]
    for position in POSITIONS:
        starts.append(spherical_mean(np.stack([value for (source, p), value in stratum_means.items() if p == position])))
    means = np.stack(list(stratum_means.values()))
    medoid_index = int(np.argmin(angular_distance(means, means).sum(axis=1)))
    starts.append(means[medoid_index])
    rng = np.random.default_rng(seed)
    while len(starts) < restarts:
        key = keys[(len(starts) - 5) % len(keys)]
        starts.append(grid[key][int(rng.integers(0, len(grid[key])))])

    best: RobustSharedCenter | None = None
    summaries: list[dict[str, Any]] = []
    candidates: list[tuple[np.ndarray, dict[str, np.ndarray], int, int, float, float]] = []
    for restart, initial in enumerate(starts[:restarts]):
        center = normalize_vector(initial)
        selected: dict[str, np.ndarray] = {}
        iteration = 0
        for iteration in range(1, maximum_iterations + 1):
            selected_vectors: list[np.ndarray] = []
            selected_weights: list[np.ndarray] = []
            for key in keys:
                distances = angular_distance(grid[key], center).reshape(-1)
                keep = max(2, int(math.ceil((1.0 - trim_fraction) * len(distances))))
                indices = np.argsort(distances, kind="stable")[:keep]
                selected[f"{key[0]}::{key[1]}"] = indices
                selected_vectors.append(grid[key][indices])
                selected_weights.append(np.full(keep, 1.0 / (len(keys) * keep)))
            updated = spherical_mean(np.concatenate(selected_vectors), np.concatenate(selected_weights))
            shift = float(angular_distance(updated[None, :], center).item())
            center = updated
            if shift <= tolerance:
                break
        q90 = [float(np.quantile(angular_distance(grid[key], center).reshape(-1), 0.90)) for key in keys]
        worst, average = max(q90), float(np.mean(q90))
        summaries.append({
            "restart": restart, "iterations": iteration,
            "worst_stratum_q90": worst, "average_stratum_q90": average,
        })
        candidates.append((center, dict(selected), restart, iteration, worst, average))
    center, selected, restart, iterations, worst, average = min(
        candidates, key=lambda item: (item[4], item[5], item[2])
    )
    best = RobustSharedCenter(
        center=center,
        objective_worst_q90=worst,
        objective_average_q90=average,
        restart=restart,
        iterations=iterations,
        inlier_indices=selected,
        restart_summaries=tuple(summaries),
    )
    return best


def calibrate_shared_radius(
    center: np.ndarray,
    vectors: Mapping[Stratum, np.ndarray],
    *,
    design_coverage: float = 0.92,
) -> tuple[float, dict[str, Any]]:
    """Use equal-source weighted position quantiles, then take the worst position."""
    grid = _validate_grid(vectors)
    sources = sorted({source for source, _ in grid})
    position_radii: dict[str, float] = {}
    source_quantiles: dict[str, dict[str, float]] = {}
    for position in POSITIONS:
        values: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        source_quantiles[position] = {}
        for source in sources:
            distances = angular_distance(grid[(source, position)], center).reshape(-1)
            values.append(distances)
            weights.append(np.full(len(distances), 1.0 / (len(sources) * len(distances))))
            source_quantiles[position][source] = float(np.quantile(distances, design_coverage))
        position_radii[position] = _weighted_quantile(
            np.concatenate(values), np.concatenate(weights), design_coverage
        )
    radius = max(position_radii.values())
    return radius, {
        "design_coverage": float(design_coverage),
        "position_radii": position_radii,
        "source_quantiles": source_quantiles,
        "shared_rule": "maximum_equal_source_position_quantile",
    }


@dataclass(frozen=True)
class FrozenCapModel:
    token_id: int
    token_text: str
    protocol: str
    centers: np.ndarray
    radii: np.ndarray
    design_coverage: float
    fit_role: str
    radius_role: str
    cap_count: int
    assignment_rule: str = "minimum_normalized_angular_distance"
    outlier_rule: str = "global_weighted_trim"

    def __post_init__(self) -> None:
        centers = normalize_rows(self.centers)
        radii = np.asarray(self.radii, dtype=np.float64).reshape(-1)
        if len(centers) != len(radii) or np.any(radii <= 0) or np.any(radii > math.pi):
            raise ValueError("invalid frozen cap model")
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "radii", radii)
        object.__setattr__(self, "cap_count", len(centers))

    def normalized_radius(self, vectors: np.ndarray) -> np.ndarray:
        return np.min(angular_distance(vectors, self.centers) / self.radii[None, :], axis=1)

    def contains(self, vectors: np.ndarray, multiplier: float = 1.0) -> np.ndarray:
        return self.normalized_radius(vectors) <= float(multiplier) + 1e-12

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["centers"] = self.centers.tolist()
        value["radii"] = self.radii.tolist()
        value["radius_degrees"] = np.degrees(self.radii).tolist()
        value["radius_chord"] = chord_distance_from_angle(self.radii).tolist()
        return value


def fit_single_cap(
    token_id: int,
    token_text: str,
    fit_vectors: Mapping[Stratum, np.ndarray],
    radius_vectors: Mapping[Stratum, np.ndarray],
    *,
    fit_role: str,
    radius_role: str,
    design_coverage: float,
    maximum_radius_degrees: float,
    trim_fraction: float,
    restarts: int,
    maximum_iterations: int,
    tolerance: float,
    seed: int,
) -> tuple[FrozenCapModel, dict[str, Any]]:
    fitted = fit_equal_strata_robust_center(
        fit_vectors, trim_fraction=trim_fraction, restarts=restarts,
        maximum_iterations=maximum_iterations, tolerance=tolerance, seed=seed,
    )
    radius, audit = calibrate_shared_radius(
        fitted.center, radius_vectors, design_coverage=design_coverage
    )
    if math.degrees(radius) > float(maximum_radius_degrees):
        raise CandidateRejectedRadius(
            f"single cap radius {math.degrees(radius):.6f} exceeds {maximum_radius_degrees} degrees"
        )
    cap = FrozenCapModel(
        token_id=int(token_id), token_text=str(token_text), protocol="P3_ST_FCA_Core",
        centers=fitted.center[None, :], radii=np.asarray([radius]),
        design_coverage=float(design_coverage), fit_role=fit_role,
        radius_role=radius_role, cap_count=1,
    )
    return cap, {"center_fit": fitted.to_dict(), "radius_calibration": audit}


def fit_position_cap(
    token_id: int,
    token_text: str,
    position: str,
    fit_vectors: Mapping[str, np.ndarray],
    radius_vectors: Mapping[str, np.ndarray],
    *,
    fit_role: str,
    radius_role: str,
    design_coverage: float,
    maximum_radius_degrees: float,
    trim_fraction: float,
    restarts: int,
    maximum_iterations: int,
    tolerance: float,
    seed: int,
    protocol: str,
) -> tuple[FrozenCapModel, dict[str, Any]]:
    """Fit a position-specific P1/P2 cap with equal total source mass."""
    if position not in POSITIONS or not fit_vectors or set(fit_vectors) != set(radius_vectors):
        raise ShapeMismatch("position cap requires matching non-empty source grids")
    # Reusing the robust shared-center solver on three identical logical views
    # is exactly equivalent to equal-source fitting for one position.
    fit_grid = {
        (str(source), logical): np.asarray(values)
        for source, values in fit_vectors.items() for logical in POSITIONS
    }
    fitted = fit_equal_strata_robust_center(
        fit_grid, trim_fraction=trim_fraction, restarts=restarts,
        maximum_iterations=maximum_iterations, tolerance=tolerance, seed=seed,
    )
    values: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    source_quantiles: dict[str, float] = {}
    for source in sorted(radius_vectors):
        distances = angular_distance(radius_vectors[source], fitted.center).reshape(-1)
        values.append(distances)
        weights.append(np.full(len(distances), 1.0 / (len(radius_vectors) * len(distances))))
        source_quantiles[str(source)] = float(np.quantile(distances, design_coverage))
    radius = _weighted_quantile(np.concatenate(values), np.concatenate(weights), design_coverage)
    if math.degrees(radius) > float(maximum_radius_degrees):
        raise CandidateRejectedRadius(
            f"{protocol}/{position} radius {math.degrees(radius):.6f} exceeds {maximum_radius_degrees} degrees"
        )
    model = FrozenCapModel(
        token_id=int(token_id), token_text=str(token_text),
        protocol=f"{protocol}:{position}", centers=fitted.center[None, :],
        radii=np.asarray([radius]), design_coverage=float(design_coverage),
        fit_role=fit_role, radius_role=radius_role, cap_count=1,
    )
    return model, {
        "position": position, "center_fit": fitted.to_dict(),
        "radius_calibration": {
            "design_coverage": design_coverage, "equal_source": True,
            "source_quantiles": source_quantiles, "radius": radius,
        },
    }


@dataclass(frozen=True)
class MultiCapFit:
    centers: np.ndarray
    assignments: dict[str, np.ndarray]
    cluster_mass: np.ndarray
    minimum_stratum_mass: np.ndarray
    objective: float
    restart: int


def fit_equal_strata_multicap(
    vectors: Mapping[Stratum, np.ndarray],
    cap_count: int,
    *,
    minimum_cluster_mass: float,
    minimum_stratum_cluster_mass: float,
    maximum_outlier_fraction: float,
    restarts: int,
    maximum_iterations: int,
    seed: int,
) -> MultiCapFit:
    grid = _validate_grid(vectors)
    if cap_count not in (2, 3, 4):
        raise ValueError("V6.2 multicap count must be 2, 3, or 4")
    keys = sorted(grid)
    all_vectors = np.concatenate([grid[key] for key in keys])
    weights = np.concatenate([
        np.full(len(grid[key]), 1.0 / (len(keys) * len(grid[key]))) for key in keys
    ])
    offsets: dict[Stratum, slice] = {}
    cursor = 0
    for key in keys:
        offsets[key] = slice(cursor, cursor + len(grid[key])); cursor += len(grid[key])
    rng = np.random.default_rng(seed)
    best: MultiCapFit | None = None
    for restart in range(restarts):
        centers = all_vectors[rng.choice(len(all_vectors), size=cap_count, replace=False)].copy()
        assignments = np.zeros(len(all_vectors), dtype=np.int64)
        outliers = np.zeros(len(all_vectors), dtype=bool)
        valid = True
        for _ in range(maximum_iterations):
            distances = angular_distance(all_vectors, centers)
            assignments = np.argmin(distances, axis=1)
            nearest = distances[np.arange(len(all_vectors)), assignments]
            cutoff = _weighted_quantile(nearest, weights, 1.0 - maximum_outlier_fraction)
            outliers = nearest > cutoff + 1e-12
            updated: list[np.ndarray] = []
            for cluster in range(cap_count):
                members = (assignments == cluster) & ~outliers
                mass = float(weights[members].sum())
                stratum_mass = [float(weights[offsets[key]][members[offsets[key]]].sum() * len(keys)) for key in keys]
                if mass < minimum_cluster_mass or min(stratum_mass) < minimum_stratum_cluster_mass:
                    valid = False; break
                updated.append(spherical_mean(all_vectors[members], weights[members]))
            if not valid:
                break
            updated_centers = np.stack(updated)
            shift = float(np.max(np.min(angular_distance(updated_centers, centers), axis=1)))
            centers = updated_centers
            if shift <= 1e-6:
                break
        if not valid:
            continue
        mass = np.asarray([weights[(assignments == c) & ~outliers].sum() for c in range(cap_count)])
        min_stratum = np.asarray([
            min(float(weights[offsets[key]][((assignments == c) & ~outliers)[offsets[key]]].sum() * len(keys)) for key in keys)
            for c in range(cap_count)
        ])
        objective = float(np.sum(weights[~outliers] * angular_distance(all_vectors, centers)[np.arange(len(all_vectors)), assignments][~outliers]))
        by_stratum = {f"{key[0]}::{key[1]}": assignments[offsets[key]].copy() for key in keys}
        candidate = MultiCapFit(centers, by_stratum, mass, min_stratum, objective, restart)
        if best is None or (candidate.objective, candidate.restart) < (best.objective, best.restart):
            best = candidate
    if best is None:
        raise CandidateRejectedDegenerateCluster("no multicap fit passed global and stratum mass gates")
    return best


def calibrate_multicap(
    centers: np.ndarray,
    vectors: Mapping[Stratum, np.ndarray],
    *,
    design_coverage: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    grid = _validate_grid(vectors)
    keys = sorted(grid)
    radii: list[float] = []
    cluster_audit: dict[str, Any] = {}
    for cluster in range(len(centers)):
        values: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        for key in keys:
            distances = angular_distance(grid[key], centers)
            assignment = np.argmin(distances, axis=1)
            selected = distances[assignment == cluster, cluster]
            if len(selected) < 2:
                raise CandidateRejectedDegenerateCluster(f"radius stratum {key} lacks mass for cap {cluster}")
            values.append(selected)
            weights.append(np.full(len(selected), 1.0 / (len(keys) * len(selected))))
        radius = _weighted_quantile(np.concatenate(values), np.concatenate(weights), design_coverage)
        radii.append(radius)
        cluster_audit[str(cluster)] = {"radius": radius, "strata": len(keys)}
    return np.asarray(radii), {"design_coverage": design_coverage, "clusters": cluster_audit}


def fit_multicap_model(
    token_id: int,
    token_text: str,
    fit_vectors: Mapping[Stratum, np.ndarray],
    radius_vectors: Mapping[Stratum, np.ndarray],
    cap_count: int,
    *,
    fit_role: str,
    radius_role: str,
    design_coverage: float,
    maximum_radius_degrees: float,
    minimum_cluster_mass: float,
    minimum_stratum_cluster_mass: float,
    maximum_outlier_fraction: float,
    restarts: int,
    maximum_iterations: int,
    seed: int,
) -> tuple[FrozenCapModel, dict[str, Any]]:
    fitted = fit_equal_strata_multicap(
        fit_vectors, cap_count,
        minimum_cluster_mass=minimum_cluster_mass,
        minimum_stratum_cluster_mass=minimum_stratum_cluster_mass,
        maximum_outlier_fraction=maximum_outlier_fraction,
        restarts=restarts, maximum_iterations=maximum_iterations, seed=seed,
    )
    radii, audit = calibrate_multicap(
        fitted.centers, radius_vectors, design_coverage=design_coverage
    )
    if float(np.max(np.degrees(radii))) > float(maximum_radius_degrees):
        raise CandidateRejectedRadius("multicap maximum radius exceeds anti-triviality limit")
    cap = FrozenCapModel(
        token_id=int(token_id), token_text=str(token_text),
        protocol="ST_mFCA_secondary", centers=fitted.centers, radii=radii,
        design_coverage=float(design_coverage), fit_role=fit_role,
        radius_role=radius_role, cap_count=cap_count,
    )
    return cap, {
        "center_fit": {
            "cluster_mass": fitted.cluster_mass.tolist(),
            "minimum_stratum_mass": fitted.minimum_stratum_mass.tolist(),
            "objective": fitted.objective, "restart": fitted.restart,
            "assignments": {key: value.tolist() for key, value in fitted.assignments.items()},
        },
        "radius_calibration": audit,
    }


def center_drift(previous: FrozenCapModel, current: FrozenCapModel) -> float:
    distances = angular_distance(previous.centers, current.centers)
    return float(max(np.max(np.min(distances, axis=0)), np.max(np.min(distances, axis=1))))
