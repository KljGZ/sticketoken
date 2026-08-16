"""Registered equal-source/equal-position single-cap geometry for V6.3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Tuple

import numpy as np

from .config import canonical_sha256
from .errors import (
    CandidateRejectedDegenerateFit,
    CandidateRejectedRadius,
    NumericalNonFinite,
    ShapeMismatch,
)


Stratum = Tuple[str, str]
POSITIONS = ("prefix", "suffix", "random")


def normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) == 0 or not np.all(np.isfinite(matrix)):
        raise NumericalNonFinite("vectors must be a finite non-empty matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise NumericalNonFinite("zero-norm vector")
    return matrix / norms


def normalize_vector(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.all(np.isfinite(vector)) or norm <= 1e-12:
        raise NumericalNonFinite("invalid center vector")
    return vector / norm


def angular_distance(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    left = normalize_rows(values)
    right = normalize_rows(centers)
    return np.arccos(np.clip(left @ right.T, -1.0, 1.0))


def spherical_mean(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    matrix = normalize_rows(values)
    if weights is None:
        result = matrix.mean(axis=0)
    else:
        weight = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(weight) != len(matrix) or np.any(weight < 0) or weight.sum() <= 0:
            raise ShapeMismatch("invalid spherical-mean weights")
        result = np.sum(matrix * (weight / weight.sum())[:, None], axis=0)
    return normalize_vector(result)


def _validate_grid(values: Mapping[Stratum, np.ndarray]) -> dict[Stratum, np.ndarray]:
    if not values:
        raise ShapeMismatch("empty source-position grid")
    grid = {(str(source), str(position)): normalize_rows(matrix) for (source, position), matrix in values.items()}
    sources = sorted({source for source, _ in grid})
    missing = [(source, position) for source in sources for position in POSITIONS if (source, position) not in grid]
    if missing:
        raise ShapeMismatch(f"incomplete source-position grid: {missing}")
    dimensions = {matrix.shape[1] for matrix in grid.values()}
    if len(dimensions) != 1:
        raise ShapeMismatch("embedding dimensions differ across strata")
    return grid


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    sample = np.asarray(values, dtype=np.float64).reshape(-1)
    mass = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(sample) == 0 or len(sample) != len(mass) or np.any(mass < 0) or not 0 < quantile < 1:
        raise ValueError("invalid weighted quantile")
    order = np.argsort(sample, kind="stable")
    cumulative = np.cumsum(mass[order])
    index = int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left"))
    return float(sample[order[min(index, len(order) - 1)]])


@dataclass(frozen=True)
class RobustCenter:
    center: np.ndarray
    restart: int
    iterations: int
    worst_stratum_q90: float
    mean_stratum_q90: float
    inlier_indices: dict[str, np.ndarray]
    restart_summaries: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "center": self.center.tolist(),
            "restart": self.restart,
            "iterations": self.iterations,
            "worst_stratum_q90": self.worst_stratum_q90,
            "mean_stratum_q90": self.mean_stratum_q90,
            "inlier_indices": {key: value.tolist() for key, value in self.inlier_indices.items()},
            "restart_summaries": list(self.restart_summaries),
            "weighting": "equal total mass per source-position stratum",
        }


def fit_robust_shared_center(
    vectors: Mapping[Stratum, np.ndarray],
    *,
    trim_fraction: float = 0.10,
    restarts: int = 20,
    maximum_iterations: int = 100,
    tolerance: float = 1e-6,
    seed: int = 0,
) -> RobustCenter:
    """Fit the registered P3 shared center; each stratum has equal mass."""
    grid = _validate_grid(vectors)
    if not 0 <= float(trim_fraction) < 0.5 or int(restarts) < 1:
        raise ValueError("invalid robust-center settings")
    keys = sorted(grid)
    means = {key: spherical_mean(grid[key]) for key in keys}
    starts: list[np.ndarray] = [spherical_mean(np.stack(list(means.values())))]
    for position in POSITIONS:
        starts.append(spherical_mean(np.stack([value for (source, pos), value in means.items() if pos == position])))
    mean_matrix = np.stack(list(means.values()))
    medoid = int(np.argmin(angular_distance(mean_matrix, mean_matrix).sum(axis=1)))
    starts.append(mean_matrix[medoid])
    rng = np.random.default_rng(int(seed))
    while len(starts) < int(restarts):
        key = keys[(len(starts) - 5) % len(keys)]
        starts.append(grid[key][int(rng.integers(0, len(grid[key])))])
    candidates: list[tuple[float, float, int, int, np.ndarray, dict[str, np.ndarray]]] = []
    summaries: list[dict[str, Any]] = []
    for restart, start in enumerate(starts[: int(restarts)]):
        center = normalize_vector(start)
        selected: dict[str, np.ndarray] = {}
        iteration = 0
        for iteration in range(1, int(maximum_iterations) + 1):
            kept_vectors: list[np.ndarray] = []
            kept_weights: list[np.ndarray] = []
            for key in keys:
                distances = angular_distance(grid[key], center[None, :]).reshape(-1)
                count = max(2, int(math.ceil((1.0 - float(trim_fraction)) * len(distances))))
                indices = np.argsort(distances, kind="stable")[:count]
                selected[f"{key[0]}::{key[1]}"] = indices
                kept_vectors.append(grid[key][indices])
                kept_weights.append(np.full(count, 1.0 / (len(keys) * count)))
            updated = spherical_mean(np.concatenate(kept_vectors), np.concatenate(kept_weights))
            shift = float(angular_distance(updated[None, :], center[None, :]).item())
            center = updated
            if shift <= float(tolerance):
                break
        q90 = [float(np.quantile(angular_distance(grid[key], center[None, :]), 0.90)) for key in keys]
        worst = max(q90)
        mean = float(np.mean(q90))
        summaries.append({"restart": restart, "iterations": iteration, "worst_stratum_q90": worst, "mean_stratum_q90": mean})
        candidates.append((worst, mean, restart, iteration, center, dict(selected)))
    if not candidates:
        raise CandidateRejectedDegenerateFit("no robust-center restart completed")
    worst, mean, restart, iterations, center, selected = min(candidates, key=lambda row: (row[0], row[1], row[2]))
    return RobustCenter(center, restart, iterations, worst, mean, selected, tuple(summaries))


def calibrate_shared_radius(
    center: np.ndarray,
    vectors: Mapping[Stratum, np.ndarray],
    *,
    design_quantile: float = 0.92,
) -> tuple[float, dict[str, Any]]:
    """Calibrate on radius-only data, source-balanced within each position."""
    grid = _validate_grid(vectors)
    sources = sorted({source for source, _ in grid})
    position_radii: dict[str, float] = {}
    source_quantiles: dict[str, dict[str, float]] = {}
    for position in POSITIONS:
        samples: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        source_quantiles[position] = {}
        for source in sources:
            distances = angular_distance(grid[(source, position)], center[None, :]).reshape(-1)
            samples.append(distances)
            weights.append(np.full(len(distances), 1.0 / (len(sources) * len(distances))))
            source_quantiles[position][source] = float(np.quantile(distances, design_quantile))
        position_radii[position] = weighted_quantile(np.concatenate(samples), np.concatenate(weights), design_quantile)
    radius = max(position_radii.values())
    return radius, {
        "design_quantile": float(design_quantile),
        "position_radii": position_radii,
        "source_quantiles": source_quantiles,
        "shared_rule": "max_position_source_balanced_quantile",
        "certification_data_used": False,
    }


@dataclass(frozen=True)
class FrozenCap:
    token_id: int
    token_text: str
    center: np.ndarray
    radius: float
    fit_role_sha256: str
    radius_role_sha256: str
    stage: str
    design_quantile: float = 0.92

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", normalize_vector(self.center))
        radius = float(self.radius)
        if not math.isfinite(radius) or radius <= 0 or radius > math.pi:
            raise NumericalNonFinite("invalid cap radius")
        object.__setattr__(self, "radius", radius)

    @property
    def radius_degrees(self) -> float:
        return float(math.degrees(self.radius))

    def normalized_radius(self, values: np.ndarray) -> np.ndarray:
        return angular_distance(values, self.center[None, :]).reshape(-1) / self.radius

    def contains(self, values: np.ndarray, multiplier: float = 1.0) -> np.ndarray:
        return self.normalized_radius(values) <= float(multiplier) + 1e-12

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["center"] = self.center.tolist()
        value["radius_degrees"] = self.radius_degrees
        value["cap_count"] = 1
        value["protocol"] = "P3_ST_FCA_CORE"
        value["cap_sha256"] = canonical_sha256(value)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenCap":
        return cls(
            int(value["token_id"]), str(value["token_text"]),
            np.asarray(value["center"], dtype=np.float64), float(value["radius"]),
            str(value["fit_role_sha256"]), str(value["radius_role_sha256"]),
            str(value["stage"]), float(value.get("design_quantile", 0.92)),
        )


def fit_single_cap(
    token_id: int,
    token_text: str,
    fit_vectors: Mapping[Stratum, np.ndarray],
    radius_vectors: Mapping[Stratum, np.ndarray],
    *,
    fit_role_sha256: str,
    radius_role_sha256: str,
    stage: str,
    trim_fraction: float,
    design_quantile: float,
    maximum_radius_degrees: float,
    restarts: int,
    maximum_iterations: int,
    tolerance: float,
    seed: int,
) -> tuple[FrozenCap, dict[str, Any]]:
    fitted = fit_robust_shared_center(
        fit_vectors, trim_fraction=trim_fraction, restarts=restarts,
        maximum_iterations=maximum_iterations, tolerance=tolerance, seed=seed,
    )
    radius, radius_audit = calibrate_shared_radius(
        fitted.center, radius_vectors, design_quantile=design_quantile
    )
    if math.degrees(radius) > float(maximum_radius_degrees):
        raise CandidateRejectedRadius(
            f"radius {math.degrees(radius):.8f} exceeds {maximum_radius_degrees} degrees"
        )
    model = FrozenCap(
        int(token_id), str(token_text), fitted.center, radius,
        str(fit_role_sha256), str(radius_role_sha256), str(stage), float(design_quantile),
    )
    return model, {"center_fit": fitted.to_dict(), "radius_calibration": radius_audit}


def center_drift(previous: FrozenCap, current: FrozenCap) -> float:
    return float(angular_distance(previous.center[None, :], current.center[None, :]).item())
