"""Prefix/suffix shared-center geometry for V7."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Tuple

import numpy as np

from sticky_lab.mode3_v6_3.errors import (
    CandidateRejectedDegenerateFit,
    NumericalNonFinite,
    ShapeMismatch,
)

from .config import canonical_sha256


Stratum = Tuple[str, str]
POSITIONS = ("prefix", "suffix")


def normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) == 0 or not np.all(np.isfinite(matrix)):
        raise NumericalNonFinite("vectors must be a finite non-empty matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise NumericalNonFinite("zero-norm embedding")
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
        mass = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(mass) != len(matrix) or np.any(mass < 0) or mass.sum() <= 0:
            raise ShapeMismatch("invalid spherical-mean weights")
        result = np.sum(matrix * (mass / mass.sum())[:, None], axis=0)
    return normalize_vector(result)


def validate_prefix_suffix_grid(
    vectors: Mapping[Stratum, np.ndarray],
) -> dict[Stratum, np.ndarray]:
    if not vectors:
        raise ShapeMismatch("empty source-position grid")
    grid = {
        (str(source), str(position)): normalize_rows(matrix)
        for (source, position), matrix in vectors.items()
    }
    observed_positions = {position for _, position in grid}
    if observed_positions != set(POSITIONS):
        raise ShapeMismatch(
            f"V7 requires exactly prefix/suffix strata, observed {sorted(observed_positions)}"
        )
    sources = sorted({source for source, _ in grid})
    missing = [
        (source, position)
        for source in sources
        for position in POSITIONS
        if (source, position) not in grid
    ]
    if missing:
        raise ShapeMismatch(f"incomplete prefix/suffix grid: {missing}")
    dimensions = {matrix.shape[1] for matrix in grid.values()}
    if len(dimensions) != 1:
        raise ShapeMismatch("embedding dimensions differ across strata")
    return grid


@dataclass(frozen=True)
class RobustSharedCenter:
    center: np.ndarray
    restart: int
    iterations: int
    worst_stratum_q90: float
    mean_stratum_q90: float
    inlier_indices: dict[str, np.ndarray]
    restart_summaries: tuple[dict[str, Any], ...]

    @property
    def center_sha256(self) -> str:
        return canonical_sha256(
            {
                "dtype": "float64",
                "dimension": int(len(self.center)),
                "center": self.center.tolist(),
            }
        )

    @property
    def restart_spread(self) -> float:
        values = [float(row["worst_stratum_q90"]) for row in self.restart_summaries]
        return max(values) - min(values)

    def to_dict(self, *, include_indices: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "center": self.center.tolist(),
            "center_sha256": self.center_sha256,
            "restart": self.restart,
            "iterations": self.iterations,
            "worst_stratum_q90": self.worst_stratum_q90,
            "mean_stratum_q90": self.mean_stratum_q90,
            "restart_spread_radians": self.restart_spread,
            "restart_summaries": list(self.restart_summaries),
            "weighting": "equal total mass per source-prefix/suffix stratum",
            "trim_fraction": 0.10,
        }
        encoded = {key: indices.tolist() for key, indices in self.inlier_indices.items()}
        if include_indices:
            value["inlier_indices"] = encoded
        else:
            value["inlier_counts"] = {key: len(indices) for key, indices in self.inlier_indices.items()}
            value["inlier_indices_sha256"] = canonical_sha256(encoded)
        return value


def fit_robust_shared_center(
    vectors: Mapping[Stratum, np.ndarray],
    *,
    trim_fraction: float = 0.10,
    restarts: int = 20,
    maximum_iterations: int = 100,
    tolerance: float = 1e-6,
    seed: int = 0,
) -> RobustSharedCenter:
    """Fit one triggered-only center with equal source/position mass."""

    grid = validate_prefix_suffix_grid(vectors)
    if not 0 <= float(trim_fraction) < 0.5 or int(restarts) < 1:
        raise ValueError("invalid robust-center settings")
    keys = sorted(grid)
    means = {key: spherical_mean(grid[key]) for key in keys}
    starts: list[np.ndarray] = [spherical_mean(np.stack(list(means.values())))]
    for position in POSITIONS:
        starts.append(
            spherical_mean(
                np.stack([value for (_, pos), value in means.items() if pos == position])
            )
        )
    mean_matrix = np.stack(list(means.values()))
    medoid = int(np.argmin(angular_distance(mean_matrix, mean_matrix).sum(axis=1)))
    starts.append(mean_matrix[medoid])
    deterministic_starts = len(starts)
    rng = np.random.default_rng(int(seed))
    while len(starts) < int(restarts):
        key = keys[(len(starts) - deterministic_starts) % len(keys)]
        starts.append(grid[key][int(rng.integers(0, len(grid[key])))])

    candidates: list[
        tuple[float, float, int, int, np.ndarray, dict[str, np.ndarray]]
    ] = []
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
        quantiles = [
            float(np.quantile(angular_distance(grid[key], center[None, :]), 0.90))
            for key in keys
        ]
        worst = max(quantiles)
        mean = float(np.mean(quantiles))
        summaries.append(
            {
                "restart": restart,
                "iterations": iteration,
                "worst_stratum_q90": worst,
                "mean_stratum_q90": mean,
                "center": center.tolist(),
            }
        )
        candidates.append((worst, mean, restart, iteration, center, dict(selected)))
    if not candidates:
        raise CandidateRejectedDegenerateFit("no center restart completed")
    worst, mean, restart, iterations, center, selected = min(
        candidates, key=lambda row: (row[0], row[1], row[2])
    )
    return RobustSharedCenter(
        center,
        restart,
        iterations,
        worst,
        mean,
        selected,
        tuple(summaries),
    )


def fit_e_star(benign_vectors: np.ndarray) -> np.ndarray:
    """Freeze the independent benign mean direction used only for diagnostics."""

    return spherical_mean(benign_vectors)


def angle_between(left: np.ndarray, right: np.ndarray) -> float:
    return float(angular_distance(normalize_vector(left)[None, :], normalize_vector(right)[None, :]).item())


def center_bootstrap_drift(
    vectors: Mapping[Stratum, np.ndarray],
    reference: np.ndarray,
    *,
    samples: int,
    seed: int,
    trim_fraction: float = 0.10,
    restarts: int = 5,
    maximum_iterations: int = 100,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Source/position-stratified document bootstrap for center uncertainty."""

    grid = validate_prefix_suffix_grid(vectors)
    rng = np.random.default_rng(int(seed))
    drift: list[float] = []
    failures: list[dict[str, Any]] = []
    for index in range(int(samples)):
        boot = {
            key: matrix[rng.integers(0, len(matrix), size=len(matrix))]
            for key, matrix in grid.items()
        }
        try:
            fitted = fit_robust_shared_center(
                boot,
                trim_fraction=trim_fraction,
                restarts=restarts,
                maximum_iterations=maximum_iterations,
                tolerance=tolerance,
                seed=int(seed) + index + 1,
            )
        except (CandidateRejectedDegenerateFit, NumericalNonFinite) as error:
            failures.append(
                {
                    "sample": index,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        else:
            drift.append(angle_between(fitted.center, reference))
    values = np.asarray(drift, dtype=np.float64)
    return {
        "samples": int(samples),
        "successful_samples": len(drift),
        "failed_samples": len(failures),
        "failures": failures,
        "status": "COMPLETE" if not failures else "PARTIAL_REPORT_ONLY",
        "seed": int(seed),
        "mean_radians": float(values.mean()) if len(values) else None,
        "median_radians": float(np.median(values)) if len(values) else None,
        "q95_radians": float(np.quantile(values, 0.95)) if len(values) else None,
        "max_radians": float(values.max()) if len(values) else None,
        "drift_radians": values.tolist(),
    }


@dataclass(frozen=True)
class FrozenOperatingPoint:
    token_id: int
    token_text: str
    center: np.ndarray
    beta: float
    radius: float
    fit_role_sha256: str
    calibration_role_sha256: str
    select_role_sha256: str
    stage: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", normalize_vector(self.center))
        if not 0 < float(self.beta) < 1:
            raise NumericalNonFinite("invalid frozen occupancy budget")
        radius = float(self.radius)
        if not math.isfinite(radius) or radius <= 0 or radius > math.pi:
            raise NumericalNonFinite("invalid frozen cap radius")
        object.__setattr__(self, "radius", radius)

    @property
    def radius_degrees(self) -> float:
        return float(math.degrees(self.radius))

    @property
    def center_sha256(self) -> str:
        return canonical_sha256(
            {
                "dtype": "float64",
                "dimension": int(len(self.center)),
                "center": self.center.tolist(),
            }
        )

    def distances(self, values: np.ndarray) -> np.ndarray:
        return angular_distance(values, self.center[None, :]).reshape(-1)

    def contains(self, values: np.ndarray) -> np.ndarray:
        return self.distances(values) <= self.radius

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["center"] = self.center.tolist()
        value["radius_degrees"] = self.radius_degrees
        value["center_sha256"] = self.center_sha256
        value["positions"] = list(POSITIONS)
        value["shared_center"] = True
        value["shared_radius"] = True
        value["protocol"] = "PS_OC_ST_FCA"
        value["operating_point_sha256"] = canonical_sha256(value)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenOperatingPoint":
        return cls(
            int(value["token_id"]),
            str(value["token_text"]),
            np.asarray(value["center"], dtype=np.float64),
            float(value["beta"]),
            float(value["radius"]),
            str(value["fit_role_sha256"]),
            str(value["calibration_role_sha256"]),
            str(value["select_role_sha256"]),
            str(value["stage"]),
        )
