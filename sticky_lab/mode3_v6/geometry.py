"""High-dimensional spherical-cap geometry for V6.

All formal calculations operate on normalized encoder outputs in the original
embedding dimension.  Reduced coordinates never enter this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence

import numpy as np


EPS = 1e-12


def normalize_rows(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("vectors must be a 2-D array")
    if not np.all(np.isfinite(x)):
        raise ValueError("vectors contain non-finite values")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= EPS):
        raise ValueError("zero-norm encoder output")
    return x / norms


def normalize_vector(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(x))
    if not np.isfinite(norm) or norm <= EPS:
        raise ValueError("invalid cap center")
    return x / norm


def angular_distance(vectors: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Return great-circle angle(s), in radians, for unit-normalized vectors."""
    x = normalize_rows(vectors)
    c = np.asarray(centers, dtype=np.float64)
    if c.ndim == 1:
        c = c[None, :]
    c = normalize_rows(c)
    return np.arccos(np.clip(x @ c.T, -1.0, 1.0))


def chord_distance_from_angle(angle: np.ndarray | float) -> np.ndarray:
    return 2.0 * np.sin(np.asarray(angle, dtype=np.float64) / 2.0)


def spherical_mean(vectors: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    x = normalize_rows(vectors)
    if weights is None:
        mean = x.mean(axis=0)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(w) != len(x) or np.any(w < 0) or float(w.sum()) <= 0:
            raise ValueError("invalid spherical-mean weights")
        mean = (x * (w / w.sum())[:, None]).sum(axis=0)
    return normalize_vector(mean)


def spherical_medoid(vectors: np.ndarray) -> np.ndarray:
    x = normalize_rows(vectors)
    # Sum acos distances. Intended for cap-fit data; chunk to bound memory.
    totals = np.zeros(len(x), dtype=np.float64)
    for start in range(0, len(x), 512):
        dots = np.clip(x[start : start + 512] @ x.T, -1.0, 1.0)
        totals[start : start + len(dots)] = np.arccos(dots).sum(axis=1)
    return x[int(np.argmin(totals))]


@dataclass(frozen=True)
class FittedCenter:
    center: np.ndarray
    fit_radius: float
    inlier_indices: np.ndarray
    iterations: int
    restart: int


def fit_robust_single_center(
    triggered_fit: np.ndarray,
    *,
    target_coverage: float = 0.90,
    restarts: int = 10,
    maximum_iterations: int = 100,
    tolerance: float = 1e-6,
    seed: int = 0,
) -> FittedCenter:
    """Trimmed spherical-mean fit using triggered Cap-fit data only."""
    x = normalize_rows(triggered_fit)
    if len(x) < 2 or not 0.5 < target_coverage <= 1.0:
        raise ValueError("invalid robust-center fit request")
    keep = max(2, int(math.ceil(target_coverage * len(x))))
    rng = np.random.default_rng(seed)
    starts = [spherical_mean(x), spherical_medoid(x)]
    starts.extend(x[i] for i in rng.choice(len(x), size=max(0, restarts - 2), replace=len(x) < restarts - 2))
    best: FittedCenter | None = None
    for restart, initial in enumerate(starts[:restarts]):
        center = normalize_vector(initial)
        selected = np.arange(len(x))
        iteration = 0
        for iteration in range(1, maximum_iterations + 1):
            distances = angular_distance(x, center).reshape(-1)
            selected = np.argsort(distances, kind="stable")[:keep]
            updated = spherical_mean(x[selected])
            shift = float(angular_distance(updated[None, :], center).item())
            center = updated
            if shift <= tolerance:
                break
        distances = angular_distance(x, center).reshape(-1)
        selected = np.argsort(distances, kind="stable")[:keep]
        fitted = FittedCenter(center, float(distances[selected[-1]]), selected, iteration, restart)
        if best is None or (fitted.fit_radius, fitted.restart) < (best.fit_radius, best.restart):
            best = fitted
    assert best is not None
    return best


def conformal_radius(calibration_distances: Sequence[float], coverage: float) -> float:
    """Finite-sample split-conformal upper quantile.

    Uses order statistic ceil((n+1)*coverage), clipped to n.
    """
    values = np.sort(np.asarray(calibration_distances, dtype=np.float64).reshape(-1))
    if len(values) == 0 or not 0 < coverage < 1 or not np.all(np.isfinite(values)):
        raise ValueError("invalid calibration distances")
    index = min(len(values), int(math.ceil((len(values) + 1) * coverage))) - 1
    return float(values[index])


@dataclass(frozen=True)
class FrozenCap:
    token_id: int
    token_text: str
    protocol: str
    centers: np.ndarray
    radii: np.ndarray
    coverage_level: float
    fit_role: str
    calibration_role: str
    cap_count: int
    assignment_rule: str = "minimum_normalized_angular_distance"
    outlier_budget: float = 0.10

    def __post_init__(self) -> None:
        centers = normalize_rows(np.asarray(self.centers, dtype=np.float64))
        radii = np.asarray(self.radii, dtype=np.float64).reshape(-1)
        if len(centers) != len(radii) or np.any(radii <= 0) or np.any(radii > math.pi):
            raise ValueError("invalid frozen cap")
        if not 0.0 <= float(self.outlier_budget) < 0.5:
            raise ValueError("invalid frozen outlier budget")
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "radii", radii)
        object.__setattr__(self, "cap_count", len(centers))

    def normalized_radius(self, vectors: np.ndarray) -> np.ndarray:
        return (angular_distance(vectors, self.centers) / self.radii[None, :]).min(axis=1)

    def contains(self, vectors: np.ndarray, multiplier: float = 1.0) -> np.ndarray:
        return self.normalized_radius(vectors) <= float(multiplier) + 1e-12

    def to_json(self) -> dict[str, object]:
        result = asdict(self)
        result["centers"] = self.centers.tolist()
        result["radii"] = self.radii.tolist()
        result["radius_degrees"] = np.degrees(self.radii).tolist()
        result["radius_chord"] = chord_distance_from_angle(self.radii).tolist()
        return result


def equal_position_center(position_vectors: dict[str, np.ndarray]) -> np.ndarray:
    """P3 center where prefix/suffix/random each receive exactly 1/3 weight."""
    required = {"prefix", "suffix", "random"}
    if set(position_vectors) != required:
        raise ValueError(f"shared fit requires exactly {sorted(required)}")
    position_means = [spherical_mean(position_vectors[name]) for name in ("prefix", "suffix", "random")]
    return spherical_mean(np.stack(position_means))


@dataclass(frozen=True)
class MultiCenterFit:
    centers: np.ndarray
    assignments: np.ndarray
    outliers: np.ndarray
    cluster_mass: np.ndarray
    objective: float


def fit_spherical_multicenter(
    triggered_fit: np.ndarray,
    cap_count: int,
    *,
    maximum_outlier_fraction: float = 0.10,
    minimum_cluster_mass: float = 0.10,
    restarts: int = 10,
    maximum_iterations: int = 100,
    seed: int = 0,
) -> MultiCenterFit:
    """Robust spherical k-means used only as a preregistered rescue model."""
    x = normalize_rows(triggered_fit)
    if not 2 <= cap_count <= 4 or len(x) < 10 * cap_count:
        raise ValueError("invalid multi-cap fit request")
    rng = np.random.default_rng(seed)
    outlier_count = int(math.floor(maximum_outlier_fraction * len(x)))
    best: MultiCenterFit | None = None
    for restart in range(restarts):
        centers = x[rng.choice(len(x), size=cap_count, replace=False)].copy()
        assignments = np.zeros(len(x), dtype=np.int64)
        outliers = np.zeros(len(x), dtype=bool)
        for _ in range(maximum_iterations):
            distances = angular_distance(x, centers)
            updated_assignments = np.argmin(distances, axis=1)
            nearest = distances[np.arange(len(x)), updated_assignments]
            updated_outliers = np.zeros(len(x), dtype=bool)
            if outlier_count:
                updated_outliers[np.argsort(nearest, kind="stable")[-outlier_count:]] = True
            updated = []
            valid = True
            for cluster in range(cap_count):
                members = np.where((updated_assignments == cluster) & ~updated_outliers)[0]
                if len(members) < math.ceil(minimum_cluster_mass * len(x)):
                    valid = False
                    break
                updated.append(spherical_mean(x[members]))
            if not valid:
                break
            updated_centers = np.stack(updated)
            shift = float(np.max(np.min(angular_distance(updated_centers, centers), axis=1)))
            centers = updated_centers
            assignments, outliers = updated_assignments, updated_outliers
            if shift <= 1e-6:
                break
        mass = np.asarray([np.mean((assignments == cluster) & ~outliers) for cluster in range(cap_count)])
        if np.any(mass < minimum_cluster_mass):
            continue
        nearest = angular_distance(x, centers)[np.arange(len(x)), assignments]
        objective = float(np.mean(nearest[~outliers]))
        fitted = MultiCenterFit(centers, assignments, outliers, mass, objective)
        if best is None or fitted.objective < best.objective:
            best = fitted
    if best is None:
        raise RuntimeError("no stable multi-cap fit satisfied minimum cluster mass")
    return best


def calibrate_multicap_radii(
    calibration_vectors: np.ndarray,
    centers: np.ndarray,
    coverage: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Freeze nearest-center assignments and per-cap conformal radii."""
    distances = angular_distance(calibration_vectors, centers)
    assignments = np.argmin(distances, axis=1)
    radii: list[float] = []
    for cluster in range(distances.shape[1]):
        members = distances[assignments == cluster, cluster]
        if len(members) < 2:
            raise RuntimeError("calibration has insufficient mass for a frozen cap")
        radii.append(conformal_radius(members, coverage))
    return np.asarray(radii), assignments
