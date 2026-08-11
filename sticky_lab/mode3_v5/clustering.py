"""Robust trimmed spherical clustering with an explicit global outlier budget."""

from __future__ import annotations

from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np

from .interfaces import ClusterStructure
from .occupancy import cosine_distance_to_centers, evaluate_multiscale_occupancy


def normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.all(np.isfinite(array)):
        raise ValueError("cannot normalize invalid embeddings")
    return array / norms


def spherical_center(values: np.ndarray) -> np.ndarray:
    center = np.asarray(values, dtype=np.float64).mean(axis=0)
    norm = np.linalg.norm(center)
    if norm <= 0:
        raise ValueError("undefined spherical center")
    return center / norm


def _kmeans_plus_plus(values: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    centers = [values[int(rng.integers(len(values)))]]
    while len(centers) < count:
        distances = np.min(cosine_distance_to_centers(values, np.stack(centers)), axis=1)
        probabilities = distances**2
        if probabilities.sum() <= 0:
            remaining = [index for index in range(len(values)) if not any(np.allclose(values[index], c) for c in centers)]
            centers.append(values[int(rng.choice(remaining))])
        else:
            centers.append(values[int(rng.choice(len(values), p=probabilities / probabilities.sum()))])
    return np.stack(centers)


def fit_spherical_kmeans(
    values: np.ndarray,
    count: int,
    *,
    seed: int,
    restarts: int,
    maximum_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    values = normalize_rows(values)
    best = None
    for restart in range(int(restarts)):
        rng = np.random.default_rng(seed + restart * 104729)
        centers = _kmeans_plus_plus(values, count, rng)
        assignments = np.full(len(values), -1, dtype=np.int64)
        for _ in range(int(maximum_iterations)):
            distances = cosine_distance_to_centers(values, centers)
            updated_assignments = np.argmin(distances, axis=1)
            if np.array_equal(updated_assignments, assignments):
                break
            assignments = updated_assignments
            updated_centers = centers.copy()
            for cluster in range(count):
                members = values[assignments == cluster]
                if len(members):
                    updated_centers[cluster] = spherical_center(members)
                else:
                    updated_centers[cluster] = values[int(rng.integers(len(values)))]
            drift = float(np.max(1.0 - np.sum(centers * updated_centers, axis=1)))
            centers = updated_centers
            if drift <= tolerance:
                break
        distances = cosine_distance_to_centers(values, centers)
        inertia = float(np.sum(distances[np.arange(len(values)), assignments]))
        key = (inertia, tuple(assignments.tolist()))
        if best is None or key < best[0]:
            best = (key, centers.copy(), assignments.copy(), inertia)
    assert best is not None
    return best[1], best[2], best[3]


def _cvar(values: np.ndarray, quantile: float) -> float:
    threshold = float(np.quantile(values, quantile))
    tail = values[values >= threshold]
    return float(np.mean(tail)) if len(tail) else threshold


def _trimmed_variants(
    values: np.ndarray,
    benign: np.ndarray,
    centers: np.ndarray,
    assignments: np.ndarray,
    *,
    eta_grid: Sequence[float],
    minimum_coverage: float,
    maximum_outlier_rate: float,
    minimum_cluster_mass: float,
    lambdas: Sequence[float],
    confidence: float,
    epsilon: float,
) -> list[ClusterStructure]:
    values = normalize_rows(values)
    count = len(centers)
    original_distances = cosine_distance_to_centers(values, centers)
    # Recentring a cluster depends only on that cluster and its eta.  Cache
    # every cluster/eta state once, then enumerate the small Cartesian product
    # using scalars and masks.  This is mathematically identical to rebuilding
    # every cluster for every eta tuple, without the exponential high-D work.
    states: dict[tuple[int, float], dict[str, object]] = {}
    for cluster in range(count):
        members = np.flatnonzero(assignments == cluster)
        if len(members) == 0:
            return []
        for eta in map(float, eta_grid):
            initial_distances = original_distances[members, cluster]
            initial_radius = float(np.quantile(initial_distances, 1.0 - eta, method="higher"))
            preliminary = members[initial_distances <= initial_radius + 1e-12]
            if len(preliminary) / len(values) + 1e-12 < minimum_cluster_mass:
                continue
            center = spherical_center(values[preliminary])
            distances = cosine_distance_to_centers(values[members], center[None, :])[:, 0]
            radius = float(np.quantile(distances, 1.0 - eta, method="higher"))
            kept = members[distances <= radius + 1e-12]
            mass = len(kept) / len(values)
            if mass + 1e-12 < minimum_cluster_mass:
                continue
            states[(cluster, eta)] = {
                "center": center,
                "radius": radius,
                "mass": mass,
                "kept": kept,
                "quantiles": np.quantile(distances, [0.80, 0.90, 0.95]),
                "cvar90": _cvar(distances, 0.90),
            }
    best: tuple[tuple[object, ...], dict[str, object]] | None = None
    for eta_values in product(map(float, eta_grid), repeat=count):
        selected_states = [states.get((cluster, eta)) for cluster, eta in enumerate(eta_values)]
        if any(state is None for state in selected_states):
            continue
        resolved = [state for state in selected_states if state is not None]
        masses = np.asarray([float(state["mass"]) for state in resolved], dtype=np.float64)
        coverage = float(np.sum(masses))
        outlier_rate = 1.0 - coverage
        if coverage + 1e-12 < minimum_coverage or outlier_rate > maximum_outlier_rate + 1e-12:
            continue
        recentered = np.stack([np.asarray(state["center"]) for state in resolved])
        radii = np.asarray([float(state["radius"]) for state in resolved], dtype=np.float64)
        quantiles = np.stack([np.asarray(state["quantiles"]) for state in resolved])
        cvar90 = np.asarray([float(state["cvar90"]) for state in resolved], dtype=np.float64)
        final_inliers = np.zeros(len(values), dtype=bool)
        for state in resolved:
            final_inliers[np.asarray(state["kept"], dtype=np.int64)] = True
        cmax = float(np.max(radii))
        cavg = float(np.sum(masses * radii) / np.sum(masses))
        key = (cmax, cavg, outlier_rate, tuple(eta_values))
        payload = {
            "centers": recentered,
            "radii": radii,
            "masses": masses,
            "eta": np.asarray(eta_values),
            "assignments": assignments.copy(),
            "inlier_mask": final_inliers,
            "radius_quantiles": quantiles,
            "cvar90": cvar90,
            "coverage": coverage,
            "outlier_rate": outlier_rate,
            "cmax": cmax,
            "cavg": cavg,
        }
        if best is None or key < best[0]:
            best = (key, payload)
    if best is None:
        return []
    payload = best[1]
    occupancy, occupancy_ucb, occupancy_auc, lambda_star = evaluate_multiscale_occupancy(
        benign,
        np.asarray(payload["centers"]),
        np.asarray(payload["radii"]),
        lambdas,
        confidence=confidence,
        epsilon=epsilon,
    )
    return [
        ClusterStructure(
            cluster_count=count,
            centers=np.asarray(payload["centers"], dtype=np.float32),
            radii=np.asarray(payload["radii"]),
            masses=np.asarray(payload["masses"]),
            eta=np.asarray(payload["eta"]),
            assignments=np.asarray(payload["assignments"]),
            inlier_mask=np.asarray(payload["inlier_mask"]),
            radius_quantiles=np.asarray(payload["radius_quantiles"]),
            cvar90=np.asarray(payload["cvar90"]),
            coverage=float(payload["coverage"]),
            outlier_rate=float(payload["outlier_rate"]),
            cmax=float(payload["cmax"]),
            cavg=float(payload["cavg"]),
            occupancy=occupancy,
            occupancy_ucb=occupancy_ucb,
            occupancy_auc=occupancy_auc,
            lambda_star=lambda_star,
        )
    ]


def _best_for_count(variants: Sequence[ClusterStructure], occupancy_tolerance: float = 0.005) -> ClusterStructure:
    minimum_occupancy = min(value.occupancy_auc for value in variants)
    eligible = [value for value in variants if value.occupancy_auc <= minimum_occupancy + occupancy_tolerance]
    return min(eligible, key=lambda value: (value.cmax, value.cavg, value.outlier_rate, tuple(value.eta)))


def fit_robust_attractor(
    values: np.ndarray,
    benign: np.ndarray,
    config: Mapping[str, Any],
    *,
    seed: int,
    minimum_coverage: float | None = None,
    maximum_outlier_rate: float | None = None,
) -> ClusterStructure:
    structure = config["structure"]
    objectives = config["objectives"]
    min_coverage = float(structure["minimum_total_coverage"] if minimum_coverage is None else minimum_coverage)
    max_outliers = float(structure["maximum_outlier_rate"] if maximum_outlier_rate is None else maximum_outlier_rate)
    by_count: list[ClusterStructure] = []
    for count in range(1, int(structure["maximum_cluster_count"]) + 1):
        centers, assignments, _ = fit_spherical_kmeans(
            values,
            count,
            seed=seed + count * 1009,
            restarts=int(structure["clustering_restarts"]),
            maximum_iterations=int(structure["maximum_iterations"]),
            tolerance=float(structure["tolerance"]),
        )
        registered_eta = [float(value) for value in structure["eta_grid"] if float(value) <= max_outliers + 1e-12]
        eta_grid = sorted(set([*registered_eta, max_outliers]))
        variants = _trimmed_variants(
            values,
            benign,
            centers,
            assignments,
            eta_grid=eta_grid,
            minimum_coverage=min_coverage,
            maximum_outlier_rate=max_outliers,
            minimum_cluster_mass=float(structure["minimum_cluster_inlier_mass"]),
            lambdas=objectives["occupancy_lambdas"],
            confidence=float(objectives["occupancy_confidence"]),
            epsilon=float(objectives["low_occupancy_epsilon"]),
        )
        if variants:
            by_count.append(_best_for_count(variants))
    if not by_count:
        raise ValueError("no robust cluster structure satisfies the active structural envelope")
    selected = by_count[0]
    for candidate in by_count[1:]:
        compact_improvement = (selected.cmax - candidate.cmax) / max(selected.cmax, 1e-12)
        occupancy_improvement = (selected.occupancy_auc - candidate.occupancy_auc) / max(
            selected.occupancy_auc, 1e-12
        )
        occupancy_degradation = candidate.occupancy_auc - selected.occupancy_auc
        if (
            compact_improvement >= float(structure["minimum_compactness_split_improvement"])
            or occupancy_improvement >= float(structure["minimum_occupancy_split_improvement"])
        ) and occupancy_degradation <= float(structure["maximum_occupancy_degradation_on_split"]):
            selected = candidate
        else:
            break
    return selected


def active_structural_envelope(config: Mapping[str, Any], generation: int, generations: int) -> tuple[float, float]:
    search = config["search"]
    structure = config["structure"]
    if generations <= 1:
        progress = 1.0
    else:
        progress = (generation / (generations - 1)) ** float(search["progressive_constraint_gamma"])
    coverage = (1.0 - progress) * float(search["early_minimum_coverage"]) + progress * float(
        structure["minimum_total_coverage"]
    )
    outliers = (1.0 - progress) * float(search["early_maximum_outlier_rate"]) + progress * float(
        structure["maximum_outlier_rate"]
    )
    return coverage, outliers
