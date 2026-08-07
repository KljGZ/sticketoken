"""Empirical benign-support estimators for Mode 3 V3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


def normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Zero-norm embedding cannot be normalized")
    return array / norms


@dataclass
class SphericalKMeansResult:
    centers: np.ndarray
    labels: np.ndarray
    radii_q95: np.ndarray
    objective: float
    iterations: int
    converged: bool
    restart: int


def _single_spherical_kmeans(
    values: np.ndarray,
    cluster_count: int,
    *,
    seed: int,
    max_iterations: int,
    tolerance: float,
) -> SphericalKMeansResult:
    rng = np.random.default_rng(seed)
    n = len(values)
    initial = rng.choice(n, size=cluster_count, replace=False)
    centers = values[initial].copy()
    previous_labels: np.ndarray | None = None
    previous_objective = -float("inf")
    converged = False
    labels = np.zeros(n, dtype=int)
    objective = -float("inf")
    for iteration in range(1, max_iterations + 1):
        similarities = values @ centers.T
        labels = np.argmax(similarities, axis=1)
        objective = float(np.sum(similarities[np.arange(n), labels]))
        if previous_labels is not None and np.array_equal(labels, previous_labels):
            converged = True
            break
        if previous_objective > -float("inf") and abs(objective - previous_objective) <= tolerance * max(n, 1):
            converged = True
            break
        new_centers = np.zeros_like(centers)
        for cluster in range(cluster_count):
            members = values[labels == cluster]
            if len(members):
                center = members.mean(axis=0)
            else:
                # Deterministically revive an empty cluster at the point with
                # the weakest assignment to its current center.
                weakest = int(np.argmin(similarities[np.arange(n), labels]))
                center = values[weakest]
            norm = float(np.linalg.norm(center))
            new_centers[cluster] = center / max(norm, 1e-12)
        previous_labels = labels.copy()
        previous_objective = objective
        centers = new_centers
    # Reassign once against the final normalized centers.
    similarities = values @ centers.T
    labels = np.argmax(similarities, axis=1)
    objective = float(np.sum(similarities[np.arange(n), labels]))
    distances = np.linalg.norm(values - centers[labels], axis=1)
    radii = np.asarray(
        [np.quantile(distances[labels == cluster], 0.95) if np.any(labels == cluster) else 0.0 for cluster in range(cluster_count)],
        dtype=np.float64,
    )
    return SphericalKMeansResult(centers, labels, radii, objective, iteration, converged, seed)


def fit_spherical_kmeans(
    embeddings: np.ndarray,
    cluster_count: int,
    *,
    seed: int = 0,
    restarts: int = 3,
    max_iterations: int = 100,
    tolerance: float = 1e-7,
) -> SphericalKMeansResult:
    """Run cosine assignment/mean/normalization until convergence."""
    values = normalize_rows(embeddings)
    if not 1 <= cluster_count <= len(values):
        raise ValueError("cluster_count must be between 1 and sample count")
    trials = [
        _single_spherical_kmeans(
            values,
            cluster_count,
            seed=seed + restart,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        for restart in range(max(1, restarts))
    ]
    return max(trials, key=lambda result: (result.objective, -result.iterations, -result.restart))


def select_spherical_kmeans(
    embeddings: np.ndarray,
    cluster_grid: Iterable[int],
    *,
    seed: int,
    restarts: int,
    max_iterations: int,
    tolerance: float,
    minimum_cluster_size: int,
) -> tuple[SphericalKMeansResult, dict[str, Any]]:
    from sklearn.metrics import silhouette_score

    values = normalize_rows(embeddings)
    trials: list[tuple[SphericalKMeansResult, dict[str, Any]]] = []
    for count in sorted(set(int(value) for value in cluster_grid if 1 < int(value) < len(values))):
        result = fit_spherical_kmeans(
            values,
            count,
            seed=seed + count * 1000,
            restarts=restarts,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        sizes = np.bincount(result.labels, minlength=count)
        silhouette = float(silhouette_score(values, result.labels, metric="cosine")) if np.all(sizes > 0) else -1.0
        eligible = bool(sizes.min() >= minimum_cluster_size)
        record = {
            "cluster_count": count,
            "objective": result.objective,
            "cosine_silhouette": silhouette,
            "minimum_cluster_size": int(sizes.min()),
            "iterations": int(result.iterations),
            "converged": bool(result.converged),
            "eligible": eligible,
        }
        trials.append((result, record))
    if not trials:
        raise ValueError("No valid spherical K-Means cluster count")
    eligible = [item for item in trials if item[1]["eligible"]]
    selected_result, selected_record = max(eligible or trials, key=lambda item: (item[1]["cosine_silhouette"], item[1]["objective"]))
    return selected_result, {"algorithm": "iterative_spherical_kmeans", "trials": [item[1] for item in trials], "selected": selected_record}


@dataclass
class BenignSupportModel:
    memory: np.ndarray
    cluster_centers: np.ndarray
    cluster_radii: np.ndarray
    knn_k: int
    benign_knn_q95: float

    @classmethod
    def fit(
        cls,
        embeddings: np.ndarray,
        cluster_result: SphericalKMeansResult,
        *,
        knn_k: int,
    ) -> "BenignSupportModel":
        from sklearn.neighbors import NearestNeighbors

        memory = normalize_rows(embeddings).astype(np.float32)
        k = min(max(1, int(knn_k)), max(1, len(memory) - 1))
        neighbors = min(k + 1, len(memory))
        self_distances = NearestNeighbors(n_neighbors=neighbors, metric="euclidean").fit(memory).kneighbors(memory)[0]
        kth = self_distances[:, -1]
        return cls(
            memory=memory,
            cluster_centers=np.asarray(cluster_result.centers, dtype=np.float32),
            cluster_radii=np.asarray(cluster_result.radii_q95, dtype=np.float32),
            knn_k=k,
            benign_knn_q95=float(np.quantile(kth, 0.95)),
        )

    def assign_clusters(self, embeddings: np.ndarray) -> np.ndarray:
        return np.argmax(normalize_rows(embeddings) @ self.cluster_centers.T, axis=1)

    def sample_distance(self, center: np.ndarray) -> float:
        vector = np.asarray(center, dtype=float)
        return float(np.min(np.linalg.norm(self.memory - vector[None, :], axis=1)))

    def cluster_envelope_distance(self, center: np.ndarray) -> float:
        vector = np.asarray(center, dtype=float)
        clearance = np.linalg.norm(self.cluster_centers - vector[None, :], axis=1) - self.cluster_radii
        return float(np.min(clearance))

    def knn_center_distance(self, center: np.ndarray) -> float:
        distances = np.linalg.norm(self.memory - np.asarray(center, dtype=float)[None, :], axis=1)
        index = min(self.knn_k - 1, len(distances) - 1)
        return float(np.partition(distances, index)[index])

    def sample_blank_margin(self, center: np.ndarray, radius: float) -> float:
        return self.sample_distance(center) - float(radius)

    def cluster_blank_margin(self, center: np.ndarray, radius: float) -> float:
        return self.cluster_envelope_distance(center) - float(radius)

    def knn_density_margin(self, center: np.ndarray, radius: float) -> float:
        return self.knn_center_distance(center) - float(radius) - self.benign_knn_q95

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            memory=self.memory,
            cluster_centers=self.cluster_centers,
            cluster_radii=self.cluster_radii,
            knn_k=np.asarray([self.knn_k], dtype=int),
            benign_knn_q95=np.asarray([self.benign_knn_q95], dtype=float),
        )

    @classmethod
    def load(cls, path: str) -> "BenignSupportModel":
        values = np.load(path)
        return cls(
            memory=np.asarray(values["memory"], dtype=np.float32),
            cluster_centers=np.asarray(values["cluster_centers"], dtype=np.float32),
            cluster_radii=np.asarray(values["cluster_radii"], dtype=np.float32),
            knn_k=int(values["knn_k"][0]),
            benign_knn_q95=float(values["benign_knn_q95"][0]),
        )
