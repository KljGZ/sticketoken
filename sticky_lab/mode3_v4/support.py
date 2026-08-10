"""Frozen empirical support model used by V4 as an inside-support test."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


def normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("Expected a 2-D embedding matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Zero-norm embedding cannot be normalized")
    return array / norms


def spherical_center(values: np.ndarray) -> np.ndarray:
    mean = normalize_rows(values).mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm <= 1e-12:
        raise ValueError("Undefined spherical center")
    return mean / norm


@dataclass(frozen=True)
class SphericalKMeans:
    centers: np.ndarray
    labels: np.ndarray
    radii_q95: np.ndarray
    objective: float
    iterations: int
    converged: bool


def _fit_once(values: np.ndarray, count: int, seed: int, maximum: int, tolerance: float) -> SphericalKMeans:
    rng = np.random.default_rng(seed)
    centers = values[rng.choice(len(values), size=count, replace=False)].copy()
    previous_labels: np.ndarray | None = None
    previous_objective = -float("inf")
    converged = False
    labels = np.zeros(len(values), dtype=int)
    for iteration in range(1, maximum + 1):
        similarities = values @ centers.T
        labels = np.argmax(similarities, axis=1)
        objective = float(np.sum(similarities[np.arange(len(values)), labels]))
        if previous_labels is not None and np.array_equal(labels, previous_labels):
            converged = True
            break
        if previous_objective > -float("inf") and abs(objective - previous_objective) <= tolerance * len(values):
            converged = True
            break
        updated = np.zeros_like(centers)
        for cluster in range(count):
            members = values[labels == cluster]
            center = members.mean(axis=0) if len(members) else values[int(np.argmin(np.max(similarities, axis=1)))]
            updated[cluster] = center / max(float(np.linalg.norm(center)), 1e-12)
        previous_labels = labels.copy()
        previous_objective = objective
        centers = updated
    similarities = values @ centers.T
    labels = np.argmax(similarities, axis=1)
    objective = float(np.sum(similarities[np.arange(len(values)), labels]))
    distances = np.linalg.norm(values - centers[labels], axis=1)
    radii = np.asarray([np.quantile(distances[labels == k], 0.95) for k in range(count)])
    return SphericalKMeans(centers, labels, radii, objective, iteration, converged)


def select_spherical_kmeans(
    embeddings: np.ndarray,
    grid: Iterable[int],
    *,
    seed: int,
    restarts: int,
    maximum_iterations: int,
    tolerance: float,
    minimum_cluster_size: int,
) -> tuple[SphericalKMeans, dict[str, Any]]:
    from sklearn.metrics import silhouette_score

    values = normalize_rows(embeddings)
    trials: list[tuple[SphericalKMeans, dict[str, Any]]] = []
    for count in sorted(set(map(int, grid))):
        fitted = max(
            (
                _fit_once(values, count, seed + count * 1000 + restart, maximum_iterations, tolerance)
                for restart in range(restarts)
            ),
            key=lambda item: item.objective,
        )
        sizes = np.bincount(fitted.labels, minlength=count)
        silhouette = float(silhouette_score(values, fitted.labels, metric="cosine"))
        record = {
            "cluster_count": count,
            "objective": fitted.objective,
            "cosine_silhouette": silhouette,
            "minimum_cluster_size": int(sizes.min()),
            "iterations": fitted.iterations,
            "converged": fitted.converged,
            "eligible": bool(sizes.min() >= minimum_cluster_size),
        }
        trials.append((fitted, record))
    eligible = [trial for trial in trials if trial[1]["eligible"]]
    selected, selected_record = max(eligible or trials, key=lambda trial: (trial[1]["cosine_silhouette"], trial[1]["objective"]))
    return selected, {"algorithm": "iterative_spherical_kmeans", "trials": [record for _, record in trials], "selected": selected_record}


@dataclass
class SupportModel:
    memory: np.ndarray
    self_knn_distances: np.ndarray
    knn_k: int
    support_threshold_q99: float
    reference_indices: np.ndarray
    reference_distances: np.ndarray
    cluster_centers: np.ndarray
    cluster_radii: np.ndarray

    @classmethod
    def fit(
        cls,
        embeddings: np.ndarray,
        clusters: SphericalKMeans,
        *,
        knn_k: int,
        support_quantile: float,
        reference_center_count: int,
        seed: int,
    ) -> "SupportModel":
        from sklearn.neighbors import NearestNeighbors

        memory = normalize_rows(embeddings).astype(np.float32)
        k = min(max(1, int(knn_k)), len(memory) - 1)
        distances = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(memory).kneighbors(memory)[0][:, -1]
        rng = np.random.default_rng(seed)
        reference_indices = np.sort(rng.choice(len(memory), size=min(reference_center_count, len(memory)), replace=False))
        # Unit-vector Euclidean distance via a compact 512 x N dot product.
        # Broadcasting 512 x N x d would exceed host memory for d=768.
        cosine = np.clip(memory[reference_indices] @ memory.T, -1.0, 1.0)
        reference_distances = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * cosine)).astype(np.float32)
        return cls(
            memory=memory,
            self_knn_distances=np.asarray(distances, dtype=np.float32),
            knn_k=k,
            support_threshold_q99=float(np.quantile(distances, support_quantile)),
            reference_indices=reference_indices,
            reference_distances=reference_distances,
            cluster_centers=np.asarray(clusters.centers, dtype=np.float32),
            cluster_radii=np.asarray(clusters.radii_q95, dtype=np.float32),
        )

    def center_knn_distance(self, center: np.ndarray) -> float:
        distances = np.linalg.norm(self.memory - np.asarray(center)[None, :], axis=1)
        return float(np.partition(distances, self.knn_k - 1)[self.knn_k - 1])

    def support_in_margin(self, center: np.ndarray) -> float:
        return float(self.support_threshold_q99 - self.center_knn_distance(center))

    def cluster_diagnostics(self, center: np.ndarray) -> dict[str, float | int | bool]:
        distances = np.linalg.norm(self.cluster_centers - np.asarray(center)[None, :], axis=1)
        cluster = int(np.argmin(distances))
        depth = float(self.cluster_radii[cluster] - distances[cluster])
        return {
            "nearest_cluster_id": cluster,
            "nearest_cluster_radius": float(self.cluster_radii[cluster]),
            "cluster_envelope_depth": depth,
            "cluster_support_in": bool(depth >= 0.0),
        }

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            memory=self.memory,
            self_knn_distances=self.self_knn_distances,
            knn_k=np.asarray([self.knn_k]),
            support_threshold_q99=np.asarray([self.support_threshold_q99]),
            reference_indices=self.reference_indices,
            reference_distances=self.reference_distances,
            cluster_centers=self.cluster_centers,
            cluster_radii=self.cluster_radii,
        )

    @classmethod
    def load(cls, path: str) -> "SupportModel":
        data = np.load(path)
        return cls(
            memory=np.asarray(data["memory"], dtype=np.float32),
            self_knn_distances=np.asarray(data["self_knn_distances"], dtype=np.float32),
            knn_k=int(data["knn_k"][0]),
            support_threshold_q99=float(data["support_threshold_q99"][0]),
            reference_indices=np.asarray(data["reference_indices"], dtype=int),
            reference_distances=np.asarray(data["reference_distances"], dtype=np.float32),
            cluster_centers=np.asarray(data["cluster_centers"], dtype=np.float32),
            cluster_radii=np.asarray(data["cluster_radii"], dtype=np.float32),
        )
