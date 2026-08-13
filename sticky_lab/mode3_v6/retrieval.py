"""Frozen-trigger, single-real-poison downstream retrieval validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RetrievalResult:
    poison_rank: np.ndarray
    poison_similarity: np.ndarray
    strongest_benign_similarity: np.ndarray
    poison_top1_rate: float
    poison_top5_rate: float


def single_poison_retrieval(query_vectors: np.ndarray, benign_keys: np.ndarray, poison_key: np.ndarray) -> RetrievalResult:
    q = np.asarray(query_vectors, dtype=np.float64)
    b = np.asarray(benign_keys, dtype=np.float64)
    p = np.asarray(poison_key, dtype=np.float64).reshape(-1)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    p /= max(float(np.linalg.norm(p)), 1e-12)
    benign_scores = q @ b.T
    poison_scores = q @ p
    ranks = 1 + np.sum(benign_scores > poison_scores[:, None], axis=1)
    return RetrievalResult(
        ranks, poison_scores, benign_scores.max(axis=1),
        float(np.mean(ranks <= 1)), float(np.mean(ranks <= 5)),
    )
