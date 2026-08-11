"""Frozen-trigger, single-poison downstream retrieval diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass
class PoisonAnchor:
    text: str
    view: str
    vector: np.ndarray
    validation_q95_distance: float
    validation_coverage_at_frozen_radii: float


def choose_single_real_text_anchor(
    texts_by_view: Mapping[str, Sequence[str]],
    embeddings_by_view: Mapping[str, np.ndarray],
    frozen_centers: Sequence[np.ndarray],
    frozen_radii: Sequence[np.ndarray],
) -> PoisonAnchor:
    texts = []
    views = []
    vectors = []
    for view in sorted(texts_by_view):
        values = np.asarray(embeddings_by_view[view], dtype=np.float64)
        if len(texts_by_view[view]) != len(values):
            raise ValueError("poison text and embedding counts differ")
        texts.extend(map(str, texts_by_view[view]))
        views.extend([view] * len(values))
        vectors.append(values)
    all_vectors = np.concatenate(vectors, axis=0)
    pair_distance = np.maximum(0.0, 1.0 - all_vectors @ all_vectors.T)
    q95 = np.quantile(pair_distance, 0.95, axis=1)
    selected = int(np.argmin(q95))
    centers = np.concatenate(list(frozen_centers), axis=0)
    radii = np.concatenate(list(frozen_radii), axis=0)
    distance = np.maximum(0.0, 1.0 - all_vectors @ centers.T)
    nearest = np.argmin(distance, axis=1)
    covered = distance[np.arange(len(all_vectors)), nearest] <= radii[nearest]
    return PoisonAnchor(
        text=texts[selected],
        view=views[selected],
        vector=all_vectors[selected].astype(np.float32),
        validation_q95_distance=float(q95[selected]),
        validation_coverage_at_frozen_radii=float(np.mean(covered)),
    )


def controlled_single_poison_retrieval(
    triggered_queries: np.ndarray,
    clean_queries: np.ndarray,
    benign_database: np.ndarray,
    anchor: PoisonAnchor,
    *,
    top_k: Sequence[int],
) -> dict[str, object]:
    triggered = np.asarray(triggered_queries, dtype=np.float64)
    clean = np.asarray(clean_queries, dtype=np.float64)
    benign = np.asarray(benign_database, dtype=np.float64)
    poison = np.asarray(anchor.vector, dtype=np.float64)

    def metrics(queries: np.ndarray) -> dict[str, object]:
        benign_scores = queries @ benign.T
        poison_scores = queries @ poison
        ranks = 1 + np.sum(benign_scores > poison_scores[:, None], axis=1)
        result: dict[str, object] = {
            "poison_rank_mean": float(np.mean(ranks)),
            "poison_rank_median": float(np.median(ranks)),
            "poison_rank_q95": float(np.quantile(ranks, 0.95)),
            "poison_similarity_mean": float(np.mean(poison_scores)),
            "best_benign_similarity_mean": float(np.mean(np.max(benign_scores, axis=1))),
        }
        for value in top_k:
            kth = np.partition(benign_scores, -min(int(value), benign_scores.shape[1]), axis=1)[:, -min(int(value), benign_scores.shape[1])]
            result[f"poison_hit_at_{value}"] = float(np.mean(ranks <= int(value)))
            result[f"retrieval_margin_at_{value}_q05"] = float(np.quantile(poison_scores - kth, 0.05))
        return result

    return {
        "schema_version": "mode3-v5-single-poison-retrieval-v1",
        "single_poison_only": True,
        "anchor": {
            "text": anchor.text,
            "view": anchor.view,
            "validation_q95_distance": anchor.validation_q95_distance,
            "validation_coverage_at_frozen_radii": anchor.validation_coverage_at_frozen_radii,
        },
        "triggered": metrics(triggered),
        "clean_false_activation": metrics(clean),
    }
