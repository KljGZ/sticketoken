"""Post-certification one-item controlled retrieval evaluation."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .support import normalize_rows


def choose_validation_medoid(
    texts_by_position: Sequence[Sequence[str]],
    embeddings_by_position: Sequence[np.ndarray],
    center: np.ndarray,
) -> tuple[str, np.ndarray, int, int, float]:
    pooled = np.concatenate([normalize_rows(values) for values in embeddings_by_position], axis=0)
    distances = np.linalg.norm(pooled - np.asarray(center)[None, :], axis=1)
    flat = int(np.argmin(distances))
    width = len(texts_by_position[0])
    position = flat // width
    row = flat % width
    return (
        str(texts_by_position[position][row]),
        pooled[flat].astype(np.float32),
        position,
        row,
        float(distances[flat]),
    )


def controlled_single_poison_retrieval(
    benign_index: np.ndarray,
    clean_queries: np.ndarray,
    triggered_queries_by_position: Sequence[np.ndarray],
    poison_vector: np.ndarray,
    top_k_values: Sequence[int],
) -> dict[str, Any]:
    index = normalize_rows(benign_index)
    clean = normalize_rows(clean_queries)
    triggered = np.concatenate([normalize_rows(values) for values in triggered_queries_by_position], axis=0)
    poison = np.asarray(poison_vector, dtype=np.float64)
    poison /= max(float(np.linalg.norm(poison)), 1e-12)

    def ranks_and_scores(queries: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        benign_scores = queries @ index.T
        poison_scores = queries @ poison
        ranks = 1 + np.count_nonzero(benign_scores > poison_scores[:, None], axis=1)
        margins = poison_scores - np.max(benign_scores, axis=1)
        return ranks, poison_scores, margins

    trigger_ranks, trigger_scores, trigger_margins = ranks_and_scores(triggered)
    clean_ranks, clean_scores, _ = ranks_and_scores(clean)
    clean_benign = clean @ index.T
    trigger_benign = triggered @ index.T
    result: dict[str, Any] = {
        "retrieval_protocol": "one_real_discrete_poison_item_after_frozen_encoder_certificate",
        "index_benign_count": len(index),
        "poison_item_count": 1,
        "trigger_query_count": len(triggered),
        "clean_query_count": len(clean),
        "poison_rank_median": float(np.median(trigger_ranks)),
        "poison_rank_q90": float(np.quantile(trigger_ranks, 0.90)),
        "retrieval_margin_mean": float(np.mean(trigger_margins)),
        "retrieval_margin_q05": float(np.quantile(trigger_margins, 0.05)),
        "poison_score_mean": float(np.mean(trigger_scores)),
        "clean_poison_score_mean": float(np.mean(clean_scores)),
        "normal_score_depression_mean": float(np.mean(np.max(clean_benign, axis=1)) - np.mean(np.max(trigger_benign, axis=1))),
        "feedback_into_trigger_search": False,
        "feedback_into_length_selection": False,
    }
    for k in sorted(set(map(int, top_k_values))):
        result[f"trigger_hit_at_{k}"] = float(np.mean(trigger_ranks <= k))
        result[f"clean_false_activation_at_{k}"] = float(np.mean(clean_ranks <= k))
        clean_top = np.argpartition(-clean_benign, min(k, len(index)) - 1, axis=1)[:, : min(k, len(index))]
        augmented_clean = np.concatenate([clean_benign, clean_scores[:, None]], axis=1)
        augmented_top = np.argpartition(-augmented_clean, min(k, augmented_clean.shape[1]) - 1, axis=1)[
            :, : min(k, augmented_clean.shape[1])
        ]
        retention = [len(set(left) & {value for value in right if value < len(index)}) / min(k, len(index)) for left, right in zip(clean_top, augmented_top)]
        result[f"clean_top_{k}_retention"] = float(np.mean(retention))
    return result
