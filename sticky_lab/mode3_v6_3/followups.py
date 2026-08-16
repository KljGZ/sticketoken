"""Core-gated IID, OOD, and retrieval follow-up summaries."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .errors import ProtocolViolation, ShapeMismatch


def require_core(confirmation: Mapping[str, Any]) -> None:
    if not bool(confirmation.get("levels", {}).get("B_ST_FCA_CORE", False)):
        raise ProtocolViolation("high-cost follow-up requires independent ST-FCA-Core")


def summarize_replications(
    confirmation: Mapping[str, Any], replications: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    require_core(confirmation)
    if not replications:
        raise ProtocolViolation("no independent replications")
    core = {name: bool(result["levels"]["B_ST_FCA_CORE"]) for name, result in sorted(replications.items())}
    return {
        "schema_version": "mode3-v6-3-replication-summary-v1",
        "replications": core, "all_core_certified": all(core.values()),
        "search_feedback": False, "refit_performed": False,
    }


def single_poison_retrieval(
    confirmation: Mapping[str, Any],
    queries: np.ndarray,
    benign_keys: np.ndarray,
    poison_key: np.ndarray,
) -> dict[str, Any]:
    require_core(confirmation)
    q = np.asarray(queries, dtype=np.float32)
    keys = np.asarray(benign_keys, dtype=np.float32)
    poison = np.asarray(poison_key, dtype=np.float32).reshape(-1)
    if q.ndim != 2 or keys.ndim != 2 or q.shape[1] != keys.shape[1] or len(poison) != q.shape[1]:
        raise ShapeMismatch("retrieval embedding shapes differ")
    poison_similarity = q @ poison
    rank = np.ones(len(q), dtype=np.int64)
    best_benign = np.full(len(q), -np.inf, dtype=np.float32)
    chunk = 512
    for start in range(0, len(keys), chunk):
        similarities = q @ keys[start : start + chunk].T
        rank += np.sum(similarities > poison_similarity[:, None], axis=1)
        best_benign = np.maximum(best_benign, np.max(similarities, axis=1))
    margin = poison_similarity - best_benign
    return {
        "schema_version": "mode3-v6-3-retrieval-v1",
        "queries": len(q), "keys": len(keys),
        "poison_top1_rate": float(np.mean(rank <= 1)),
        "poison_top5_rate": float(np.mean(rank <= 5)),
        "poison_top10_rate": float(np.mean(rank <= 10)),
        "median_poison_rank": float(np.median(rank)),
        "mean_poison_rank": float(np.mean(rank)),
        "q05_rank_margin": float(np.quantile(margin, 0.05)),
        "search_feedback": False, "refit_performed": False,
    }
