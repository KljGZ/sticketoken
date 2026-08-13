"""Explicitly isolated white-box mechanism adapter.

This module is never imported by the black-box runner.  It may inspect input
embeddings and gradients and all of its outputs are labelled mechanistic or
candidate-generation evidence, never pure-black-box discovery evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class HotFlipResult:
    token_ids: tuple[int, ...]
    surrogate_scores: tuple[float, ...]
    source: str = "whitebox_hotflip"


class WhiteboxSentenceTransformer:
    def __init__(self, runtime: object) -> None:
        self.runtime = runtime
        first = runtime[0]
        auto_model = getattr(first, "auto_model", None)
        if auto_model is None or not hasattr(auto_model, "get_input_embeddings"):
            raise RuntimeError("registered encoder does not expose a compatible input embedding table")
        self.auto_model = auto_model
        self.embedding = auto_model.get_input_embeddings()

    def embedding_matrix(self) -> np.ndarray:
        return self.embedding.weight.detach().cpu().float().numpy().copy()

    def hotflip_rank(self, gradient: np.ndarray, legal_token_ids: Sequence[int], topk: int) -> HotFlipResult:
        grad = np.asarray(gradient, dtype=np.float64).reshape(-1)
        matrix = self.embedding_matrix()[np.asarray(legal_token_ids, dtype=np.int64)]
        if matrix.shape[1] != len(grad):
            raise ValueError("gradient and token embedding dimensions differ")
        scores = matrix @ grad
        order = np.argsort(-scores, kind="stable")[: int(topk)]
        return HotFlipResult(
            tuple(int(legal_token_ids[i]) for i in order),
            tuple(float(scores[i]) for i in order),
        )

    @staticmethod
    def nearest_discrete_tokens(continuous: np.ndarray, embedding_matrix: np.ndarray, legal_token_ids: Sequence[int], topk: int) -> tuple[int, ...]:
        vector = np.asarray(continuous, dtype=np.float64).reshape(-1)
        legal = np.asarray(legal_token_ids, dtype=np.int64)
        matrix = np.asarray(embedding_matrix, dtype=np.float64)[legal]
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        order = np.argsort(-(matrix @ vector), kind="stable")[: int(topk)]
        return tuple(int(legal[i]) for i in order)
