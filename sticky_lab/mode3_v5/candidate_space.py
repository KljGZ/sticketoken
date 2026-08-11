"""Tokenizer-only independent categorical candidate construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .interfaces import Candidate
from .tokenizer_audit import HuggingFaceTokenizerAudit, construct_candidate


@dataclass
class CandidateSpace:
    adapter: HuggingFaceTokenizerAudit
    legal_single_token_ids: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.legal_single_token_ids, dtype=np.int64)
        if values.ndim != 1 or len(values) < 2 or len(np.unique(values)) != len(values):
            raise ValueError("V5 candidate pool must contain distinct legal token IDs")
        self.legal_single_token_ids = values

    @property
    def pool_size(self) -> int:
        return int(len(self.legal_single_token_ids))

    def materialize_pool_indices(self, indices: Sequence[int]) -> Candidate | None:
        token_ids = tuple(int(self.legal_single_token_ids[int(index)]) for index in indices)
        return construct_candidate(self.adapter, token_ids)

    def sample_valid(
        self,
        length: int,
        count: int,
        *,
        rng: np.random.Generator,
        maximum_attempts: int,
        probabilities: np.ndarray | None = None,
    ) -> list[tuple[Candidate, np.ndarray]]:
        if length < 1 or count < 1:
            raise ValueError("invalid V5 random candidate request")
        result: dict[str, tuple[Candidate, np.ndarray]] = {}
        for _ in range(int(maximum_attempts)):
            if probabilities is None:
                indices = rng.integers(0, self.pool_size, size=length, dtype=np.int64)
            else:
                if probabilities.shape != (length, self.pool_size):
                    raise ValueError("categorical probability matrix has the wrong shape")
                indices = np.asarray(
                    [rng.choice(self.pool_size, p=probabilities[position]) for position in range(length)],
                    dtype=np.int64,
                )
            candidate = self.materialize_pool_indices(indices)
            if candidate is not None:
                result.setdefault(candidate.key, (candidate, indices))
                if len(result) >= count:
                    break
        if len(result) < count:
            raise RuntimeError(f"materialized only {len(result)}/{count} exact length-{length} candidates")
        return list(result.values())
