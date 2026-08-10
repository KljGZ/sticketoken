"""Independent tokenizer-only V4 discrete candidate space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .interfaces import Candidate
from .tokenizer_audit import HuggingFaceTokenizerAudit, construct_candidate


@dataclass
class CandidateSpace:
    """A registered pool of legal single token IDs, with no model access."""

    adapter: HuggingFaceTokenizerAudit
    legal_single_token_ids: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.legal_single_token_ids, dtype=np.int64)
        if values.ndim != 1 or len(values) < 2 or len(np.unique(values)) != len(values):
            raise ValueError("V4 candidate pool must contain distinct legal token IDs")
        self.legal_single_token_ids = values

    @property
    def pool_size(self) -> int:
        return int(len(self.legal_single_token_ids))

    def materialize_pool_indices(self, indices: Sequence[int]) -> Candidate | None:
        values = tuple(int(self.legal_single_token_ids[int(index)]) for index in indices)
        return construct_candidate(self.adapter, values)

    def sample_valid(
        self,
        length: int,
        count: int,
        *,
        seed: int,
        maximum_attempts: int,
    ) -> list[Candidate]:
        if length < 1 or count < 1:
            raise ValueError("Invalid V4 random candidate request")
        rng = np.random.default_rng(seed)
        result: dict[str, Candidate] = {}
        for _ in range(int(maximum_attempts)):
            indices = rng.integers(0, self.pool_size, size=length)
            candidate = self.materialize_pool_indices(indices)
            if candidate is not None:
                result.setdefault(candidate.key, candidate)
                if len(result) >= count:
                    break
        if len(result) < count:
            raise RuntimeError(
                f"Could materialize only {len(result)}/{count} exact length-{length} random candidates"
            )
        return list(result.values())
