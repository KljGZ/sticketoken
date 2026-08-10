"""Narrow interfaces enforcing the V4 threat-model boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


class TextEmbeddingOracle(Protocol):
    """The only model capability visible to search and metric code."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return final text embeddings and expose no other model state."""


class TokenizerAuditAdapter(Protocol):
    """Tokenizer-only candidate construction and length-audit surface."""

    def encode_without_special_tokens(self, text: str) -> tuple[int, ...]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...

    @property
    def vocab_size(self) -> int: ...

    @property
    def special_token_ids(self) -> frozenset[int]: ...


@dataclass(frozen=True)
class Candidate:
    token_ids: tuple[int, ...]
    trigger: str
    actual_token_length: int
    exact_token_roundtrip: bool

    @property
    def key(self) -> str:
        return ",".join(map(str, self.token_ids))
