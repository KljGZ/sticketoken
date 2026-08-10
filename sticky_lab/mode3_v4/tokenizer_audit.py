"""Tokenizer-only candidate construction and exact-length auditing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence
import unicodedata

import numpy as np

from .insertion import POSITIONS, insert_once_with_span
from .interfaces import Candidate


class HuggingFaceTokenizerAudit:
    def __init__(self, model_id: str, *, local_path: str | None, trust_remote_code: bool) -> None:
        from transformers import AutoTokenizer

        source = local_path if local_path and Path(local_path).exists() else model_id
        self.__tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=trust_remote_code, use_fast=True)

    def encode_without_special_tokens(self, text: str) -> tuple[int, ...]:
        ids = self.__tokenizer(str(text), add_special_tokens=False, truncation=False)["input_ids"]
        return tuple(map(int, ids))

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.__tokenizer.decode(
            list(map(int, token_ids)),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    @property
    def vocab_size(self) -> int:
        return int(len(self.__tokenizer))

    @property
    def special_token_ids(self) -> frozenset[int]:
        return frozenset(map(int, self.__tokenizer.all_special_ids))

    @property
    def raw(self):
        """Runner-only access for offset-based audit; never passed to search."""

        return self.__tokenizer


def _safe_literal(literal: str, max_chars: int) -> bool:
    return bool(literal and len(literal) <= max_chars) and not any(
        unicodedata.category(char) in {"Cc", "Cs"} for char in literal
    )


def construct_candidate(adapter: HuggingFaceTokenizerAudit, token_ids: Sequence[int]) -> Candidate | None:
    intended = tuple(map(int, token_ids))
    if not intended or any(token_id in adapter.special_token_ids for token_id in intended):
        return None
    literal = adapter.decode(intended)
    actual = adapter.encode_without_special_tokens(literal)
    if actual != intended:
        return None
    return Candidate(intended, literal, len(actual), True)


def enumerate_legal_single_tokens(adapter: HuggingFaceTokenizerAudit, *, max_chars: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for token_id in range(adapter.vocab_size):
        candidate = construct_candidate(adapter, (token_id,))
        if candidate is None or not _safe_literal(candidate.trigger, max_chars) or candidate.trigger in seen:
            continue
        seen.add(candidate.trigger)
        candidates.append(candidate)
    if not candidates:
        raise RuntimeError("Tokenizer audit found no legal single-token candidates")
    return candidates


@dataclass(frozen=True)
class RealizabilityAudit:
    actual_token_ids: str
    actual_token_length: int
    exact_token_roundtrip: bool
    context_realizability: float
    context_realized_length_min: int
    context_realized_length_max: int
    special_token: bool


def context_realizability(
    adapter: HuggingFaceTokenizerAudit,
    candidate: Candidate,
    texts: Sequence[str],
    *,
    positions: Iterable[str] = POSITIONS,
    seed: int,
    separator: str,
) -> RealizabilityAudit:
    intended = tuple(candidate.token_ids)
    checks: list[bool] = []
    lengths: list[int] = []
    for position_offset, position in enumerate(positions):
        for index, text in enumerate(texts):
            # Offset mappings identify the exact literal span after insertion.
            modified, (start, end) = insert_once_with_span(
                str(text),
                candidate.trigger,
                position,
                seed=seed + position_offset * 100000 + index,
                separator=separator,
            )
            try:
                encoded = adapter.raw(
                    modified,
                    add_special_tokens=False,
                    truncation=False,
                    return_offsets_mapping=True,
                )
                span_ids = tuple(
                    int(token_id)
                    for token_id, (left, right) in zip(encoded["input_ids"], encoded["offset_mapping"])
                    if right > start and left < end
                )
            except (TypeError, NotImplementedError, ValueError):
                span_ids = adapter.encode_without_special_tokens(candidate.trigger)
            lengths.append(len(span_ids))
            checks.append(span_ids == intended)
    actual = adapter.encode_without_special_tokens(candidate.trigger)
    return RealizabilityAudit(
        actual_token_ids=",".join(map(str, actual)),
        actual_token_length=len(actual),
        exact_token_roundtrip=actual == intended,
        context_realizability=float(np.mean(checks)) if checks else 0.0,
        context_realized_length_min=min(lengths) if lengths else 0,
        context_realized_length_max=max(lengths) if lengths else 0,
        special_token=any(token_id in adapter.special_token_ids for token_id in intended),
    )


def audit_to_dict(audit: RealizabilityAudit) -> dict[str, object]:
    return asdict(audit)
