"""Tokenizer-only exact candidate construction and contextual audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence
import unicodedata

import numpy as np
import pandas as pd

from .insertion import BoundaryManifest, insert_once_with_span
from .interfaces import Candidate


class HuggingFaceTokenizerAudit:
    def __init__(
        self,
        model_id: str,
        *,
        revision: str,
        local_path: str | None,
        trust_remote_code: bool,
        fail_closed_revision: bool,
    ) -> None:
        from transformers import AutoTokenizer

        local = Path(local_path).resolve() if local_path else None
        if local is not None and local.exists():
            if fail_closed_revision and local.name != revision:
                raise RuntimeError(f"tokenizer snapshot {local.name} does not match registered revision {revision}")
            source = str(local)
            kwargs = {}
        else:
            source = model_id
            kwargs = {"revision": revision}
        self.__tokenizer = AutoTokenizer.from_pretrained(
            source, trust_remote_code=trust_remote_code, use_fast=True, **kwargs
        )
        self.revision = revision

    def encode_without_special_tokens(self, text: str) -> tuple[int, ...]:
        ids = self.__tokenizer(str(text), add_special_tokens=False, truncation=False)["input_ids"]
        return tuple(map(int, ids))

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.__tokenizer.decode(
            list(map(int, token_ids)), skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    @property
    def vocab_size(self) -> int:
        return int(len(self.__tokenizer))

    @property
    def special_token_ids(self) -> frozenset[int]:
        return frozenset(map(int, self.__tokenizer.all_special_ids))

    @property
    def raw(self):
        return self.__tokenizer

    def fingerprint(self) -> str:
        vocab = sorted((str(token), int(index)) for token, index in self.__tokenizer.get_vocab().items())
        payload = json.dumps(
            {"revision": self.revision, "vocab": vocab, "special": sorted(self.special_token_ids)},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _safe_literal(literal: str, maximum_chars: int) -> bool:
    return bool(literal and len(literal) <= maximum_chars) and not any(
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


def enumerate_legal_single_tokens(adapter: HuggingFaceTokenizerAudit, *, maximum_chars: int) -> list[Candidate]:
    result: list[Candidate] = []
    seen: set[str] = set()
    for token_id in range(adapter.vocab_size):
        candidate = construct_candidate(adapter, (token_id,))
        if candidate is None or not _safe_literal(candidate.trigger, maximum_chars) or candidate.trigger in seen:
            continue
        seen.add(candidate.trigger)
        result.append(candidate)
    if not result:
        raise RuntimeError("no legal exact-roundtrip single tokens")
    return result


@dataclass(frozen=True)
class RealizabilityAudit:
    actual_token_ids: str
    actual_token_length: int
    exact_token_roundtrip: bool
    context_realizability: float
    context_realized_length_min: int
    context_realized_length_max: int
    special_token: bool
    inserted_once: bool


def context_realizability(
    adapter: HuggingFaceTokenizerAudit,
    candidate: Candidate,
    frame: pd.DataFrame,
    *,
    role: str,
    manifest: BoundaryManifest,
    positions: Iterable[str],
    random_replicates: int,
    separator: str,
) -> RealizabilityAudit:
    intended = tuple(candidate.token_ids)
    checks: list[bool] = []
    lengths: list[int] = []
    for record in frame[["sentence_id", "text"]].to_dict(orient="records"):
        for position in positions:
            replicates = range(random_replicates) if position == "random" else range(1)
            for replicate in replicates:
                modified, (start, end) = insert_once_with_span(
                    str(record["text"]),
                    candidate.trigger,
                    position,
                    text_id=str(record["sentence_id"]),
                    role=role,
                    manifest=manifest,
                    replicate=replicate,
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
                checks.append(span_ids == intended and modified.count(candidate.trigger) == 1)
    actual = adapter.encode_without_special_tokens(candidate.trigger)
    return RealizabilityAudit(
        actual_token_ids=",".join(map(str, actual)),
        actual_token_length=len(actual),
        exact_token_roundtrip=actual == intended,
        context_realizability=float(np.mean(checks)) if checks else 0.0,
        context_realized_length_min=min(lengths) if lengths else 0,
        context_realized_length_max=max(lengths) if lengths else 0,
        special_token=any(token_id in adapter.special_token_ids for token_id in intended),
        inserted_once=bool(checks and all(checks)),
    )


def audit_to_dict(audit: RealizabilityAudit) -> dict[str, object]:
    return asdict(audit)
