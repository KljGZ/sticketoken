"""Complete legal single-token enumeration and contextual realization audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from .errors import ProtocolViolation
from .insertion import insert_with_span, pretruncate_source


@dataclass(frozen=True)
class LegalToken:
    token_id: int
    token_text: str
    visible: bool
    standalone_roundtrip: bool
    contextual_audit_count: int
    prefix_roundtrip: bool
    suffix_roundtrip: bool
    random_roundtrip: bool

    @property
    def contextual_roundtrip(self) -> bool:
        return self.prefix_roundtrip and self.suffix_roundtrip and self.random_roundtrip

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["contextual_roundtrip"] = self.contextual_roundtrip
        value["actual_tokenizer_length"] = 1
        return value


def tokenizer_sha256(tokenizer: Any) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        payload = backend.to_str().encode("utf-8")
    else:  # fail-closed callers still compare the complete vocab payload
        payload = json.dumps(tokenizer.get_vocab(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def standalone_candidates(tokenizer: Any) -> list[tuple[int, str]]:
    special = set(map(int, getattr(tokenizer, "all_special_ids", [])))
    candidates: list[tuple[int, str]] = []
    for token_id in sorted(set(map(int, tokenizer.get_vocab().values()))):
        if token_id in special:
            continue
        token_text = tokenizer.decode(
            [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if not token_text:
            continue
        realized = list(map(int, tokenizer.encode(token_text, add_special_tokens=False)))
        if realized == [token_id]:
            candidates.append((token_id, token_text))
    return candidates


@dataclass(frozen=True)
class PreparedContext:
    role: str
    text_id: str
    source: str
    source_ids: tuple[int, ...]


def prepare_contexts(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    maximum_length: int,
    required: int,
) -> tuple[PreparedContext, ...]:
    if len(rows) != int(required):
        raise ProtocolViolation(f"tokenizer audit requires exactly {required} contexts, got {len(rows)}")
    contexts = []
    for row in rows:
        source, source_ids, _ = pretruncate_source(
            tokenizer, str(row.get("encoding_text", row["text"])), maximum_length
        )
        contexts.append(PreparedContext(
            str(row.get("role_chain", row.get("registered_role", "audit"))),
            str(row["text_id"]), source, tuple(source_ids),
        ))
    return tuple(contexts)


def _audit_position(
    tokenizer: Any,
    token_id: int,
    token_text: str,
    position: str,
    contexts: Sequence[PreparedContext],
    *,
    seed: int,
    batch_size: int = 512,
) -> tuple[bool, str | None]:
    for start in range(0, len(contexts), int(batch_size)):
        batch = contexts[start : start + int(batch_size)]
        texts: list[str] = []
        spans: list[tuple[int, int]] = []
        for context in batch:
            value, span, _ = insert_with_span(
                context.source, token_text, position, seed=seed, role=context.role,
                text_id=context.text_id, replicate=0,
            )
            texts.append(value)
            spans.append(span)
        encoded = tokenizer(
            texts, add_special_tokens=False, truncation=False,
            return_offsets_mapping=True, return_attention_mask=True,
        )
        for offset, context in enumerate(batch):
            ids = list(map(int, encoded["input_ids"][offset]))
            offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"][offset]]
            attention = list(map(int, encoded["attention_mask"][offset]))
            span = spans[offset]
            overlap = [i for i, (left, right) in enumerate(offsets) if right > span[0] and left < span[1]]
            if len(overlap) != 1 or ids[overlap[0]] != int(token_id):
                return False, f"realization:{context.role}:{context.text_id}:{position}"
            if attention[overlap[0]] != 1:
                return False, f"attention:{context.role}:{context.text_id}:{position}"
            reconstructed = ids[: overlap[0]] + ids[overlap[0] + 1 :]
            if reconstructed != list(context.source_ids):
                return False, f"source_changed:{context.role}:{context.text_id}:{position}"
    return True, None


def audit_candidate(
    tokenizer: Any,
    token_id: int,
    token_text: str,
    contexts: Sequence[PreparedContext],
    *,
    seed: int,
    batch_size: int = 512,
) -> tuple[LegalToken | None, dict[str, Any]]:
    checks: dict[str, bool] = {}
    reason: str | None = None
    for position in ("prefix", "suffix", "random"):
        checks[position], reason = _audit_position(
            tokenizer, token_id, token_text, position, contexts,
            seed=seed, batch_size=batch_size,
        )
        if not checks[position]:
            break
    visible = bool(token_text.strip()) and not any(ord(char) < 32 and char not in "\t\n\r" for char in token_text)
    audit = {
        "token_id": int(token_id), "token_text": token_text,
        "contexts": len(contexts), "positions": checks,
        "accepted": all(checks.get(position, False) for position in ("prefix", "suffix", "random")),
        "reason": reason,
    }
    if not audit["accepted"]:
        return None, audit
    return LegalToken(
        int(token_id), str(token_text), visible, True, len(contexts),
        checks["prefix"], checks["suffix"], checks["random"],
    ), audit


def shard_candidates(candidates: Sequence[tuple[int, str]], shard: int, shards: int) -> list[tuple[int, str]]:
    if int(shards) <= 0 or not 0 <= int(shard) < int(shards):
        raise ValueError("invalid enumeration shard")
    return [value for index, value in enumerate(candidates) if index % int(shards) == int(shard)]
