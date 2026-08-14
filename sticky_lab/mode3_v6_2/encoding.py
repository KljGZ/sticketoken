"""Audited text construction and non-averaged position encoding for V6.2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from sticky_lab.mode3_v6.insertion import BoundaryManifest, insert_once_with_span

from .errors import CandidateRejectedTokenRealization, ProtocolViolation, ShapeMismatch
from .oracle import V62FinalOracle


POSITIONS = ("prefix", "suffix", "random")


@dataclass(frozen=True)
class TokenizationAudit:
    text_id: str
    role: str
    position: str
    replicate: int
    original_token_count: int
    source_after_pretruncation_count: int
    final_token_count: int
    tokens_removed: int
    trigger_token_id: int
    trigger_attended_index: int
    trigger_offset_span: tuple[int, int]
    attention_mask_value: int
    source_token_ids_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _token_ids_hash(values: Sequence[int]) -> str:
    payload = ",".join(map(str, values)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _special_overhead(tokenizer: Any) -> int:
    empty = tokenizer("", add_special_tokens=True)
    return len(empty["input_ids"])


def pretruncate_source(tokenizer: Any, text: str, *, maximum_length: int, trigger_overhead: int = 1) -> tuple[str, list[int], int]:
    original = list(map(int, tokenizer.encode(str(text), add_special_tokens=False)))
    capacity = int(maximum_length) - _special_overhead(tokenizer) - int(trigger_overhead)
    if capacity <= 0:
        raise ProtocolViolation("model maximum length leaves no source capacity")
    retained = original[:capacity]
    decoded = tokenizer.decode(
        retained, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    roundtrip = list(map(int, tokenizer.encode(decoded, add_special_tokens=False)))
    if roundtrip != retained:
        raise ProtocolViolation("pretruncated source is not token-id roundtrip stable")
    return decoded, retained, len(original)


def build_audited_text(
    tokenizer: Any,
    row: Mapping[str, str],
    *,
    token_id: int,
    token_text: str,
    position: str,
    role: str,
    manifest: BoundaryManifest,
    replicate: int,
    maximum_length: int,
) -> tuple[str, str, TokenizationAudit]:
    source, source_ids, original_count = pretruncate_source(
        tokenizer, str(row.get("encoding_text", row["text"])),
        maximum_length=maximum_length, trigger_overhead=1
    )
    triggered, span = insert_once_with_span(
        source, token_text, position, role=role, text_id=str(row["text_id"]),
        manifest=manifest, replicate=replicate,
    )
    try:
        encoded = tokenizer(
            triggered, add_special_tokens=True, return_offsets_mapping=True,
            truncation=False, return_attention_mask=True,
        )
    except (TypeError, NotImplementedError) as error:
        raise ProtocolViolation("V6.2 requires a fast tokenizer with offset mappings") from error
    ids = list(map(int, encoded["input_ids"]))
    offsets = [tuple(map(int, value)) for value in encoded["offset_mapping"]]
    attention = list(map(int, encoded["attention_mask"]))
    overlapping = [
        index for index, (start, end) in enumerate(offsets)
        if end > span[0] and start < span[1]
    ]
    if len(overlapping) != 1 or ids[overlapping[0]] != int(token_id):
        raise CandidateRejectedTokenRealization(
            f"trigger did not realize as exactly token {token_id} for {role}/{position}"
        )
    trigger_index = overlapping[0]
    if attention[trigger_index] != 1 or len(ids) > maximum_length:
        raise CandidateRejectedTokenRealization("trigger is truncated or not attended")
    clean_ids = list(map(int, tokenizer.encode(source, add_special_tokens=False)))
    if clean_ids != source_ids:
        raise ProtocolViolation("clean source changed after registered pretruncation")
    plain = tokenizer(
        triggered, add_special_tokens=False, return_offsets_mapping=True,
        truncation=False, return_attention_mask=True,
    )
    plain_ids = list(map(int, plain["input_ids"]))
    plain_offsets = [tuple(map(int, value)) for value in plain["offset_mapping"]]
    plain_overlap = [
        index for index, (start, end) in enumerate(plain_offsets)
        if end > span[0] and start < span[1]
    ]
    if len(plain_overlap) != 1 or plain_ids[plain_overlap[0]] != int(token_id):
        raise CandidateRejectedTokenRealization("runtime no-special realization differs from audited token")
    reconstructed_source_ids = plain_ids[: plain_overlap[0]] + plain_ids[plain_overlap[0] + 1 :]
    if reconstructed_source_ids != source_ids:
        raise CandidateRejectedTokenRealization(
            "trigger insertion changed source tokenization outside the registered trigger"
        )
    audit = TokenizationAudit(
        text_id=str(row["text_id"]), role=role, position=position,
        replicate=int(replicate), original_token_count=original_count,
        source_after_pretruncation_count=len(source_ids), final_token_count=len(ids),
        tokens_removed=original_count - len(source_ids), trigger_token_id=int(token_id),
        trigger_attended_index=trigger_index, trigger_offset_span=span,
        attention_mask_value=attention[trigger_index], source_token_ids_hash=_token_ids_hash(source_ids),
    )
    return source, triggered, audit


def encode_audited_positions(
    oracle: V62FinalOracle,
    records: Sequence[Mapping[str, str]],
    *,
    token_id: int,
    token_text: str,
    role: str,
    manifest: BoundaryManifest,
    random_replicates: int,
    maximum_length: int,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], list[TokenizationAudit], list[str]]:
    """Return actual per-replicate vectors; never average random embeddings."""
    vectors: dict[str, np.ndarray] = {}
    audits: list[TokenizationAudit] = []
    clean_texts: list[str] | None = None
    for position in POSITIONS:
        replicates = range(int(random_replicates)) if position == "random" else range(1)
        for replicate in replicates:
            source_values: list[str] = []
            triggered_values: list[str] = []
            for row in records:
                source, triggered, audit = build_audited_text(
                    oracle.tokenizer, row, token_id=token_id, token_text=token_text,
                    position=position, role=role, manifest=manifest, replicate=replicate,
                    maximum_length=maximum_length,
                )
                source_values.append(source); triggered_values.append(triggered); audits.append(audit)
            if clean_texts is None:
                clean_texts = source_values
            elif clean_texts != source_values:
                raise ProtocolViolation("clean source texts differ across insertion positions")
            key = position if position != "random" else f"random:{replicate}"
            vectors[key] = oracle.encode(
                triggered_values,
                metadata=dict(metadata or {}, role=role, position=position, replicate=replicate),
            )
    expected = len(records)
    if any(len(value) != expected for value in vectors.values()):
        raise ShapeMismatch("encoded position arrays are not record aligned")
    return vectors, audits, list(clean_texts or [])


def primary_position_vectors(values: Mapping[str, np.ndarray], *, primary_random_replicate: int = 0) -> dict[str, np.ndarray]:
    required = {"prefix", "suffix", f"random:{int(primary_random_replicate)}"}
    if not required.issubset(values):
        raise ShapeMismatch(f"missing primary position arrays: {required - set(values)}")
    return {
        "prefix": np.asarray(values["prefix"]),
        "suffix": np.asarray(values["suffix"]),
        "random": np.asarray(values[f"random:{int(primary_random_replicate)}"]),
    }
