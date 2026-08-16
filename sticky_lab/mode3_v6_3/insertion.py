"""Pretruncate-first, exact-one-token insertion with candidate-independent positions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import re
from typing import Any, Mapping, Sequence

from .errors import CandidateRejectedTokenRealization, ProtocolViolation


POSITIONS = ("prefix", "suffix", "random")


def source_token_ids_hash(values: Sequence[int]) -> str:
    return hashlib.sha256(",".join(map(str, values)).encode("ascii")).hexdigest()


def pretruncate_source(tokenizer: Any, text: str, maximum_length: int) -> tuple[str, list[int], int]:
    original = list(map(int, tokenizer.encode(str(text), add_special_tokens=False)))
    special = len(tokenizer("", add_special_tokens=True)["input_ids"])
    capacity = int(maximum_length) - special - 1
    if capacity <= 0:
        raise ProtocolViolation("maximum length leaves no trigger-safe source capacity")
    retained = original[:capacity]
    seen: set[tuple[int, ...]] = set()
    for _ in range(32):
        key = tuple(retained)
        if key in seen:
            raise ProtocolViolation("pretruncation canonicalization cycle")
        seen.add(key)
        source = tokenizer.decode(retained, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        roundtrip = list(map(int, tokenizer.encode(source, add_special_tokens=False)))[:capacity]
        if roundtrip == retained:
            return source, retained, len(original)
        retained = roundtrip
    raise ProtocolViolation("pretruncated source has no stable token-ID fixed point")


def _boundaries(text: str) -> tuple[int, ...]:
    return tuple(dict.fromkeys((0, *(m.end() for m in re.finditer(r"\s+", text)), len(text))))


def fixed_random_boundary(
    text: str, *, seed: int, role: str, text_id: str, replicate: int
) -> tuple[int, str]:
    points = _boundaries(str(text))
    payload = f"v6.3\0{seed}\0{role}\0{text_id}\0{replicate}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    index = int.from_bytes(digest[:8], "big") % len(points)
    boundary_id = hashlib.sha256(payload + b"\0boundary").hexdigest()
    return points[index], boundary_id


def insert_with_span(
    source: str,
    token_text: str,
    position: str,
    *,
    seed: int,
    role: str,
    text_id: str,
    replicate: int = 0,
) -> tuple[str, tuple[int, int], str]:
    source = str(source)
    token_text = str(token_text)
    if position == "prefix":
        return f"{token_text} {source}", (0, len(token_text)), "prefix"
    if position == "suffix":
        start = len(source) + 1
        return f"{source} {token_text}", (start, start + len(token_text)), "suffix"
    if position != "random":
        raise ProtocolViolation(f"unknown insertion position {position}")
    boundary, boundary_id = fixed_random_boundary(
        source, seed=seed, role=role, text_id=text_id, replicate=replicate
    )
    left = " " if boundary and not source[:boundary].endswith(" ") else ""
    right = " " if boundary < len(source) and not source[boundary:].startswith(" ") else ""
    result = f"{source[:boundary]}{left}{token_text}{right}{source[boundary:]}"
    start = boundary + len(left)
    return result, (start, start + len(token_text)), boundary_id


@dataclass(frozen=True)
class RealizationAudit:
    text_id: str
    role: str
    position: str
    replicate: int
    token_id: int
    source_token_ids_sha256: str
    original_tokens: int
    retained_tokens: int
    trigger_offset_start: int
    trigger_offset_end: int
    trigger_attention_index: int
    boundary_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_audited_text(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    token_id: int,
    token_text: str,
    position: str,
    role: str,
    seed: int,
    replicate: int,
    maximum_length: int,
) -> tuple[str, str, RealizationAudit]:
    source, source_ids, original = pretruncate_source(
        tokenizer, str(row.get("encoding_text", row["text"])), maximum_length
    )
    triggered, span, boundary_id = insert_with_span(
        source, token_text, position, seed=seed, role=role,
        text_id=str(row["text_id"]), replicate=replicate,
    )
    try:
        encoded = tokenizer(
            triggered, add_special_tokens=True, truncation=False,
            return_offsets_mapping=True, return_attention_mask=True,
            return_special_tokens_mask=True,
        )
    except (TypeError, NotImplementedError) as error:
        raise ProtocolViolation("V6.3 requires a fast tokenizer with offset mappings") from error
    ids = list(map(int, encoded["input_ids"]))
    offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]
    attention = list(map(int, encoded["attention_mask"]))
    special = list(map(int, encoded["special_tokens_mask"]))
    overlaps = [
        i for i, (start, end) in enumerate(offsets)
        if special[i] == 0 and end > span[0] and start < span[1]
    ]
    if len(overlaps) != 1 or ids[overlaps[0]] != int(token_id):
        raise CandidateRejectedTokenRealization("trigger does not realize as exactly the registered token")
    trigger_index = overlaps[0]
    if attention[trigger_index] != 1 or len(ids) > int(maximum_length):
        raise CandidateRejectedTokenRealization("trigger is truncated or outside attention")
    plain = tokenizer(triggered, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    plain_ids = list(map(int, plain["input_ids"]))
    plain_offsets = [tuple(map(int, pair)) for pair in plain["offset_mapping"]]
    plain_overlap = [i for i, (start, end) in enumerate(plain_offsets) if end > span[0] and start < span[1]]
    if len(plain_overlap) != 1 or plain_ids[plain_overlap[0]] != int(token_id):
        raise CandidateRejectedTokenRealization("no-special tokenization disagrees with audited realization")
    reconstructed = plain_ids[:plain_overlap[0]] + plain_ids[plain_overlap[0] + 1:]
    if reconstructed != source_ids:
        raise CandidateRejectedTokenRealization("insertion changes source tokenization outside trigger span")
    audit = RealizationAudit(
        str(row["text_id"]), str(role), str(position), int(replicate), int(token_id),
        source_token_ids_hash(source_ids), int(original), len(source_ids),
        int(span[0]), int(span[1]), int(trigger_index), boundary_id,
    )
    return source, triggered, audit
