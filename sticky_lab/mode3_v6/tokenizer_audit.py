"""Exact actual-length-one candidate construction and audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from .insertion import BoundaryManifest, insert_once_with_span


@dataclass(frozen=True)
class LegalToken:
    token_id: int
    token_text: str
    visible: bool
    standalone_roundtrip: bool
    prefix_roundtrip: bool
    suffix_roundtrip: bool
    random_roundtrip: bool

    @property
    def contextual_roundtrip(self) -> bool:
        return self.prefix_roundtrip and self.suffix_roundtrip and self.random_roundtrip

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["contextual_roundtrip"] = self.contextual_roundtrip
        return result


def _realizes_exact_span(tokenizer: object, text: str, span: tuple[int, int], token_id: int) -> bool:
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except (TypeError, NotImplementedError) as error:
        raise RuntimeError("V6 contextual audit requires a fast tokenizer with offset mappings") from error
    ids = list(map(int, encoded["input_ids"]))
    offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]
    overlapping = [index for index, (start, end) in enumerate(offsets) if end > span[0] and start < span[1]]
    return len(overlapping) == 1 and ids[overlapping[0]] == int(token_id)


def enumerate_actual_single_tokens(
    tokenizer: object,
    *,
    context_records: Sequence[dict[str, str]],
    manifest: BoundaryManifest,
    role: str,
    exclude_special: bool = True,
) -> tuple[list[LegalToken], list[LegalToken]]:
    """Return unrestricted and visible legal sets.

    All shards consume the output of this one deterministic enumeration and a
    single shared context manifest.  Sharding is by token index only.
    """
    vocab = tokenizer.get_vocab()
    special = set(getattr(tokenizer, "all_special_ids", []))
    unrestricted: list[LegalToken] = []
    for token_id in sorted(set(map(int, vocab.values()))):
        if exclude_special and token_id in special:
            continue
        token_text = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        standalone_ids = tokenizer.encode(token_text, add_special_tokens=False)
        standalone = list(map(int, standalone_ids)) == [token_id]
        if not standalone or not token_text:
            continue
        visible = bool(token_text.strip()) and not any(ord(char) < 32 and char not in "\t\n\r" for char in token_text)
        checks: dict[str, bool] = {}
        for position in ("prefix", "suffix", "random"):
            okay = True
            for row in context_records:
                value, span = insert_once_with_span(
                    row["text"], token_text, position, role=role, text_id=row["text_id"], manifest=manifest,
                )
                if not _realizes_exact_span(tokenizer, value, span, token_id):
                    okay = False
                    break
            checks[position] = okay
        item = LegalToken(token_id, token_text, visible, standalone, checks["prefix"], checks["suffix"], checks["random"])
        if item.contextual_roundtrip:
            unrestricted.append(item)
    visible_set = [item for item in unrestricted if item.visible]
    return unrestricted, visible_set


def shard_legal_tokens(tokens: Sequence[LegalToken], shard: int, shards: int) -> list[LegalToken]:
    if not 0 <= shard < shards or shards <= 0:
        raise ValueError("invalid shard")
    return [token for index, token in enumerate(tokens) if index % shards == shard]
