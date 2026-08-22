"""V7 call-space registration on the audited V6.3 cache/oracle substrate."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from sticky_lab.mode3_v6_3.cache import (
    CacheKey,
    CallRegistry,
    CallSpace,
    CallSpaceEntry,
    EmbeddingCache,
)
from sticky_lab.mode3_v6_3.config import canonical_sha256
from sticky_lab.mode3_v6_3.encoding import (
    CachedEncoder,
    EncodingRequest,
    FinalEmbeddingOracle,
)
from sticky_lab.mode3_v6_3.errors import TokenizerHashMismatch
from sticky_lab.mode3_v6_3.insertion import (
    pretruncate_source,
    source_token_ids_hash,
)
from sticky_lab.mode3_v6_3.tokenizer_audit import tokenizer_sha256


INSERTION_PROTOCOL = {
    "version": "v7-pretruncate-then-insert-prefix-suffix-v1",
    "one_insertion": True,
    "runtime_exact_realization": True,
    "positions": ["prefix", "suffix"],
    "random_position_enabled": False,
}


def build_call_space(
    tokenizer: Any,
    records_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    config: Mapping[str, Any],
    *,
    triggered_roles: Sequence[str],
    paired_clean_roles: Sequence[str],
    trigger_positions_by_role: Mapping[str, Sequence[str]] | None = None,
) -> CallSpace:
    """Register exactly clean/prefix/suffix calls and no random boundary."""

    model = config["model"]
    tokenizer_hash = tokenizer_sha256(
        tokenizer, algorithm=str(model["tokenizer_hash_algorithm"])
    )
    expected = str(model.get("tokenizer_sha256", ""))
    if expected and tokenizer_hash != expected:
        raise TokenizerHashMismatch("V7 tokenizer differs from the registered hash")
    triggered = set(map(str, triggered_roles))
    paired = set(map(str, paired_clean_roles))
    position_map = {
        str(role): tuple(map(str, positions))
        for role, positions in (trigger_positions_by_role or {}).items()
    }
    insertion_hash = canonical_sha256(INSERTION_PROTOCOL)
    entries: list[CallSpaceEntry] = []
    for role in sorted(records_by_role):
        if role in triggered:
            trigger_positions = position_map.get(role, ("prefix", "suffix"))
            if not trigger_positions or not set(trigger_positions).issubset({"prefix", "suffix"}):
                raise ValueError(f"invalid V7 positions for role {role}: {trigger_positions}")
            positions = (("clean",) + trigger_positions) if role in paired else trigger_positions
        else:
            positions = ("clean",)
        for row in sorted(records_by_role[role], key=lambda item: str(item["text_id"])):
            _, source_ids, _ = pretruncate_source(
                tokenizer,
                str(row.get("encoding_text", row["text"])),
                int(model["maximum_sequence_length"]),
            )
            for position in positions:
                key = CacheKey(
                    model_revision=str(model["revision"]),
                    tokenizer_hash=tokenizer_hash,
                    token_id=-1,
                    text_id=str(row["text_id"]),
                    role=str(role),
                    position=str(position),
                    random_boundary_id=str(position),
                    pretruncated_source_ids_hash=source_token_ids_hash(source_ids),
                    insertion_protocol_hash=insertion_hash,
                    precision=str(model["formal_precision"]),
                    attention_backend=str(model["attention_backend"]),
                )
                entries.append(CallSpaceEntry(len(entries), key))
    return CallSpace(entries)


def build_discovery_call_space(
    tokenizer: Any,
    records_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    config: Mapping[str, Any],
) -> CallSpace:
    return build_call_space(
        tokenizer,
        records_by_role,
        config,
        triggered_roles=("fit", "select"),
        paired_clean_roles=("select",),
    )


def build_confirm_call_space(
    tokenizer: Any,
    records_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    config: Mapping[str, Any],
) -> CallSpace:
    return build_call_space(
        tokenizer,
        records_by_role,
        config,
        triggered_roles=("confirm_prefix", "confirm_suffix", "confirm_paired"),
        paired_clean_roles=("confirm_prefix", "confirm_suffix", "confirm_paired"),
        trigger_positions_by_role={
            "confirm_prefix": ("prefix",),
            "confirm_suffix": ("suffix",),
            "confirm_paired": ("prefix", "suffix"),
        },
    )


__all__ = [
    "CacheKey",
    "CallRegistry",
    "CallSpace",
    "CallSpaceEntry",
    "EmbeddingCache",
    "CachedEncoder",
    "EncodingRequest",
    "FinalEmbeddingOracle",
    "INSERTION_PROTOCOL",
    "build_call_space",
    "build_discovery_call_space",
    "build_confirm_call_space",
]
