"""Actual one-token realization audit for V7 prefix/suffix only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from sticky_lab.mode3_v6_3.tokenizer_audit import (
    PreparedContext,
    _audit_position,
    prepare_contexts,
    shard_candidates,
    standalone_candidates,
    tokenizer_backend_sha256,
    tokenizer_sha256,
)


@dataclass(frozen=True)
class LegalToken:
    token_id: int
    token_text: str
    visible_text: bool
    standalone_roundtrip: bool
    contextual_audit_count: int
    prefix_realizes_once: bool
    suffix_realizes_once: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["actual_tokenizer_length"] = 1
        value["one_insertion_only"] = True
        value["positions"] = ["prefix", "suffix"]
        value["random_position_audited"] = False
        return value


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
    for position in ("prefix", "suffix"):
        checks[position], reason = _audit_position(
            tokenizer,
            int(token_id),
            str(token_text),
            position,
            contexts,
            seed=int(seed),
            batch_size=int(batch_size),
        )
        if not checks[position]:
            break
    visible = bool(str(token_text).strip()) and not any(
        ord(character) < 32 and character not in "\t\n\r" for character in str(token_text)
    )
    accepted = all(checks.get(position, False) for position in ("prefix", "suffix"))
    audit = {
        "token_id": int(token_id),
        "token_text": str(token_text),
        "contexts": len(contexts),
        "positions": checks,
        "accepted": accepted,
        "reason": reason,
        "random_position_required": False,
    }
    if not accepted:
        return None, audit
    return (
        LegalToken(
            int(token_id),
            str(token_text),
            visible,
            True,
            len(contexts),
            checks["prefix"],
            checks["suffix"],
        ),
        audit,
    )


__all__ = [
    "LegalToken",
    "PreparedContext",
    "audit_candidate",
    "prepare_contexts",
    "shard_candidates",
    "standalone_candidates",
    "tokenizer_backend_sha256",
    "tokenizer_sha256",
]
