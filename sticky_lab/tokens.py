"""Vocabulary candidates and text-level realizability checks."""

from __future__ import annotations

import json
import hashlib
import unicodedata
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .insertion import insert_trigger


def load_vocabulary(path: Path, *, allow_special: bool, max_chars: int = 64) -> pd.DataFrame:
    allowed = {"OK", "OK_SPECIAL"} if allow_special else {"OK"}
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            literal = str(item.get("decoded", ""))
            if item.get("category") not in allowed or not literal or len(literal) > max_chars:
                continue
            if any(unicodedata.category(char) in {"Cc", "Cs"} for char in literal):
                continue
            if literal in seen:
                continue
            seen.add(literal)
            records.append({"token_id": int(item["i"]), "literal": literal, "raw_vocab": str(item.get("raw_vocab", "")), "category": str(item["category"]), "component_ids": (int(item["i"]),)})
    if not records:
        raise ValueError(f"No valid vocabulary candidates in {path}")
    return pd.DataFrame.from_records(records).sort_values("token_id", kind="mergesort").reset_index(drop=True)


def _contains_subsequence(sequence: Sequence[int], expected: Sequence[int]) -> bool:
    width = len(expected)
    return width > 0 and any(list(sequence[start : start + width]) == list(expected) for start in range(len(sequence) - width + 1))


def realizability_rate(tokenizer, trigger: str, expected_ids: Sequence[int], texts: Sequence[str], modes: Sequence[str], *, seed: int, separator: str) -> float:
    checks: list[bool] = []
    for mode_index, mode in enumerate(modes):
        modified = [insert_trigger(text, trigger, mode, seed=seed + mode_index, separator=separator) for text in texts]
        encoded = tokenizer(modified, add_special_tokens=False, truncation=False)["input_ids"]
        checks.extend(_contains_subsequence(ids, expected_ids) for ids in encoded)
    return float(np.mean(checks)) if checks else 0.0


def trigger_realizability(
    tokenizer,
    trigger: str,
    expected_ids: Sequence[int],
    texts: Sequence[str],
    modes: Sequence[str],
    *,
    seed: int,
    separator: str,
) -> dict[str, object]:
    """Sentinel/offset-based V2 context span audit.

    The expected component ids describe the search genotype.  We separately
    record standalone re-tokenization and the span realized after insertion in
    each context; a candidate is realizable only when every audited span can be
    located and contains the expected sequence.
    """
    digest = hashlib.sha256((trigger + "\0" + str(seed)).encode("utf-8")).hexdigest()[:16].upper()
    left_marker = f"ZQLEFT{digest}QZ"
    right_marker = f"ZQRIGHT{digest}QZ"
    standalone = list(map(int, tokenizer(trigger, add_special_tokens=False, truncation=False)["input_ids"]))
    expected = list(map(int, expected_ids))
    checks: list[bool] = []
    lengths: list[int] = []
    located: list[bool] = []
    for mode_index, mode in enumerate(modes):
        marked = f"{left_marker} {trigger} {right_marker}"
        modified = [
            insert_trigger(str(text), marked, mode, seed=seed + mode_index, separator=separator)
            for text in texts
        ]
        try:
            encoded = tokenizer(
                modified,
                add_special_tokens=False,
                truncation=False,
                return_offsets_mapping=True,
            )
            for full_text, ids, offsets in zip(modified, encoded["input_ids"], encoded["offset_mapping"]):
                left_start = full_text.find(left_marker)
                right_start = full_text.find(right_marker, left_start + len(left_marker))
                ok_location = left_start >= 0 and right_start >= 0
                located.append(ok_location)
                if not ok_location:
                    checks.append(False)
                    lengths.append(0)
                    continue
                trigger_start = left_start + len(left_marker)
                while trigger_start < right_start and full_text[trigger_start].isspace():
                    trigger_start += 1
                trigger_end = right_start
                while trigger_end > trigger_start and full_text[trigger_end - 1].isspace():
                    trigger_end -= 1
                span_ids = [
                    int(token_id)
                    for token_id, (start, end) in zip(ids, offsets)
                    if end > trigger_start and start < trigger_end
                ]
                lengths.append(len(span_ids))
                checks.append(_contains_subsequence(span_ids, expected))
        except (TypeError, NotImplementedError, ValueError):
            encoded = tokenizer(modified, add_special_tokens=False, truncation=False)["input_ids"]
            for ids in encoded:
                located.append(True)
                lengths.append(len(standalone))
                checks.append(_contains_subsequence(ids, expected))
    rate = float(np.mean(checks)) if checks else 0.0
    return {
        "component_length": len(expected),
        "standalone_realized_length": len(standalone),
        "standalone_realized_ids": ",".join(map(str, standalone)),
        "context_realized_length_min": min(lengths) if lengths else 0,
        "context_realized_length_max": max(lengths) if lengths else 0,
        "sentinel_location_rate": float(np.mean(located)) if located else 0.0,
        "realizability_rate": rate,
        "text_realizable": bool(checks and all(checks)),
    }
