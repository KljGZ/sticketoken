"""Vocabulary candidates and text-level realizability checks."""

from __future__ import annotations

import json
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

