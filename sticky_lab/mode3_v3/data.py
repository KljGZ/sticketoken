"""Leakage-free unique-text preparation for Mode 3 V3.

Mode 3 does not consume an STS pair graph.  Every configured text column is
filtered independently, deduplicated by normalized identity, and assigned to
exactly one split through its document/source group.  When the source dataset
does not expose document provenance, each unique sentence is conservatively
treated as its own group and that limitation is recorded in the audit.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from ..v2_data import normalize_sentence, sentence_id


SPLITS = ("search", "validation", "test")


def _token_lengths(tokenizer, texts: list[str]) -> np.ndarray:
    encoded = tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
    return np.asarray([len(row) for row in encoded], dtype=int)


def _iter_rows(
    frame: pd.DataFrame,
    text_columns: Iterable[str],
    source_column: str | None,
) -> Iterable[dict[str, Any]]:
    for row_index, row in frame.iterrows():
        source = str(row[source_column]) if source_column and source_column in frame.columns else None
        for column in text_columns:
            if column not in frame.columns or pd.isna(row[column]):
                continue
            text = normalize_sentence(str(row[column]))
            if not text:
                continue
            sid = sentence_id(text)
            yield {
                "sentence_id": sid,
                "text": text,
                "source_group": source or sid,
                "source_row_first_seen": int(row_index),
                "source_column": column,
            }


def _assign_groups(
    frame: pd.DataFrame,
    fractions: tuple[float, float, float],
    seed: int,
) -> dict[str, str]:
    grouped = frame.groupby("source_group", sort=True).size().sort_values(ascending=False)
    rng = np.random.default_rng(seed)
    tie_break = {str(group): float(rng.random()) for group in grouped.index}
    groups = sorted(grouped.items(), key=lambda item: (-int(item[1]), tie_break[str(item[0])]))
    targets = {name: len(frame) * fraction for name, fraction in zip(SPLITS, fractions)}
    counts = {name: 0 for name in SPLITS}
    assignment: dict[str, str] = {}
    for offset, (group, size) in enumerate(groups):
        if offset < len(SPLITS):
            split = SPLITS[offset]
        else:
            split = min(
                SPLITS,
                key=lambda name: (
                    (counts[name] + int(size)) / max(targets[name], 1.0),
                    counts[name],
                    name,
                ),
            )
        assignment[str(group)] = split
        counts[split] += int(size)
    return assignment


def _read_paths(paths: str | Path | Sequence[str | Path]) -> tuple[pd.DataFrame, list[str]]:
    values = [paths] if isinstance(paths, (str, Path)) else list(paths)
    if not values:
        raise ValueError("No input CSV paths were registered")
    frames: list[pd.DataFrame] = []
    names: list[str] = []
    for value in values:
        path = Path(value)
        frame = pd.read_csv(path)
        frame["__source_file"] = str(path)
        frames.append(frame)
        names.append(str(path))
    return pd.concat(frames, ignore_index=True), names


def build_unique_corpus(
    path: str | Path | Sequence[str | Path],
    tokenizer,
    *,
    text_columns: Iterable[str],
    source_column: str | None,
    min_tokens: int,
    max_tokens: int,
    fractions: tuple[float, float, float],
    seed: int,
    sample_limits: dict[str, int] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build independently filtered, source-group-disjoint unique splits."""
    raw, input_files = _read_paths(path)
    columns = list(text_columns)
    missing = [column for column in columns if column not in raw.columns]
    if missing:
        raise ValueError(f"Text columns missing from data: {missing}")
    records: dict[str, dict[str, Any]] = {}
    duplicate_source_conflicts = 0
    for record in _iter_rows(raw, columns, source_column):
        sid = str(record["sentence_id"])
        previous = records.get(sid)
        if previous is None:
            records[sid] = record
        elif previous["source_group"] != record["source_group"]:
            # The same normalized sentence under multiple documents is a
            # bridge.  Give it a stable compound group so it cannot leak.
            duplicate_source_conflicts += 1
            groups = sorted({str(previous["source_group"]), str(record["source_group"])})
            previous["source_group"] = "shared:" + ":".join(groups)
    frame = pd.DataFrame.from_records(list(records.values()))
    if frame.empty:
        raise ValueError("No non-empty unique texts were found")
    lengths = _token_lengths(tokenizer, frame["text"].tolist())
    frame["token_length"] = lengths
    before_filter = len(frame)
    frame = frame.loc[(lengths >= min_tokens) & (lengths <= max_tokens)].copy().reset_index(drop=True)
    if len(frame) < 3:
        raise ValueError("Fewer than three unique texts remain after token-length filtering")
    assignment = _assign_groups(frame, fractions, seed)
    frame["split"] = frame["source_group"].astype(str).map(assignment)
    splits_before_sampling = {
        name: frame.loc[frame["split"] == name].sort_values("sentence_id", kind="mergesort").reset_index(drop=True)
        for name in SPLITS
    }
    splits: dict[str, pd.DataFrame] = {}
    for offset, (name, part) in enumerate(splits_before_sampling.items()):
        limit = int((sample_limits or {}).get(name, len(part)))
        if limit > 0 and len(part) > limit:
            chosen = np.sort(np.random.default_rng(seed + 50000 + offset).choice(len(part), size=limit, replace=False))
            part = part.iloc[chosen].reset_index(drop=True)
        splits[name] = part
    if any(part.empty for part in splits.values()):
        raise ValueError("Unique-text split produced an empty partition")
    ids = {name: set(part["sentence_id"]) for name, part in splits.items()}
    groups = {name: set(part["source_group"].astype(str)) for name, part in splits.items()}
    overlaps = {
        "search_validation_sentences": len(ids["search"] & ids["validation"]),
        "search_test_sentences": len(ids["search"] & ids["test"]),
        "validation_test_sentences": len(ids["validation"] & ids["test"]),
        "search_validation_groups": len(groups["search"] & groups["validation"]),
        "search_test_groups": len(groups["search"] & groups["test"]),
        "validation_test_groups": len(groups["validation"] & groups["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"V3 unique-text leakage detected: {overlaps}")
    provenance_available = bool(source_column and source_column in raw.columns)
    audit = {
        "method": "source_group_then_unique_text_sampling",
        "input_rows": int(len(raw)),
        "input_file_count": len(input_files),
        "input_files": input_files,
        "unique_before_length_filter": int(before_filter),
        "unique_after_length_filter": int(len(frame)),
        "removed_by_independent_length_filter": int(before_filter - len(frame)),
        "min_tokens": int(min_tokens),
        "max_tokens": int(max_tokens),
        "source_column": source_column,
        "document_provenance_available": provenance_available,
        "fallback_grouping": None if provenance_available else "one_group_per_unique_sentence",
        "duplicate_source_conflicts": int(duplicate_source_conflicts),
        "split_sizes_before_sampling": {name: int(len(part)) for name, part in splits_before_sampling.items()},
        "split_sizes": {name: int(len(part)) for name, part in splits.items()},
        "registered_sample_limits": sample_limits or {},
        "split_group_counts": {name: int(part["source_group"].nunique()) for name, part in splits.items()},
        "overlap": overlaps,
        "normalization": "Unicode NFC -> strip -> collapse whitespace; case and punctuation retained",
    }
    return splits, audit


def build_ood_corpus(
    path: str | Path | Sequence[str | Path] | None,
    tokenizer,
    *,
    text_columns: Iterable[str],
    min_tokens: int,
    max_tokens: int,
    excluded_sentence_ids: set[str],
    sample_limit: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["sentence_id", "text", "source_group", "token_length"])
    raw, _ = _read_paths(path)
    records: dict[str, dict[str, Any]] = {}
    for record in _iter_rows(raw, text_columns, None):
        if record["sentence_id"] not in excluded_sentence_ids:
            records.setdefault(str(record["sentence_id"]), record)
    frame = pd.DataFrame.from_records(list(records.values()))
    if frame.empty:
        return frame
    lengths = _token_lengths(tokenizer, frame["text"].tolist())
    frame["token_length"] = lengths
    frame = frame.loc[(lengths >= min_tokens) & (lengths <= max_tokens)].reset_index(drop=True)
    if sample_limit and len(frame) > sample_limit:
        chosen = np.sort(np.random.default_rng(seed).choice(len(frame), size=sample_limit, replace=False))
        frame = frame.iloc[chosen].reset_index(drop=True)
    return frame
