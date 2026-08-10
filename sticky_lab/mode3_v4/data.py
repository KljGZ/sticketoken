"""Leakage-free role-separated corpus construction for V4."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Iterable, Sequence
import unicodedata

import numpy as np
import pandas as pd


IID_ROLES = (
    "search_trigger",
    "search_benign",
    "validation_trigger",
    "validation_benign",
    "test_trigger",
    "test_benign",
)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value)).strip())


def text_id(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _read(paths: Sequence[Path]) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    names: list[str] = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["__source_file"] = str(path)
        frames.append(frame)
        names.append(str(path))
    if not frames:
        raise ValueError("No V4 input CSV files were resolved")
    return pd.concat(frames, ignore_index=True), names


def _records(
    frame: pd.DataFrame,
    columns: Iterable[str],
    source_column: str | None,
) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row_index, row in frame.iterrows():
        source = str(row[source_column]) if source_column and source_column in frame.columns else None
        for column in columns:
            if column not in frame.columns or pd.isna(row[column]):
                continue
            text = normalize_text(str(row[column]))
            if not text:
                continue
            identity = text_id(text)
            record = {
                "sentence_id": identity,
                "text": text,
                "source_group": source or identity,
                "source_row_first_seen": int(row_index),
                "source_column": str(column),
                "source_file": str(row.get("__source_file", "")),
            }
            previous = unique.get(identity)
            if previous is None:
                unique[identity] = record
            elif previous["source_group"] != record["source_group"]:
                previous["source_group"] = "shared:" + hashlib.sha256(
                    "\0".join(sorted({str(previous["source_group"]), str(record["source_group"])})).encode("utf-8")
                ).hexdigest()
    return list(unique.values())


def _length_filter(adapter, frame: pd.DataFrame, minimum: int, maximum: int) -> pd.DataFrame:
    lengths = np.asarray([len(adapter.encode_without_special_tokens(text)) for text in frame["text"]], dtype=int)
    result = frame.copy()
    result["token_length"] = lengths
    return result.loc[(lengths >= minimum) & (lengths <= maximum)].reset_index(drop=True)


def _ordered_groups(frame: pd.DataFrame, seed: int) -> list[str]:
    groups = sorted(set(frame["source_group"].astype(str)))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    return groups


def _assign_exact_roles(frame: pd.DataFrame, sizes: dict[str, int], seed: int) -> dict[str, pd.DataFrame]:
    if set(sizes) != set(IID_ROLES):
        raise ValueError(f"V4 IID sizes must define exactly {IID_ROLES}")
    group_rows = {group: part.index.to_numpy() for group, part in frame.groupby("source_group", sort=False)}
    ordered = _ordered_groups(frame, seed)
    cursor = 0
    roles: dict[str, pd.DataFrame] = {}
    used_indices: set[int] = set()
    for role in IID_ROLES:
        target = int(sizes[role])
        selected: list[int] = []
        while len(selected) < target and cursor < len(ordered):
            rows = list(map(int, group_rows[ordered[cursor]]))
            cursor += 1
            if len(selected) + len(rows) > target:
                # Exact size is possible for the current sentence-level fallback.
                # For real multi-row groups, preserve group integrity and stop short.
                continue
            selected.extend(rows)
        if len(selected) != target:
            raise ValueError(f"Could not allocate exact V4 role size {role}={target}; obtained {len(selected)}")
        used_indices.update(selected)
        part = frame.loc[selected].copy().sort_values("sentence_id", kind="mergesort").reset_index(drop=True)
        part["role"] = role
        roles[role] = part
    if sum(map(len, roles.values())) != len(used_indices):
        raise AssertionError("V4 role allocation reused a sentence")
    return roles


def build_v4_corpus(
    iid_paths: Sequence[Path],
    ood_paths: Sequence[Path],
    adapter,
    *,
    text_columns: Sequence[str],
    ood_text_columns: Sequence[str],
    source_column: str | None,
    minimum_tokens: int,
    maximum_tokens: int,
    iid_sizes: dict[str, int],
    ood_size: int,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    raw, iid_names = _read(iid_paths)
    before = _records(raw, text_columns, source_column)
    iid = _length_filter(adapter, pd.DataFrame.from_records(before), minimum_tokens, maximum_tokens)
    roles = _assign_exact_roles(iid, iid_sizes, seed)
    used_ids = set().union(*(set(part["sentence_id"]) for part in roles.values()))
    used_groups = {name: set(part["source_group"].astype(str)) for name, part in roles.items()}
    overlaps: dict[str, int] = {}
    for left_index, left in enumerate(IID_ROLES):
        for right in IID_ROLES[left_index + 1 :]:
            overlaps[f"{left}__{right}__groups"] = len(used_groups[left] & used_groups[right])
            overlaps[f"{left}__{right}__sentences"] = len(
                set(roles[left]["sentence_id"]) & set(roles[right]["sentence_id"])
            )
    if any(overlaps.values()):
        raise AssertionError(f"V4 role leakage detected: {overlaps}")

    ood_raw, ood_names = _read(ood_paths)
    ood_records = [record for record in _records(ood_raw, ood_text_columns, None) if record["sentence_id"] not in used_ids]
    ood = _length_filter(adapter, pd.DataFrame.from_records(ood_records), minimum_tokens, maximum_tokens)
    if len(ood) < ood_size:
        raise ValueError(f"V4 requested {ood_size} OOD texts but only {len(ood)} are available")
    chosen = np.sort(np.random.default_rng(seed + 900001).choice(len(ood), size=ood_size, replace=False))
    ood = ood.iloc[chosen].copy().sort_values("sentence_id", kind="mergesort").reset_index(drop=True)
    ood["role"] = "ood"
    provenance = bool(source_column and source_column in raw.columns)
    audit = {
        "protocol_version": 4,
        "method": "global_unique_text_then_disjoint_role_allocation",
        "iid_input_files": iid_names,
        "ood_input_files": ood_names,
        "iid_input_rows": int(len(raw)),
        "unique_before_length_filter": int(len(before)),
        "unique_after_length_filter": int(len(iid)),
        "role_sizes": {name: int(len(part)) for name, part in roles.items()},
        "ood_size": int(len(ood)),
        "overlap": overlaps,
        "ood_overlap_with_iid": int(len(set(ood["sentence_id"]) & used_ids)),
        "document_provenance_available": provenance,
        "fallback_grouping": None if provenance else "one_group_per_unique_sentence",
        "minimum_tokens": int(minimum_tokens),
        "maximum_tokens": int(maximum_tokens),
        "normalization": "Unicode NFC -> strip -> collapse whitespace; case and punctuation retained",
    }
    return roles, ood, audit
