"""Fresh, leakage-audited V5 calibration/search/validation/test/OOD roles."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

import numpy as np
import pandas as pd


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value)).strip())


def text_id(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _read(paths: Sequence[Path]) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    names = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["__source_file"] = str(path)
        frames.append(frame)
        names.append(str(path))
    if not frames:
        raise ValueError("no V5 input CSV files resolved")
    return pd.concat(frames, ignore_index=True), names


def _records(frame: pd.DataFrame, columns: Iterable[str], source_column: str | None) -> list[dict[str, Any]]:
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
                    "\0".join(sorted({str(previous["source_group"]), str(record["source_group"])})).encode()
                ).hexdigest()
    return list(unique.values())


def _length_filter(adapter, frame: pd.DataFrame, minimum: int, maximum: int) -> pd.DataFrame:
    lengths = np.asarray([len(adapter.encode_without_special_tokens(text)) for text in frame["text"]], dtype=int)
    result = frame.copy()
    result["token_length"] = lengths
    return result.loc[(lengths >= minimum) & (lengths <= maximum)].reset_index(drop=True)


def _assign_exact_roles(frame: pd.DataFrame, sizes: Mapping[str, int], seed: int) -> dict[str, pd.DataFrame]:
    group_rows = {str(group): part.index.to_numpy() for group, part in frame.groupby("source_group", sort=False)}
    groups = np.asarray(sorted(group_rows), dtype=object)
    np.random.default_rng(seed).shuffle(groups)
    cursor = 0
    roles: dict[str, pd.DataFrame] = {}
    used: set[int] = set()
    for role, raw_target in sizes.items():
        target = int(raw_target)
        selected: list[int] = []
        while len(selected) < target and cursor < len(groups):
            rows = list(map(int, group_rows[str(groups[cursor])]))
            cursor += 1
            if len(selected) + len(rows) <= target:
                selected.extend(rows)
        if len(selected) != target:
            raise ValueError(f"could not allocate exact V5 role {role}={target}; obtained {len(selected)}")
        if used.intersection(selected):
            raise AssertionError("V5 role allocation reused a sentence")
        used.update(selected)
        part = frame.loc[selected].copy().sort_values("sentence_id", kind="mergesort").reset_index(drop=True)
        part["role"] = role
        roles[role] = part
    return roles


def _overlap_audit(roles: Mapping[str, pd.DataFrame]) -> dict[str, int]:
    names = list(roles)
    result: dict[str, int] = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            result[f"{left}__{right}__sentences"] = len(
                set(roles[left]["sentence_id"]) & set(roles[right]["sentence_id"])
            )
            result[f"{left}__{right}__groups"] = len(
                set(roles[left]["source_group"].astype(str)) & set(roles[right]["source_group"].astype(str))
            )
    return result


def build_v5_corpus(
    iid_paths: Sequence[Path],
    ood_paths: Sequence[Path],
    adapter,
    *,
    text_columns: Sequence[str],
    ood_text_columns: Sequence[str],
    source_column: str | None,
    minimum_tokens: int,
    maximum_tokens: int,
    iid_sizes: Mapping[str, int],
    ood_sizes: Mapping[str, int],
    seed: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, Any]]:
    raw, iid_names = _read(iid_paths)
    iid_records = _records(raw, text_columns, source_column)
    iid = _length_filter(adapter, pd.DataFrame.from_records(iid_records), minimum_tokens, maximum_tokens)
    roles = _assign_exact_roles(iid, iid_sizes, seed)
    overlaps = _overlap_audit(roles)
    if any(overlaps.values()):
        raise AssertionError(f"V5 IID leakage: {overlaps}")
    used_ids = set().union(*(set(frame["sentence_id"]) for frame in roles.values()))

    ood_raw, ood_names = _read(ood_paths)
    ood_records = [record for record in _records(ood_raw, ood_text_columns, None) if record["sentence_id"] not in used_ids]
    ood = _length_filter(adapter, pd.DataFrame.from_records(ood_records), minimum_tokens, maximum_tokens)
    ood_roles = _assign_exact_roles(ood, ood_sizes, seed + 900001)
    ood_overlaps = _overlap_audit(ood_roles)
    if any(ood_overlaps.values()):
        raise AssertionError(f"V5 OOD leakage: {ood_overlaps}")
    if any(set(frame["sentence_id"]) & used_ids for frame in ood_roles.values()):
        raise AssertionError("V5 OOD overlaps IID")

    audit = {
        "protocol_version": 5,
        "seed": int(seed),
        "method": "global_unique_text_then_disjoint_group_role_allocation",
        "iid_input_files": iid_names,
        "ood_input_files": ood_names,
        "iid_input_rows": int(len(raw)),
        "unique_before_length_filter": int(len(iid_records)),
        "unique_after_length_filter": int(len(iid)),
        "iid_role_sizes": {role: int(len(frame)) for role, frame in roles.items()},
        "ood_role_sizes": {role: int(len(frame)) for role, frame in ood_roles.items()},
        "iid_overlap": overlaps,
        "ood_overlap": ood_overlaps,
        "ood_overlap_with_iid": 0,
        "document_provenance_available": bool(source_column and source_column in raw.columns),
        "fallback_grouping": None if source_column and source_column in raw.columns else "one_group_per_unique_sentence",
        "minimum_tokens": int(minimum_tokens),
        "maximum_tokens": int(maximum_tokens),
        "normalization": "Unicode NFC -> strip -> collapse whitespace; case and punctuation retained",
        "test_and_ood_embeddings_sealed": True,
    }
    return roles, ood_roles, audit
