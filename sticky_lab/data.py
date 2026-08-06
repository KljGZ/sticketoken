"""Pair loading and deterministic stratified three-way splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PairDataset:
    frame: pd.DataFrame
    sentence1_embeddings: np.ndarray
    sentence2_embeddings: np.ndarray
    baseline: np.ndarray
    split_indices: dict[str, np.ndarray]


def load_pairs(path: Path, tokenizer, *, min_tokens: int, max_tokens: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"sentence1", "sentence2"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Pair data missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=sorted(required)).copy()
    frame.insert(0, "source_row", frame.index.astype(int))
    for column in required:
        frame[column] = frame[column].astype(str)
    lengths = [len(ids) for ids in tokenizer(frame["sentence2"].tolist(), add_special_tokens=False, truncation=False)["input_ids"]]
    frame["sentence2_token_length"] = lengths
    return frame[frame["sentence2_token_length"].between(min_tokens, max_tokens)].reset_index(drop=True)


def stratified_split(similarities: np.ndarray, *, fractions: tuple[float, float, float], seed: int, bins: int = 10) -> dict[str, np.ndarray]:
    values = np.asarray(similarities, dtype=float)
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("At least three finite similarities are required")
    rng = np.random.default_rng(seed)
    groups = np.array_split(np.argsort(values, kind="mergesort"), min(bins, len(values)))
    output: dict[str, list[int]] = {"search": [], "validation": [], "test": []}
    for group in groups:
        shuffled = np.asarray(group, dtype=int).copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        search_n = int(round(n * fractions[0]))
        validation_n = int(round(n * fractions[1]))
        if n >= 3:
            search_n = max(1, min(search_n, n - 2))
            validation_n = max(1, min(validation_n, n - search_n - 1))
        test_start = search_n + validation_n
        output["search"].extend(shuffled[:search_n].tolist())
        output["validation"].extend(shuffled[search_n:test_start].tolist())
        output["test"].extend(shuffled[test_start:].tolist())
    arrays = {name: np.asarray(sorted(indices), dtype=int) for name, indices in output.items()}
    flattened = np.concatenate(list(arrays.values()))
    if len(flattened) != len(values) or len(np.unique(flattened)) != len(values):
        raise AssertionError("Split is not a disjoint exhaustive partition")
    return arrays


def build_dataset(frame: pd.DataFrame, encoder, *, batch_size: int, seed: int, fractions: tuple[float, float, float], show_progress: bool) -> PairDataset:
    first = encoder.encode_texts(frame["sentence1"].tolist(), batch_size=batch_size, show_progress=show_progress)
    second = encoder.encode_texts(frame["sentence2"].tolist(), batch_size=batch_size, show_progress=show_progress)
    baseline = np.einsum("ij,ij->i", first, second, optimize=True)
    return PairDataset(frame, first, second, baseline, stratified_split(baseline, fractions=fractions, seed=seed))

