"""Leakage-free data preparation for Sticky / Attractor V2.

The V1 experiment split pair rows.  V2 instead treats normalized sentences as
graph vertices and keeps every connected component inside one split.  This
module deliberately contains no search logic so that every mode consumes the
same immutable partition.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .data import PairDataset


SPLITS = ("search", "validation", "test")


def normalize_sentence(text: str) -> str:
    """Apply the registered V2 identity normalization."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text)).strip())


def sentence_id(text: str) -> str:
    return hashlib.sha256(normalize_sentence(text).encode("utf-8")).hexdigest()


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


@dataclass
class SplitAudit:
    method: str
    component_count: int
    largest_component_pairs: int
    cross_split_pairs_dropped: int
    drop_rate: float
    pair_counts: dict[str, int]
    sentence_counts: dict[str, int]
    sentence_overlap_counts: dict[str, int]


def _quantile_bins(values: np.ndarray, bins: int = 10) -> np.ndarray:
    ranks = pd.Series(np.asarray(values, dtype=float)).rank(method="first").to_numpy()
    return np.minimum(bins - 1, ((ranks - 1) * bins / max(len(ranks), 1)).astype(int))


def _component_split(
    frame: pd.DataFrame,
    similarities: np.ndarray,
    *,
    fractions: tuple[float, float, float],
    seed: int,
) -> tuple[dict[str, np.ndarray], SplitAudit]:
    """Greedily balance sentence-graph components over the three splits."""
    uf = _UnionFind()
    for left, right in zip(frame["sentence1_id"], frame["sentence2_id"]):
        uf.union(str(left), str(right))
    roots = np.asarray([uf.find(str(value)) for value in frame["sentence1_id"]], dtype=object)
    sim_bins = _quantile_bins(similarities)
    pair_lengths = (
        frame["sentence1_token_length"].to_numpy(dtype=float)
        + frame["sentence2_token_length"].to_numpy(dtype=float)
    ) / 2.0
    length_bins = _quantile_bins(pair_lengths, bins=5)
    source_values = frame["source"].astype(str).to_numpy() if "source" in frame else np.asarray(["__single_source__"] * len(frame))
    source_names = sorted(set(source_values.tolist()))
    source_index = {name: index for index, name in enumerate(source_names)}
    source_codes = np.asarray([source_index[value] for value in source_values], dtype=int)
    components: list[dict[str, Any]] = []
    for root in sorted(set(roots.tolist())):
        indices = np.flatnonzero(roots == root)
        nodes = set(frame.iloc[indices]["sentence1_id"]) | set(frame.iloc[indices]["sentence2_id"])
        components.append(
            {
                "root": root,
                "indices": indices,
                "pairs": len(indices),
                "sentences": len(nodes),
                "bin_hist": np.bincount(sim_bins[indices], minlength=10).astype(float),
                "length_hist": np.bincount(length_bins[indices], minlength=5).astype(float),
                "source_hist": np.bincount(source_codes[indices], minlength=len(source_names)).astype(float),
                "mean_length": float(
                    np.mean(
                        np.concatenate(
                            [
                                frame.iloc[indices]["sentence1_token_length"].to_numpy(),
                                frame.iloc[indices]["sentence2_token_length"].to_numpy(),
                            ]
                        )
                    )
                ),
            }
        )

    # Largest and most distributionally unusual components are allocated
    # first; the seeded jitter makes exact ties reproducible.
    rng = np.random.default_rng(seed)
    jitter = {item["root"]: float(rng.random()) for item in components}
    components.sort(key=lambda item: (-item["pairs"], -item["sentences"], jitter[item["root"]]))
    fraction_map = dict(zip(SPLITS, fractions))
    target_pairs = {name: len(frame) * fraction_map[name] for name in SPLITS}
    total_hist = np.bincount(sim_bins, minlength=10).astype(float)
    target_hist = {name: total_hist * fraction_map[name] for name in SPLITS}
    total_length_hist = np.bincount(length_bins, minlength=5).astype(float)
    target_length_hist = {name: total_length_hist * fraction_map[name] for name in SPLITS}
    total_source_hist = np.bincount(source_codes, minlength=len(source_names)).astype(float)
    target_source_hist = {name: total_source_hist * fraction_map[name] for name in SPLITS}
    assigned: dict[str, list[int]] = {name: [] for name in SPLITS}
    pair_counts = {name: 0 for name in SPLITS}
    hist_counts = {name: np.zeros(10, dtype=float) for name in SPLITS}
    length_hist_counts = {name: np.zeros(5, dtype=float) for name in SPLITS}
    source_hist_counts = {name: np.zeros(len(source_names), dtype=float) for name in SPLITS}

    for offset, component in enumerate(components):
        if offset < len(SPLITS):
            split = SPLITS[offset]
        else:
            costs: dict[str, float] = {}
            for name in SPLITS:
                new_pairs = pair_counts[name] + component["pairs"]
                pair_cost = abs(new_pairs - target_pairs[name]) / max(target_pairs[name], 1.0)
                new_hist = hist_counts[name] + component["bin_hist"]
                hist_cost = float(np.mean(np.abs(new_hist - target_hist[name]) / np.maximum(target_hist[name], 1.0)))
                new_length_hist = length_hist_counts[name] + component["length_hist"]
                length_cost = float(
                    np.mean(
                        np.abs(new_length_hist - target_length_hist[name])
                        / np.maximum(target_length_hist[name], 1.0)
                    )
                )
                new_source_hist = source_hist_counts[name] + component["source_hist"]
                source_cost = float(
                    np.mean(
                        np.abs(new_source_hist - target_source_hist[name])
                        / np.maximum(target_source_hist[name], 1.0)
                    )
                )
                overflow = max(0.0, new_pairs - target_pairs[name]) / max(target_pairs[name], 1.0)
                costs[name] = pair_cost + 0.35 * hist_cost + 0.20 * length_cost + 0.20 * source_cost + 2.0 * overflow
            split = min(SPLITS, key=lambda name: (costs[name], pair_counts[name], name))
        assigned[split].extend(map(int, component["indices"]))
        pair_counts[split] += int(component["pairs"])
        hist_counts[split] += component["bin_hist"]
        length_hist_counts[split] += component["length_hist"]
        source_hist_counts[split] += component["source_hist"]

    arrays = {name: np.asarray(sorted(assigned[name]), dtype=int) for name in SPLITS}
    sentence_sets = {
        name: set(frame.iloc[indices]["sentence1_id"]) | set(frame.iloc[indices]["sentence2_id"])
        for name, indices in arrays.items()
    }
    overlap = {
        "search_validation": len(sentence_sets["search"] & sentence_sets["validation"]),
        "search_test": len(sentence_sets["search"] & sentence_sets["test"]),
        "validation_test": len(sentence_sets["validation"] & sentence_sets["test"]),
    }
    if any(overlap.values()):
        raise AssertionError(f"Sentence leakage detected: {overlap}")
    combined = np.concatenate(list(arrays.values()))
    if len(combined) != len(frame) or len(np.unique(combined)) != len(frame):
        raise AssertionError("Component split is not disjoint and exhaustive")
    audit = SplitAudit(
        method="sentence_graph_connected_components",
        component_count=len(components),
        largest_component_pairs=max((item["pairs"] for item in components), default=0),
        cross_split_pairs_dropped=0,
        drop_rate=0.0,
        pair_counts={name: len(indices) for name, indices in arrays.items()},
        sentence_counts={name: len(values) for name, values in sentence_sets.items()},
        sentence_overlap_counts=overlap,
    )
    return arrays, audit


def load_normalized_pairs(
    path: Path,
    tokenizer,
    *,
    min_tokens: int,
    max_tokens: int,
    source_token_budget: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path)
    required = {"sentence1", "sentence2"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Pair data missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=sorted(required)).copy()
    frame.insert(0, "source_row", frame.index.astype(int))
    for side in ("sentence1", "sentence2"):
        original = frame[side].astype(str).map(normalize_sentence)
        encoded = tokenizer(original.tolist(), add_special_tokens=False, truncation=False)["input_ids"]
        lengths = np.asarray([len(ids) for ids in encoded], dtype=int)
        frame[f"{side}_original"] = original
        frame[f"{side}_id"] = original.map(sentence_id)
        frame[f"{side}_token_length"] = lengths
        frame[side] = [tokenizer.decode(ids[:source_token_budget], clean_up_tokenization_spaces=False) for ids in encoded]
        frame[f"{side}_tokens_removed"] = np.maximum(lengths - source_token_budget, 0)
    keep = frame["sentence2_token_length"].between(min_tokens, max_tokens)
    frame = frame.loc[keep].reset_index(drop=True)
    removed = np.concatenate(
        [frame["sentence1_tokens_removed"].to_numpy(), frame["sentence2_tokens_removed"].to_numpy()]
    )
    audit = {
        "source_token_budget": int(source_token_budget),
        "truncated_sentence_rate": float(np.mean(removed > 0)),
        "total_source_tokens_removed": int(removed.sum()),
        "normalization": "Unicode NFC -> strip -> collapse whitespace; case and punctuation retained",
        "sentence_identity": "SHA-256(normalized UTF-8 text)",
    }
    return frame, audit


def build_v2_dataset(
    frame: pd.DataFrame,
    encoder,
    *,
    batch_size: int,
    seed: int,
    fractions: tuple[float, float, float],
    show_progress: bool,
) -> tuple[PairDataset, SplitAudit]:
    first = encoder.encode_texts(frame["sentence1"].tolist(), batch_size=batch_size, show_progress=show_progress)
    second = encoder.encode_texts(frame["sentence2"].tolist(), batch_size=batch_size, show_progress=show_progress)
    baseline = np.einsum("ij,ij->i", first, second, optimize=True)
    split_indices, split_audit = _component_split(frame, baseline, fractions=fractions, seed=seed)
    return PairDataset(frame, first, second, baseline, split_indices), split_audit


def unique_sentences(dataset: PairDataset, split: str) -> pd.DataFrame:
    rows: dict[str, dict[str, Any]] = {}
    for index in dataset.split_indices[split]:
        pair = dataset.frame.iloc[int(index)]
        for side in ("sentence1", "sentence2"):
            sid = str(pair[f"{side}_id"])
            rows.setdefault(
                sid,
                {
                    "sentence_id": sid,
                    "text": str(pair[side]),
                    "normalized_original": str(pair[f"{side}_original"]),
                    "source_row_first_seen": int(pair["source_row"]),
                },
            )
    return pd.DataFrame.from_records(list(rows.values())).sort_values("sentence_id", kind="mergesort").reset_index(drop=True)


def write_prepared_dataset(dataset: PairDataset, output: Path, split_audit: SplitAudit, truncation_audit: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    split_by_index = np.empty(len(dataset.frame), dtype=object)
    for name, indices in dataset.split_indices.items():
        split_by_index[indices] = name
    frame = dataset.frame.copy()
    frame.insert(1, "split", split_by_index)
    frame["baseline_similarity"] = dataset.baseline
    frame.to_csv(output / "prepared_pairs.csv", index=False)
    np.savez_compressed(
        output / "prepared_pair_embeddings.npz",
        sentence1=dataset.sentence1_embeddings,
        sentence2=dataset.sentence2_embeddings,
        baseline=dataset.baseline,
    )
    audit = {**split_audit.__dict__, **truncation_audit}
    (output / "split_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")


def load_prepared_dataset(output: Path) -> PairDataset:
    frame = pd.read_csv(output / "prepared_pairs.csv", keep_default_na=False)
    arrays = np.load(output / "prepared_pair_embeddings.npz")
    split_indices = {name: np.flatnonzero(frame["split"].to_numpy() == name) for name in SPLITS}
    return PairDataset(
        frame,
        np.asarray(arrays["sentence1"], dtype=np.float32),
        np.asarray(arrays["sentence2"], dtype=np.float32),
        np.asarray(arrays["baseline"], dtype=np.float32),
        split_indices,
    )


def assert_sentence_disjoint(dataset: PairDataset) -> None:
    ids = {
        name: set(dataset.frame.iloc[indices]["sentence1_id"]) | set(dataset.frame.iloc[indices]["sentence2_id"])
        for name, indices in dataset.split_indices.items()
    }
    assert ids["search"].isdisjoint(ids["validation"])
    assert ids["search"].isdisjoint(ids["test"])
    assert ids["validation"].isdisjoint(ids["test"])


def select_balanced(indices: Sequence[int], values: np.ndarray, count: int, seed: int) -> np.ndarray:
    positions = np.asarray(indices, dtype=int)
    if count <= 0 or len(positions) <= count:
        return positions.copy()
    rng = np.random.default_rng(seed)
    ordered = positions[np.argsort(np.asarray(values)[positions], kind="mergesort")]
    return np.asarray([group[int(rng.integers(0, len(group)))] for group in np.array_split(ordered, count)], dtype=int)
