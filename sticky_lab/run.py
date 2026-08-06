"""CLI runner for the three-mode Sticky / Attractor experiment framework."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from .config import load_config, resolve_path
from .data import PairDataset, build_dataset, load_pairs
from .insertion import insert_trigger, repeat_literal
from .metrics import (
    booster_metrics,
    exact_pairwise_mean,
    prefix_path_metrics,
    repulsive_attractor_metrics,
    single_sticky_score,
)
from .model import SentenceTransformerEncoder
from .search import run_search
from .seed import seed_everything
from .tokens import load_vocabulary, realizability_rate
from .visualization import curves_to_frame, plot_embedding_projection, plot_similarity_curves


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _rank_frame(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    if "feasible" not in ranked:
        ranked["feasible"] = False
    if "constraint_violation" not in ranked:
        ranked["constraint_violation"] = np.inf
    ranked = ranked.sort_values(["feasible", "constraint_violation", "objective"], ascending=[False, True, False], kind="mergesort").reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def _select_across_range(indices: Sequence[int], similarities: np.ndarray, count: int, seed: int) -> np.ndarray:
    positions = np.asarray(indices, dtype=int)
    if count <= 0 or len(positions) <= count:
        return positions
    rng = np.random.default_rng(seed)
    ordered = positions[np.argsort(similarities[positions], kind="mergesort")]
    return np.asarray([group[int(rng.integers(0, len(group)))] for group in np.array_split(ordered, count)], dtype=int)


def _tail_thresholds(dataset: PairDataset, config: dict[str, Any]) -> tuple[float, float]:
    grouping = config["similarity_groups"]
    search_values = dataset.baseline[dataset.split_indices["search"]]
    if grouping["method"] == "fixed":
        low, high = float(grouping["low_threshold"]), float(grouping["high_threshold"])
    elif grouping["method"] == "quantile":
        low = float(np.quantile(search_values, float(grouping["low_quantile"])))
        high = float(np.quantile(search_values, float(grouping["high_quantile"])))
    else:
        raise ValueError("similarity_groups.method must be fixed or quantile")
    if low >= high:
        raise ValueError("Low threshold must be below high threshold")
    return low, high


def _mode_indices(dataset: PairDataset, split: str, low: float, high: float, per_group: int, seed: int) -> np.ndarray:
    indices = dataset.split_indices[split]
    low_indices = indices[dataset.baseline[indices] <= low]
    high_indices = indices[dataset.baseline[indices] >= high]
    if not len(low_indices) or not len(high_indices):
        raise ValueError(f"{split} split lacks a low or high similarity group")
    selected_low = _select_across_range(low_indices, dataset.baseline, per_group, seed)
    selected_high = _select_across_range(high_indices, dataset.baseline, per_group, seed + 1)
    return np.concatenate([selected_low, selected_high])


def _one_sided_similarities(
    encoder: SentenceTransformerEncoder,
    dataset: PairDataset,
    indices: np.ndarray,
    triggers: Sequence[str],
    modes: Sequence[str],
    *,
    seed: int,
    separator: str,
    batch_size: int,
) -> np.ndarray:
    pairs = dataset.frame.iloc[indices]
    texts = [
        insert_trigger(str(text), trigger, mode, seed=seed + mode_index, separator=separator)
        for trigger in triggers
        for mode_index, mode in enumerate(modes)
        for text in pairs["sentence2"].tolist()
    ]
    embeddings = encoder.encode_texts(texts, batch_size=batch_size)
    embeddings = embeddings.reshape(len(triggers), len(modes), len(indices), -1)
    return np.einsum("pd,cmpd->cmp", dataset.sentence1_embeddings[indices], embeddings, optimize=True)


def _shared_embeddings(
    encoder: SentenceTransformerEncoder,
    dataset: PairDataset,
    indices: np.ndarray,
    triggers: Sequence[str],
    modes: Sequence[str],
    *,
    seed: int,
    separator: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    pairs = dataset.frame.iloc[indices]
    first_texts = [insert_trigger(str(text), trigger, mode, seed=seed + mode_index, separator=separator) for trigger in triggers for mode_index, mode in enumerate(modes) for text in pairs["sentence1"].tolist()]
    second_texts = [insert_trigger(str(text), trigger, mode, seed=seed + mode_index, separator=separator) for trigger in triggers for mode_index, mode in enumerate(modes) for text in pairs["sentence2"].tolist()]
    first = encoder.encode_texts(first_texts, batch_size=batch_size).reshape(len(triggers), len(modes), len(indices), -1)
    second = encoder.encode_texts(second_texts, batch_size=batch_size).reshape(len(triggers), len(modes), len(indices), -1)
    return first, second


def _write_split_manifest(dataset: PairDataset, output: Path, low: float | None, high: float | None) -> None:
    records: list[dict[str, Any]] = []
    for split, indices in dataset.split_indices.items():
        for index in indices:
            similarity = float(dataset.baseline[index])
            group = "all"
            if low is not None and high is not None:
                group = "low" if similarity <= low else "high" if similarity >= high else "middle"
            row = dataset.frame.iloc[index]
            records.append({"split": split, "group": group, "filtered_row": int(index), "source_row": int(row["source_row"]), "baseline_similarity": similarity, "sentence2_token_length": int(row["sentence2_token_length"]), "sentence1": row["sentence1"], "sentence2": row["sentence2"]})
    pd.DataFrame.from_records(records).to_csv(output / "split_manifest.csv", index=False)


def _plot_indices(dataset: PairDataset, count: int, seed: int) -> np.ndarray:
    return _select_across_range(dataset.split_indices["test"], dataset.baseline, count, seed)


def _repeat_curves(
    encoder: SentenceTransformerEncoder,
    dataset: PairDataset,
    indices: np.ndarray,
    trigger: str,
    *,
    mode: str,
    max_count: int,
    seed: int,
    separator: str,
    batch_size: int,
    shared: bool,
) -> np.ndarray:
    curves = np.empty((len(indices), max_count + 1), dtype=float)
    curves[:, 0] = dataset.baseline[indices]
    for count in range(1, max_count + 1):
        repeated = repeat_literal(trigger, count, separator="")
        if shared:
            first, second = _shared_embeddings(encoder, dataset, indices, [repeated], [mode], seed=seed, separator=separator, batch_size=batch_size)
            curves[:, count] = np.einsum("pd,pd->p", first[0, 0], second[0, 0], optimize=True)
        else:
            curves[:, count] = _one_sided_similarities(encoder, dataset, indices, [repeated], [mode], seed=seed, separator=separator, batch_size=batch_size)[0, 0]
    return curves


def _screen_booster_tokens(
    encoder: SentenceTransformerEncoder,
    dataset: PairDataset,
    indices: np.ndarray,
    vocabulary: pd.DataFrame,
    modes: Sequence[str],
    constraints: dict[str, float],
    *,
    seed: int,
    separator: str,
    batch_size: int,
    chunk_size: int,
    low: float,
    high: float,
) -> pd.DataFrame:
    baseline = dataset.baseline[indices]
    low_mask, high_mask = baseline <= low, baseline >= high
    frames: list[pd.DataFrame] = []
    for start in range(0, len(vocabulary), chunk_size):
        chunk = vocabulary.iloc[start : start + chunk_size].reset_index(drop=True)
        print(f"token screen {start + 1}-{start + len(chunk)} / {len(vocabulary)}", flush=True)
        similarities = _one_sided_similarities(encoder, dataset, indices, chunk["literal"].tolist(), modes, seed=seed, separator=separator, batch_size=batch_size)
        metrics = booster_metrics(similarities, baseline, low_mask, high_mask, constraints)
        frames.append(pd.concat([chunk, pd.DataFrame(metrics)], axis=1))
    return _rank_frame(pd.concat(frames, ignore_index=True))


def _build_candidate_pool(screen: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    if size > len(screen):
        size = len(screen)
    selected: list[int] = []
    selected_set: set[int] = set()

    def take(frame: pd.DataFrame, count: int) -> None:
        for index in frame.index:
            if int(index) not in selected_set:
                selected_set.add(int(index))
                selected.append(int(index))
            if len(selected) >= count:
                return

    half = size // 2
    take(screen, half)
    preservation = screen[screen["low_gain_mean"] > 0].sort_values(["high_gain_q05", "low_gain_mean"], ascending=[False, False], kind="mergesort")
    target = min(size, half + size // 4)
    take(preservation, target)
    rng = np.random.default_rng(seed)
    remaining = np.asarray([index for index in screen.index if int(index) not in selected_set], dtype=int)
    rng.shuffle(remaining)
    for index in remaining:
        if len(selected) >= size:
            break
        selected.append(int(index))
    result = screen.loc[selected].drop(columns=["rank"], errors="ignore").reset_index(drop=True)
    result.insert(0, "pool_index", np.arange(len(result)))
    return result


def _sequence_trigger(sequence: Sequence[int], pool: pd.DataFrame) -> str:
    return "".join(str(pool.iloc[int(index)]["literal"]) for index in sequence)


def _sequence_component_ids(sequence: Sequence[int], pool: pd.DataFrame) -> tuple[int, ...]:
    return tuple(int(pool.iloc[int(index)]["token_id"]) for index in sequence)


def _search_records_frame(records: list[dict[str, Any]], pool: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for record in records:
        sequence = tuple(record["sequence"])
        output.append({"pool_sequence": ",".join(map(str, sequence)), "component_token_ids": ",".join(map(str, _sequence_component_ids(sequence, pool))), "trigger": _sequence_trigger(sequence, pool), **{key: value for key, value in record.items() if key != "sequence"}})
    return _rank_frame(pd.DataFrame.from_records(output))


def run_single_sticky(config: dict[str, Any], encoder: SentenceTransformerEncoder, dataset: PairDataset, vocabulary: pd.DataFrame, output: Path) -> dict[str, Any]:
    insertion = config["insertion"]
    modes = list(insertion["modes"])
    seed, batch_size = int(config["seed"]), int(config["runtime"]["batch_size"])
    chunk_size = int(config["runtime"]["candidate_chunk_size"])
    print("Encoding the full realizable vocabulary for exact pairwise mean...", flush=True)
    token_embeddings = encoder.encode_texts(vocabulary["literal"].tolist(), batch_size=batch_size, show_progress=True)
    np.save(output / "token_embeddings.npy", token_embeddings)
    mean_similarity = exact_pairwise_mean(token_embeddings)
    _write_json(output / "model_similarity_mean.json", {"u": mean_similarity, "formula": "(||sum(z)||^2-M)/(M(M-1))", "accumulator_dtype": "float64", "valid_vocab_size": len(vocabulary)})

    search_available = dataset.split_indices["search"]
    search_available = search_available[dataset.baseline[search_available] < mean_similarity - float(config["single_sticky"]["pair_filter_margin"])]
    search_indices = _select_across_range(search_available, dataset.baseline, int(config["single_sticky"]["sample_pair_count"]), seed)
    if not len(search_indices):
        raise ValueError("No below-mean search pairs")
    repeat_count = int(config["single_sticky"]["repeat_count"])
    reference_mean = dataset.sentence1_embeddings[search_indices].mean(axis=0)
    screen_frames: list[pd.DataFrame] = []
    for start in range(0, len(vocabulary), chunk_size):
        chunk = vocabulary.iloc[start : start + chunk_size].reset_index(drop=True)
        print(f"sticky screen {start + 1}-{start + len(chunk)} / {len(vocabulary)}", flush=True)
        triggers = [repeat_literal(value, repeat_count) for value in chunk["literal"].tolist()]
        similarities = _one_sided_similarities(encoder, dataset, search_indices, triggers, modes, seed=seed, separator=insertion["separator"], batch_size=batch_size)
        semantic = token_embeddings[start : start + len(chunk)] @ reference_mean
        scores, parts = single_sticky_score(similarities, dataset.baseline[search_indices], semantic, alpha=float(config["single_sticky"]["alpha"]), beta=float(config["single_sticky"]["beta"]), gamma=float(config["single_sticky"]["gamma"]))
        scored = chunk.copy()
        scored["sticky_score"] = scores
        scored["semantic_penalty"] = semantic
        for name, values in parts.items():
            scored[name] = values
        screen_frames.append(scored)
    screen = pd.concat(screen_frames, ignore_index=True).sort_values("sticky_score", ascending=False, kind="mergesort").reset_index(drop=True)
    screen.insert(0, "rank", np.arange(1, len(screen) + 1))
    screen.to_csv(output / "all_token_scores.csv", index=False)
    candidate_count = max(1, int(np.ceil(len(screen) * float(config["single_sticky"]["candidate_ratio"]))))
    candidates = screen.head(candidate_count).copy()

    validation_available = dataset.split_indices["validation"]
    validation_available = validation_available[dataset.baseline[validation_available] < mean_similarity - float(config["single_sticky"]["pair_filter_margin"])]
    validation_indices = _select_across_range(validation_available, dataset.baseline, int(config["single_sticky"]["validation_pair_limit"]), seed + 2)
    ge_values: list[np.ndarray] = []
    for start in range(0, len(candidates), max(1, chunk_size // 2)):
        chunk = candidates.iloc[start : start + max(1, chunk_size // 2)]
        triggers = [repeat_literal(value, repeat_count) for value in chunk["literal"]]
        sims = _one_sided_similarities(encoder, dataset, validation_indices, triggers, modes, seed=seed, separator=insertion["separator"], batch_size=batch_size)
        ge_values.append(np.max(np.abs(sims - mean_similarity), axis=(1, 2)))
    candidates["validation_GE_max"] = np.concatenate(ge_values)
    validation = config["single_sticky"]["validation"]
    if validation["epsilon_mode"] == "fixed":
        epsilon = float(validation["epsilon"])
    else:
        q1, q3 = np.quantile(candidates["validation_GE_max"], [0.25, 0.75])
        epsilon = float(q3 + 1.5 * (q3 - q1))
    candidates["validation_pass"] = candidates["validation_GE_max"] <= epsilon

    test_available = dataset.split_indices["test"]
    test_available = test_available[dataset.baseline[test_available] < mean_similarity - float(config["single_sticky"]["pair_filter_margin"])]
    test_indices = _select_across_range(test_available, dataset.baseline, int(config["single_sticky"]["test_pair_limit"]), seed + 3)
    test_gaps: list[np.ndarray] = []
    for start in range(0, len(candidates), max(1, chunk_size // 2)):
        chunk = candidates.iloc[start : start + max(1, chunk_size // 2)]
        triggers = [repeat_literal(value, repeat_count) for value in chunk["literal"]]
        sims = _one_sided_similarities(encoder, dataset, test_indices, triggers, modes, seed=seed, separator=insertion["separator"], batch_size=batch_size)
        test_gaps.append(np.max(np.abs(sims - mean_similarity), axis=(1, 2)))
    candidates["test_GE_max"] = np.concatenate(test_gaps)
    candidates["test_pass"] = candidates["test_GE_max"] <= epsilon
    candidates["certified"] = candidates["validation_pass"] & candidates["test_pass"]
    candidates = candidates.sort_values(["certified", "test_GE_max", "sticky_score"], ascending=[False, True, False], kind="mergesort").reset_index(drop=True)
    candidates.to_csv(output / "candidate_validation.csv", index=False)
    candidates[candidates["certified"]].to_csv(output / "validated_sticky_tokens.csv", index=False)
    best = candidates.iloc[0]
    plot_indices = _plot_indices(dataset, int(config["plot"]["pair_count"]), seed + 4)
    curves = _repeat_curves(encoder, dataset, plot_indices, str(best["literal"]), mode=str(config["plot"]["insertion_mode"]), max_count=int(config["plot"]["max_insertions"]), seed=seed, separator=insertion["separator"], batch_size=batch_size, shared=False)
    curves_to_frame(curves, dataset.frame.iloc[plot_indices]).to_csv(output / "similarity_curves.csv", index=False)
    plot_similarity_curves(curves, output / "similarity_curves.png", xlabel="Inserted number of sticky token", dpi=int(config["plot"]["dpi"]))
    _write_json(output / "pair_filter_statistics.json", {"search_below_mean": len(search_available), "search_used": len(search_indices), "validation_below_mean": len(validation_available), "validation_used": len(validation_indices), "test_below_mean": len(test_available), "test_used": len(test_indices)})
    return {"mode": "single_sticky", "u": mean_similarity, "vocab_size": int(getattr(encoder.tokenizer, "vocab_size", len(vocabulary))), "valid_vocab_size": len(vocabulary), "repeat_count": repeat_count, "sample_pair_count": len(search_indices), "candidate_count": candidate_count, "epsilon": epsilon, "validated_count": int(candidates["certified"].sum()), "best_trigger": str(best["literal"]), "best_token_id": int(best["token_id"]), "best_test_GE_max": float(best["test_GE_max"])}


def _run_multi_mode(config: dict[str, Any], encoder: SentenceTransformerEncoder, dataset: PairDataset, vocabulary: pd.DataFrame, output: Path, *, repulsive: bool) -> dict[str, Any]:
    mode_name = "repulsive_attractor" if repulsive else "multi_booster"
    insertion = config["insertion"]
    modes = list(insertion["modes"])
    seed, batch_size = int(config["seed"]), int(config["runtime"]["batch_size"])
    low, high = _tail_thresholds(dataset, config)
    search_indices = _mode_indices(dataset, "search", low, high, int(config["data"]["pairs_per_group"]["search"]), seed)
    validation_indices = _mode_indices(dataset, "validation", low, high, int(config["data"]["pairs_per_group"]["validation"]), seed + 2)
    test_indices = _mode_indices(dataset, "test", low, high, int(config["data"]["pairs_per_group"]["test"]), seed + 4)
    constraints = {key: float(value) for key, value in config["constraints"].items()}

    screen_path = resolve_path(config, config.get("candidate_pool", {}).get("reuse_screen", "")) if config.get("candidate_pool", {}).get("reuse_screen") else None
    if screen_path is not None and screen_path.exists():
        print(f"Reusing token screen: {screen_path}", flush=True)
        screen = pd.read_csv(screen_path, keep_default_na=False)
    else:
        screen = _screen_booster_tokens(encoder, dataset, search_indices, vocabulary, modes, constraints if not repulsive else {**constraints, "min_low_coverage": float(config["candidate_pool"]["screen_min_low_coverage"]), "global_drop_tolerance": float(config["candidate_pool"]["screen_global_drop_tolerance"]), "max_global_drop_rate": float(config["candidate_pool"]["screen_max_global_drop_rate"]), "min_range_ratio": float(config["candidate_pool"]["screen_min_range_ratio"]), "min_spearman": float(config["candidate_pool"]["screen_min_spearman"]), "low_gain_margin": float(config["candidate_pool"]["screen_low_gain_margin"]), "coverage_weight": float(config["candidate_pool"]["screen_coverage_weight"])}, seed=seed, separator=insertion["separator"], batch_size=batch_size, chunk_size=int(config["runtime"]["candidate_chunk_size"]), low=low, high=high)
        screen.to_csv(output / "single_token_screen.csv", index=False)
    pool = _build_candidate_pool(screen, int(config["candidate_pool"]["size"]), seed)
    pool.to_csv(output / "candidate_pool.csv", index=False)
    search_cfg = config["search"]
    cache: dict[tuple[int, ...], dict[str, Any]] = {}

    def score_fn(sequences: list[tuple[int, ...]], iteration: int) -> list[dict[str, Any]]:
        missing = [sequence for sequence in sequences if sequence not in cache]
        if missing:
            triggers = [_sequence_trigger(sequence, pool) for sequence in missing]
            base = dataset.baseline[search_indices]
            low_mask, high_mask = base <= low, base >= high
            if repulsive:
                first, second = _shared_embeddings(encoder, dataset, search_indices, triggers, modes, seed=seed, separator=insertion["separator"], batch_size=batch_size)
                current_constraints = dict(constraints)
                curriculum = search_cfg.get("repulsion_curriculum")
                if curriculum:
                    fraction = min(1.0, iteration / max(int(search_cfg["iterations"]) - 1, 1))
                    current_constraints["min_displacement_q05"] = float(curriculum["start"] + fraction * (curriculum["end"] - curriculum["start"]))
                metrics = repulsive_attractor_metrics(first, second, dataset.sentence1_embeddings[search_indices], dataset.sentence2_embeddings[search_indices], base, low_mask, high_mask, current_constraints)
            else:
                similarities = _one_sided_similarities(encoder, dataset, search_indices, triggers, modes, seed=seed, separator=insertion["separator"], batch_size=batch_size)
                metrics = booster_metrics(similarities, base, low_mask, high_mask, constraints)
            cache.update(zip(missing, metrics))
        return [dict(cache[sequence]) for sequence in sequences]

    result = run_search(str(search_cfg["algorithm"]), len(pool), int(search_cfg["trigger_length"]), score_fn, search_cfg, seed)
    search_frame = _search_records_frame(result.candidates, pool)
    search_frame.to_csv(output / "search_candidates.csv", index=False)
    pd.DataFrame(result.history).to_csv(output / "search_history.csv", index=False)

    validation_count = min(int(config["validation"]["candidate_count"]), len(result.candidates))
    validation_records = result.candidates[:validation_count]
    validation_sequences = [tuple(record["sequence"]) for record in validation_records]
    validation_triggers = [_sequence_trigger(sequence, pool) for sequence in validation_sequences]
    validation_base = dataset.baseline[validation_indices]
    validation_low, validation_high = validation_base <= low, validation_base >= high
    if repulsive:
        first, second = _shared_embeddings(encoder, dataset, validation_indices, validation_triggers, modes, seed=seed, separator=insertion["separator"], batch_size=batch_size)
        validation_metrics = repulsive_attractor_metrics(first, second, dataset.sentence1_embeddings[validation_indices], dataset.sentence2_embeddings[validation_indices], validation_base, validation_low, validation_high, constraints)
    else:
        similarities = _one_sided_similarities(encoder, dataset, validation_indices, validation_triggers, modes, seed=seed, separator=insertion["separator"], batch_size=batch_size)
        validation_metrics = booster_metrics(similarities, validation_base, validation_low, validation_high, constraints)
    validation_frame = pd.DataFrame([{"pool_sequence": ",".join(map(str, sequence)), "component_token_ids": ",".join(map(str, _sequence_component_ids(sequence, pool))), "trigger": trigger, **metric} for sequence, trigger, metric in zip(validation_sequences, validation_triggers, validation_metrics)])
    validation_frame = _rank_frame(validation_frame)
    validation_frame.to_csv(output / "validation_candidates.csv", index=False)

    test_count = min(int(config["validation"]["test_candidate_count"]), len(validation_frame))
    test_input = validation_frame.head(test_count).copy()
    test_triggers = test_input["trigger"].tolist()
    test_sequences = [tuple(map(int, value.split(","))) for value in test_input["pool_sequence"]]
    test_base = dataset.baseline[test_indices]
    test_low, test_high = test_base <= low, test_base >= high
    if repulsive:
        test_first, test_second = _shared_embeddings(encoder, dataset, test_indices, test_triggers, modes, seed=seed, separator=insertion["separator"], batch_size=batch_size)
        test_metrics = repulsive_attractor_metrics(test_first, test_second, dataset.sentence1_embeddings[test_indices], dataset.sentence2_embeddings[test_indices], test_base, test_low, test_high, constraints)
    else:
        test_similarities = _one_sided_similarities(encoder, dataset, test_indices, test_triggers, modes, seed=seed, separator=insertion["separator"], batch_size=batch_size)
        test_metrics = booster_metrics(test_similarities, test_base, test_low, test_high, constraints)

    final_records: list[dict[str, Any]] = []
    for offset, (sequence, trigger, metric) in enumerate(zip(test_sequences, test_triggers, test_metrics)):
        intended_ids = _sequence_component_ids(sequence, pool)
        realized_ids = tuple(encoder.tokenize([trigger], add_special_tokens=False)[0])
        realization = realizability_rate(encoder.tokenizer, trigger, intended_ids, dataset.frame.iloc[test_indices]["sentence2"].tolist()[:32], modes, seed=seed, separator=insertion["separator"])
        extra: dict[str, Any] = {}
        if not repulsive:
            prefix_similarities = []
            for prefix_length in range(1, len(sequence) + 1):
                prefix = _sequence_trigger(sequence[:prefix_length], pool)
                prefix_similarities.append(_one_sided_similarities(encoder, dataset, test_indices, [prefix], modes, seed=seed, separator=insertion["separator"], batch_size=batch_size)[0])
            extra = prefix_path_metrics(np.asarray(prefix_similarities), test_base, drop_tolerance=float(config["validation"]["step_drop_tolerance"]), max_failure_rate=float(config["validation"]["max_step_failure_rate"]))
        certified = bool(metric["feasible"] and realization >= float(config["validation"]["min_realizability"]) and (repulsive or extra["path_pass"]))
        final_records.append({"pool_sequence": ",".join(map(str, sequence)), "component_token_ids": ",".join(map(str, intended_ids)), "realized_trigger_ids": ",".join(map(str, realized_ids)), "trigger": trigger, **metric, **extra, "realizability_rate": realization, "text_realizable": realization >= float(config["validation"]["min_realizability"]), "certified": certified})
    final = pd.DataFrame.from_records(final_records).sort_values(["certified", "feasible", "constraint_violation", "objective"], ascending=[False, False, True, False], kind="mergesort").reset_index(drop=True)
    final.insert(0, "rank", np.arange(1, len(final) + 1))
    final.to_csv(output / "test_candidates.csv", index=False)
    final[final["certified"]].to_csv(output / "certified_candidates.csv", index=False)
    best = final.iloc[0]
    plot_indices = _plot_indices(dataset, int(config["plot"]["pair_count"]), seed + 6)
    curves = _repeat_curves(encoder, dataset, plot_indices, str(best["trigger"]), mode=str(config["plot"]["insertion_mode"]), max_count=int(config["plot"]["max_insertions"]), seed=seed, separator=insertion["separator"], batch_size=batch_size, shared=repulsive)
    curves_to_frame(curves, dataset.frame.iloc[plot_indices]).to_csv(output / "similarity_curves.csv", index=False)
    xlabel = "Inserted number of attractor string" if repulsive else "Inserted number of sticky_high token"
    plot_similarity_curves(curves, output / ("inserted_number_of_attractor_string.png" if repulsive else "inserted_number_of_sticky_high_token.png"), xlabel=xlabel, dpi=int(config["plot"]["dpi"]))
    if repulsive:
        projection_first, projection_second = _shared_embeddings(encoder, dataset, plot_indices, [str(best["trigger"])], [str(config["plot"]["insertion_mode"])], seed=seed, separator=insertion["separator"], batch_size=batch_size)
        original = np.concatenate([dataset.sentence1_embeddings[plot_indices], dataset.sentence2_embeddings[plot_indices]], axis=0)
        triggered = np.concatenate([projection_first[0, 0], projection_second[0, 0]], axis=0)
        plot_embedding_projection(original, triggered, output / "embedding_projection.png", dpi=int(config["plot"]["dpi"]))
    return {"mode": mode_name, "low_threshold": low, "high_threshold": high, "vocab_screen_count": len(screen), "candidate_pool_size": len(pool), "search_archive_size": len(search_frame), "validation_candidate_count": len(validation_frame), "test_candidate_count": len(final), "certified_count": int(final["certified"].sum()), "best_trigger": str(best["trigger"]), "best_component_token_ids": str(best["component_token_ids"]), "best_feasible": bool(best["feasible"]), "best_certified": bool(best["certified"]), "best_constraint_violation": float(best["constraint_violation"]), "best_objective": float(best["objective"]), "best_metrics": {key: value for key, value in best.to_dict().items() if key not in {"trigger", "pool_sequence"}}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a deterministic, explicitly non-scientific 64-token pipeline check.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config["model"]["device"] = args.device
    if args.smoke:
        config["vocabulary"]["max_candidates"] = 64
        config["runtime"]["batch_size"] = 32
        config["runtime"]["candidate_chunk_size"] = 16
        config["runtime"]["show_progress"] = False
        config["plot"]["pair_count"] = 5
        config["plot"]["max_insertions"] = 3
        if config["mode"] == "single_sticky":
            config["single_sticky"]["candidate_ratio"] = 0.10
            config["single_sticky"]["validation_pair_limit"] = 8
            config["single_sticky"]["test_pair_limit"] = 8
        else:
            config["data"]["pairs_per_group"] = {"search": 2, "validation": 3, "test": 4}
            config["candidate_pool"]["size"] = 16
            config["candidate_pool"]["reuse_screen"] = None
            config["search"]["population_size"] = 12
            config["search"]["iterations"] = 2
            config["validation"]["candidate_count"] = 4
            config["validation"]["test_candidate_count"] = 2
    seed_everything(int(config["seed"]))
    repo_root = Path(__file__).resolve().parents[1]
    output = (args.output_dir.resolve() if args.output_dir else resolve_path(config, config["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "resolved_config.json", config)
    started = time.time()
    encoder = SentenceTransformerEncoder(
        config["model"]["id"],
        config["model"]["device"],
        str(resolve_path(config, config["model"]["cache_folder"])) if config["model"].get("cache_folder") else None,
        bool(config["model"].get("trust_remote_code", False)),
        config["model"].get("local_path"),
    )
    data_path = resolve_path(config, config["data"]["path"])
    frame = load_pairs(data_path, encoder.tokenizer, min_tokens=int(config["data"]["min_tokens"]), max_tokens=int(config["data"]["max_tokens"]))
    fractions = config["data"]["split"]
    dataset = build_dataset(frame, encoder, batch_size=int(config["runtime"]["batch_size"]), seed=int(config["seed"]), fractions=(float(fractions["search"]), float(fractions["validation"]), float(fractions["test"])), show_progress=bool(config["runtime"].get("show_progress", True)))
    vocabulary = load_vocabulary(resolve_path(config, config["vocabulary"]["analysis_path"]), allow_special=bool(config["vocabulary"]["allow_special_tokens"]), max_chars=int(config["vocabulary"]["max_chars"]))
    if config["vocabulary"].get("max_candidates"):
        vocabulary = vocabulary.head(int(config["vocabulary"]["max_candidates"])).reset_index(drop=True)
    low = high = None
    if config["mode"] != "single_sticky":
        low, high = _tail_thresholds(dataset, config)
    _write_split_manifest(dataset, output, low, high)
    if config["mode"] == "single_sticky":
        summary = run_single_sticky(config, encoder, dataset, vocabulary, output)
    elif config["mode"] == "multi_booster":
        summary = _run_multi_mode(config, encoder, dataset, vocabulary, output, repulsive=False)
    else:
        summary = _run_multi_mode(config, encoder, dataset, vocabulary, output, repulsive=True)
    summary.update({"model_id": config["model"]["id"], "model_revision": encoder.revision, "pooling": "model-provided SentenceTransformer pooling", "precision": "float32 search and final validation", "seed": int(config["seed"]), "data_path": str(data_path), "data_rows_after_filter": len(frame), "split_sizes": {key: len(value) for key, value in dataset.split_indices.items()}, "insertion_modes": config["insertion"]["modes"], "git_commit": _git_commit(repo_root), "runtime_seconds": time.time() - started, "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__, "sentence_transformers": _package_version("sentence-transformers"), "transformers": _package_version("transformers"), "device": config["model"]["device"], "cuda_name": torch.cuda.get_device_name(config["model"]["device"]) if str(config["model"]["device"]).startswith("cuda") and torch.cuda.is_available() else None}})
    _write_json(output / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
