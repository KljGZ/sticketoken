"""Sticky / Attractor V2 reproducible experiment runner.

Examples
--------
python -m sticky_lab.v2 --config configs/v2_multi_booster.yaml --phase prepare
python -m sticky_lab.v2 --config configs/v2_multi_booster.yaml --phase search --restart 0
python -m sticky_lab.v2 --config configs/v2_multi_booster.yaml --phase finalize
python -m sticky_lab.v2 --config configs/v2_single_sticky.yaml --phase full
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from .insertion import insert_trigger, repeat_literal
from .metrics import exact_pairwise_mean, single_sticky_score
from .model import SentenceTransformerEncoder
from .seed import seed_everything
from .tokens import load_vocabulary, trigger_realizability
from .v2_data import (
    assert_sentence_disjoint,
    build_v2_dataset,
    load_normalized_pairs,
    load_prepared_dataset,
    select_balanced,
    unique_sentences,
    write_prepared_dataset,
)
from .v2_metrics import dose_certification, mode2_metrics, mode2_sort_key, mode3_metrics, mode3_sort_key
from .v2_search import cem_search_v2, expand_warm_sequences
from .v2_visualization import plot_embedding_progression, plot_length_frontier
from .visualization import plot_similarity_curves


MODES = {"single_sticky", "multi_booster", "repulsive_attractor"}


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
    path.parent.mkdir(parents=True, exist_ok=True)
    def clean(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: clean(current) for key, current in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(current) for current in item]
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return None
        return item

    path.write_text(
        json.dumps(clean(value), indent=2, ensure_ascii=False, default=_json_default, allow_nan=False),
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_config(path: Path) -> dict[str, Any]:
    source = path.resolve()
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("mode") not in MODES:
        raise ValueError(f"V2 config mode must be one of {sorted(MODES)}")
    config["config_path"] = str(source)
    if config.get("protocol_version") != 2:
        raise ValueError("V2 configs must set protocol_version: 2")
    if config["mode"] != "single_sticky":
        search = config["search"]
        minimum = int(search["min_trigger_length"])
        maximum = int(search["max_trigger_length"])
        step = int(search["length_step"])
        if minimum < 2 or maximum < minimum or step < 1:
            raise ValueError("Invalid registered V2 length schedule")
        if int(config.get("repeat_count", 1)) != 1:
            raise ValueError("Modes 2 and 3 insert each optimized combination exactly once")
    return config


def _resolve(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    config_path = Path(config["config_path"])
    root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    return (root / path).resolve()


def _encoder(config: dict[str, Any]) -> SentenceTransformerEncoder:
    model = config["model"]
    return SentenceTransformerEncoder(
        model["id"],
        model["device"],
        str(_resolve(config, model["cache_folder"])) if model.get("cache_folder") else None,
        bool(model.get("trust_remote_code", False)),
        model.get("local_path"),
    )


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment(config: dict[str, Any], encoder: SentenceTransformerEncoder) -> dict[str, Any]:
    device = str(config["model"]["device"])
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sentence_transformers": _package_version("sentence-transformers"),
        "transformers": _package_version("transformers"),
        "device": device,
        "cuda_name": torch.cuda.get_device_name(device) if device.startswith("cuda") and torch.cuda.is_available() else None,
        "model_revision": encoder.revision,
    }


def _lengths(config: dict[str, Any]) -> list[int]:
    search = config["search"]
    return list(
        range(
            int(search["min_trigger_length"]),
            int(search["max_trigger_length"]) + 1,
            int(search["length_step"]),
        )
    )


def _sequence_trigger(sequence: Sequence[int], pool: pd.DataFrame) -> str:
    return "".join(str(pool.iloc[int(index)]["literal"]) for index in sequence)


def _sequence_ids(sequence: Sequence[int], pool: pd.DataFrame) -> tuple[int, ...]:
    return tuple(int(pool.iloc[int(index)]["token_id"]) for index in sequence)


def _parse_ids(value: Any) -> tuple[int, ...]:
    text = str(value).strip()
    return tuple(int(item) for item in text.split(",") if item != "")


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    output = dict(record)
    if "per_mode_metrics" in output:
        output["per_mode_metrics_json"] = json.dumps(output.pop("per_mode_metrics"), ensure_ascii=False)
    return output


def _tail_thresholds(dataset, config: dict[str, Any]) -> tuple[float, float]:
    search_values = dataset.baseline[dataset.split_indices["search"]]
    groups = config["similarity_groups"]
    low = float(np.quantile(search_values, float(groups.get("low_quantile", 0.30))))
    high = float(np.quantile(search_values, float(groups.get("high_quantile", 0.70))))
    if low >= high:
        raise ValueError("Frozen low threshold must be below high threshold")
    return low, high


def _one_sided_similarities(
    encoder: SentenceTransformerEncoder,
    dataset,
    indices: np.ndarray,
    triggers: Sequence[str],
    modes: Sequence[str],
    *,
    seed: int,
    separator: str,
    batch_size: int,
) -> np.ndarray:
    frame = dataset.frame.iloc[indices]
    texts = [
        insert_trigger(str(text), trigger, mode, seed=seed + mode_index, separator=separator)
        for trigger in triggers
        for mode_index, mode in enumerate(modes)
        for text in frame["sentence2"].tolist()
    ]
    embeddings = encoder.encode_texts(texts, batch_size=batch_size)
    embeddings = embeddings.reshape(len(triggers), len(modes), len(indices), -1)
    return np.einsum("pd,cmpd->cmp", dataset.sentence1_embeddings[indices], embeddings, optimize=True)


def _shared_embeddings(
    encoder: SentenceTransformerEncoder,
    texts: Sequence[str],
    triggers: Sequence[str],
    modes: Sequence[str],
    *,
    seed: int,
    separator: str,
    batch_size: int,
) -> np.ndarray:
    modified = [
        insert_trigger(str(text), trigger, mode, seed=seed + mode_index, separator=separator)
        for trigger in triggers
        for mode_index, mode in enumerate(modes)
        for text in texts
    ]
    return encoder.encode_texts(modified, batch_size=batch_size).reshape(
        len(triggers), len(modes), len(texts), -1
    )


def _prepare_common(config: dict[str, Any], encoder: SentenceTransformerEncoder, output: Path):
    runtime = config["runtime"]
    max_trigger_tokens = int(runtime["max_trigger_tokens_reserved"])
    source_budget = max(1, encoder.max_length - max_trigger_tokens - int(runtime.get("reserved_special_tokens", 1)))
    frame, truncation_audit = load_normalized_pairs(
        _resolve(config, config["data"]["path"]),
        encoder.tokenizer,
        min_tokens=int(config["data"]["min_tokens"]),
        max_tokens=int(config["data"]["max_tokens"]),
        source_token_budget=source_budget,
    )
    fractions = config["data"]["split"]
    dataset, split_audit = build_v2_dataset(
        frame,
        encoder,
        batch_size=int(runtime["batch_size"]),
        seed=int(config["seed"]),
        fractions=(float(fractions["search"]), float(fractions["validation"]), float(fractions["test"])),
        show_progress=bool(runtime.get("show_progress", True)),
    )
    assert_sentence_disjoint(dataset)
    write_prepared_dataset(dataset, output, split_audit, truncation_audit)
    return dataset


def _mode2_evaluate(
    encoder: SentenceTransformerEncoder,
    dataset,
    indices: np.ndarray,
    triggers: Sequence[str],
    config: dict[str, Any],
    low: float,
    high: float,
) -> list[dict[str, Any]]:
    modes = list(config["insertion"]["modes"])
    chunk_size = int(config["runtime"].get("evaluation_candidate_chunk_size", 16))
    records: list[dict[str, Any]] = []
    base = dataset.baseline[indices]
    low_mask, high_mask = base <= low, base >= high
    for start in range(0, len(triggers), chunk_size):
        chunk = list(triggers[start : start + chunk_size])
        similarities = _one_sided_similarities(
            encoder,
            dataset,
            indices,
            chunk,
            modes,
            seed=int(config["seed"]),
            separator=str(config["insertion"].get("separator", "")),
            batch_size=int(config["runtime"]["batch_size"]),
        )
        records.extend(
            mode2_metrics(
                similarities,
                base,
                low_mask,
                high_mask,
                {key: float(value) for key, value in config["constraints"].items()},
                low_threshold=low,
                high_threshold=high,
            )
        )
    return records


def _load_unique(output: Path, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(output / f"unique_{split}.csv", keep_default_na=False)
    embeddings = np.load(output / f"unique_{split}_embeddings.npy")
    return frame, np.asarray(embeddings, dtype=np.float32)


def _mode3_evaluate(
    encoder: SentenceTransformerEncoder,
    unique_frame: pd.DataFrame,
    original: np.ndarray,
    triggers: Sequence[str],
    config: dict[str, Any],
    centers: np.ndarray,
    radii: np.ndarray,
    *,
    include_density: bool,
    search_benign: np.ndarray | None = None,
    benign_knn_q95: float | None = None,
) -> list[dict[str, Any]]:
    modes = list(config["insertion"]["modes"])
    chunk_size = int(config["runtime"].get("evaluation_candidate_chunk_size", 16))
    records: list[dict[str, Any]] = []
    labels = unique_frame["source_cluster"].to_numpy(dtype=int)
    for start in range(0, len(triggers), chunk_size):
        chunk = list(triggers[start : start + chunk_size])
        triggered = _shared_embeddings(
            encoder,
            unique_frame["text"].tolist(),
            chunk,
            modes,
            seed=int(config["seed"]),
            separator=str(config["insertion"].get("separator", "")),
            batch_size=int(config["runtime"]["batch_size"]),
        )
        metrics = mode3_metrics(
            triggered,
            original,
            labels,
            centers,
            radii,
            {key: float(value) for key, value in config["constraints"].items()},
            pairwise_sample_size=int(config["mode3"].get("pairwise_sample_size", 20000)),
            seed=int(config["seed"]) + start,
        )
        if include_density:
            if search_benign is None or benign_knn_q95 is None:
                raise ValueError("Density diagnostics require frozen search benign references")
            from sklearn.neighbors import NearestNeighbors

            k = min(int(config["mode3"].get("knn_k", 10)), len(search_benign))
            neighbor = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(search_benign)
            for candidate_index, record in enumerate(metrics):
                distances = []
                for mode_values in triggered[candidate_index]:
                    distances.extend(neighbor.kneighbors(mode_values, return_distance=True)[0][:, -1].tolist())
                numerator = float(np.quantile(distances, 0.05))
                record["triggered_benign_knn_q05"] = numerator
                record["benign_knn_q95"] = float(benign_knn_q95)
                record["density_ratio"] = numerator / max(float(benign_knn_q95), 1e-12)
                record["blank_region_knn_feasible"] = record["density_ratio"] > 1.0
        records.extend(metrics)
    return records


def _fit_spherical_clusters(
    embeddings: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    values = np.asarray(embeddings, dtype=float)
    grid = [int(value) for value in config["mode3"]["cluster_count_grid"] if int(value) < len(values)]
    seed = int(config["seed"])
    minimum_size = int(config["mode3"].get("minimum_cluster_size", 5))
    trials: list[dict[str, Any]] = []
    fitted: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for count in grid:
        labels_by_seed: list[np.ndarray] = []
        primary_centers = None
        for offset in range(int(config["mode3"].get("clustering_restarts", 3))):
            model = KMeans(n_clusters=count, n_init=5, random_state=seed + offset)
            model.fit(values)
            centers = model.cluster_centers_
            centers /= np.linalg.norm(centers, axis=1, keepdims=True)
            labels = np.argmax(values @ centers.T, axis=1)
            labels_by_seed.append(labels)
            if offset == 0:
                primary_centers = centers
        assert primary_centers is not None
        primary = labels_by_seed[0]
        minimum = int(np.bincount(primary, minlength=count).min())
        silhouette = float(silhouette_score(values, primary, metric="cosine")) if len(np.unique(primary)) > 1 else -1.0
        stability_values = [
            adjusted_rand_score(labels_by_seed[left], labels_by_seed[right])
            for left in range(len(labels_by_seed))
            for right in range(left + 1, len(labels_by_seed))
        ]
        stability = float(np.mean(stability_values)) if stability_values else 1.0
        eligible = minimum >= minimum_size
        trials.append(
            {
                "cluster_count": count,
                "cosine_silhouette": silhouette,
                "minimum_cluster_size": minimum,
                "restart_stability_ari": stability,
                "eligible": eligible,
                "selection_score": silhouette + 0.10 * stability if eligible else -float("inf"),
            }
        )
        fitted[count] = (primary_centers, primary)
    eligible_trials = [trial for trial in trials if trial["eligible"]]
    chosen = max(eligible_trials or trials, key=lambda trial: (trial["selection_score"], trial["cosine_silhouette"]))
    centers, labels = fitted[int(chosen["cluster_count"])]
    distances = np.linalg.norm(values - centers[labels], axis=1)
    radii = np.asarray(
        [np.quantile(distances[labels == cluster], 0.95) for cluster in range(len(centers))],
        dtype=float,
    )
    return centers.astype(np.float32), radii.astype(np.float32), labels.astype(int), {"trials": trials, "selected": chosen}


def _pareto_indices(screen: pd.DataFrame, escape_column: str, radius_column: str) -> list[int]:
    ordered = screen.sort_values([escape_column, radius_column], ascending=[False, True], kind="mergesort")
    best_radius = float("inf")
    output: list[int] = []
    for index, row in ordered.iterrows():
        radius = float(row[radius_column])
        if radius < best_radius:
            output.append(int(index))
            best_radius = radius
    return output


def _candidate_pool(
    screen: pd.DataFrame,
    vocabulary: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    settings = config["candidate_pool"]
    seed = int(config["seed"])
    selected: list[int] = []
    sources: dict[int, set[str]] = defaultdict(set)

    def take(indices: Iterable[int], count: int, source: str) -> None:
        added = 0
        for index in indices:
            index = int(index)
            sources[index].add(source)
            if index not in selected:
                selected.append(index)
                added += 1
            if added >= count:
                break

    if config["mode"] == "multi_booster":
        take(screen.sort_values(["low_gain_q10", "low_coverage"], ascending=[False, False]).index, int(settings["top_low_gain_tokens"]), "low_gain")
        take(screen.sort_values(["high_gain_q05", "high_state_retention"], ascending=[False, False]).index, int(settings["top_high_preservation_tokens"]), "high_preservation")
    else:
        take(screen.sort_values(["absolute_escape_q05", "relative_outward_q05"], ascending=[False, False]).index, int(settings["top_escape_tokens"]), "escape")
        take(screen.sort_values(["compact_radius_q95", "triggered_pairwise_q05"], ascending=[True, False]).index, int(settings["top_compacting_tokens"]), "compact")
        take(_pareto_indices(screen, "absolute_escape_q05", "compact_radius_q95"), int(settings["pareto_tokens"]), "pareto")

    sticky_path = settings.get("sticky_screen")
    if sticky_path:
        path = _resolve(config, sticky_path)
        if path.exists():
            sticky = pd.read_csv(path).sort_values("sticky_score", ascending=False, kind="mergesort")
            by_token = {int(token_id): int(index) for index, token_id in screen["token_id"].items()}
            sticky_indices = [by_token[int(token_id)] for token_id in sticky["token_id"] if int(token_id) in by_token]
            take(sticky_indices, int(settings["top_sticky_tokens"]), "sticky")
    rng = np.random.default_rng(seed)
    random_indices = np.arange(len(screen))
    rng.shuffle(random_indices)
    random_added: list[int] = []
    for index in random_indices:
        index = int(index)
        if index in selected:
            continue
        selected.append(index)
        random_added.append(index)
        sources[index].add("random")
        if len(random_added) >= int(settings["random_tokens"]):
            break
    maximum = int(settings["candidate_pool_max_size"])
    if len(selected) > maximum:
        # Preserve an independent exploration reservoir even when the union of
        # targeted lists exceeds the configured pool cap.
        random_quota = min(len(random_added), int(settings["random_tokens"]), maximum // 3)
        targeted = [index for index in selected if index not in set(random_added)]
        selected = targeted[: maximum - random_quota] + random_added[:random_quota]
        if len(selected) < maximum:
            used = set(selected)
            selected.extend(index for index in targeted[maximum - random_quota :] if index not in used)
            selected = selected[:maximum]
    result = screen.loc[selected].copy().reset_index(drop=True)
    result.insert(0, "pool_index", np.arange(len(result)))
    result["pool_source"] = ["+".join(sorted(sources[int(index)])) for index in selected]
    return result


def prepare(config: dict[str, Any], encoder: SentenceTransformerEncoder, output: Path) -> dict[str, Any]:
    if config["mode"] == "single_sticky":
        raise ValueError("Mode 1 uses --phase full")
    dataset = _prepare_common(config, encoder, output)
    vocabulary = load_vocabulary(
        _resolve(config, config["vocabulary"]["analysis_path"]),
        allow_special=bool(config["vocabulary"]["allow_special_tokens"]),
        max_chars=int(config["vocabulary"]["max_chars"]),
    )
    if config["vocabulary"].get("max_candidates"):
        vocabulary = vocabulary.head(int(config["vocabulary"]["max_candidates"])).reset_index(drop=True)
    low = high = None
    screen_frames: list[pd.DataFrame] = []
    chunk_size = int(config["runtime"]["screen_candidate_chunk_size"])
    if config["mode"] == "multi_booster":
        low, high = _tail_thresholds(dataset, config)
        search = dataset.split_indices["search"]
        low_indices = search[dataset.baseline[search] <= low]
        high_indices = search[dataset.baseline[search] >= high]
        per_group = int(config["runtime"]["screen_examples_per_group"])
        screen_indices = np.concatenate(
            [
                select_balanced(low_indices, dataset.baseline, per_group, int(config["seed"])),
                select_balanced(high_indices, dataset.baseline, per_group, int(config["seed"]) + 1),
            ]
        )
        for start in range(0, len(vocabulary), chunk_size):
            chunk = vocabulary.iloc[start : start + chunk_size].reset_index(drop=True)
            print(f"mode2 exhaustive token screen {start + 1}-{start + len(chunk)} / {len(vocabulary)}", flush=True)
            metrics = _mode2_evaluate(encoder, dataset, screen_indices, chunk["literal"].tolist(), config, low, high)
            screen_frames.append(pd.concat([chunk, pd.DataFrame([_flatten_record(item) for item in metrics])], axis=1))
        thresholds = {"low_threshold": low, "high_threshold": high, "derived_from": "search split only"}
    else:
        for split in ("search", "validation", "test"):
            unique = unique_sentences(dataset, split)
            embeddings = encoder.encode_texts(
                unique["text"].tolist(),
                batch_size=int(config["runtime"]["batch_size"]),
                show_progress=bool(config["runtime"].get("show_progress", True)),
            )
            unique.to_csv(output / f"unique_{split}.csv", index=False)
            np.save(output / f"unique_{split}_embeddings.npy", embeddings)
        search_frame, search_embeddings = _load_unique(output, "search")
        centers, radii, search_labels, selection = _fit_spherical_clusters(search_embeddings, config)
        np.save(output / "cluster_centers.npy", centers)
        pd.DataFrame({"cluster": np.arange(len(radii)), "radius_q95": radii}).to_csv(output / "cluster_radii.csv", index=False)
        search_frame["source_cluster"] = search_labels
        search_frame["source_cluster_distance"] = np.linalg.norm(search_embeddings - centers[search_labels], axis=1)
        search_frame.to_csv(output / "unique_search.csv", index=False)
        for split in ("validation", "test"):
            frame, embeddings = _load_unique(output, split)
            labels = np.argmax(embeddings @ centers.T, axis=1)
            frame["source_cluster"] = labels
            frame["source_cluster_distance"] = np.linalg.norm(embeddings - centers[labels], axis=1)
            frame.to_csv(output / f"unique_{split}.csv", index=False)
        _write_json(output / "clustering_selection.json", selection)
        from sklearn.neighbors import NearestNeighbors

        k = min(int(config["mode3"].get("knn_k", 10)) + 1, len(search_embeddings))
        self_distances = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(search_embeddings).kneighbors(search_embeddings, return_distance=True)[0][:, -1]
        benign_knn_q95 = float(np.quantile(self_distances, 0.95))
        _write_json(output / "benign_density_reference.json", {"knn_k": k - 1, "benign_knn_q95": benign_knn_q95})
        count = min(int(config["runtime"]["screen_unique_sentences"]), len(search_frame))
        chosen = np.random.default_rng(int(config["seed"])).choice(len(search_frame), size=count, replace=False)
        screen_frame = search_frame.iloc[chosen].reset_index(drop=True)
        screen_embeddings = search_embeddings[chosen]
        for start in range(0, len(vocabulary), chunk_size):
            chunk = vocabulary.iloc[start : start + chunk_size].reset_index(drop=True)
            print(f"mode3 exhaustive token screen {start + 1}-{start + len(chunk)} / {len(vocabulary)}", flush=True)
            metrics = _mode3_evaluate(
                encoder,
                screen_frame,
                screen_embeddings,
                chunk["literal"].tolist(),
                config,
                centers,
                radii,
                include_density=False,
            )
            screen_frames.append(pd.concat([chunk, pd.DataFrame([_flatten_record(item) for item in metrics])], axis=1))
        thresholds = {
            "cluster_count": len(centers),
            "cluster_radii_quantile": 0.95,
            "clustering_fit_split": "search",
        }
    screen = pd.concat(screen_frames, ignore_index=True)
    key = mode2_sort_key if config["mode"] == "multi_booster" else mode3_sort_key
    screen = pd.DataFrame(sorted(screen.to_dict("records"), key=key))
    screen.insert(0, "rank", np.arange(1, len(screen) + 1))
    screen.to_csv(output / "single_token_screen.csv", index=False)
    pool = _candidate_pool(screen, vocabulary, config)
    pool.to_csv(output / "candidate_pool.csv", index=False)
    _write_json(output / "frozen_search_geometry.json", thresholds)
    return {
        "phase": "prepare",
        "mode": config["mode"],
        "valid_vocabulary_size": len(vocabulary),
        "candidate_pool_size": len(pool),
        "split_sizes": {name: len(indices) for name, indices in dataset.split_indices.items()},
        **thresholds,
    }


def _dynamic_mode2_indices(dataset, low: float, high: float, count: int, seed: int) -> np.ndarray:
    search = dataset.split_indices["search"]
    low_indices = search[dataset.baseline[search] <= low]
    high_indices = search[dataset.baseline[search] >= high]
    rng = np.random.default_rng(seed)
    low_chosen = rng.choice(low_indices, size=min(count, len(low_indices)), replace=False)
    high_chosen = rng.choice(high_indices, size=min(count, len(high_indices)), replace=False)
    return np.concatenate([low_chosen, high_chosen])


def search(config: dict[str, Any], encoder: SentenceTransformerEncoder, output: Path, restart: int, lengths: Sequence[int]) -> dict[str, Any]:
    if not (output / "candidate_pool.csv").exists():
        raise FileNotFoundError("Run --phase prepare before search")
    dataset = load_prepared_dataset(output)
    assert_sentence_disjoint(dataset)
    pool = pd.read_csv(output / "candidate_pool.csv", keep_default_na=False)
    settings = config["search"]
    mode = config["mode"]
    geometry = _read_json(output / "frozen_search_geometry.json")
    if mode == "multi_booster":
        low, high = float(geometry["low_threshold"]), float(geometry["high_threshold"])
    else:
        unique_frame, unique_embeddings = _load_unique(output, "search")
        centers = np.load(output / "cluster_centers.npy")
        radii = pd.read_csv(output / "cluster_radii.csv")["radius_q95"].to_numpy(dtype=float)
    previous_elites: list[tuple[int, ...]] = []
    written: list[str] = []
    restart_dir = output / "search" / f"restart_{restart:02d}"
    restart_dir.mkdir(parents=True, exist_ok=True)
    for length in lengths:
        if length not in _lengths(config):
            raise ValueError(f"Length {length} is outside the registered schedule {_lengths(config)}")

        def evaluate_sequences(sequences: list[tuple[int, ...]], iteration: int, *, full: bool) -> list[dict[str, Any]]:
            triggers = [_sequence_trigger(sequence, pool) for sequence in sequences]
            if mode == "multi_booster":
                if full:
                    indices = dataset.split_indices["search"]
                else:
                    indices = _dynamic_mode2_indices(
                        dataset,
                        low,
                        high,
                        int(config["runtime"]["search_examples_per_group"]),
                        int(config["seed"]) + restart * 100000 + length * 1000 + iteration,
                    )
                metrics = _mode2_evaluate(encoder, dataset, indices, triggers, config, low, high)
            else:
                if full:
                    chosen = np.arange(len(unique_frame))
                else:
                    rng = np.random.default_rng(int(config["seed"]) + restart * 100000 + length * 1000 + iteration)
                    size = min(int(config["runtime"]["search_unique_sentences"]), len(unique_frame))
                    chosen = rng.choice(len(unique_frame), size=size, replace=False)
                metrics = _mode3_evaluate(
                    encoder,
                    unique_frame.iloc[chosen].reset_index(drop=True),
                    unique_embeddings[chosen],
                    triggers,
                    config,
                    centers,
                    radii,
                    include_density=False,
                )
            for record in metrics:
                record["component_length"] = length
            return metrics

        warm = expand_warm_sequences(
            previous_elites,
            length,
            len(pool),
            int(settings["population_size"]) // 2,
            int(config["seed"]) + restart * 1000 + length,
        )
        sort_key = mode2_sort_key if mode == "multi_booster" else mode3_sort_key
        result = cem_search_v2(
            len(pool),
            length,
            lambda sequences, iteration: evaluate_sequences(sequences, iteration, full=False),
            sort_key=sort_key,
            population_size=int(settings["population_size"]),
            elite_ratio=float(settings["elite_ratio"]),
            iterations=int(settings["iterations"]),
            update_alpha=float(settings["update_alpha"]),
            probability_floor=float(settings["probability_floor"]),
            uniform_mixture=float(settings["uniform_mixture"]),
            entropy_min_fraction=float(settings["entropy_min_fraction"]),
            stall_patience=int(settings["stall_patience"]),
            elite_min_hamming_fraction=float(settings["elite_min_hamming_fraction"]),
            seed=int(config["seed"]) + restart * 10000 + length,
            full_score_fn=lambda sequences, iteration: evaluate_sequences(sequences, iteration, full=True),
            full_evaluation_interval=int(settings["full_evaluation_interval"]),
            initial_sequences=warm,
        )
        records = []
        for rank, record in enumerate(result.candidates[: int(settings["archive_size"])], 1):
            sequence = tuple(record["sequence"])
            records.append(
                {
                    "rank": rank,
                    "restart": restart,
                    "component_length": length,
                    "pool_sequence": ",".join(map(str, sequence)),
                    "component_token_ids": ",".join(map(str, _sequence_ids(sequence, pool))),
                    "trigger": _sequence_trigger(sequence, pool),
                    **_flatten_record({key: value for key, value in record.items() if key != "sequence"}),
                }
            )
        candidate_path = restart_dir / f"length_{length:02d}_candidates.csv"
        history_path = restart_dir / f"length_{length:02d}_history.csv"
        pd.DataFrame.from_records(records).to_csv(candidate_path, index=False)
        pd.DataFrame.from_records(result.history).to_csv(history_path, index=False)
        previous_elites = [
            _parse_ids(value)
            for value in pd.DataFrame.from_records(records).head(int(settings["warm_elite_count"]))["pool_sequence"]
        ]
        written.append(str(candidate_path))
    return {"phase": "search", "mode": mode, "restart": restart, "lengths": list(lengths), "files": written}


def _rank_records(records: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    key = mode2_sort_key if mode == "multi_booster" else mode3_sort_key
    return sorted(records, key=key)


def _validation_inputs(output: Path, config: dict[str, Any], length: int) -> list[dict[str, Any]]:
    count = int(config["validation"]["candidate_count_per_length"])
    if length == 1:
        screen = pd.read_csv(output / "single_token_screen.csv", keep_default_na=False).head(count)
        return [
            {
                "component_length": 1,
                "component_token_ids": str(int(row["token_id"])),
                "trigger": str(row["literal"]),
                "source_restart": -1,
                "search_rank": int(row["rank"]),
            }
            for _, row in screen.iterrows()
        ]
    per_restart = int(config["validation"]["candidates_per_restart"])
    records: list[dict[str, Any]] = []
    for restart in range(int(config["search"]["restarts_per_length"])):
        path = output / "search" / f"restart_{restart:02d}" / f"length_{length:02d}_candidates.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing registered search shard: {path}")
        frame = pd.read_csv(path, keep_default_na=False).head(per_restart)
        for row in frame.to_dict("records"):
            row["source_restart"] = restart
            row["search_rank"] = int(row["rank"])
            records.append(row)
    deduplicated: dict[str, dict[str, Any]] = {}
    for record in _rank_records(records, config["mode"]):
        deduplicated.setdefault(str(record["component_token_ids"]), record)
    return list(deduplicated.values())[:count]


def _realizability_records(
    encoder: SentenceTransformerEncoder,
    candidates: Sequence[dict[str, Any]],
    texts: Sequence[str],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    sample_count = min(int(config["validation"]["realizability_contexts"]), len(texts))
    chosen = list(texts[:sample_count])
    threshold = float(config["validation"]["min_realizability"])
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        audit = trigger_realizability(
            encoder.tokenizer,
            str(candidate["trigger"]),
            _parse_ids(candidate["component_token_ids"]),
            chosen,
            config["insertion"]["modes"],
            seed=int(config["seed"]),
            separator=str(config["insertion"].get("separator", "")),
        )
        audit["text_realizable"] = float(audit["realizability_rate"]) >= threshold
        output.append(audit)
    return output


def _evaluate_validation_length(
    encoder: SentenceTransformerEncoder,
    dataset,
    output: Path,
    config: dict[str, Any],
    length: int,
    candidates: list[dict[str, Any]],
) -> pd.DataFrame:
    triggers = [str(item["trigger"]) for item in candidates]
    if config["mode"] == "multi_booster":
        geometry = _read_json(output / "frozen_search_geometry.json")
        metrics = _mode2_evaluate(
            encoder,
            dataset,
            dataset.split_indices["validation"],
            triggers,
            config,
            float(geometry["low_threshold"]),
            float(geometry["high_threshold"]),
        )
        texts = dataset.frame.iloc[dataset.split_indices["validation"]]["sentence2"].tolist()
    else:
        unique_frame, embeddings = _load_unique(output, "validation")
        search_frame, search_embeddings = _load_unique(output, "search")
        centers = np.load(output / "cluster_centers.npy")
        radii = pd.read_csv(output / "cluster_radii.csv")["radius_q95"].to_numpy(dtype=float)
        density = _read_json(output / "benign_density_reference.json")
        metrics = _mode3_evaluate(
            encoder,
            unique_frame,
            embeddings,
            triggers,
            config,
            centers,
            radii,
            include_density=True,
            search_benign=search_embeddings,
            benign_knn_q95=float(density["benign_knn_q95"]),
        )
        texts = unique_frame["text"].tolist()
    realizability = _realizability_records(encoder, candidates, texts, config)
    rows: list[dict[str, Any]] = []
    for candidate, metric, audit in zip(candidates, metrics, realizability):
        record = {**candidate, **_flatten_record(metric), **audit}
        record["component_length"] = length
        record["core_certified"] = bool(metric["core_feasible"] and audit["text_realizable"])
        if config["mode"] == "multi_booster":
            record["structure_certified"] = bool(metric["structure_feasible"] and audit["text_realizable"])
        rows.append(record)
    ranked = _rank_records(rows, config["mode"])
    frame = pd.DataFrame.from_records(ranked)
    frame.insert(0, "validation_rank", np.arange(1, len(frame) + 1))
    return frame


def _test_once(
    encoder: SentenceTransformerEncoder,
    dataset,
    output: Path,
    config: dict[str, Any],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    trigger = str(frozen["trigger"])
    if config["mode"] == "multi_booster":
        geometry = _read_json(output / "frozen_search_geometry.json")
        metric = _mode2_evaluate(
            encoder,
            dataset,
            dataset.split_indices["test"],
            [trigger],
            config,
            float(geometry["low_threshold"]),
            float(geometry["high_threshold"]),
        )[0]
        texts = dataset.frame.iloc[dataset.split_indices["test"]]["sentence2"].tolist()
    else:
        unique_frame, embeddings = _load_unique(output, "test")
        _, search_embeddings = _load_unique(output, "search")
        centers = np.load(output / "cluster_centers.npy")
        radii = pd.read_csv(output / "cluster_radii.csv")["radius_q95"].to_numpy(dtype=float)
        density = _read_json(output / "benign_density_reference.json")
        metric = _mode3_evaluate(
            encoder,
            unique_frame,
            embeddings,
            [trigger],
            config,
            centers,
            radii,
            include_density=True,
            search_benign=search_embeddings,
            benign_knn_q95=float(density["benign_knn_q95"]),
        )[0]
        texts = unique_frame["text"].tolist()
    audit = _realizability_records(encoder, [frozen], texts, config)[0]
    result = {**frozen, **_flatten_record(metric), **audit}
    result["test_core_certified"] = bool(metric["core_feasible"] and audit["text_realizable"])
    if config["mode"] == "multi_booster":
        result["test_structure_certified"] = bool(metric["structure_feasible"] and audit["text_realizable"])
    return result


def _mode2_ablations(
    encoder: SentenceTransformerEncoder,
    dataset,
    output: Path,
    config: dict[str, Any],
    frozen: dict[str, Any],
) -> None:
    ids = _parse_ids(frozen["component_token_ids"])
    tokenizer = encoder.tokenizer
    geometry = _read_json(output / "frozen_search_geometry.json")
    low, high = float(geometry["low_threshold"]), float(geometry["high_threshold"])
    validation = dataset.split_indices["validation"]

    def evaluate(items: list[tuple[str, tuple[int, ...]]]) -> pd.DataFrame:
        triggers = [value for value, _ in items]
        metrics = _mode2_evaluate(encoder, dataset, validation, triggers, config, low, high)
        return pd.DataFrame(
            [
                {"trigger": trigger, "component_token_ids": ",".join(map(str, token_ids)), **metric}
                for (trigger, token_ids), metric in zip(items, metrics)
            ]
        )

    singles = [(tokenizer.decode([token_id], clean_up_tokenization_spaces=False), (token_id,)) for token_id in ids]
    evaluate(singles).to_csv(output / "single_token_ablation.csv", index=False)
    prefixes = [
        (tokenizer.decode(list(ids[:end]), clean_up_tokenization_spaces=False), ids[:end])
        for end in range(1, len(ids) + 1)
    ]
    prefix_frame = evaluate(prefixes)
    prefix_frame.insert(0, "prefix_length", np.arange(1, len(prefix_frame) + 1))
    prefix_frame.to_csv(output / "prefix_metrics.csv", index=False)
    leave_one_out = [
        (tokenizer.decode(list(ids[:index] + ids[index + 1 :]), clean_up_tokenization_spaces=False), ids[:index] + ids[index + 1 :])
        for index in range(len(ids))
        if len(ids) > 1
    ]
    loo_frame = evaluate(leave_one_out) if leave_one_out else pd.DataFrame()
    if not loo_frame.empty:
        loo_frame.insert(0, "removed_position", np.arange(len(loo_frame)))
    loo_frame.to_csv(output / "leave_one_out_ablation.csv", index=False)
    rng = np.random.default_rng(int(config["seed"]) + 8000)
    permutations: list[tuple[str, tuple[int, ...]]] = []
    seen: set[tuple[int, ...]] = set()
    for _ in range(int(config["validation"].get("permutation_count", 16)) * 4):
        permuted = tuple(map(int, rng.permutation(ids)))
        if permuted not in seen:
            seen.add(permuted)
            permutations.append((tokenizer.decode(list(permuted), clean_up_tokenization_spaces=False), permuted))
        if len(permutations) >= int(config["validation"].get("permutation_count", 16)):
            break
    evaluate(permutations).to_csv(output / "permutation_ablation.csv", index=False)
    single_frame = pd.read_csv(output / "single_token_ablation.csv")
    synergy = float(frozen["objective"]) - float(single_frame["objective"].max())
    contributions = []
    if not loo_frame.empty:
        contributions = (float(frozen["objective"]) - loo_frame["objective"].to_numpy(dtype=float)).tolist()
    _write_json(output / "combination_synergy.json", {"synergy_vs_best_single": synergy, "leave_one_out_contributions": contributions})


def _mode3_baselines_and_projection(
    encoder: SentenceTransformerEncoder,
    output: Path,
    config: dict[str, Any],
    frontier: pd.DataFrame,
    frozen: dict[str, Any],
) -> None:
    unique_frame, embeddings = _load_unique(output, "validation")
    search_frame, search_embeddings = _load_unique(output, "search")
    centers = np.load(output / "cluster_centers.npy")
    radii = pd.read_csv(output / "cluster_radii.csv")["radius_q95"].to_numpy(dtype=float)
    density = _read_json(output / "benign_density_reference.json")
    screen = pd.read_csv(output / "single_token_screen.csv", keep_default_na=False)
    pool = pd.read_csv(output / "candidate_pool.csv", keep_default_na=False)
    rng = np.random.default_rng(int(config["seed"]) + 9000)
    v1_ids = tuple(int(value) for value in config["mode3"].get("v1_component_token_ids", []))
    baselines: list[dict[str, Any]] = []
    stage_embeddings: list[np.ndarray] = []
    stage_labels: list[str] = []
    sample_count = min(int(config["plot"].get("projection_sentence_count", 128)), len(unique_frame))
    sample = np.arange(sample_count)
    for _, best in frontier.iterrows():
        length = int(best["component_length"])
        optimized = str(best["trigger"])
        random_ids = tuple(int(value) for value in rng.choice(pool["token_id"], size=length, replace=True))
        frequent_ids = tuple(int(value) for value in config["mode3"].get("frequent_token_ids", [8, 5, 6]))
        frequent_ids = tuple(frequent_ids[index % len(frequent_ids)] for index in range(length))
        natural_ids = tuple(encoder.tokenize([str(search_frame.iloc[0]["text"])], add_special_tokens=False)[0][:length])
        if len(natural_ids) < length:
            natural_ids = tuple((natural_ids * (math.ceil(length / max(len(natural_ids), 1))))[:length])
        sticky_rows = pool[pool["pool_source"].astype(str).str.contains("sticky", regex=False)]
        sticky_id = int((sticky_rows.iloc[0] if len(sticky_rows) else screen.sort_values("rank").iloc[0])["token_id"])
        escape_ids = tuple(screen.sort_values("absolute_escape_q05", ascending=False).head(length)["token_id"].astype(int))
        compact_ids = tuple(screen.sort_values("compact_radius_q95", ascending=True).head(length)["token_id"].astype(int))
        adjusted_v1 = tuple(v1_ids[index % len(v1_ids)] for index in range(length)) if v1_ids else random_ids
        variants = {
            "random_legal": random_ids,
            "frequent_ordinary": frequent_ids,
            "natural_phrase": natural_ids,
            "best_single_sticky_repeated": (sticky_id,) * length,
            "top_escape_concatenation": escape_ids,
            "top_compact_concatenation": compact_ids,
            "v1_attractor_adjusted": adjusted_v1,
            "v2_optimized": _parse_ids(best["component_token_ids"]),
        }
        for name, token_ids in variants.items():
            trigger = encoder.decode(token_ids)
            metric = _mode3_evaluate(
                encoder,
                unique_frame,
                embeddings,
                [trigger],
                config,
                centers,
                radii,
                include_density=True,
                search_benign=search_embeddings,
                benign_knn_q95=float(density["benign_knn_q95"]),
            )[0]
            realization = trigger_realizability(
                encoder.tokenizer,
                trigger,
                token_ids,
                unique_frame["text"].tolist()[: int(config["validation"]["realizability_contexts"])],
                config["insertion"]["modes"],
                seed=int(config["seed"]),
                separator=str(config["insertion"].get("separator", "")),
            )
            baselines.append(
                {
                    "component_length": length,
                    "baseline": name,
                    "component_token_ids": ",".join(map(str, token_ids)),
                    "trigger": trigger,
                    **_flatten_record(metric),
                    **realization,
                }
            )
        triggered = _shared_embeddings(
            encoder,
            unique_frame.iloc[sample]["text"].tolist(),
            [optimized],
            [str(config["plot"]["insertion_mode"])],
            seed=int(config["seed"]),
            separator=str(config["insertion"].get("separator", "")),
            batch_size=int(config["runtime"]["batch_size"]),
        )[0, 0]
        stage_embeddings.append(triggered)
        stage_labels.append(f"L={length}")
    pd.DataFrame.from_records(baselines).to_csv(output / "equal_length_baselines.csv", index=False)
    # Keep the explanatory plot readable while preserving the complete numeric
    # frontier in CSV.
    if len(stage_embeddings) > 8:
        selected = np.linspace(0, len(stage_embeddings) - 1, 8, dtype=int)
        stage_embeddings = [stage_embeddings[index] for index in selected]
        stage_labels = [stage_labels[index] for index in selected]
    projection_method = plot_embedding_progression(
        embeddings[sample],
        stage_embeddings,
        stage_labels,
        output / "embedding_length_progression.png",
        cluster_centers=centers,
        projection_config=config["plot"].get("projection", {}),
        dpi=int(config["plot"]["dpi"]),
    )
    _write_json(output / "projection_metadata.json", {"joint_fit": True, "method_used": projection_method, "stage_labels": stage_labels})
    diagnostic = frozen
    if int(frozen["component_length"]) == 1 and np.any(frontier["component_length"].to_numpy(dtype=int) > 1):
        diagnostic = _rank_records(
            frontier[frontier["component_length"].astype(int) > 1].to_dict("records"),
            "repulsive_attractor",
        )[0]
    selected_length = int(diagnostic["component_length"])
    selected_restart = int(diagnostic.get("source_restart", -1))
    if selected_length > 1 and selected_restart >= 0:
        history_path = output / "search" / f"restart_{selected_restart:02d}" / f"length_{selected_length:02d}_history.csv"
        if history_path.exists():
            history = pd.read_csv(history_path, keep_default_na=False)
            requested = [0, 5, 10, 20, int(history["iteration"].max())]
            rows = []
            for target in requested:
                row = history.iloc[(history["iteration"].astype(int) - target).abs().argsort()[:1]].iloc[0]
                if int(row["iteration"]) not in {int(item["iteration"]) for item in rows}:
                    rows.append(row.to_dict())
            iteration_embeddings: list[np.ndarray] = []
            iteration_labels: list[str] = []
            for row in rows:
                sequence = _parse_ids(row["best_sequence"])
                trigger = _sequence_trigger(sequence, pool)
                values = _shared_embeddings(
                    encoder,
                    unique_frame.iloc[sample]["text"].tolist(),
                    [trigger],
                    [str(config["plot"]["insertion_mode"])],
                    seed=int(config["seed"]),
                    separator=str(config["insertion"].get("separator", "")),
                    batch_size=int(config["runtime"]["batch_size"]),
                )[0, 0]
                iteration_embeddings.append(values)
                iteration_labels.append(f"iteration {int(row['iteration'])}")
            final_values = _shared_embeddings(
                encoder,
                unique_frame.iloc[sample]["text"].tolist(),
                [str(diagnostic["trigger"])],
                [str(config["plot"]["insertion_mode"])],
                seed=int(config["seed"]),
                separator=str(config["insertion"].get("separator", "")),
                batch_size=int(config["runtime"]["batch_size"]),
            )[0, 0]
            iteration_embeddings.append(final_values)
            iteration_labels.append("validation-selected diagnostic")
            plot_embedding_progression(
                embeddings[sample],
                iteration_embeddings,
                iteration_labels,
                output / "embedding_search_iteration_progression.png",
                cluster_centers=centers,
                projection_config=config["plot"].get("projection", {}),
                dpi=int(config["plot"]["dpi"]),
            )


def finalize(config: dict[str, Any], encoder: SentenceTransformerEncoder, output: Path) -> dict[str, Any]:
    dataset = load_prepared_dataset(output)
    assert_sentence_disjoint(dataset)
    scheduled = [1, *_lengths(config)]
    best_by_length: list[dict[str, Any]] = []
    for length in scheduled:
        candidates = _validation_inputs(output, config, length)
        frame = _evaluate_validation_length(encoder, dataset, output, config, length, candidates)
        length_dir = output / "validation" / f"length_{length:02d}"
        length_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(length_dir / "validation_candidates.csv", index=False)
        best = frame.iloc[0].to_dict()
        if length == 1:
            success_rate = float(bool(best["core_certified"]))
        else:
            restart_success = []
            for restart in range(int(config["search"]["restarts_per_length"])):
                subset = frame[frame["source_restart"].astype(int) == restart]
                restart_success.append(bool(subset.iloc[0]["core_certified"]) if len(subset) else False)
            success_rate = float(np.mean(restart_success))
        best["search_seed_success_rate"] = success_rate
        best_by_length.append(best)
        _write_json(length_dir / "length_best.json", best)
    frontier = pd.DataFrame.from_records(best_by_length).sort_values("component_length", kind="mergesort").reset_index(drop=True)
    frontier.to_csv(output / "length_frontier.csv", index=False)
    plot_length_frontier(frontier, output / "length_frontier.png", mode=config["mode"], dpi=int(config["plot"]["dpi"]))
    feasible = frontier[frontier["core_certified"].astype(bool)]
    if len(feasible):
        selected = feasible.sort_values("component_length", kind="mergesort").iloc[0].to_dict()
        status = "validation_core_certified"
    else:
        selected = _rank_records(frontier.to_dict("records"), config["mode"])[0]
        status = "no_validation_feasible_registered_fallback"
    frozen = {
        **selected,
        "selection_status": status,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "minimality_status": "budgeted_search",
        "shortest_found_length": int(selected["component_length"]) if status == "validation_core_certified" else None,
        "shorter_lengths_exhaustive": [1],
        "shorter_lengths_heuristically_searched": [
            value for value in _lengths(config) if value < int(selected["component_length"])
        ],
        "registered_length_schedule": scheduled,
    }
    _write_json(output / "frozen_candidate.json", frozen)
    test_result = _test_once(encoder, dataset, output, config, frozen)
    pd.DataFrame.from_records([test_result]).to_csv(output / "test_result.csv", index=False)
    _write_json(output / "test_result.json", test_result)
    if config["mode"] == "multi_booster":
        _mode2_ablations(encoder, dataset, output, config, frozen)
    else:
        _mode3_baselines_and_projection(encoder, output, config, frontier, frozen)
    return {
        "phase": "finalize",
        "mode": config["mode"],
        "selection_status": status,
        "selected_length": int(selected["component_length"]),
        "selected_trigger": str(selected["trigger"]),
        "selected_component_token_ids": str(selected["component_token_ids"]),
        "validation_core_certified": bool(selected["core_certified"]),
        "test_core_certified": bool(test_result["test_core_certified"]),
        "test_result": test_result,
    }


def _single_sticky_gaps(
    encoder: SentenceTransformerEncoder,
    dataset,
    indices: np.ndarray,
    triggers: Sequence[str],
    config: dict[str, Any],
    mean_similarity: float,
) -> np.ndarray:
    values = _one_sided_similarities(
        encoder,
        dataset,
        indices,
        triggers,
        config["insertion"]["modes"],
        seed=int(config["seed"]),
        separator=str(config["insertion"].get("separator", "")),
        batch_size=int(config["runtime"]["batch_size"]),
    )
    return np.abs(values - mean_similarity)


def run_mode1(config: dict[str, Any], encoder: SentenceTransformerEncoder, output: Path) -> dict[str, Any]:
    dataset = _prepare_common(config, encoder, output)
    vocabulary = load_vocabulary(
        _resolve(config, config["vocabulary"]["analysis_path"]),
        allow_special=bool(config["vocabulary"]["allow_special_tokens"]),
        max_chars=int(config["vocabulary"]["max_chars"]),
    )
    if config["vocabulary"].get("max_candidates"):
        vocabulary = vocabulary.head(int(config["vocabulary"]["max_candidates"])).reset_index(drop=True)
    batch_size = int(config["runtime"]["batch_size"])
    chunk_size = int(config["runtime"]["screen_candidate_chunk_size"])
    token_embeddings = encoder.encode_texts(vocabulary["literal"].tolist(), batch_size=batch_size, show_progress=True)
    mean_similarity = exact_pairwise_mean(token_embeddings)
    _write_json(output / "model_similarity_mean.json", {"u": mean_similarity, "valid_vocab_size": len(vocabulary)})
    paper = config["single_sticky"]["paper_replication"]
    below_mean = np.flatnonzero(dataset.baseline < mean_similarity - float(paper.get("pair_filter_margin", 0.0)))
    repeat_count = int(paper["repeat_count"])

    def score_vocabulary(available: np.ndarray, label: str, seed_offset: int) -> pd.DataFrame:
        score_indices = select_balanced(
            available,
            dataset.baseline,
            int(paper["sample_pair_count"]),
            int(config["seed"]) + seed_offset,
        )
        reference_mean = dataset.sentence1_embeddings[score_indices].mean(axis=0)
        score_frames: list[pd.DataFrame] = []
        for start in range(0, len(vocabulary), chunk_size):
            chunk = vocabulary.iloc[start : start + chunk_size].reset_index(drop=True)
            print(
                f"mode1 {label} exhaustive token screen {start + 1}-{start + len(chunk)} / {len(vocabulary)}",
                flush=True,
            )
            triggers = [repeat_literal(value, repeat_count) for value in chunk["literal"]]
            similarities = _one_sided_similarities(
                encoder,
                dataset,
                score_indices,
                triggers,
                config["insertion"]["modes"],
                seed=int(config["seed"]),
                separator=str(config["insertion"].get("separator", "")),
                batch_size=batch_size,
            )
            semantic = token_embeddings[start : start + len(chunk)] @ reference_mean
            scores, parts = single_sticky_score(
                similarities,
                dataset.baseline[score_indices],
                semantic,
                alpha=float(paper["alpha"]),
                beta=float(paper["beta"]),
                gamma=float(paper["gamma"]),
            )
            frame = chunk.copy()
            frame["sticky_score"] = scores
            frame["semantic_penalty"] = semantic
            for name, values in parts.items():
                frame[name] = values
            score_frames.append(frame)
        ranked = pd.concat(score_frames, ignore_index=True).sort_values(
            "sticky_score", ascending=False, kind="mergesort"
        )
        ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
        return ranked

    # Paper replication intentionally has no held-out partition.  The second
    # independent screen below is the only candidate source for holdout
    # certification, preventing test sentence content from influencing it.
    screen = score_vocabulary(below_mean, "paper", 0)
    screen.to_csv(output / "all_token_scores.csv", index=False)
    candidate_count = max(1, int(math.ceil(len(screen) * float(paper["candidate_ratio"]))))
    candidates = screen.head(candidate_count).copy().reset_index(drop=True)
    paper_records: list[dict[str, Any]] = []
    all_ge_max: list[float] = []
    stored_gaps: list[np.ndarray] = []
    for start in range(0, len(candidates), max(1, chunk_size // 2)):
        chunk = candidates.iloc[start : start + max(1, chunk_size // 2)]
        triggers = [repeat_literal(value, repeat_count) for value in chunk["literal"]]
        gaps = _single_sticky_gaps(encoder, dataset, below_mean, triggers, config, mean_similarity)
        stored_gaps.extend(gaps)
        all_ge_max.extend(np.max(gaps, axis=(1, 2)).tolist())
    q1, q3 = np.quantile(all_ge_max, [0.25, 0.75])
    paper_epsilon = float(q3 + 1.5 * (q3 - q1))
    for (_, row), gaps in zip(candidates.iterrows(), stored_gaps):
        flat = gaps.ravel()
        paper_records.append(
            {
                **row.to_dict(),
                "paper_epsilon": paper_epsilon,
                "paper_GE_max": float(flat.max()),
                "paper_GE_q95": float(np.quantile(flat, 0.95)),
                "paper_pass_rate": float(np.mean(flat <= paper_epsilon)),
                "paper_validated": bool(flat.max() <= paper_epsilon),
            }
        )
    paper_frame = pd.DataFrame.from_records(paper_records).sort_values(
        ["paper_validated", "paper_GE_max", "sticky_score"], ascending=[False, True, False], kind="mergesort"
    )
    paper_frame.to_csv(output / "paper_replication_validation.csv", index=False)

    holdout = config["single_sticky"]["holdout_certification"]
    epsilon = float(holdout.get("epsilon", paper_epsilon))
    search_available = dataset.split_indices["search"]
    search_below_mean = search_available[dataset.baseline[search_available] < mean_similarity]
    holdout_screen = score_vocabulary(search_below_mean, "holdout-search", 100)
    holdout_screen.to_csv(output / "holdout_search_token_scores.csv", index=False)
    holdout_candidates = holdout_screen.head(candidate_count).copy().reset_index(drop=True)
    validation_available = dataset.split_indices["validation"]
    validation_indices = validation_available[dataset.baseline[validation_available] < mean_similarity]
    holdout_rows: list[dict[str, Any]] = []
    for start in range(0, len(holdout_candidates), max(1, chunk_size // 2)):
        chunk = holdout_candidates.iloc[start : start + max(1, chunk_size // 2)]
        gaps = _single_sticky_gaps(
            encoder,
            dataset,
            validation_indices,
            [repeat_literal(value, repeat_count) for value in chunk["literal"]],
            config,
            mean_similarity,
        )
        for (_, row), candidate_gaps in zip(chunk.iterrows(), gaps):
            holdout_rows.append({**row.to_dict(), **dose_certification(candidate_gaps, epsilon, float(holdout["coverage_target"]))})
    holdout_frame = pd.DataFrame.from_records(holdout_rows).sort_values(
        ["q95_certified", "coverage_certified", "GE_q95", "GE_max", "sticky_score"],
        ascending=[False, False, True, True, False],
        kind="mergesort",
    )
    holdout_frame.to_csv(output / "holdout_validation_candidates.csv", index=False)
    frozen = holdout_frame.iloc[0].to_dict()
    frozen.update({"selection_split": "validation", "test_used_for_selection": False, "epsilon": epsilon})
    _write_json(output / "frozen_candidate.json", frozen)
    test_available = dataset.split_indices["test"]
    test_indices = test_available[dataset.baseline[test_available] < mean_similarity]
    test_gaps = _single_sticky_gaps(
        encoder,
        dataset,
        test_indices,
        [repeat_literal(str(frozen["literal"]), repeat_count)],
        config,
        mean_similarity,
    )[0]
    test_result = {**frozen, **{f"test_{key}": value for key, value in dose_certification(test_gaps, epsilon, float(holdout["coverage_target"])).items()}}
    _write_json(output / "test_result.json", test_result)
    pd.DataFrame.from_records([test_result]).to_csv(output / "test_result.csv", index=False)

    dose_rows: list[dict[str, Any]] = []
    for count in range(0, int(config["single_sticky"]["dose_curve"]["max_repeat_count"]) + 1):
        if count == 0:
            gaps = np.abs(dataset.baseline[test_indices] - mean_similarity)[None, None, :]
        else:
            gaps = _single_sticky_gaps(
                encoder,
                dataset,
                test_indices,
                [repeat_literal(str(frozen["literal"]), count)],
                config,
                mean_similarity,
            )[0]
        dose_rows.append({"repeat_count": count, **dose_certification(gaps, epsilon, float(holdout["coverage_target"]))})
    dose_frame = pd.DataFrame.from_records(dose_rows)
    dose_frame.to_csv(output / "dose_curve.csv", index=False)
    effective = dose_frame[dose_frame["coverage_certified"].astype(bool)]
    minimum_effective = int(effective.iloc[0]["repeat_count"]) if len(effective) else None
    plot_count = min(int(config["plot"]["pair_count"]), len(test_indices))
    plot_indices = select_balanced(
        test_indices,
        dataset.baseline,
        plot_count,
        int(config["seed"]) + 400,
    )
    maximum = int(config["single_sticky"]["dose_curve"]["max_repeat_count"])
    curves = np.empty((len(plot_indices), maximum + 1), dtype=float)
    curves[:, 0] = dataset.baseline[plot_indices]
    for count in range(1, maximum + 1):
        similarities = _one_sided_similarities(
            encoder,
            dataset,
            plot_indices,
            [repeat_literal(str(frozen["literal"]), count)],
            [str(config["plot"]["insertion_mode"])],
            seed=int(config["seed"]),
            separator=str(config["insertion"].get("separator", "")),
            batch_size=batch_size,
        )
        curves[:, count] = similarities[0, 0]
    pd.DataFrame(curves).to_csv(output / "dose_similarity_curves.csv", index=False)
    plot_similarity_curves(
        curves,
        output / "dose_similarity_curves.png",
        xlabel="Inserted number of sticky token",
        dpi=int(config["plot"]["dpi"]),
    )
    return {
        "phase": "full",
        "mode": "single_sticky",
        "u": mean_similarity,
        "paper_epsilon": paper_epsilon,
        "paper_validated_count": int(paper_frame["paper_validated"].sum()),
        "frozen_token": str(frozen["literal"]),
        "frozen_token_id": int(frozen["token_id"]),
        "minimum_effective_repeat_count": minimum_effective,
        "test": test_result,
    }


def _smoke_overrides(config: dict[str, Any]) -> None:
    config["vocabulary"]["max_candidates"] = 48
    config["runtime"]["batch_size"] = 32
    config["runtime"]["screen_candidate_chunk_size"] = 12
    config["runtime"]["evaluation_candidate_chunk_size"] = 6
    config["runtime"]["screen_examples_per_group"] = 3
    config["runtime"]["screen_unique_sentences"] = 8
    config["runtime"]["search_examples_per_group"] = 2
    config["runtime"]["search_unique_sentences"] = 8
    if config["mode"] != "single_sticky":
        config["candidate_pool"]["candidate_pool_max_size"] = 24
        config["candidate_pool"]["random_tokens"] = 12
        config["search"]["max_trigger_length"] = 2
        config["search"]["population_size"] = 8
        config["search"]["iterations"] = 2
        config["search"]["restarts_per_length"] = 1
        config["search"]["archive_size"] = 8
        config["validation"]["candidate_count_per_length"] = 4
        config["validation"]["candidates_per_restart"] = 4
        if config["mode"] == "repulsive_attractor":
            config["mode3"]["cluster_count_grid"] = [2, 3]
            config["mode3"]["minimum_cluster_size"] = 1
    else:
        config["single_sticky"]["paper_replication"]["candidate_ratio"] = 0.10
        config["single_sticky"]["dose_curve"]["max_repeat_count"] = 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=["prepare", "search", "finalize", "full"], required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--restart", type=int, default=0)
    parser.add_argument("--lengths", default=None, help="Comma-separated registered component lengths")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = _load_config(args.config)
    if args.device:
        config["model"]["device"] = args.device
    if args.smoke:
        _smoke_overrides(config)
    seed_everything(int(config["seed"]))
    output = args.output_dir.resolve() if args.output_dir else _resolve(config, config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "resolved_config.json", config)
    started = time.time()
    encoder = _encoder(config)
    if args.phase == "prepare":
        summary = prepare(config, encoder, output)
    elif args.phase == "search":
        if config["mode"] == "single_sticky":
            raise ValueError("Mode 1 does not use the CEM search phase")
        lengths = [int(value) for value in args.lengths.split(",")] if args.lengths else _lengths(config)
        summary = search(config, encoder, output, args.restart, lengths)
    elif args.phase == "finalize":
        if config["mode"] == "single_sticky":
            raise ValueError("Mode 1 uses --phase full")
        summary = finalize(config, encoder, output)
    else:
        if config["mode"] != "single_sticky":
            raise ValueError("Modes 2 and 3 use prepare/search/finalize phases")
        summary = run_mode1(config, encoder, output)
    repo_root = Path(__file__).resolve().parents[1]
    summary.update(
        {
            "protocol_version": 2,
            "git_commit": _git_commit(repo_root),
            "runtime_seconds": time.time() - started,
            "environment": _environment(config, encoder),
        }
    )
    suffix = f"_{args.restart:02d}" if args.phase == "search" else ""
    _write_json(output / f"{args.phase}_summary{suffix}.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
