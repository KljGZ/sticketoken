"""Registered Mode 3 V3 experiment runner.

The runner keeps search, validation selection, and the one-shot test strictly
separated.  It supports independent prefix/suffix/random protocols, 3A
separator and 3B compact blank-region objectives, a final universal-position
audit, continuous soft-prompt feasibility, exhaustive token screening, CEM,
and optional HotFlip multi-coordinate refinement.
"""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import math
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from ..insertion import insert_trigger
from ..model import SentenceTransformerEncoder
from ..seed import seed_everything
from ..tokens import load_vocabulary, trigger_realizability
from .cem_search import cem_search, expand_warm_sequences
from .data import SPLITS, build_ood_corpus, build_unique_corpus
from .gradient_search import gradient_beam_search
from .metrics import (
    blank_sort_key,
    evaluate_mode3,
    grouped_bootstrap,
    separator_sort_key,
)
from .retrieval_bridge import optimize_anchor
from .soft_prompt import optimize_soft_prompt
from .support import BenignSupportModel, normalize_rows, select_spherical_kmeans
from .visualization import plot_joint_progression, plot_length_protocols


ROOT = Path(__file__).resolve().parents[2]


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if int(config.get("protocol_version", 0)) != 3:
        raise ValueError("Mode 3 V3 requires protocol_version: 3")
    return config


def _resolve(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _environment(config: dict[str, Any], encoder: SentenceTransformerEncoder | None) -> dict[str, Any]:
    device = str(config["model"].get("device", "cpu"))
    cuda_name = None
    if device.startswith("cuda") and torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(torch.device(device))
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sentence_transformers": _package_version("sentence-transformers"),
        "transformers": _package_version("transformers"),
        "scikit_learn": _package_version("scikit-learn"),
        "umap_learn": _package_version("umap-learn"),
        "device": device,
        "cuda_name": cuda_name,
        "model_id": str(config["model"]["id"]),
        "model_revision": None if encoder is None else str(encoder.revision),
        "embedding_dimension": None if encoder is None else int(encoder.embedding_dim),
        "precision": "float32 hard-text evaluation",
    }


def _encoder(config: dict[str, Any]) -> SentenceTransformerEncoder:
    model = config["model"]
    return SentenceTransformerEncoder(
        model_id=str(model["id"]),
        device=str(model["device"]),
        cache_folder=model.get("cache_folder"),
        local_path=model.get("local_path"),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
    )


def _lengths(config: dict[str, Any]) -> list[int]:
    search = config["search"]
    minimum = int(search["min_trigger_length"])
    maximum = int(search["max_trigger_length"])
    step = int(search["length_step"])
    if minimum < 1 or maximum < minimum or step < 1:
        raise ValueError("Invalid V3 length schedule")
    values = list(range(minimum, maximum + 1, step))
    if values[-1] != maximum:
        raise ValueError("max_trigger_length must lie on the registered length schedule")
    return values


def _positions(config: dict[str, Any]) -> list[str]:
    values = list(map(str, config["insertion"]["positions"]))
    if not values or any(value not in {"prefix", "suffix", "random"} for value in values):
        raise ValueError("V3 positions must be a non-empty subset of prefix/suffix/random")
    return values


def _subprotocols(config: dict[str, Any]) -> list[str]:
    values = list(map(str, config["mode3"]["subprotocols"]))
    if not values or any(value not in {"separator", "blank"} for value in values):
        raise ValueError("V3 subprotocols must be separator and/or blank")
    return values


def _load_split(output: Path, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(output / f"unique_{split}.csv", keep_default_na=False)
    embeddings = np.asarray(np.load(output / f"unique_{split}_embeddings.npy"), dtype=np.float32)
    if len(frame) != len(embeddings):
        raise AssertionError(f"Frame/embedding mismatch for {split}")
    return frame, embeddings


def _data_paths(data: dict[str, Any], *, ood: bool = False) -> list[Path]:
    prefix = "ood_" if ood else ""
    explicit = data.get(prefix + "paths")
    single = data.get(prefix + "path")
    pattern = data.get(prefix + "paths_glob")
    paths: list[Path] = []
    if explicit:
        paths.extend(path for value in explicit if (path := _resolve(value)) is not None)
    if single:
        resolved = _resolve(single)
        if resolved is not None:
            paths.append(resolved)
    if pattern:
        paths.extend(sorted(ROOT.glob(str(pattern))))
    excluded = {str(_resolve(value)) for value in data.get(prefix + "exclude_paths", [])}
    paths = [path for path in paths if str(path) not in excluded]
    return list(dict.fromkeys(paths))


def _vocabulary(config: dict[str, Any]) -> pd.DataFrame:
    vocabulary = load_vocabulary(
        _resolve(config["vocabulary"]["analysis_path"]),
        allow_special=bool(config["vocabulary"].get("allow_special_tokens", False)),
        max_chars=int(config["vocabulary"].get("max_chars", 64)),
    )
    maximum = config["vocabulary"].get("max_candidates")
    return vocabulary.head(int(maximum)).reset_index(drop=True) if maximum else vocabulary.reset_index(drop=True)


def _encode_insertions(
    encoder: SentenceTransformerEncoder,
    texts: Sequence[str],
    triggers: Sequence[str],
    position: str,
    config: dict[str, Any],
) -> np.ndarray:
    chunk_size = int(config["runtime"].get("evaluation_candidate_chunk_size", 16))
    output: list[np.ndarray] = []
    separator = str(config["insertion"].get("separator", ""))
    for start in range(0, len(triggers), chunk_size):
        chunk = list(triggers[start : start + chunk_size])
        inserted = [
            insert_trigger(
                text,
                trigger,
                position,
                seed=int(config["seed"]),
                separator=separator,
            )
            for trigger in chunk
            for text in texts
        ]
        values = encoder.encode_texts(
            inserted,
            batch_size=int(config["runtime"]["batch_size"]),
            show_progress=False,
        )
        output.extend(values.reshape(len(chunk), len(texts), -1))
    return np.asarray(output, dtype=np.float32)


def _evaluate_triggers(
    encoder: SentenceTransformerEncoder,
    frame: pd.DataFrame,
    original: np.ndarray,
    support: BenignSupportModel,
    triggers: Sequence[str],
    position: str,
    config: dict[str, Any],
    *,
    bootstrap: bool,
    seed_offset: int = 0,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    triggered = _encode_insertions(encoder, frame["text"].tolist(), triggers, position, config)
    records = _evaluate_encoded_triggers(
        frame,
        original,
        support,
        triggered,
        position,
        config,
        bootstrap=bootstrap,
        seed_offset=seed_offset,
    )
    return records, triggered


def _evaluate_encoded_triggers(
    frame: pd.DataFrame,
    original: np.ndarray,
    support: BenignSupportModel,
    triggered: np.ndarray,
    position: str,
    config: dict[str, Any],
    *,
    bootstrap: bool,
    seed_offset: int = 0,
) -> list[dict[str, Any]]:
    """Evaluate already encoded triggers without changing statistical seeds.

    Full-search checkpoints repeatedly revisit formal CEM champions.  Encoding
    is deterministic for the frozen texts, trigger literal, insertion mode,
    seed, and model, whereas pairwise/bootstrap sampling is intentionally
    re-evaluated at every call.  Separating these operations permits an exact
    embedding cache without caching or weakening any hard metric.
    """
    constraints = {key: float(value) for key, value in config["constraints"].items()}
    groups = (
        frame["source_group"].astype(str).to_numpy()
        if "source_group" in frame.columns
        else support.assign_clusters(original)
    )
    records: list[dict[str, Any]] = []
    for index, values in enumerate(triggered):
        if bootstrap:
            record = grouped_bootstrap(
                original,
                values,
                support,
                constraints,
                groups,
                replicates=int(config["statistics"]["bootstrap_replicates"]),
                confidence=float(config["statistics"].get("confidence", 0.95)),
                pairwise_sample_size=int(config["mode3"].get("pairwise_sample_size", 20000)),
                seed=int(config["seed"]) + seed_offset + index * 1009,
            )
        else:
            record = asdict(
                evaluate_mode3(
                    original,
                    values,
                    support,
                    constraints,
                    pairwise_sample_size=int(config["mode3"].get("pairwise_sample_size", 20000)),
                    seed=int(config["seed"]) + seed_offset + index * 1009,
                )
            )
        record["position"] = position
        record["min_displacement_q05"] = constraints["min_displacement_q05"]
        record["max_compact_radius_q95"] = constraints["max_compact_radius_q95"]
        records.append(record)
    return records


def _token_ids(sequence: Sequence[int], pool: pd.DataFrame) -> tuple[int, ...]:
    return tuple(int(pool.iloc[int(index)]["token_id"]) for index in sequence)


def _trigger_from_ids(encoder: SentenceTransformerEncoder, token_ids: Sequence[int]) -> str:
    return encoder.decode(tuple(map(int, token_ids)))


def _trigger_audit(encoder: SentenceTransformerEncoder, token_ids: Sequence[int]) -> dict[str, Any]:
    intended = tuple(map(int, token_ids))
    trigger = _trigger_from_ids(encoder, intended)
    actual = tuple(encoder.tokenize([trigger], add_special_tokens=False)[0])
    return {
        "trigger": trigger,
        "component_token_ids": ",".join(map(str, intended)),
        "actual_token_ids": ",".join(map(str, actual)),
        "actual_token_length": len(actual),
        "exact_token_roundtrip": actual == intended,
    }


def _parse_ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, (tuple, list, np.ndarray)):
        return tuple(map(int, value))
    return tuple(int(item) for item in str(value).split(",") if str(item).strip())


def _sort_key(protocol: str):
    base = separator_sort_key if protocol == "separator" else blank_sort_key

    def key(record: dict[str, Any]) -> tuple[Any, ...]:
        return (0 if bool(record.get("exact_token_roundtrip", True)) else 1, *base(record))

    return key


def prepare_common(config: dict[str, Any], encoder: SentenceTransformerEncoder, output: Path) -> dict[str, Any]:
    data = config["data"]
    fractions = data["split"]
    input_paths = _data_paths(data)
    splits, audit = build_unique_corpus(
        input_paths,
        encoder.tokenizer,
        text_columns=data.get("text_columns", ["sentence1", "sentence2"]),
        source_column=data.get("source_column"),
        min_tokens=int(data["min_tokens"]),
        max_tokens=int(data["max_tokens"]),
        fractions=(float(fractions["search"]), float(fractions["validation"]), float(fractions["test"])),
        seed=int(config["seed"]),
        sample_limits={name: int(value) for name, value in data.get("sample_limits", {}).items()},
    )
    output.mkdir(parents=True, exist_ok=True)
    all_ids: set[str] = set()
    for split, frame in splits.items():
        frame.to_csv(output / f"unique_{split}.csv", index=False)
        embeddings = encoder.encode_texts(
            frame["text"].tolist(),
            batch_size=int(config["runtime"]["batch_size"]),
            show_progress=bool(config["runtime"].get("show_progress", True)),
        )
        np.save(output / f"unique_{split}_embeddings.npy", embeddings)
        all_ids.update(frame["sentence_id"].astype(str))
    ood = build_ood_corpus(
        _data_paths(data, ood=True),
        encoder.tokenizer,
        text_columns=data.get("ood_text_columns", data.get("text_columns", ["sentence1", "sentence2"])),
        min_tokens=int(data["min_tokens"]),
        max_tokens=int(data["max_tokens"]),
        excluded_sentence_ids=all_ids,
        sample_limit=int(data.get("ood_sample_limit", 0)) or None,
        seed=int(config["seed"]) + 90000,
    )
    if len(ood):
        ood.to_csv(output / "unique_ood.csv", index=False)
        np.save(
            output / "unique_ood_embeddings.npy",
            encoder.encode_texts(ood["text"].tolist(), batch_size=int(config["runtime"]["batch_size"])),
        )
    audit["ood_size"] = int(len(ood))
    audit["recommended_minimums"] = dict(config["data"].get("recommended_minimums", {}))
    audit["minimums_met"] = {
        split: len(splits[split]) >= int(config["data"].get("recommended_minimums", {}).get(split, 0))
        for split in SPLITS
    }
    audit["minimums_met"]["ood"] = len(ood) >= int(config["data"].get("recommended_minimums", {}).get("ood", 0))
    _write_json(output / "data_audit.json", audit)
    search_frame, search_embeddings = _load_split(output, "search")
    support_settings = config["support"]
    clustering, selection = select_spherical_kmeans(
        search_embeddings,
        support_settings["cluster_count_grid"],
        seed=int(config["seed"]),
        restarts=int(support_settings.get("clustering_restarts", 3)),
        max_iterations=int(support_settings.get("max_iterations", 100)),
        tolerance=float(support_settings.get("tolerance", 1e-7)),
        minimum_cluster_size=int(support_settings.get("minimum_cluster_size", 5)),
    )
    support = BenignSupportModel.fit(
        search_embeddings,
        clustering,
        knn_k=int(support_settings.get("knn_k", 10)),
    )
    support.save(str(output / "benign_support.npz"))
    search_frame["semantic_cluster"] = clustering.labels
    search_frame.to_csv(output / "unique_search.csv", index=False)
    _write_json(output / "spherical_kmeans_audit.json", selection)
    _write_json(
        output / "support_audit.json",
        {
            "memory_size": len(support.memory),
            "embedding_dimension": support.memory.shape[1],
            "cluster_count": len(support.cluster_centers),
            "knn_k": support.knn_k,
            "benign_knn_q95": support.benign_knn_q95,
            "support_estimators": ["sample_support", "cluster_envelope", "knn_density"],
        },
    )
    return {"phase": "prepare-common", "split_sizes": audit["split_sizes"], "ood_size": len(ood), "cluster_count": len(support.cluster_centers)}


def screen_shard(
    config: dict[str, Any],
    encoder: SentenceTransformerEncoder,
    output: Path,
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    if not 0 <= shard_index < shard_count:
        raise ValueError("Invalid shard index/count")
    vocabulary = _vocabulary(config)
    indices = np.array_split(np.arange(len(vocabulary)), shard_count)[shard_index]
    shard = vocabulary.iloc[indices].reset_index(drop=True)
    frame, embeddings = _load_split(output, "search")
    sample_size = min(int(config["runtime"]["screen_unique_sentences"]), len(frame))
    chosen = np.sort(np.random.default_rng(int(config["seed"])).choice(len(frame), size=sample_size, replace=False))
    sample_frame = frame.iloc[chosen].reset_index(drop=True)
    sample_embeddings = embeddings[chosen]
    support = BenignSupportModel.load(str(output / "benign_support.npz"))
    chunk_size = int(config["runtime"].get("screen_candidate_chunk_size", 32))
    rows: list[dict[str, Any]] = []
    for start in range(0, len(shard), chunk_size):
        chunk = shard.iloc[start : start + chunk_size].reset_index(drop=True)
        print(f"V3 screen shard {shard_index + 1}/{shard_count}: {start + 1}-{start + len(chunk)} / {len(shard)}", flush=True)
        for position in _positions(config):
            metrics, _ = _evaluate_triggers(
                encoder,
                sample_frame,
                sample_embeddings,
                support,
                chunk["literal"].astype(str).tolist(),
                position,
                config,
                bootstrap=False,
                seed_offset=shard_index * 100000 + start,
            )
            for token, metric in zip(chunk.to_dict("records"), metrics):
                rows.append({**token, "position": position, **metric})
    shard_dir = output / "screen_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / f"shard_{shard_index:02d}_of_{shard_count:02d}.csv"
    pd.DataFrame.from_records(rows).to_csv(path, index=False)
    return {"phase": "screen-shard", "shard_index": shard_index, "shard_count": shard_count, "token_count": len(shard), "row_count": len(rows), "output": str(path)}


def merge_prepare(config: dict[str, Any], output: Path, shard_count: int) -> dict[str, Any]:
    paths = [output / "screen_shards" / f"shard_{index:02d}_of_{shard_count:02d}.csv" for index in range(shard_count)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing V3 screen shards: {missing}")
    screen = pd.concat([pd.read_csv(path, keep_default_na=False) for path in paths], ignore_index=True)
    vocabulary = _vocabulary(config)
    expected = len(vocabulary) * len(_positions(config))
    if len(screen) != expected or screen.groupby(["token_id", "position"]).size().max() != 1:
        raise AssertionError("V3 screen merge is not exhaustive and duplicate-free")
    screen.to_csv(output / "single_token_screen.csv", index=False)
    settings = config["candidate_pool"]
    selected: list[int] = []
    sources: dict[int, set[str]] = defaultdict(set)

    def take(rows: Iterable[int], count: int, source: str) -> None:
        for rank, token_id in enumerate(rows):
            if rank >= count:
                break
            token = int(token_id)
            sources[token].add(source)
            if token not in selected:
                selected.append(token)

    for position in _positions(config):
        subset = screen[screen["position"] == position]
        take(subset.sort_values("separation_margin", ascending=False)["token_id"], int(settings["top_separator_tokens_per_position"]), f"{position}_separator")
        take(subset.sort_values("sample_blank_margin", ascending=False)["token_id"], int(settings["top_blank_tokens_per_position"]), f"{position}_blank")
        take(subset.sort_values("compact_radius_q95", ascending=True)["token_id"], int(settings["top_compact_tokens_per_position"]), f"{position}_compact")
    rng = np.random.default_rng(int(config["seed"]))
    random_ids = vocabulary["token_id"].astype(int).to_numpy(copy=True)
    rng.shuffle(random_ids)
    take(random_ids, int(settings["random_tokens"]), "random")
    maximum = int(settings["candidate_pool_max_size"])
    if len(selected) > maximum:
        raise ValueError(
            f"Candidate union has {len(selected)} unique tokens but candidate_pool_max_size={maximum}; "
            "silent truncation would bias earlier insertion positions"
        )
    metadata = vocabulary.set_index("token_id").loc[selected].reset_index()
    metadata.insert(0, "pool_index", np.arange(len(metadata)))
    metadata["pool_source"] = ["+".join(sorted(sources[int(token)])) for token in metadata["token_id"]]
    metadata.to_csv(output / "candidate_pool.csv", index=False)
    return {
        "phase": "merge-prepare",
        "screen_rows": len(screen),
        "valid_vocabulary_size": len(vocabulary),
        "candidate_pool_size": len(metadata),
        "candidate_pool_max_size": maximum,
        "source_membership_counts": {
            source: sum(source in memberships for memberships in sources.values())
            for source in sorted({name for memberships in sources.values() for name in memberships})
        },
    }


def soft_prompt_phase(
    config: dict[str, Any],
    encoder: SentenceTransformerEncoder,
    output: Path,
    positions: Sequence[str],
    protocols: Sequence[str],
    lengths: Sequence[int],
) -> dict[str, Any]:
    frame, embeddings = _load_split(output, "search")
    support = BenignSupportModel.load(str(output / "benign_support.npz"))
    vocabulary = _vocabulary(config)
    settings = config["soft_prompt"]
    sample_size = min(int(settings.get("sample_size", len(frame))), len(frame))
    chosen = np.sort(np.random.default_rng(int(config["seed"]) + 700).choice(len(frame), size=sample_size, replace=False))
    records: list[dict[str, Any]] = []
    for position in positions:
        if position == "random":
            continue
        for protocol in protocols:
            for length in lengths:
                result = optimize_soft_prompt(
                    encoder,
                    frame.iloc[chosen]["text"].tolist(),
                    embeddings[chosen],
                    support,
                    {key: float(value) for key, value in config["constraints"].items()},
                    vocabulary["token_id"].astype(int).tolist(),
                    length=length,
                    position=position,
                    subprotocol=protocol,
                    iterations=int(settings["iterations"]),
                    learning_rate=float(settings["learning_rate"]),
                    temperature=float(settings["temperature"]),
                    batch_size=int(settings["batch_size"]),
                    seed=int(config["seed"]) + length * 100 + (0 if position == "prefix" else 10000) + (0 if protocol == "separator" else 20000),
                )
                directory = output / "soft_prompt" / position / protocol / f"length_{length:02d}"
                directory.mkdir(parents=True, exist_ok=True)
                np.save(directory / "prompt_embeddings.npy", result.prompt_embeddings)
                pd.DataFrame.from_records(result.history).to_csv(directory / "history.csv", index=False)
                record = {"position": position, "subprotocol": protocol, "length": length, "nearest_token_ids": ",".join(map(str, result.nearest_token_ids)), **result.continuous_metrics}
                _write_json(directory / "result.json", record)
                records.append(record)
    summary_frame = pd.DataFrame.from_records(records)
    for (position, protocol), group in summary_frame.groupby(["position", "subprotocol"], sort=True):
        directory = output / "soft_prompt" / str(position) / str(protocol)
        directory.mkdir(parents=True, exist_ok=True)
        group.to_csv(directory / "summary.csv", index=False)
    # Registered multi-GPU runs use one process per position/protocol.  Only a
    # single process spanning multiple tasks may safely write the convenience
    # aggregate; task-local summaries above never collide.
    if len(set(zip(summary_frame["position"], summary_frame["subprotocol"]))) > 1:
        summary_frame.to_csv(output / "soft_prompt_summary.csv", index=False)
    return {"phase": "soft-prompt", "runs": len(records), "lengths": list(lengths)}


def _score_id_sequences(
    encoder: SentenceTransformerEncoder,
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    support: BenignSupportModel,
    sequences: Sequence[tuple[int, ...]],
    position: str,
    protocol: str,
    config: dict[str, Any],
    *,
    seed_offset: int,
    encoded_cache: dict[tuple[str, tuple[int, ...]], np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    if position == "universal":
        by_position = [
            _score_id_sequences(
                encoder,
                frame,
                embeddings,
                support,
                sequences,
                registered_position,
                protocol,
                config,
                seed_offset=seed_offset + offset * 100000,
                encoded_cache=encoded_cache,
            )
            for offset, registered_position in enumerate(_positions(config))
        ]
        return [
            _aggregate_position_records(
                {registered_position: by_position[offset][index] for offset, registered_position in enumerate(_positions(config))},
                protocol,
            )
            for index in range(len(sequences))
        ]
    normalized = [tuple(map(int, sequence)) for sequence in sequences]
    audits = [_trigger_audit(encoder, sequence) for sequence in normalized]
    if encoded_cache is None:
        metrics, _ = _evaluate_triggers(
            encoder,
            frame,
            embeddings,
            support,
            [audit["trigger"] for audit in audits],
            position,
            config,
            bootstrap=False,
            seed_offset=seed_offset,
        )
    else:
        missing_indices = [
            index
            for index, sequence in enumerate(normalized)
            if (position, sequence) not in encoded_cache
        ]
        if missing_indices:
            missing_values = _encode_insertions(
                encoder,
                frame["text"].tolist(),
                [audits[index]["trigger"] for index in missing_indices],
                position,
                config,
            )
            for index, values in zip(missing_indices, missing_values):
                encoded_cache[(position, normalized[index])] = values
        triggered = np.asarray(
            [encoded_cache[(position, sequence)] for sequence in normalized],
            dtype=np.float32,
        )
        metrics = _evaluate_encoded_triggers(
            frame,
            embeddings,
            support,
            triggered,
            position,
            config,
            bootstrap=False,
            seed_offset=seed_offset,
        )
    output: list[dict[str, Any]] = []
    for sequence, audit, metric in zip(sequences, audits, metrics):
        metric.update(audit)
        metric["component_length"] = len(sequence)
        if not audit["exact_token_roundtrip"]:
            metric["separator_certified"] = False
            metric["blank_region_certified"] = False
        metric["search_subprotocol"] = protocol
        output.append(metric)
    return output


def _aggregate_position_records(records: dict[str, dict[str, Any]], protocol: str) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot aggregate an empty position record set")
    first = dict(next(iter(records.values())))
    minimize = [
        "displacement_q05",
        "displacement_median",
        "displacement_q95",
        "separation_margin",
        "mean_separation",
        "linear_auc",
        "balanced_accuracy",
        "pairwise_mean",
        "pairwise_q05",
        "sample_blank_margin",
        "cluster_blank_margin",
        "density_blank_margin",
        "source_escape_q05",
        "center_norm_pre_normalization",
    ]
    maximize = ["fpr_at_95_tpr", "compact_radius_q95"]
    for name in minimize:
        available = [float(record[name]) for record in records.values() if name in record]
        if available:
            first[name] = min(available)
    for name in maximize:
        available = [float(record[name]) for record in records.values() if name in record]
        if available:
            first[name] = max(available)
    for name in [
        "shift_certified",
        "separator_certified",
        "compact_certified",
        "sample_blank_certified",
        "cluster_blank_certified",
        "density_blank_certified",
        "blank_region_certified",
        "random_baseline_exceeded",
        "validation_certified",
        "test_certified",
    ]:
        available = [bool(record[name]) for record in records.values() if name in record]
        if available:
            first[name] = all(available)
    first["position"] = "universal"
    first["position_universal_certified"] = bool(
        all(
            record.get("separator_certified", False) if protocol == "separator" else record.get("blank_region_certified", False)
            for record in records.values()
        )
    )
    first["per_position_metrics"] = records
    first["per_position_metrics_json"] = json.dumps(records, ensure_ascii=False, default=_json_default)
    return first


def search(
    config: dict[str, Any],
    encoder: SentenceTransformerEncoder,
    output: Path,
    *,
    restart: int,
    position: str,
    protocol: str,
    lengths: Sequence[int],
) -> dict[str, Any]:
    if position not in [*_positions(config), "universal"] or protocol not in _subprotocols(config):
        raise ValueError("Unregistered V3 position/subprotocol")
    registered = _lengths(config)
    if any(length not in registered for length in lengths):
        raise ValueError(f"Lengths must belong to {registered}")
    pool = pd.read_csv(output / "candidate_pool.csv", keep_default_na=False)
    frame, embeddings = _load_split(output, "search")
    support = BenignSupportModel.load(str(output / "benign_support.npz"))
    settings = config["search"]
    key = _sort_key(protocol)
    previous: list[tuple[int, ...]] = []
    files: list[str] = []
    directory = output / "search" / position / protocol / f"restart_{restart:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    for length in lengths:
        full_embedding_cache: dict[tuple[str, tuple[int, ...]], np.ndarray] = {}

        def score_pool(sequences: list[tuple[int, ...]], iteration: int, *, full: bool) -> list[dict[str, Any]]:
            if full:
                chosen = np.arange(len(frame))
            else:
                rng = np.random.default_rng(int(config["seed"]) + restart * 1000000 + length * 10000 + iteration)
                size = min(int(config["runtime"]["search_unique_sentences"]), len(frame))
                chosen = np.sort(rng.choice(len(frame), size=size, replace=False))
            ids = [_token_ids(sequence, pool) for sequence in sequences]
            return _score_id_sequences(
                encoder,
                frame.iloc[chosen].reset_index(drop=True),
                embeddings[chosen],
                support,
                ids,
                position,
                protocol,
                config,
                seed_offset=restart * 1000000 + length * 10000 + iteration,
                encoded_cache=full_embedding_cache if full else None,
            )

        warm = expand_warm_sequences(
            previous,
            length,
            len(pool),
            int(settings["population_size"]) // 2,
            int(config["seed"]) + restart * 1000 + length,
        )
        result = cem_search(
            len(pool),
            length,
            lambda values, iteration: score_pool(values, iteration, full=False),
            lambda values, iteration: score_pool(values, iteration, full=True),
            sort_key=key,
            population_size=int(settings["population_size"]),
            elite_ratio=float(settings["elite_ratio"]),
            iterations=int(settings["iterations"]),
            update_alpha=float(settings["update_alpha"]),
            probability_floor=float(settings["probability_floor"]),
            uniform_mixture=float(settings["uniform_mixture"]),
            entropy_min_fraction=float(settings["entropy_min_fraction"]),
            stall_patience=int(settings["stall_patience"]),
            elite_min_hamming_fraction=float(settings["elite_min_hamming_fraction"]),
            full_evaluation_interval=int(settings["full_evaluation_interval"]),
            archive_size=int(settings["archive_size"]),
            seed=int(config["seed"]) + restart * 10000 + length + (0 if position == "prefix" else 100000) + (0 if protocol == "separator" else 200000),
            initial_sequences=warm,
        )
        rows: list[dict[str, Any]] = []
        for rank, record in enumerate(result.candidates, 1):
            pool_sequence = tuple(record["sequence"])
            ids = _token_ids(pool_sequence, pool)
            rows.append(
                {
                    "rank": rank,
                    "algorithm": "cem_v3",
                    "restart": restart,
                    "position": position,
                    "subprotocol": protocol,
                    "component_length": length,
                    "pool_sequence": ",".join(map(str, pool_sequence)),
                    **_trigger_audit(encoder, ids),
                    **{name: value for name, value in record.items() if name not in {"sequence", "trigger", "component_token_ids", "actual_token_ids", "actual_token_length", "exact_token_roundtrip"}},
                }
            )
        # Optional white-box refinement starts from the formal CEM champions
        # and proposes from the full legal vocabulary.  Every proposal is
        # accepted/ranked only after a true hard-text forward pass.
        gradient = config.get("gradient_search", {})
        gradient_lengths = set(map(int, gradient.get("lengths", lengths)))
        gradient_restarts = set(map(int, gradient.get("restarts", [0])))
        if (
            bool(gradient.get("enabled", False))
            and rows
            and position != "universal"
            and length in gradient_lengths
            and restart in gradient_restarts
        ):
            initial_ids = [_parse_ids(row["component_token_ids"]) for row in rows[: int(gradient.get("beam_width", 8))]]
            vocabulary = _vocabulary(config)

            sample_size = min(int(gradient.get("sample_size", 64)), len(frame))
            chosen = np.sort(np.random.default_rng(int(config["seed"]) + length).choice(len(frame), size=sample_size, replace=False))

            def hard(values: list[tuple[int, ...]], iteration: int) -> list[dict[str, Any]]:
                return _score_id_sequences(
                    encoder,
                    frame.iloc[chosen].reset_index(drop=True),
                    embeddings[chosen],
                    support,
                    values,
                    position,
                    protocol,
                    config,
                    seed_offset=50000000 + restart * 100000 + length * 1000 + iteration,
                )

            refined = gradient_beam_search(
                encoder,
                frame.iloc[chosen]["text"].tolist(),
                embeddings[chosen],
                support,
                initial_ids,
                vocabulary["token_id"].astype(int).tolist(),
                hard,
                sort_key=key,
                position=position,
                subprotocol=protocol,
                gradient_top_m=int(gradient["gradient_top_m"]),
                beam_width=int(gradient["beam_width"]),
                candidate_batch=int(gradient["candidate_batch"]),
                iterations=int(gradient["iterations"]),
                temperature=float(gradient["temperature"]),
                seed=int(config["seed"]) + restart * 10000 + length,
            )
            refined_limit = min(
                int(settings["archive_size"]),
                int(gradient.get("full_candidate_count", config["validation"]["candidate_count_per_length"])),
            )
            refined_ids = [tuple(record["sequence"]) for record in refined.candidates[:refined_limit]]
            refined_full = _score_id_sequences(
                encoder,
                frame,
                embeddings,
                support,
                refined_ids,
                position,
                protocol,
                config,
                seed_offset=60000000 + restart * 100000 + length * 1000,
            )
            for ids, full_record in zip(refined_ids, refined_full):
                rows.append(
                    {
                        "rank": 0,
                        "algorithm": "hotflip_multicoordinate",
                        "restart": restart,
                        "position": position,
                        "subprotocol": protocol,
                        "component_length": length,
                        "pool_sequence": "",
                        **_trigger_audit(encoder, ids),
                        **{name: value for name, value in full_record.items() if name not in {"sequence", "trigger", "component_token_ids", "actual_token_ids", "actual_token_length", "exact_token_roundtrip"}},
                    }
                )
            pd.DataFrame.from_records(refined.history).to_csv(directory / f"length_{length:02d}_gradient_history.csv", index=False)
        rows = sorted(rows, key=key)[: int(settings["archive_size"])]
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
        candidate_path = directory / f"length_{length:02d}_candidates.csv"
        pd.DataFrame.from_records(rows).to_csv(candidate_path, index=False)
        pd.DataFrame.from_records(result.history).to_csv(directory / f"length_{length:02d}_history.csv", index=False)
        pd.DataFrame.from_records(result.full_champion_history).to_csv(directory / f"length_{length:02d}_full_champions.csv", index=False)
        previous = [_parse_ids(row["pool_sequence"]) for row in rows if row.get("pool_sequence")][: int(settings["warm_elite_count"])]
        files.append(str(candidate_path))
    return {"phase": "search", "position": position, "subprotocol": protocol, "restart": restart, "lengths": list(lengths), "files": files}


def _candidate_inputs(output: Path, config: dict[str, Any], position: str, protocol: str, length: int) -> list[dict[str, Any]]:
    count = int(config["validation"]["candidate_count_per_length"])
    if length == 1:
        screen = pd.read_csv(output / "single_token_screen.csv", keep_default_na=False)
        if position == "universal":
            metric = "separation_margin" if protocol == "separator" else "sample_blank_margin"
            grouped = screen.groupby("token_id", as_index=False).agg(
                literal=("literal", "first"),
                worst_metric=(metric, "min"),
            )
            subset = grouped.sort_values("worst_metric", ascending=False)
        else:
            subset = screen[screen["position"] == position].copy()
            subset = subset.sort_values("separation_margin" if protocol == "separator" else "sample_blank_margin", ascending=False)
        return [
            {
                "algorithm": "exhaustive_single_token",
                "source_restart": -1,
                "component_length": 1,
                "component_token_ids": str(int(row["token_id"])),
                "trigger": str(row["literal"]),
            }
            for _, row in subset.head(count).iterrows()
        ]
    records: list[dict[str, Any]] = []
    for restart in range(int(config["search"]["restarts_per_length"])):
        path = output / "search" / position / protocol / f"restart_{restart:02d}" / f"length_{length:02d}_candidates.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing registered V3 search output: {path}")
        for row in pd.read_csv(path, keep_default_na=False).head(int(config["validation"]["candidates_per_restart"])).to_dict("records"):
            row["source_restart"] = restart
            records.append(row)
    deduplicated: dict[str, dict[str, Any]] = {}
    for record in sorted(records, key=_sort_key(protocol)):
        deduplicated.setdefault(str(record["component_token_ids"]), record)
    return list(deduplicated.values())[:count]


def _random_baseline_threshold(
    encoder: SentenceTransformerEncoder,
    output: Path,
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    support: BenignSupportModel,
    pool: pd.DataFrame,
    position: str,
    protocol: str,
    length: int,
    config: dict[str, Any],
) -> tuple[float, list[dict[str, Any]]]:
    count = int(config["validation"].get("random_baseline_count", 32))
    rng = np.random.default_rng(int(config["seed"]) + length * 1000 + _positions(config).index(position) * 100 + _subprotocols(config).index(protocol))
    sequences = [tuple(map(int, rng.choice(pool["token_id"].astype(int), size=length, replace=True))) for _ in range(count)]
    records = _score_id_sequences(encoder, frame, embeddings, support, sequences, position, protocol, config, seed_offset=70000000 + length * 1000)
    objective_name = "separation_margin" if protocol == "separator" else "sample_blank_margin"
    threshold = float(np.quantile([record[objective_name] for record in records], float(config["validation"].get("random_baseline_quantile", 0.99))))
    rows = [{"baseline": "random_legal", "component_length": length, **_trigger_audit(encoder, sequence), **record} for sequence, record in zip(sequences, records)]
    screen = pd.read_csv(output / "single_token_screen.csv", keep_default_na=False)
    position_screen = screen[screen["position"] == position]
    best_separator = int(position_screen.sort_values("separation_margin", ascending=False).iloc[0]["token_id"])
    best_blank = int(position_screen.sort_values("sample_blank_margin", ascending=False).iloc[0]["token_id"])
    best_compact = int(position_screen.sort_values("compact_radius_q95", ascending=True).iloc[0]["token_id"])
    best_mean_distance = int(position_screen.sort_values("displacement_median", ascending=False).iloc[0]["token_id"])
    frequent = tuple(map(int, config["mode3"].get("frequent_token_ids", [8, 5, 6])))
    v2 = tuple(map(int, config["mode3"].get("v2_fallback_token_ids", [best_blank])))
    sticky = int(config["mode3"].get("sticky_token_id", best_blank))
    natural = tuple(encoder.tokenize([str(frame.iloc[0]["text"])], add_special_tokens=False)[0])

    def resize(values: tuple[int, ...]) -> tuple[int, ...]:
        if not values:
            values = (best_blank,)
        return tuple(values[index % len(values)] for index in range(length))

    controls = {
        "frequent_ordinary": resize(frequent),
        "natural_phrase": resize(natural),
        "best_sticky_repeated": (sticky,) * length,
        "v2_fallback_adjusted": resize(v2),
        "separator_only_repeated": (best_separator,) * length,
        "blank_only_repeated": (best_blank,) * length,
        "compact_only_repeated": (best_compact,) * length,
        "agentpoison_mean_distance_proxy": (best_mean_distance,) * length,
    }
    control_sequences = list(controls.values())
    control_metrics = _score_id_sequences(
        encoder,
        frame,
        embeddings,
        support,
        control_sequences,
        position,
        protocol,
        config,
        seed_offset=71000000 + length * 1000,
    )
    for (name, sequence), record in zip(controls.items(), control_metrics):
        rows.append({"baseline": name, "component_length": length, **_trigger_audit(encoder, sequence), **record})
    return threshold, rows


def _evaluate_candidates(
    encoder: SentenceTransformerEncoder,
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    support: BenignSupportModel,
    candidates: Sequence[dict[str, Any]],
    position: str,
    protocol: str,
    config: dict[str, Any],
    *,
    baseline_threshold: float,
    seed_offset: int,
    bootstrap: bool = True,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    triggers = [str(record["trigger"]) for record in candidates]
    metrics, triggered = _evaluate_triggers(
        encoder,
        frame,
        embeddings,
        support,
        triggers,
        position,
        config,
        bootstrap=bootstrap,
        seed_offset=seed_offset,
    )
    rows: list[dict[str, Any]] = []
    for candidate, metric in zip(candidates, metrics):
        ids = _parse_ids(candidate["component_token_ids"])
        audit = _trigger_audit(encoder, ids)
        realizability = trigger_realizability(
            encoder.tokenizer,
            audit["trigger"],
            ids,
            frame["text"].tolist()[: int(config["validation"]["realizability_contexts"])],
            [position],
            seed=int(config["seed"]),
            separator=str(config["insertion"].get("separator", "")),
        )
        objective = metric["separation_margin"] if protocol == "separator" else metric["sample_blank_margin"]
        baseline_exceeded = objective > baseline_threshold
        core = bool(metric["separator_certified"] if protocol == "separator" else metric["blank_region_certified"])
        certified = bool(
            core
            and audit["exact_token_roundtrip"]
            and float(realizability["realizability_rate"]) >= float(config["validation"]["min_realizability"])
            and (baseline_exceeded or not bool(config["validation"].get("require_random_baseline_exceedance", True)))
        )
        rows.append(
            {
                **candidate,
                **audit,
                **metric,
                **realizability,
                "position": position,
                "subprotocol": protocol,
                "random_baseline_threshold": baseline_threshold,
                "random_baseline_exceeded": baseline_exceeded,
                "validation_point_feasible": certified,
                "ci_evaluated": bootstrap,
                "validation_certified": certified if bootstrap else False,
            }
        )
    return rows, list(triggered)


def _anchor_test(anchor: np.ndarray, benign: np.ndarray, triggered: np.ndarray) -> dict[str, Any]:
    positive = normalize_rows(triggered) @ anchor
    negative = normalize_rows(benign) @ anchor
    margin = float(np.quantile(positive, 0.05) - np.quantile(negative, 0.95))
    return {
        "test_anchor_margin": margin,
        "test_triggered_score_q05": float(np.quantile(positive, 0.05)),
        "test_benign_score_q95": float(np.quantile(negative, 0.95)),
        "test_top_k_oracle_coverage": float(np.mean(positive > np.quantile(negative, 0.95))),
        "test_retrieval_anchor_certified": bool(margin > 0.0),
    }


def _prefix_growth(
    encoder: SentenceTransformerEncoder,
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    support: BenignSupportModel,
    frozen: dict[str, Any],
    position: str,
    protocol: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    ids = _parse_ids(frozen["component_token_ids"])
    sequences = [ids[:length] for length in range(1, len(ids) + 1)]
    records = _score_id_sequences(encoder, frame, embeddings, support, sequences, position, protocol, config, seed_offset=80000000)
    return pd.DataFrame.from_records([{"prefix_length": len(sequence), **record} for sequence, record in zip(sequences, records)])


def _plot_iteration_progression(
    encoder: SentenceTransformerEncoder,
    output: Path,
    task_dir: Path,
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    support: BenignSupportModel,
    pool: pd.DataFrame,
    frozen: dict[str, Any],
    position: str,
    protocol: str,
    config: dict[str, Any],
) -> None:
    restart = int(frozen.get("source_restart", frozen.get("restart", -1)))
    length = int(frozen["component_length"])
    if restart < 0 or length == 1:
        return
    path = output / "search" / position / protocol / f"restart_{restart:02d}" / f"length_{length:02d}_history.csv"
    if not path.exists():
        return
    history = pd.read_csv(path, keep_default_na=False)
    history = history[history["cumulative_full_sequence"].astype(str).str.len() > 0]
    if history.empty:
        return
    requested = [0, 5, 10, 20, int(history["iteration"].max())]
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for target in requested:
        row = history.iloc[(history["iteration"].astype(int) - target).abs().argsort()[:1]].iloc[0]
        iteration = int(row["iteration"])
        if iteration not in used:
            selected.append(row.to_dict())
            used.add(iteration)
    count = min(int(config["plot"].get("projection_sentence_count", 128)), len(frame))
    sample = np.arange(count)
    stages: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    labels: list[str] = []
    metric_labels: list[str] = []
    for row in selected:
        pool_sequence = _parse_ids(row["cumulative_full_sequence"])
        ids = _token_ids(pool_sequence, pool)
        trigger = _trigger_from_ids(encoder, ids)
        values = _encode_insertions(encoder, frame.iloc[sample]["text"].tolist(), [trigger], position, config)[0]
        metric = evaluate_mode3(
            embeddings[sample],
            values,
            support,
            {key: float(value) for key, value in config["constraints"].items()},
            seed=int(config["seed"]),
        )
        center = values.mean(axis=0)
        center /= max(float(np.linalg.norm(center)), 1e-12)
        stages.append(values)
        centers.append(center)
        labels.append(f"iteration {int(row['iteration'])}")
        metric_labels.append(
            f"M_sep={metric.separation_margin:.3f}  rho95={metric.compact_radius_q95:.3f}\n"
            f"M_sample={metric.sample_blank_margin:.3f}  M_cluster={metric.cluster_blank_margin:.3f}\n"
            f"M_density={metric.density_blank_margin:.3f}  pair_q05={metric.pairwise_q05:.3f}"
        )
    if stages:
        plot_joint_progression(
            embeddings[sample],
            stages,
            centers,
            labels,
            metric_labels,
            task_dir / "embedding_search_iteration_progression",
            seed=int(config["seed"]),
            dpi=int(config["plot"]["dpi"]),
        )


def _evaluate_universal_candidates(
    encoder: SentenceTransformerEncoder,
    output: Path,
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    support: BenignSupportModel,
    pool: pd.DataFrame,
    candidates: Sequence[dict[str, Any]],
    protocol: str,
    length: int,
    config: dict[str, Any],
    *,
    bootstrap: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_position_rows: dict[str, list[dict[str, Any]]] = {}
    baseline_rows: list[dict[str, Any]] = []
    for position in _positions(config):
        threshold, baselines = _random_baseline_threshold(
            encoder,
            output,
            frame,
            embeddings,
            support,
            pool,
            position,
            protocol,
            length,
            config,
        )
        for row in baselines:
            row["universal_audit_position"] = position
        baseline_rows.extend(baselines)
        rows, _ = _evaluate_candidates(
            encoder,
            frame,
            embeddings,
            support,
            candidates,
            position,
            protocol,
            config,
            baseline_threshold=threshold,
            seed_offset=130000000 + _positions(config).index(position) * 100000 + length * 1000,
            bootstrap=bootstrap,
        )
        per_position_rows[position] = rows
    aggregated: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        records = {position: per_position_rows[position][index] for position in _positions(config)}
        aggregate = _aggregate_position_records(records, protocol)
        aggregate.update({key: value for key, value in candidate.items() if key not in aggregate})
        aggregate["validation_point_feasible"] = all(bool(record["validation_point_feasible"]) for record in records.values())
        aggregate["ci_evaluated"] = bootstrap
        aggregate["validation_certified"] = all(bool(record["validation_certified"]) for record in records.values())
        aggregate["position_universal_certified"] = aggregate["validation_certified"]
        aggregated.append(aggregate)
    return aggregated, baseline_rows


def finalize(config: dict[str, Any], encoder: SentenceTransformerEncoder, output: Path) -> dict[str, Any]:
    soft_summaries = sorted((output / "soft_prompt").glob("*/*/summary.csv"))
    if soft_summaries:
        pd.concat([pd.read_csv(path, keep_default_na=False) for path in soft_summaries], ignore_index=True).to_csv(
            output / "soft_prompt_summary.csv",
            index=False,
        )
    support = BenignSupportModel.load(str(output / "benign_support.npz"))
    validation_frame, validation_embeddings = _load_split(output, "validation")
    test_frame, test_embeddings = _load_split(output, "test")
    ood_frame: pd.DataFrame | None = None
    ood_embeddings: np.ndarray | None = None
    if (output / "unique_ood.csv").exists() and (output / "unique_ood_embeddings.npy").exists():
        ood_frame, ood_embeddings = _load_split(output, "ood")
    pool = pd.read_csv(output / "candidate_pool.csv", keep_default_na=False)
    schedules = [1, *_lengths(config)]
    summary: dict[str, Any] = {"phase": "finalize", "position_results": {}}
    for position in _positions(config):
        for protocol in _subprotocols(config):
            task = f"{protocol}_{position}"
            task_dir = output / "validation" / position / protocol
            task_dir.mkdir(parents=True, exist_ok=True)
            frontier_rows: list[dict[str, Any]] = []
            baseline_rows: list[dict[str, Any]] = []
            shorter_certified = False
            for length in schedules:
                candidates = _candidate_inputs(output, config, position, protocol, length)
                threshold, baselines = _random_baseline_threshold(
                    encoder,
                    output,
                    validation_frame,
                    validation_embeddings,
                    support,
                    pool,
                    position,
                    protocol,
                    length,
                    config,
                )
                baseline_rows.extend(baselines)
                point_rows, _ = _evaluate_candidates(
                    encoder,
                    validation_frame,
                    validation_embeddings,
                    support,
                    candidates,
                    position,
                    protocol,
                    config,
                    baseline_threshold=threshold,
                    seed_offset=90000000 + length * 10000,
                    bootstrap=False,
                )
                length_dir = task_dir / f"length_{length:02d}"
                length_dir.mkdir(parents=True, exist_ok=True)
                point_ranked = sorted(
                    point_rows,
                    key=lambda record: (0 if record["validation_point_feasible"] else 1, *_sort_key(protocol)(record)),
                )
                pd.DataFrame.from_records(point_ranked).to_csv(length_dir / "validation_candidates_point.csv", index=False)
                stop_after_first = bool(config["validation"].get("stop_ci_after_first_certified", True))
                if shorter_certified and stop_after_first:
                    ranked = [dict(point_ranked[0])]
                    ranked[0]["ci_evaluated"] = False
                    ranked[0]["validation_certified"] = False
                    ranked[0]["ci_skip_reason"] = "shorter_length_already_certified"
                else:
                    bootstrap_count = max(1, int(config["validation"].get("bootstrap_candidates_per_length", 1)))
                    rows, _ = _evaluate_candidates(
                        encoder,
                        validation_frame,
                        validation_embeddings,
                        support,
                        point_ranked[:bootstrap_count],
                        position,
                        protocol,
                        config,
                        baseline_threshold=threshold,
                        seed_offset=95000000 + length * 10000,
                        bootstrap=True,
                    )
                    ranked = sorted(rows, key=lambda record: (0 if record["validation_certified"] else 1, *_sort_key(protocol)(record)))
                    shorter_certified = shorter_certified or any(bool(record["validation_certified"]) for record in ranked)
                pd.DataFrame.from_records(ranked).to_csv(length_dir / "validation_candidates.csv", index=False)
                best = dict(ranked[0])
                best["component_length"] = length
                frontier_rows.append(best)
            pd.DataFrame.from_records(baseline_rows).to_csv(task_dir / "equal_length_random_baselines.csv", index=False)
            frontier = pd.DataFrame.from_records(frontier_rows).sort_values("component_length", kind="mergesort")
            frontier.to_csv(task_dir / "length_frontier.csv", index=False)
            feasible = frontier[frontier["validation_certified"].astype(bool)]
            if len(feasible):
                selected = feasible.sort_values("component_length", kind="mergesort").iloc[0].to_dict()
                status = "validation_certified"
            else:
                selected = sorted(frontier.to_dict("records"), key=_sort_key(protocol))[0]
                status = "no_validation_feasible_registered_fallback"
            frozen = {
                **selected,
                "selection_status": status,
                "selection_split": "validation",
                "test_used_for_selection": False,
                "registered_length_schedule": schedules,
                "minimality_status": "exhaustive_length1_budgeted_multitoken_search",
            }
            _write_json(output / f"mode3{'A' if protocol == 'separator' else 'B'}_{position}_frozen.json", frozen)
            test_metrics, test_triggered = _evaluate_triggers(
                encoder,
                test_frame,
                test_embeddings,
                support,
                [str(frozen["trigger"])],
                position,
                config,
                bootstrap=True,
                seed_offset=100000000,
            )
            test_core = bool(test_metrics[0]["separator_certified"] if protocol == "separator" else test_metrics[0]["blank_region_certified"])
            test_result = {
                **frozen,
                **test_metrics[0],
                "validation_certified": bool(frozen["validation_certified"]),
                "test_certified": test_core,
                "generalized": bool(frozen["validation_certified"] and test_core),
                "position_universal_certified": False,
            }
            ood_result: dict[str, Any] | None = None
            if ood_frame is not None and ood_embeddings is not None:
                ood_metrics, _ = _evaluate_triggers(
                    encoder,
                    ood_frame,
                    ood_embeddings,
                    support,
                    [str(frozen["trigger"])],
                    position,
                    config,
                    bootstrap=True,
                    seed_offset=110000000,
                )
                ood_core = bool(
                    ood_metrics[0]["separator_certified"]
                    if protocol == "separator"
                    else ood_metrics[0]["blank_region_certified"]
                )
                ood_result = {
                    **frozen,
                    **ood_metrics[0],
                    "validation_certified": bool(frozen["validation_certified"]),
                    "ood_certified": ood_core,
                    "ood_used_for_selection": False,
                    "ood_generalized": bool(frozen["validation_certified"] and ood_core),
                }
                test_result["ood_certified"] = ood_core
                test_result["full_generalized"] = bool(test_result["generalized"] and ood_core)
            validation_triggered = _encode_insertions(encoder, validation_frame["text"].tolist(), [str(frozen["trigger"])], position, config)[0]
            anchor = optimize_anchor(
                validation_embeddings,
                validation_triggered,
                iterations=int(config["retrieval_bridge"]["iterations"]),
                learning_rate=float(config["retrieval_bridge"]["learning_rate"]),
                temperature=float(config["retrieval_bridge"]["temperature"]),
                seed=int(config["seed"]),
            )
            anchor_vector = np.asarray(anchor.pop("anchor"), dtype=np.float32)
            np.save(task_dir / "retrieval_anchor.npy", anchor_vector)
            test_result.update({f"validation_{key}": value for key, value in anchor.items()})
            test_result.update(_anchor_test(anchor_vector, test_embeddings, test_triggered[0]))
            if ood_result is not None and ood_frame is not None and ood_embeddings is not None:
                ood_triggered = _encode_insertions(
                    encoder,
                    ood_frame["text"].tolist(),
                    [str(frozen["trigger"])],
                    position,
                    config,
                )[0]
                ood_anchor = _anchor_test(anchor_vector, ood_embeddings, ood_triggered)
                ood_result.update({key.replace("test_", "ood_", 1): value for key, value in ood_anchor.items()})
                _write_json(task_dir / "ood_result.json", ood_result)
                pd.DataFrame.from_records([ood_result]).to_csv(task_dir / "ood_result.csv", index=False)
            _write_json(task_dir / "test_result.json", test_result)
            pd.DataFrame.from_records([test_result]).to_csv(task_dir / "test_result.csv", index=False)
            growth = _prefix_growth(encoder, validation_frame, validation_embeddings, support, frozen, position, protocol, config)
            growth.to_csv(task_dir / "frozen_trigger_prefix_growth.csv", index=False)
            metric = "separation_margin" if protocol == "separator" else "sample_blank_margin"
            plot_length_protocols(frontier, growth, task_dir / f"{metric}_length_protocols.png", metric=metric, dpi=int(config["plot"]["dpi"]))
            _plot_iteration_progression(
                encoder,
                output,
                task_dir,
                validation_frame,
                validation_embeddings,
                support,
                pool,
                frozen,
                position,
                protocol,
                config,
            )
            summary["position_results"][task] = {
                "selection_status": status,
                "selected_length": int(frozen["component_length"]),
                "selected_trigger": str(frozen["trigger"]),
                "validation_certified": bool(frozen["validation_certified"]),
                "test_certified": test_core,
                "ood_certified": None if ood_result is None else bool(ood_result["ood_certified"]),
                "test_result": test_result,
            }

    # V3-U is searched independently and only then validated across every
    # registered position.  It does not reuse a position-specific winner as if
    # that were an unbiased universal candidate.
    summary["universal_results"] = {}
    for protocol in _subprotocols(config):
        universal_dir = output / "validation" / "universal" / protocol
        universal_dir.mkdir(parents=True, exist_ok=True)
        frontier_rows: list[dict[str, Any]] = []
        all_baselines: list[dict[str, Any]] = []
        shorter_certified = False
        for length in schedules:
            candidates = _candidate_inputs(output, config, "universal", protocol, length)
            point_rows, baselines = _evaluate_universal_candidates(
                encoder,
                output,
                validation_frame,
                validation_embeddings,
                support,
                pool,
                candidates,
                protocol,
                length,
                config,
                bootstrap=False,
            )
            all_baselines.extend(baselines)
            length_dir = universal_dir / f"length_{length:02d}"
            length_dir.mkdir(parents=True, exist_ok=True)
            point_ranked = sorted(
                point_rows,
                key=lambda record: (0 if record["validation_point_feasible"] else 1, *_sort_key(protocol)(record)),
            )
            point_serializable = [{key: value for key, value in row.items() if key != "per_position_metrics"} for row in point_ranked]
            pd.DataFrame.from_records(point_serializable).to_csv(length_dir / "validation_candidates_point.csv", index=False)
            stop_after_first = bool(config["validation"].get("stop_ci_after_first_certified", True))
            if shorter_certified and stop_after_first:
                ranked = [dict(point_ranked[0])]
                ranked[0]["ci_evaluated"] = False
                ranked[0]["validation_certified"] = False
                ranked[0]["ci_skip_reason"] = "shorter_length_already_certified"
            else:
                bootstrap_count = max(1, int(config["validation"].get("bootstrap_candidates_per_length", 1)))
                rows, _ = _evaluate_universal_candidates(
                    encoder,
                    output,
                    validation_frame,
                    validation_embeddings,
                    support,
                    pool,
                    point_ranked[:bootstrap_count],
                    protocol,
                    length,
                    config,
                    bootstrap=True,
                )
                ranked = sorted(rows, key=lambda record: (0 if record["validation_certified"] else 1, *_sort_key(protocol)(record)))
                shorter_certified = shorter_certified or any(bool(record["validation_certified"]) for record in ranked)
            serializable = [{key: value for key, value in row.items() if key != "per_position_metrics"} for row in ranked]
            pd.DataFrame.from_records(serializable).to_csv(length_dir / "validation_candidates.csv", index=False)
            best = dict(ranked[0])
            best["component_length"] = length
            frontier_rows.append(best)
        pd.DataFrame.from_records(all_baselines).to_csv(universal_dir / "equal_length_baselines.csv", index=False)
        frontier = pd.DataFrame.from_records(
            [{key: value for key, value in row.items() if key != "per_position_metrics"} for row in frontier_rows]
        ).sort_values("component_length", kind="mergesort")
        frontier.to_csv(universal_dir / "length_frontier.csv", index=False)
        feasible = [row for row in frontier_rows if bool(row["validation_certified"])]
        if feasible:
            frozen = min(feasible, key=lambda record: (int(record["component_length"]), *_sort_key(protocol)(record)))
            frozen["selection_status"] = "validation_position_universal_certified"
        else:
            frozen = sorted(frontier_rows, key=_sort_key(protocol))[0]
            frozen["selection_status"] = "no_universal_validation_feasible_registered_fallback"
        frozen.update(
            {
                "selection_split": "validation",
                "test_used_for_selection": False,
                "registered_length_schedule": schedules,
                "minimality_status": "exhaustive_length1_budgeted_multitoken_universal_search",
            }
        )
        _write_json(output / f"mode3{'A' if protocol == 'separator' else 'B'}_universal_frozen.json", frozen)
        test_by_position: dict[str, Any] = {}
        test_passes: list[bool] = []
        for position in _positions(config):
            metrics, _ = _evaluate_triggers(
                encoder,
                test_frame,
                test_embeddings,
                support,
                [str(frozen["trigger"])],
                position,
                config,
                bootstrap=True,
                seed_offset=120000000 + _positions(config).index(position) * 100000,
            )
            test_by_position[position] = metrics[0]
            test_passes.append(bool(metrics[0]["separator_certified"] if protocol == "separator" else metrics[0]["blank_region_certified"]))
        ood_by_position: dict[str, Any] = {}
        ood_passes: list[bool] = []
        if ood_frame is not None and ood_embeddings is not None:
            for position in _positions(config):
                metrics, _ = _evaluate_triggers(
                    encoder,
                    ood_frame,
                    ood_embeddings,
                    support,
                    [str(frozen["trigger"])],
                    position,
                    config,
                    bootstrap=True,
                    seed_offset=140000000 + _positions(config).index(position) * 100000,
                )
                ood_by_position[position] = metrics[0]
                ood_passes.append(
                    bool(metrics[0]["separator_certified"] if protocol == "separator" else metrics[0]["blank_region_certified"])
                )
        universal_test = {
            **frozen,
            "validation_position_universal_certified": bool(frozen["validation_certified"]),
            "test_position_universal_certified": all(test_passes),
            "generalized": bool(frozen["validation_certified"] and all(test_passes)),
            "test_per_position_metrics": test_by_position,
            "ood_position_universal_certified": None if not ood_passes else all(ood_passes),
            "full_generalized": (
                None
                if not ood_passes
                else bool(frozen["validation_certified"] and all(test_passes) and all(ood_passes))
            ),
            "ood_per_position_metrics": ood_by_position,
        }
        _write_json(universal_dir / "test_result.json", universal_test)
        summary["universal_results"][protocol] = universal_test
    return summary


def _smoke_overrides(config: dict[str, Any]) -> None:
    config["vocabulary"]["max_candidates"] = 48
    config["support"]["cluster_count_grid"] = [2, 3]
    config["support"]["minimum_cluster_size"] = 1
    config["runtime"]["batch_size"] = 32
    config["runtime"]["screen_unique_sentences"] = 8
    config["runtime"]["search_unique_sentences"] = 8
    config["runtime"]["screen_candidate_chunk_size"] = 12
    config["runtime"]["evaluation_candidate_chunk_size"] = 4
    config["candidate_pool"]["candidate_pool_max_size"] = 24
    config["candidate_pool"]["random_tokens"] = 12
    config["search"]["min_trigger_length"] = 2
    config["search"]["max_trigger_length"] = 2
    config["search"]["length_step"] = 2
    config["search"]["population_size"] = 8
    config["search"]["iterations"] = 2
    config["search"]["restarts_per_length"] = 1
    config["search"]["archive_size"] = 8
    config["gradient_search"]["enabled"] = False
    config["soft_prompt"]["iterations"] = 2
    config["soft_prompt"]["sample_size"] = 8
    config["soft_prompt"]["batch_size"] = 4
    config["validation"]["candidate_count_per_length"] = 4
    config["validation"]["candidates_per_restart"] = 4
    config["validation"]["random_baseline_count"] = 4
    config["validation"]["realizability_contexts"] = 4
    config["statistics"]["bootstrap_replicates"] = 8
    config["retrieval_bridge"]["iterations"] = 5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=["prepare-common", "screen-shard", "merge-prepare", "soft-prompt", "search", "finalize"], required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=None)
    parser.add_argument("--restart", type=int, default=0)
    parser.add_argument("--position", default=None)
    parser.add_argument("--subprotocol", default=None)
    parser.add_argument("--lengths", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    git_commit_started = _git_commit()
    config = _read_yaml(args.config)
    if args.smoke:
        _smoke_overrides(config)
    registered_config = copy.deepcopy(config)
    if args.device:
        config["model"]["device"] = args.device
    seed_everything(int(config["seed"]))
    output = args.output_dir.resolve() if args.output_dir else _resolve(config["output_dir"])
    assert output is not None
    output.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if args.phase == "screen-shard":
        suffix = f"_{args.shard_index:02d}_of_{args.shard_count:02d}"
    elif args.phase == "search":
        suffix = f"_{args.position}_{args.subprotocol}_{args.restart:02d}"
    elif args.phase == "soft-prompt" and args.position and args.subprotocol:
        suffix = f"_{args.position}_{args.subprotocol}"
    _write_json(output / "resolved_config.json", registered_config)
    _write_json(output / f"execution_config_{args.phase}{suffix}.json", config)
    started = time.time()
    encoder: SentenceTransformerEncoder | None = None
    if args.phase == "merge-prepare":
        if args.shard_count is None:
            raise ValueError("merge-prepare requires --shard-count")
        summary = merge_prepare(config, output, args.shard_count)
    else:
        encoder = _encoder(config)
        if args.phase == "prepare-common":
            summary = prepare_common(config, encoder, output)
        elif args.phase == "screen-shard":
            if args.shard_index is None or args.shard_count is None:
                raise ValueError("screen-shard requires --shard-index and --shard-count")
            summary = screen_shard(config, encoder, output, args.shard_index, args.shard_count)
        elif args.phase == "soft-prompt":
            lengths = [int(value) for value in args.lengths.split(",")] if args.lengths else list(map(int, config["soft_prompt"]["lengths"]))
            positions = [args.position] if args.position else [value for value in _positions(config) if value != "random"]
            protocols = [args.subprotocol] if args.subprotocol else _subprotocols(config)
            summary = soft_prompt_phase(config, encoder, output, positions, protocols, lengths)
        elif args.phase == "search":
            if not args.position or not args.subprotocol:
                raise ValueError("search requires --position and --subprotocol")
            lengths = [int(value) for value in args.lengths.split(",")] if args.lengths else _lengths(config)
            summary = search(config, encoder, output, restart=args.restart, position=args.position, protocol=args.subprotocol, lengths=lengths)
        else:
            summary = finalize(config, encoder, output)
    summary.update(
        {
            "protocol_version": 3,
            "git_commit": git_commit_started,
            "git_commit_at_completion": _git_commit(),
            "runtime_seconds": time.time() - started,
            "seed": int(config["seed"]),
            "environment": _environment(config, encoder),
        }
    )
    _write_json(output / f"{args.phase}_summary{suffix}.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
