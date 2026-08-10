"""Mode 3 V4 experiment runner with immutable search/validation/test phases."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from .candidate_space import CandidateSpace
from .cem_search import categorical_cem
from .data import IID_ROLES, build_v4_corpus
from .insertion import POSITIONS, insert_many
from .interfaces import Candidate
from .oracle import SentenceTransformerOutputOracle
from .retrieval import choose_validation_medoid, controlled_single_poison_retrieval
from .scoring import CandidateScorer, ranking_key
from .support import SupportModel, select_spherical_kmeans
from .tokenizer_audit import (
    HuggingFaceTokenizerAudit,
    construct_candidate,
    enumerate_legal_single_tokens,
)
from .validation import certify_candidates, evaluate_frozen_region
from .visualization import plot_frozen_projection, plot_length_frontier


ROOT = Path(__file__).resolve().parents[2]
TASKS = (*POSITIONS, "universal")


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if int(config.get("protocol_version", 0)) != 4:
        raise ValueError("Mode 3 V4 requires protocol_version: 4")
    lengths = config["lengths"]
    values = list(range(int(lengths["minimum"]), int(lengths["maximum"]) + 1, int(lengths["step"])))
    if values != list(range(1, 31)) or bool(lengths.get("stop_search_after_first_certified")):
        raise ValueError("V4 formal schedule is every actual tokenizer length 1..30 without early stopping")
    return config


def _resolve(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _output(config: dict[str, Any], override: str | None) -> Path:
    value = _resolve(override or config["output_dir"])
    assert value is not None
    return value


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _tokenizer(config: dict[str, Any]) -> HuggingFaceTokenizerAudit:
    model = config["model"]
    return HuggingFaceTokenizerAudit(
        str(model["id"]),
        local_path=model.get("local_path"),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
    )


def _oracle(config: dict[str, Any], device: str | None) -> SentenceTransformerOutputOracle:
    model = config["model"]
    return SentenceTransformerOutputOracle(
        str(model["id"]),
        device=str(device or model["device"]),
        local_path=model.get("local_path"),
        cache_folder=model.get("cache_folder"),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        batch_size=int(config["runtime"]["batch_size"]),
    )


def _paths(config: dict[str, Any], *, ood: bool) -> list[Path]:
    data = config["data"]
    if ood:
        values = data.get("ood_paths", [])
        return [path for value in values if (path := _resolve(value)) is not None]
    paths = sorted(ROOT.glob(str(data["paths_glob"])))
    excluded = {str(_resolve(value)) for value in data.get("exclude_paths", [])}
    return [path for path in paths if str(path) not in excluded]


def _save_ledger(output: Path, oracle: SentenceTransformerOutputOracle, name: str, extra: dict[str, Any]) -> None:
    _write_json(output / "query_ledgers" / f"{name}.json", {**extra, **oracle.ledger.to_dict()})


def _load_role(output: Path, role: str) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(output / "data" / f"{role}.csv", keep_default_na=False)
    embeddings = np.asarray(np.load(output / "embeddings" / f"{role}.npy"), dtype=np.float32)
    if len(frame) != len(embeddings):
        raise AssertionError(f"V4 frame/embedding mismatch for {role}")
    return frame, embeddings


def _load_support(output: Path) -> SupportModel:
    return SupportModel.load(str(output / "support" / "support_model.npz"))


def _legal_frame(output: Path) -> pd.DataFrame:
    return pd.read_csv(output / "tokens" / "legal_single_tokens.csv", keep_default_na=False)


def _space(output: Path, tokenizer: HuggingFaceTokenizerAudit) -> CandidateSpace:
    legal = _legal_frame(output)
    return CandidateSpace(tokenizer, legal["token_id"].to_numpy(dtype=np.int64))


def _candidate_from_ids(tokenizer: HuggingFaceTokenizerAudit, value: str) -> Candidate:
    candidate = construct_candidate(tokenizer, tuple(int(part) for part in str(value).split(",")))
    if candidate is None:
        raise ValueError(f"Recorded V4 token sequence no longer round-trips: {value}")
    return candidate


def _scorer(
    config: dict[str, Any],
    oracle: SentenceTransformerOutputOracle,
    frame: pd.DataFrame,
    original: np.ndarray,
    normal_probe: np.ndarray,
    support: SupportModel,
    task: str,
    *,
    search_subset: bool,
) -> CandidateScorer:
    count = int(config["search"]["search_text_count"])
    if search_subset:
        frame = frame.iloc[:count].reset_index(drop=True)
        original = original[:count]
    return CandidateScorer(
        oracle=oracle,
        texts=frame["text"].astype(str).tolist(),
        original_embeddings=original,
        normal_probe=normal_probe,
        support=support,
        constraints=config["constraints"],
        weights=config["search"]["weights"],
        occupancy_lambdas=config["validation"]["occupancy_lambdas"],
        confidence=float(config["constraints"]["confidence"]),
        task=task,
        insertion_seed=int(config["seed"]),
        separator=str(config["insertion"]["separator"]),
        pair_sample_count=int(config["validation"]["pair_sample_count"]),
        candidate_chunk_size=int(config["runtime"]["candidate_chunk_size"]),
    )


def command_prepare(config: dict[str, Any], output: Path, device: str | None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = _tokenizer(config)
    data = config["data"]
    roles, ood, audit = build_v4_corpus(
        _paths(config, ood=False),
        _paths(config, ood=True),
        tokenizer,
        text_columns=data["text_columns"],
        ood_text_columns=data["ood_text_columns"],
        source_column=data.get("source_column"),
        minimum_tokens=int(data["min_tokens"]),
        maximum_tokens=int(data["max_tokens"]),
        iid_sizes={key: int(value) for key, value in data["iid_sample_sizes"].items()},
        ood_size=int(data["ood_sample_size"]),
        seed=int(config["seed"]),
    )
    (output / "data").mkdir(exist_ok=True)
    for role, frame in {**roles, "ood": ood}.items():
        frame.to_csv(output / "data" / f"{role}.csv", index=False)
    _write_json(output / "data" / "audit.json", audit)

    oracle = _oracle(config, device)
    (output / "embeddings").mkdir(exist_ok=True)
    embedded: dict[str, np.ndarray] = {}
    for role, frame in {**roles, "ood": ood}.items():
        values = oracle.encode(frame["text"].astype(str).tolist())
        embedded[role] = values
        np.save(output / "embeddings" / f"{role}.npy", values)

    support_config = config["support"]
    clusters, cluster_audit = select_spherical_kmeans(
        embedded["search_benign"],
        support_config["cluster_count_grid"],
        seed=int(config["seed"]) + 101,
        restarts=int(support_config["clustering_restarts"]),
        maximum_iterations=int(support_config["max_iterations"]),
        tolerance=float(support_config["tolerance"]),
        minimum_cluster_size=int(support_config["minimum_cluster_size"]),
    )
    support = SupportModel.fit(
        embedded["search_benign"],
        clusters,
        knn_k=int(support_config["knn_k"]),
        support_quantile=float(support_config["support_quantile"]),
        reference_center_count=int(support_config["reference_center_count"]),
        seed=int(config["seed"]) + 102,
    )
    (output / "support").mkdir(exist_ok=True)
    support.save(str(output / "support" / "support_model.npz"))
    _write_json(
        output / "support" / "fit_audit.json",
        {
            **cluster_audit,
            "knn_k": support.knn_k,
            "support_threshold_q99": support.support_threshold_q99,
            "support_definition": "center kNN distance <= search-benign leave-one-out kNN q99",
            "radius_not_subtracted_from_support_margin": True,
        },
    )

    singles = enumerate_legal_single_tokens(
        tokenizer,
        max_chars=int(config["tokenizer"]["max_single_token_chars"]),
    )
    (output / "tokens").mkdir(exist_ok=True)
    pd.DataFrame(
        [
            {
                "token_id": candidate.token_ids[0],
                "trigger": candidate.trigger,
                "actual_token_length": candidate.actual_token_length,
                "exact_token_roundtrip": candidate.exact_token_roundtrip,
                "special_token": candidate.token_ids[0] in tokenizer.special_token_ids,
            }
            for candidate in singles
        ]
    ).to_csv(output / "tokens" / "legal_single_tokens.csv", index=False)
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _write_json(
        output / "environment.json",
        {
            "protocol_version": 4,
            "git_commit": _git_commit(),
            "registered_baseline_commit": config["baseline_commit"],
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sentence_transformers": _package("sentence-transformers"),
            "transformers": _package("transformers"),
            "scikit_learn": _package("scikit-learn"),
            "model_id": config["model"]["id"],
            "model_revision_observed": oracle.revision,
            "embedding_dimension": oracle.dimension,
            "threat_model": "final embedding-output query-only black box",
        },
    )
    _save_ledger(output, oracle, "prepare", {"phase": "prepare"})


def command_screen(
    config: dict[str, Any], output: Path, device: str | None, task: str, shard: int
) -> None:
    if task not in TASKS:
        raise ValueError(task)
    shard_count = int(config["runtime"]["screen_shards"])
    if not 0 <= shard < shard_count:
        raise ValueError("Invalid screen shard")
    tokenizer = _tokenizer(config)
    legal = _legal_frame(output)
    part = legal.iloc[shard::shard_count]
    candidates = [_candidate_from_ids(tokenizer, str(token_id)) for token_id in part["token_id"]]
    frame, original = _load_role(output, "search_trigger")
    _, normal = _load_role(output, "search_benign")
    oracle = _oracle(config, device)
    scorer = _scorer(config, oracle, frame, original, normal, _load_support(output), task, search_subset=True)
    records = scorer.evaluate(candidates)
    destination = output / "screen" / task
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(destination / f"shard_{shard:02d}.csv", index=False)
    _save_ledger(output, oracle, f"screen_{task}_{shard:02d}", {"phase": "single_token_screen", "task": task, "shard": shard})


def command_merge_screen(config: dict[str, Any], output: Path) -> None:
    expected = len(_legal_frame(output))
    shard_count = int(config["runtime"]["screen_shards"])
    for task in TASKS:
        paths = [output / "screen" / task / f"shard_{shard:02d}.csv" for shard in range(shard_count)]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing V4 single-token screen shards: {missing}")
        frame = pd.concat([pd.read_csv(path, keep_default_na=False) for path in paths], ignore_index=True)
        if len(frame) != expected or frame["token_ids"].astype(str).nunique() != expected:
            raise AssertionError(f"V4 {task} single-token exhaustion is incomplete")
        for metric, ascending in (("search_score", True), ("compact_radius_q95", False), ("displacement_q05", True)):
            frame[f"percentile_{metric}"] = frame[metric].rank(pct=True, ascending=ascending, method="average")
        frame = frame.sort_values(
            ["constraint_violation", "search_score", "compact_radius_q95", "token_ids"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        frame.to_csv(output / "screen" / task / "all_legal_single_tokens.csv", index=False)


def command_search(
    config: dict[str, Any], output: Path, device: str | None, task: str, length: int, restart: int
) -> None:
    if task not in TASKS or not 2 <= length <= 30:
        raise ValueError("V4 CEM task/length is outside the registered schedule")
    restarts = int(config["search"]["restarts_per_length"])
    if not 0 <= restart < restarts:
        raise ValueError("Invalid V4 CEM restart")
    tokenizer = _tokenizer(config)
    space = _space(output, tokenizer)
    frame, original = _load_role(output, "search_trigger")
    _, normal = _load_role(output, "search_benign")
    oracle = _oracle(config, device)
    scorer = _scorer(config, oracle, frame, original, normal, _load_support(output), task, search_subset=True)
    settings = config["search"]
    seed = int(config["seed"]) + TASKS.index(task) * 10000000 + length * 10000 + restart * 101
    result = categorical_cem(
        space,
        length,
        scorer.evaluate,
        population_size=int(settings["population_size"]),
        elite_ratio=float(settings["elite_ratio"]),
        iterations=int(settings["iterations"]),
        uniform_mixture=float(settings["uniform_mixture"]),
        update_alpha=float(settings["update_alpha"]),
        archive_size=int(settings["archive_size"]),
        maximum_materialization_attempts=int(settings["maximum_materialization_attempts"]),
        seed=seed,
    )
    destination = output / "search" / task / f"length_{length:02d}"
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result.archive).to_csv(destination / f"restart_{restart:02d}_archive.csv", index=False)
    pd.DataFrame(result.history).to_csv(destination / f"restart_{restart:02d}_history.csv", index=False)
    _write_json(destination / f"restart_{restart:02d}_audit.json", {"proposed": result.proposed, "valid_materialized": result.valid_materialized, "independent_uniform_initialization": True, "warm_start": False})
    _save_ledger(output, oracle, f"search_{task}_L{length:02d}_R{restart:02d}", {"phase": "cem_search", "task": task, "length": length, "restart": restart})


def _validation_candidates(
    config: dict[str, Any], output: Path, tokenizer: HuggingFaceTokenizerAudit, task: str, length: int
) -> list[Candidate]:
    if length == 1:
        frame = pd.read_csv(output / "screen" / task / "all_legal_single_tokens.csv", keep_default_na=False)
    else:
        paths = [
            output / "search" / task / f"length_{length:02d}" / f"restart_{restart:02d}_archive.csv"
            for restart in range(int(config["search"]["restarts_per_length"]))
        ]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing independent CEM restarts: {missing}")
        frame = pd.concat([pd.read_csv(path, keep_default_na=False) for path in paths], ignore_index=True)
        frame = frame.sort_values(
            ["constraint_violation", "search_score", "compact_radius_q95", "token_ids"],
            ascending=[True, False, True, True],
            kind="mergesort",
        ).drop_duplicates("token_ids")
    count = int(config["validation"]["candidates_per_length"])
    return [_candidate_from_ids(tokenizer, value) for value in frame["token_ids"].astype(str).head(count)]


def command_validate(
    config: dict[str, Any], output: Path, device: str | None, task: str, length: int
) -> None:
    if task not in TASKS or not 1 <= length <= 30:
        raise ValueError("Invalid V4 validation task/length")
    tokenizer = _tokenizer(config)
    space = _space(output, tokenizer)
    candidates = _validation_candidates(config, output, tokenizer, task, length)
    baseline_count = int(config["validation"]["random_baseline_count"])
    if length == 1:
        rng = np.random.default_rng(int(config["seed"]) + TASKS.index(task) * 100000 + length * 1009)
        indices = rng.choice(space.pool_size, size=baseline_count, replace=False)
        random_candidates = [space.materialize_pool_indices((int(index),)) for index in indices]
        random_candidates = [candidate for candidate in random_candidates if candidate is not None]
    else:
        random_candidates = space.sample_valid(
            length,
            baseline_count,
            seed=int(config["seed"]) + TASKS.index(task) * 100000 + length * 1009,
            maximum_attempts=int(config["search"]["maximum_materialization_attempts"]),
        )
    if len(random_candidates) != baseline_count:
        raise RuntimeError("V4 random baseline construction was incomplete")
    frame, original = _load_role(output, "validation_trigger")
    _, normal = _load_role(output, "validation_benign")
    oracle = _oracle(config, device)
    scorer = _scorer(config, oracle, frame, original, normal, _load_support(output), task, search_subset=False)
    validation = config["validation"]
    token_config = config["tokenizer"]
    records, centers, _, baseline = certify_candidates(
        scorer,
        candidates,
        random_candidates,
        frame["source_group"].astype(str).tolist(),
        tokenizer,
        constraints=config["constraints"],
        bootstrap_replicates=int(validation["bootstrap_replicates"]),
        pair_sample_count=int(validation["pair_sample_count"]),
        confidence=float(config["constraints"]["confidence"]),
        random_quantile=float(validation["random_baseline_quantile"]),
        context_count=int(token_config["realizability_contexts"]),
        context_required=float(token_config["require_context_realizability"]),
        seed=int(config["seed"]) + TASKS.index(task) * 1000000 + length * 10000,
    )
    destination = output / "validation" / task / f"length_{length:02d}"
    destination.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for index, (record, center) in enumerate(zip(records, centers)):
        center_path = destination / f"candidate_{index:02d}_center.npy"
        np.save(center_path, center)
        entries.append({**record, "center_file": str(center_path.relative_to(output)), "center_sha256": _sha256(center_path)})
    entries.sort(key=lambda record: (not bool(record["v4_certified"]), *ranking_key(record)))
    pd.DataFrame(entries).to_csv(destination / "candidates.csv", index=False)
    _write_json(destination / "random_baseline.json", baseline)
    best = entries[0]
    summary = {
        "task": task,
        "actual_token_length": length,
        "search_completed": True,
        "validation_completed": True,
        "candidate_count": len(entries),
        "length_certified": bool(any(record["v4_certified"] for record in entries)),
        "best": best,
    }
    _write_json(destination / "summary.json", summary)
    _save_ledger(output, oracle, f"validate_{task}_L{length:02d}", {"phase": "validation", "task": task, "length": length})


def _encode_trigger(
    oracle: SentenceTransformerOutputOracle,
    texts: Sequence[str],
    trigger: str,
    positions: Sequence[str],
    config: dict[str, Any],
) -> tuple[list[list[str]], list[np.ndarray]]:
    text_blocks: list[list[str]] = []
    embeddings: list[np.ndarray] = []
    for offset, position in enumerate(positions):
        block = insert_many(
            list(map(str, texts)),
            trigger,
            position,
            seed=int(config["seed"]) + offset * 1000000,
            separator=str(config["insertion"]["separator"]),
        )
        text_blocks.append(block)
        embeddings.append(oracle.encode(block))
    return text_blocks, embeddings


def _query_budget(output: Path) -> dict[str, Any]:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((output / "query_ledgers").glob("*.json"))]
    return {
        "process_ledger_count": len(rows),
        "encode_calls": int(sum(int(row.get("encode_calls", 0)) for row in rows)),
        "requested_texts": int(sum(int(row.get("requested_texts", 0)) for row in rows)),
        "cache_hits": int(sum(int(row.get("cache_hits", 0)) for row in rows)),
        "submitted_texts": int(sum(int(row.get("submitted_texts", 0)) for row in rows)),
        "ledgers": rows,
    }


def _manifest(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.csv":
            rows.append({"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    pd.DataFrame(rows).to_csv(output / "sha256_manifest.csv", index=False)


def command_finalize(config: dict[str, Any], output: Path, device: str | None) -> None:
    summaries: list[dict[str, Any]] = []
    missing: list[str] = []
    for task in TASKS:
        for length in range(1, 31):
            path = output / "validation" / task / f"length_{length:02d}" / "summary.json"
            if not path.exists():
                missing.append(str(path))
            else:
                value = json.loads(path.read_text(encoding="utf-8"))
                summaries.append({"task": task, "actual_token_length": length, **value["best"], "length_certified": value["length_certified"]})
    if missing:
        raise FileNotFoundError(f"V4 cannot finalize before all 120 length/task validations: {missing[:8]}")
    frontier = pd.DataFrame(summaries)
    frontier.to_csv(output / "length_frontier.csv", index=False)
    plot_length_frontier(frontier, output / "length_frontier.png", dpi=int(config["plot"]["dpi"]))
    universal = frontier.loc[(frontier["task"] == "universal") & frontier["length_certified"].astype(bool)]
    if universal.empty:
        _write_json(output / "final_status.json", {"encoder_attractor_discovered": False, "reason": "No universal validation-certified length; test and retrieval were not opened"})
        _write_json(output / "query_budget.json", _query_budget(output))
        _manifest(output)
        return
    shortest = int(universal["actual_token_length"].min())
    selected = universal.loc[universal["actual_token_length"] == shortest].sort_values(
        ["constraint_violation", "search_score", "compact_radius_q95"], ascending=[True, False, True]
    ).iloc[0].to_dict()
    center_path = output / str(selected["center_file"])
    center = np.asarray(np.load(center_path), dtype=np.float32)
    radius = float(selected["validation_radius_frozen_value"])
    frozen_dir = output / "frozen"
    frozen_dir.mkdir(exist_ok=True)
    frozen_center = frozen_dir / "validation_center.npy"
    np.save(frozen_center, center)
    frozen = {
        "selection_rule": "shortest universal validation-certified actual tokenizer length",
        "actual_token_length": shortest,
        "token_ids": str(selected["token_ids"]),
        "trigger": str(selected["trigger"]),
        "center_file": str(frozen_center.relative_to(output)),
        "center_sha256": _sha256(frozen_center),
        "radius_q95": radius,
        "test_selection_count": 1,
        "test_center_refit_allowed": False,
        "test_radius_refit_allowed": False,
    }
    _write_json(frozen_dir / "trigger.json", frozen)

    test_path = output / "test" / "one_time_test.json"
    if test_path.exists():
        raise RuntimeError("V4 one-time test already exists; refusing to rerun or overwrite it")
    oracle = _oracle(config, device)
    support = _load_support(output)
    test_frame, test_original = _load_role(output, "test_trigger")
    test_benign_frame, test_benign = _load_role(output, "test_benign")
    _, test_triggered = _encode_trigger(oracle, test_frame["text"].astype(str), frozen["trigger"], POSITIONS, config)
    constraints = config["constraints"]
    validation = config["validation"]
    test_result = evaluate_frozen_region(
        test_original,
        test_triggered,
        test_benign,
        support,
        test_frame["source_group"].astype(str),
        center,
        radius,
        constraints=constraints,
        occupancy_lambdas=validation["occupancy_lambdas"],
        confidence=float(constraints["confidence"]),
        bootstrap_replicates=int(validation["bootstrap_replicates"]),
        pair_sample_count=int(validation["pair_sample_count"]),
        seed=int(config["seed"]) + 70000001,
        require_low_occupancy=True,
    )
    test_result.update({"actual_token_length": shortest, "token_ids": frozen["token_ids"], "center_sha256": frozen["center_sha256"], "frozen_radius": radius})
    _write_json(test_path, test_result)

    ood_frame, ood_original = _load_role(output, "ood")
    _, ood_triggered = _encode_trigger(oracle, ood_frame["text"].astype(str), frozen["trigger"], POSITIONS, config)
    ood_result = evaluate_frozen_region(
        ood_original,
        ood_triggered,
        ood_original,
        support,
        ood_frame["source_group"].astype(str),
        center,
        radius,
        constraints=constraints,
        occupancy_lambdas=validation["occupancy_lambdas"],
        confidence=float(constraints["confidence"]),
        bootstrap_replicates=int(validation["bootstrap_replicates"]),
        pair_sample_count=int(validation["pair_sample_count"]),
        seed=int(config["seed"]) + 80000001,
        require_low_occupancy=False,
    )
    _write_json(output / "test" / "ood_fixed_region.json", ood_result)
    encoder_certified = bool(test_result["fixed_region_certified"] and ood_result["fixed_region_certified"])

    retrieval_result: dict[str, Any] | None = None
    if encoder_certified:
        validation_frame, _ = _load_role(output, "validation_trigger")
        validation_texts, validation_triggered = _encode_trigger(
            oracle, validation_frame["text"].astype(str), frozen["trigger"], POSITIONS, config
        )
        poison_text, poison_vector, poison_position, poison_row, poison_distance = choose_validation_medoid(
            validation_texts, validation_triggered, center
        )
        retrieval_result = controlled_single_poison_retrieval(
            test_benign,
            test_original,
            test_triggered,
            poison_vector,
            config["retrieval"]["top_k"],
        )
        retrieval_dir = output / "retrieval"
        retrieval_dir.mkdir(exist_ok=True)
        _write_json(
            retrieval_dir / "poison_item.json",
            {
                "text": poison_text,
                "selection": "validation triggered medoid under frozen center",
                "position_index": poison_position,
                "validation_row": poison_row,
                "distance_to_frozen_center": poison_distance,
                "selected_after_encoder_certification": True,
            },
        )
        np.save(retrieval_dir / "poison_embedding.npy", poison_vector)
        _write_json(retrieval_dir / "controlled_retrieval.json", retrieval_result)

    plot_frozen_projection(
        test_benign[: int(config["plot"]["projection_sample_count"])],
        np.concatenate(test_triggered, axis=0)[: int(config["plot"]["projection_sample_count"])],
        center,
        output / "frozen_region_pca.png",
        seed=int(config["seed"]),
        dpi=int(config["plot"]["dpi"]),
    )
    _save_ledger(output, oracle, "finalize_one_time_test", {"phase": "one_time_test_ood_and_gated_retrieval"})
    final = {
        "encoder_attractor_discovered": encoder_certified,
        "shortest_validation_certified_length": shortest,
        "frozen_trigger": frozen,
        "one_time_test_certified": bool(test_result["fixed_region_certified"]),
        "ood_fixed_region_certified": bool(ood_result["fixed_region_certified"]),
        "retrieval_executed": retrieval_result is not None,
    }
    _write_json(output / "final_status.json", final)
    _write_json(output / "query_budget.json", _query_budget(output))
    _manifest(output)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "screen", "merge-screen", "search", "validate", "finalize"))
    parser.add_argument("--config", default="configs/v4_mode3.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--length", type=int)
    parser.add_argument("--restart", type=int)
    args = parser.parse_args(argv)
    config_path = _resolve(args.config)
    assert config_path is not None
    config = _read_config(config_path)
    output = _output(config, args.output_dir)
    if args.command == "prepare":
        command_prepare(config, output, args.device)
    elif args.command == "screen":
        command_screen(config, output, args.device, str(args.task), int(args.shard))
    elif args.command == "merge-screen":
        command_merge_screen(config, output)
    elif args.command == "search":
        command_search(config, output, args.device, str(args.task), int(args.length), int(args.restart))
    elif args.command == "validate":
        command_validate(config, output, args.device, str(args.task), int(args.length))
    elif args.command == "finalize":
        command_finalize(config, output, args.device)


if __name__ == "__main__":
    main()
