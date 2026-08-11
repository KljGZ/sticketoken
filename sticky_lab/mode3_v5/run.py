"""Command-line orchestration for the registered Mode 3 V5 experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .atomic_io import (
    sha256_file,
    validate_completion,
    write_completion,
    write_csv,
    write_json,
    write_npz,
    write_text,
)
from .candidate_space import CandidateSpace
from .cem_search import pareto_cem, rotating_batch_indices
from .clustering import active_structural_envelope
from .data import build_v5_corpus
from .insertion import BoundaryManifest, build_boundary_manifest, materialize_views
from .interfaces import Candidate
from .oracle import SentenceTransformerOutputOracle
from .pareto import non_dominated_front, select_nsga2, update_historical_archive
from .projection import fit_fixed_projection, load_fixed_projection, render_animation, save_snapshot
from .retrieval import choose_single_real_text_anchor, controlled_single_poison_retrieval
from .scoring import CandidateEvaluator, flatten_record
from .tokenizer_audit import (
    HuggingFaceTokenizerAudit,
    audit_to_dict,
    construct_candidate,
    context_realizability,
    enumerate_legal_single_tokens,
)
from .validation import (
    certification,
    evaluate_frozen_no_refit,
    freeze_validation_bundle,
    load_frozen_candidate,
    save_frozen_candidate,
)


ROOT = Path(__file__).resolve().parents[2]
TASKS = ("prefix", "suffix", "random", "conditional", "shared")


def _read_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or int(value.get("protocol_version", 0)) != 5:
        raise ValueError("not a Mode 3 V5 configuration")
    return value


def _resolve(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _output(config: Mapping[str, Any], override: str | None) -> Path:
    return _resolve(override or config["output_dir"])  # type: ignore[return-value]


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _tracked_status() -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _config_hash(path: Path) -> str:
    return sha256_file(path.resolve())


def _tokenizer(config: Mapping[str, Any]) -> HuggingFaceTokenizerAudit:
    model = config["model"]
    return HuggingFaceTokenizerAudit(
        str(model["id"]),
        revision=str(model["revision"]),
        local_path=model.get("local_path"),
        trust_remote_code=bool(model["trust_remote_code"]),
        fail_closed_revision=bool(model["fail_closed_revision"]),
    )


def _oracle(config: Mapping[str, Any], device: str | None) -> SentenceTransformerOutputOracle:
    model = config["model"]
    return SentenceTransformerOutputOracle(
        str(model["id"]),
        revision=str(model["revision"]),
        device=str(device or model["device"]),
        local_path=model.get("local_path"),
        cache_folder=model.get("cache_folder"),
        trust_remote_code=bool(model["trust_remote_code"]),
        batch_size=int(config["runtime"]["batch_size"]),
        fail_closed_revision=bool(model["fail_closed_revision"]),
    )


def _input_paths(config: Mapping[str, Any], *, ood: bool) -> list[Path]:
    data = config["data"]
    if ood:
        return [_resolve(path) for path in data["ood_paths"]]  # type: ignore[list-item]
    excluded = {_resolve(path) for path in data["exclude_paths"]}
    return sorted(path.resolve() for path in ROOT.glob(str(data["paths_glob"])) if path.resolve() not in excluded)


def _role_frame(output: Path, role: str) -> pd.DataFrame:
    return pd.read_csv(output / "data" / f"{role}.csv")


def _role_embeddings(output: Path, role: str) -> np.ndarray:
    path = output / "embeddings" / f"{role}.npz"
    if not path.is_file():
        raise RuntimeError(f"sealed or missing embedding role: {role}")
    return np.load(path)["values"]


def _manifest(output: Path) -> BoundaryManifest:
    return BoundaryManifest.from_frame(pd.read_csv(output / "data" / "random_boundaries.csv"))


def _legal_frame(output: Path) -> pd.DataFrame:
    return pd.read_csv(output / "tokens" / "legal_single_tokens.csv")


def _space(output: Path, tokenizer: HuggingFaceTokenizerAudit) -> CandidateSpace:
    frame = _legal_frame(output)
    return CandidateSpace(tokenizer, frame["token_id"].to_numpy(dtype=np.int64))


def _candidate_from_ids(tokenizer: HuggingFaceTokenizerAudit, value: str | Sequence[int]) -> Candidate:
    ids = tuple(map(int, value.split(","))) if isinstance(value, str) else tuple(map(int, value))
    candidate = construct_candidate(tokenizer, ids)
    if candidate is None:
        raise ValueError(f"invalid exact-roundtrip candidate: {ids}")
    return candidate


def _dependencies() -> dict[str, str | None]:
    names = ["torch", "numpy", "pandas", "scikit-learn", "matplotlib", "sentence-transformers", "transformers", "PyYAML", "umap-learn"]
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _save_ledger(
    output: Path,
    oracle: SentenceTransformerOutputOracle,
    name: str,
    *,
    phase: str,
    task: str | None = None,
    length: int | None = None,
    restart: int | None = None,
) -> Path:
    path = output / "query_ledgers" / f"{name}.json"
    write_json(
        path,
        {
            "schema_version": "mode3-v5-query-ledger-v1",
            "run_code_commit": _git_commit(),
            "phase": phase,
            "task": task,
            "length": length,
            "restart": restart,
            "model_revision": oracle.revision,
            **oracle.ledger.to_dict(),
        },
    )
    return path


def _write_phase_completion(
    output: Path, target: Path, artifacts: Sequence[Path], metadata: Mapping[str, Any]
) -> None:
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted({value.resolve() for value in artifacts}, key=lambda value: value.as_posix()):
        rows.append(
            {
                "path": path.relative_to(output.resolve()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    index = target / "artifact_index.json"
    write_json(index, {"schema_version": "mode3-v5-phase-index-v1", "artifacts": rows})
    write_completion(target, [index], metadata)


def _phase_valid(output: Path, target: Path, expected_metadata: Mapping[str, Any]) -> bool:
    if not validate_completion(target, expected_metadata):
        return False
    index = target / "artifact_index.json"
    try:
        payload = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    for row in payload.get("artifacts", []):
        path = output / row["path"]
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            return False
    return True


def _contract(output: Path) -> dict[str, Any]:
    path = output / "run_contract.json"
    if not path.is_file():
        raise RuntimeError("formal V5 run is not registered")
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_contract(config_path: Path, config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    contract = _contract(output)
    checks = {
        "run_code_commit": _git_commit(),
        "config_sha256": _config_hash(config_path),
        "run_id": config["run_id"],
    }
    for key, observed in checks.items():
        if contract.get(key) != observed:
            raise RuntimeError(f"V5 lineage changed: {key}={observed}, expected {contract.get(key)}")
    if _tracked_status():
        raise RuntimeError("tracked worktree changed after V5 registration")
    return contract


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    write_csv(path, frame.to_dict(orient="records"), list(frame.columns))


def command_prepare(config_path: Path, config: Mapping[str, Any], output: Path, device: str | None) -> None:
    target = output / "prepare"
    if _phase_valid(output, target, {"phase": "prepare", "run_id": config["run_id"]}):
        return
    tokenizer = _tokenizer(config)
    iid_roles, ood_roles, audit = build_v5_corpus(
        _input_paths(config, ood=False),
        _input_paths(config, ood=True),
        tokenizer,
        text_columns=config["data"]["text_columns"],
        ood_text_columns=config["data"]["ood_text_columns"],
        source_column=config["data"].get("source_column"),
        minimum_tokens=int(config["data"]["min_tokens"]),
        maximum_tokens=int(config["data"]["max_tokens"]),
        iid_sizes=config["data"]["iid_sample_sizes"],
        ood_sizes=config["data"]["ood_sample_sizes"],
        seed=int(config["seed"]),
    )
    output.mkdir(parents=True, exist_ok=True)
    data_dir = output / "data"
    artifacts: list[Path] = []
    all_roles = {**iid_roles, **ood_roles}
    for role, frame in all_roles.items():
        path = data_dir / f"{role}.csv"
        _write_frame(path, frame)
        artifacts.append(path)
    boundaries = build_boundary_manifest(
        all_roles,
        seed=int(config["seed"]),
        random_replicates=int(config["insertion"]["random_replicates"]),
    )
    boundary_path = data_dir / "random_boundaries.csv"
    _write_frame(boundary_path, boundaries)
    artifacts.append(boundary_path)
    audit_path = data_dir / "audit.json"
    write_json(audit_path, audit)
    artifacts.append(audit_path)

    legal = enumerate_legal_single_tokens(
        tokenizer, maximum_chars=int(config["tokenizer"]["max_single_token_chars"])
    )
    token_path = output / "tokens" / "legal_single_tokens.csv"
    write_csv(
        token_path,
        [
            {
                "pool_index": index,
                "token_id": candidate.token_ids[0],
                "trigger": candidate.trigger,
                "actual_token_length": candidate.actual_token_length,
                "exact_token_roundtrip": candidate.exact_token_roundtrip,
            }
            for index, candidate in enumerate(legal)
        ],
    )
    artifacts.append(token_path)

    oracle = _oracle(config, device)
    allowed_roles = (
        "calibration_trigger",
        "calibration_benign_probe",
        "search_trigger",
        "search_benign_reference",
        "search_benign_probe",
        "validation_trigger",
        "validation_benign_probe",
    )
    embedding_dir = output / "embeddings"
    for role in allowed_roles:
        values = oracle.encode(iid_roles[role]["text"].astype(str).tolist())
        path = embedding_dir / f"{role}.npz"
        write_npz(path, values=values)
        artifacts.append(path)
    if any((embedding_dir / f"{role}.npz").exists() for role in ("test_trigger", "test_benign_probe", "ood_trigger", "ood_benign_probe")):
        raise RuntimeError("test/OOD embeddings were encoded before the validation gate")

    calibration_benign = _role_embeddings(output, "calibration_benign_probe")
    calibration_clean = _role_embeddings(output, "calibration_trigger")
    count_benign = min(int(config["calibration"]["projection_benign_count"]), len(calibration_benign))
    count_clean = min(int(config["calibration"]["projection_clean_count"]), len(calibration_clean))
    projection = output / "projection"
    fit_fixed_projection(
        calibration_benign[:count_benign],
        calibration_clean[:count_clean],
        projection,
        seed=int(config["seed"]) + 400001,
        neighbors=int(config["calibration"]["umap_neighbors"]),
        minimum_distance=float(config["calibration"]["umap_min_dist"]),
        axis_quantile=float(config["plot"]["fixed_axis_quantile"]),
    )
    artifacts.extend(
        [
            projection / "pca_basis.npz",
            projection / "umap_model.joblib",
            projection / "projection_metadata.json",
            projection / "fit_coordinates.csv",
        ]
    )
    environment = output / "environment.json"
    write_json(
        environment,
        {
            "run_code_commit": _git_commit(),
            "model_id": config["model"]["id"],
            "registered_model_revision": config["model"]["revision"],
            "observed_model_revision": oracle.revision,
            "tokenizer_sha256": tokenizer.fingerprint(),
            "dependencies": _dependencies(),
            "python": sys.version,
        },
    )
    dependency_lock = output / "dependency_lock.json"
    write_json(dependency_lock, _dependencies())
    resolved = output / "config_resolved.yaml"
    write_text(resolved, yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True))
    sealed = output / "sealed_state.json"
    write_json(
        sealed,
        {
            "test_encoded": False,
            "ood_encoded": False,
            "gate_open": False,
            "sealed_roles": ["test_trigger", "test_benign_probe", "ood_trigger", "ood_benign_probe"],
        },
    )
    ledger = _save_ledger(output, oracle, "prepare", phase="prepare")
    artifacts.extend([environment, dependency_lock, resolved, sealed, ledger])
    _write_phase_completion(output, target, artifacts, {"phase": "prepare", "run_id": config["run_id"]})


def _make_evaluator(
    config: Mapping[str, Any],
    output: Path,
    oracle: SentenceTransformerOutputOracle,
    *,
    task: str,
    role: str,
    indices: np.ndarray | None,
    minimum_coverage: float,
    maximum_outlier_rate: float,
) -> CandidateEvaluator:
    frame = _role_frame(output, role)
    clean = _role_embeddings(output, role)
    if indices is not None:
        frame = frame.iloc[indices].reset_index(drop=True)
        clean = clean[indices]
    probe_role = role.replace("trigger", "benign_probe")
    if probe_role not in {"calibration_benign_probe", "search_benign_probe", "validation_benign_probe"}:
        raise ValueError(f"no benign probe registered for {role}")
    reference_role = "search_benign_reference" if role == "search_trigger" else probe_role
    return CandidateEvaluator(
        oracle=oracle,
        frame=frame,
        clean_embeddings=clean,
        benign_probe=_role_embeddings(output, probe_role),
        benign_reference=_role_embeddings(output, reference_role),
        manifest=_manifest(output),
        role=role,
        task=task,
        config=config,
        active_minimum_coverage=minimum_coverage,
        active_maximum_outlier_rate=maximum_outlier_rate,
    )


def command_calibrate(
    config_path: Path,
    config: Mapping[str, Any],
    output: Path,
    device: str | None,
    *,
    shard: int,
    shards: int,
) -> None:
    if not 0 <= shard < shards:
        raise ValueError(f"invalid calibration shard {shard}/{shards}")
    target = output / "calibration_shards" / f"shard_{shard:02d}"
    metadata = {"phase": "calibration_shard", "shard": shard, "shards": shards}
    if _phase_valid(output, target, metadata):
        return
    if not _phase_valid(output, output / "prepare", {"phase": "prepare", "run_id": config["run_id"]}):
        raise RuntimeError("prepare must complete before calibration")
    tokenizer = _tokenizer(config)
    space = _space(output, tokenizer)
    oracle = _oracle(config, device)
    rng = np.random.default_rng(int(config["seed"]) + 500003)
    records = []
    jobs = []
    for length in config["calibration"]["random_lengths"]:
        sampled = space.sample_valid(
            int(length),
            int(config["calibration"]["matched_random_count_per_length"]),
            rng=rng,
            maximum_attempts=int(config["search"]["maximum_materialization_attempts"]),
        )
        for rank, (candidate, pool_indices) in enumerate(sampled):
            jobs.append((int(length), rank, candidate, pool_indices))
    evaluator = _make_evaluator(
        config,
        output,
        oracle,
        task="shared",
        role="calibration_trigger",
        indices=None,
        minimum_coverage=float(config["structure"]["minimum_total_coverage"]),
        maximum_outlier_rate=float(config["structure"]["maximum_outlier_rate"]),
    )
    for global_index, (length, rank, candidate, pool_indices) in enumerate(jobs):
        if global_index % shards != shard:
            continue
        record = evaluator.evaluate(candidate).record
        record.update(
            {
                "pool_indices": ",".join(map(str, pool_indices.tolist())),
                "calibration_length": length,
                "calibration_rank": rank,
                "calibration_global_index": global_index,
                "calibration_shard": shard,
            }
        )
        records.append(record)
    target.mkdir(parents=True, exist_ok=True)
    calibration_csv = target / "matched_random_calibration.csv"
    write_csv(calibration_csv, [flatten_record(record) for record in records])
    ledger = _save_ledger(
        output, oracle, f"calibration_shard_{shard:02d}", phase="calibration_shard"
    )
    _write_phase_completion(output, target, [calibration_csv, ledger], metadata)


def command_merge_calibration(config: Mapping[str, Any], output: Path, *, shards: int) -> None:
    target = output / "calibration"
    metadata = {"phase": "calibration", "run_id": config["run_id"], "shards": shards}
    if _phase_valid(output, target, metadata):
        return
    records = []
    for shard in range(shards):
        source = output / "calibration_shards" / f"shard_{shard:02d}"
        expected = {"phase": "calibration_shard", "shard": shard, "shards": shards}
        if not _phase_valid(output, source, expected):
            raise RuntimeError(f"incomplete V5 calibration shard: {shard}/{shards}")
        records.extend(pd.read_csv(source / "matched_random_calibration.csv").to_dict(orient="records"))
    records.sort(key=lambda value: int(value["calibration_global_index"]))
    for record in records:
        if pd.isna(record.get("evaluation_error")):
            record["evaluation_error"] = ""
    successful = [record for record in records if not record.get("evaluation_error")]
    diagnostic_quantile = (
        float(np.quantile([record["cmax"] for record in successful], 0.10)) if successful else None
    )
    maximum_cmax = float(config["certification"]["maximum_cmax"])
    target.mkdir(parents=True, exist_ok=True)
    calibration_csv = target / "matched_random_calibration.csv"
    thresholds = target / "frozen_thresholds.json"
    write_csv(calibration_csv, [flatten_record(record) for record in records])
    write_json(
        thresholds,
        {
            "schema_version": "mode3-v5-frozen-thresholds-v1",
            "maximum_cmax": maximum_cmax,
            "maximum_cmax_source": config["certification"]["compactness_threshold_source"],
            "diagnostic_random_cmax_q10": diagnostic_quantile,
            "calibration_candidate_count": len(records),
            "successful_candidate_count": len(successful),
            "search_feedback": False,
            "matched_random_is_gate": False,
            "frozen_before_formal_search": True,
        },
    )
    _write_phase_completion(
        output,
        target,
        [calibration_csv, thresholds],
        metadata,
    )


def command_register(config_path: Path, config: Mapping[str, Any], output: Path) -> None:
    if _tracked_status():
        raise RuntimeError("cannot register V5 from a dirty tracked worktree")
    for phase in ("prepare", "calibration"):
        expected = {"phase": phase, "run_id": config["run_id"]}
        if phase == "calibration":
            expected["shards"] = int(config["runtime"]["calibration_shards"])
        if not _phase_valid(output, output / phase, expected):
            raise RuntimeError(f"V5 {phase} is incomplete")
    tokenizer = _tokenizer(config)
    data_files = sorted((output / "data").glob("*.csv")) + [output / "data" / "audit.json"]
    split_digest = hashlib.sha256()
    for path in data_files:
        split_digest.update(f"{path.name}\0{sha256_file(path)}\n".encode())
    contract = {
        "schema_version": "mode3-v5-run-contract-v1",
        "run_id": config["run_id"],
        "run_code_commit": _git_commit(),
        "config_sha256": _config_hash(config_path),
        "data_split_sha256": split_digest.hexdigest(),
        "model_revision": config["model"]["revision"],
        "tokenizer_sha256": tokenizer.fingerprint(),
        "dependency_lock_sha256": sha256_file(output / "dependency_lock.json"),
        "thresholds_sha256": sha256_file(output / "calibration" / "frozen_thresholds.json"),
        "test_and_ood_embeddings_absent": not any(
            (output / "embeddings" / f"{role}.npz").exists()
            for role in ("test_trigger", "test_benign_probe", "ood_trigger", "ood_benign_probe")
        ),
    }
    if not contract["test_and_ood_embeddings_absent"]:
        raise RuntimeError("test/OOD seal was broken before registration")
    write_json(output / "run_contract.json", contract)
    write_json(output / "FORMAL_RUN_REGISTERED.json", {**contract, "registered": True})


def _decode_record(record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record)
    for key in ("constraint_violations", "position_records", "structure_summaries"):
        value = result.get(key)
        if isinstance(value, str) and value:
            try:
                result[key] = json.loads(value)
            except json.JSONDecodeError:
                pass
    for key in (
        "occupancy_auc",
        "cmax",
        "cavg",
        "coverage",
        "worst_position_coverage",
        "outlier_rate",
        "lambda_star",
    ):
        if key in result:
            result[key] = float(result[key])
    return result


def _formal_thresholds(output: Path) -> dict[str, Any]:
    return json.loads((output / "calibration" / "frozen_thresholds.json").read_text(encoding="utf-8"))


def command_screen(
    config: Mapping[str, Any],
    output: Path,
    device: str | None,
    *,
    task: str,
    shard: int,
    shards: int,
) -> None:
    if task not in TASKS:
        raise ValueError(task)
    target = output / "screen" / task / f"shard_{shard:02d}"
    metadata = {"phase": "screen", "task": task, "shard": shard, "shards": shards}
    if _phase_valid(output, target, metadata):
        return
    tokenizer = _tokenizer(config)
    legal = _legal_frame(output)
    selected = legal.iloc[np.arange(len(legal)) % shards == shard].reset_index(drop=True)
    batch = rotating_batch_indices(
        len(_role_frame(output, "search_trigger")),
        int(config["search"]["single_token_batch_size"]),
        shard,
        int(config["seed"]) + sum(map(ord, task)) + 600011,
    )
    oracle = _oracle(config, device)
    evaluator = _make_evaluator(
        config,
        output,
        oracle,
        task=task,
        role="search_trigger",
        indices=batch,
        minimum_coverage=float(config["structure"]["minimum_total_coverage"]),
        maximum_outlier_rate=float(config["structure"]["maximum_outlier_rate"]),
    )
    records = []
    for row in selected.to_dict(orient="records"):
        candidate = _candidate_from_ids(tokenizer, (int(row["token_id"]),))
        record = evaluator.evaluate(candidate).record
        record.update(
            {
                "pool_indices": str(int(row["pool_index"])),
                "screen_shard": shard,
                "evaluation_scope": "single_token_rotating_batch",
            }
        )
        records.append(flatten_record(record))
    target.mkdir(parents=True, exist_ok=True)
    results = target / "population.csv"
    batch_path = target / "batch_manifest.json"
    write_csv(results, records)
    write_json(batch_path, {"indices": batch.tolist(), "candidate_count": len(records)})
    ledger = _save_ledger(
        output,
        oracle,
        f"screen_{task}_shard_{shard:02d}",
        phase="single_token_screen",
        task=task,
        length=1,
    )
    _write_phase_completion(output, target, [results, batch_path, ledger], metadata)


def command_formalize_screen(
    config: Mapping[str, Any], output: Path, device: str | None, *, task: str
) -> None:
    target = output / "search" / task / "length_01"
    metadata = {"phase": "formalize_single_token", "task": task, "length": 1}
    if _phase_valid(output, target, metadata):
        return
    shard_count = int(config["runtime"]["screen_shards_per_task"])
    records = []
    for shard in range(shard_count):
        source = output / "screen" / task / f"shard_{shard:02d}"
        if not _phase_valid(
            output,
            source,
            {"phase": "screen", "task": task, "shard": shard, "shards": shard_count},
        ):
            raise RuntimeError(f"incomplete single-token shard: {task}/{shard}")
        records.extend(_decode_record(row) for row in pd.read_csv(source / "population.csv").to_dict(orient="records"))
    selected_indices = select_nsga2(
        records, min(int(config["search"]["single_token_full_candidates"]), len(records))
    )
    tokenizer = _tokenizer(config)
    oracle = _oracle(config, device)
    evaluator = _make_evaluator(
        config,
        output,
        oracle,
        task=task,
        role="search_trigger",
        indices=None,
        minimum_coverage=float(config["structure"]["minimum_total_coverage"]),
        maximum_outlier_rate=float(config["structure"]["maximum_outlier_rate"]),
    )
    full = []
    for index in selected_indices:
        source = records[index]
        candidate = _candidate_from_ids(tokenizer, str(source["token_ids"]))
        record = evaluator.evaluate(candidate).record
        record.update(
            {
                "pool_indices": source["pool_indices"],
                "evaluation_scope": "full_search",
                "restart": -1,
                "generation": -1,
            }
        )
        full.append(record)
    formal = update_historical_archive([], full, int(config["search"]["formal_archive_size"]))
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "formal_archive.json"
    full_path = target / "full_search_evaluations.csv"
    write_json(archive, formal)
    write_csv(full_path, [flatten_record(record) for record in full])
    ledger = _save_ledger(
        output, oracle, f"formalize_single_{task}", phase="formalize_single_token", task=task, length=1
    )
    _write_phase_completion(output, target, [archive, full_path, ledger], metadata)


def command_search(
    config: Mapping[str, Any],
    output: Path,
    device: str | None,
    *,
    task: str,
    length: int,
    restart: int,
) -> None:
    if task not in TASKS or length < 2 or length > int(config["lengths"]["maximum"]):
        raise ValueError(f"invalid V5 search job: {task}/{length}/{restart}")
    tokenizer = _tokenizer(config)
    space = _space(output, tokenizer)
    total = len(_role_frame(output, "search_trigger"))
    run_output = output / "search" / task / f"length_{length:02d}" / f"restart_{restart:02d}"
    job_metadata = {"phase": "search_job", "task": task, "length": length, "restart": restart}
    if _phase_valid(output, run_output / "job_complete", job_metadata):
        return
    oracle = _oracle(config, device)
    evaluator_cache: dict[tuple[int, tuple[int, ...]], CandidateEvaluator] = {}
    projection = load_fixed_projection(output / "projection")

    def evaluator_for(indices: np.ndarray, generation: int) -> CandidateEvaluator:
        key = (generation, tuple(map(int, indices)))
        if key not in evaluator_cache:
            coverage, outliers = active_structural_envelope(
                config, generation, int(config["search"]["iterations"])
            )
            evaluator_cache[key] = _make_evaluator(
                config,
                output,
                oracle,
                task=task,
                role="search_trigger",
                indices=indices,
                minimum_coverage=coverage,
                maximum_outlier_rate=outliers,
            )
        return evaluator_cache[key]

    full_evaluator = _make_evaluator(
        config,
        output,
        oracle,
        task=task,
        role="search_trigger",
        indices=None,
        minimum_coverage=float(config["structure"]["minimum_total_coverage"]),
        maximum_outlier_rate=float(config["structure"]["maximum_outlier_rate"]),
    )

    def score(candidate: Candidate, indices: np.ndarray, generation: int) -> dict[str, Any]:
        return evaluator_for(indices, generation).evaluate(candidate).record

    def full_score(candidate: Candidate, generation: int) -> dict[str, Any]:
        return full_evaluator.evaluate(candidate).record

    def snapshot(candidate: Candidate, generation: int, label: str, target: Path) -> Sequence[Path]:
        indices = rotating_batch_indices(
            total,
            int(config["search"]["rotating_batch_size"]),
            generation,
            int(config["seed"]) + restart * 10000019 + length * 1009 + sum(map(ord, task)) + 700001,
        )
        bundle = evaluator_for(indices, generation).evaluate(candidate, retain_embeddings=True)
        return save_snapshot(
            bundle,
            task=task,
            benign=_role_embeddings(output, "search_benign_probe"),
            projection=projection,
            output=target,
            metadata={
                "title": f"V5 {task} L={length} restart={restart} generation={generation} {label}",
                "task": task,
                "length": length,
                "restart": restart,
                "generation": generation,
                "snapshot_role": label,
                "candidate_key": candidate.key,
                "trigger": candidate.trigger,
            },
            sample_count=int(config["plot"]["projection_sample_count"]),
            dpi=int(config["plot"]["dpi"]),
        )

    pareto_cem(
        space,
        length=length,
        restart=restart,
        task=task,
        total_search_texts=total,
        config=config,
        output=run_output,
        score=score,
        full_score=full_score,
        snapshot=snapshot,
        query_ledger=oracle.ledger.to_dict,
    )
    frames = sorted(run_output.glob("generation_*/snapshots/leader/cluster.png"))
    animation_record = render_animation(
        frames,
        run_output / "optimization.gif",
        run_output / "optimization.mp4",
        fps=int(config["plot"]["fps"]),
        dpi=max(80, int(config["plot"]["dpi"]) // 2),
    )
    if bool(config["plot"]["make_mp4"]) and not animation_record.get("mp4"):
        raise RuntimeError("registered V5 MP4 generation failed; ffmpeg is required")
    write_json(run_output / "animation.json", animation_record)
    ledger = _save_ledger(
        output,
        oracle,
        f"search_{task}_length_{length:02d}_restart_{restart:02d}",
        phase="pareto_cem_search",
        task=task,
        length=length,
        restart=restart,
    )
    job_artifacts = [
        run_output / "COMPLETE.json",
        run_output / "formal_archive.json",
        run_output / "historical_archive.json",
        run_output / "animation.json",
        run_output / "optimization.gif",
        ledger,
    ]
    if (run_output / "optimization.mp4").is_file():
        job_artifacts.append(run_output / "optimization.mp4")
    _write_phase_completion(output, run_output / "job_complete", job_artifacts, job_metadata)


def command_merge_search(
    config: Mapping[str, Any],
    output: Path,
    device: str | None,
    *,
    task: str,
    length: int,
) -> None:
    target = output / "search" / task / f"length_{length:02d}" / "merged"
    metadata = {"phase": "merge_search", "task": task, "length": length}
    if _phase_valid(output, target, metadata):
        return
    records = []
    for restart in range(int(config["search"]["restarts_per_length"])):
        source = output / "search" / task / f"length_{length:02d}" / f"restart_{restart:02d}"
        if not _phase_valid(
            output,
            source / "job_complete",
            {"phase": "search_job", "task": task, "length": length, "restart": restart},
        ):
            raise RuntimeError(f"incomplete V5 search restart: {task}/{length}/{restart}")
        records.extend(json.loads((source / "historical_archive.json").read_text(encoding="utf-8")))
    if not records:
        raise RuntimeError(f"no historical CEM candidates to formalize: {task}/{length}")
    records = update_historical_archive([], records, max(len(records), 1))
    selected_indices = select_nsga2(
        records, min(int(config["search"]["formal_archive_size"]), len(records))
    )
    tokenizer = _tokenizer(config)
    oracle = _oracle(config, device)
    evaluator = _make_evaluator(
        config,
        output,
        oracle,
        task=task,
        role="search_trigger",
        indices=None,
        minimum_coverage=float(config["structure"]["minimum_total_coverage"]),
        maximum_outlier_rate=float(config["structure"]["maximum_outlier_rate"]),
    )
    full = []
    for index in selected_indices:
        source_record = records[index]
        candidate = _candidate_from_ids(tokenizer, str(source_record["token_ids"]))
        record = evaluator.evaluate(candidate).record
        record.update(
            {
                "pool_indices": source_record["pool_indices"],
                "source_restart": source_record.get("restart"),
                "source_generation": source_record.get("generation"),
                "evaluation_scope": "full_search",
            }
        )
        full.append(record)
    formal = update_historical_archive([], full, int(config["search"]["formal_archive_size"]))
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "formal_archive.json"
    full_path = target / "full_search_evaluations.csv"
    write_json(archive, formal)
    write_csv(full_path, [flatten_record(record) for record in full])
    ledger = _save_ledger(
        output,
        oracle,
        f"formalize_{task}_length_{length:02d}",
        phase="formalize_multi_token",
        task=task,
        length=length,
    )
    _write_phase_completion(output, target, [archive, full_path, ledger], metadata)


def _formal_archive_path(output: Path, task: str, length: int) -> Path:
    if length == 1:
        return output / "search" / task / "length_01" / "formal_archive.json"
    return output / "search" / task / f"length_{length:02d}" / "merged" / "formal_archive.json"


def _validation_positions(task: str) -> tuple[str, ...]:
    return (task,) if task in {"prefix", "suffix", "random"} else ("prefix", "suffix", "random")


def command_validate(
    config: Mapping[str, Any], output: Path, device: str | None, *, task: str, length: int
) -> None:
    target = output / "validation" / task / f"length_{length:02d}"
    metadata = {"phase": "validation", "task": task, "length": length}
    if _phase_valid(output, target, metadata):
        return
    archive_path = _formal_archive_path(output, task, length)
    if not archive_path.is_file():
        raise RuntimeError(f"formal full-search archive is missing: {archive_path}")
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    if not archive:
        raise RuntimeError(f"formal full-search archive is empty: {task}/{length}")
    selected_indices = select_nsga2(archive, min(int(config["validation"]["candidates_per_length"]), len(archive)))
    tokenizer = _tokenizer(config)
    oracle = _oracle(config, device)
    evaluator = _make_evaluator(
        config,
        output,
        oracle,
        task=task,
        role="validation_trigger",
        indices=None,
        minimum_coverage=float(config["structure"]["minimum_total_coverage"]),
        maximum_outlier_rate=float(config["structure"]["maximum_outlier_rate"]),
    )
    thresholds = _formal_thresholds(output)
    validation_frame = _role_frame(output, "validation_trigger")
    realizability_frame = validation_frame.iloc[: int(config["tokenizer"]["realizability_contexts"])].copy()
    manifest = _manifest(output)
    projection = load_fixed_projection(output / "projection")
    candidate_summaries = []
    candidate_artifacts: list[Path] = []
    for rank, index in enumerate(selected_indices):
        source = archive[index]
        candidate = _candidate_from_ids(tokenizer, str(source["token_ids"]))
        candidate_dir = target / f"candidate_{rank:02d}"
        bundle = evaluator.evaluate(candidate, retain_embeddings=True)
        audit = context_realizability(
            tokenizer,
            candidate,
            realizability_frame,
            role="validation_trigger",
            manifest=manifest,
            positions=_validation_positions(task),
            random_replicates=int(config["insertion"]["random_replicates"]),
            separator=str(config["insertion"]["separator"]),
        )
        if bundle.record.get("evaluation_error"):
            candidate_dir.mkdir(parents=True, exist_ok=True)
            evaluation_path = candidate_dir / "evaluation_record.json"
            realizability_path = candidate_dir / "realizability.json"
            summary_path = candidate_dir / "validation_summary.json"
            certificate_path = candidate_dir / "certification.json"
            failure_summary = {
                "evaluation_error": bundle.record["evaluation_error"],
                "occupancy_auc": float(bundle.record["occupancy_auc"]),
                "cmax": float(bundle.record["cmax"]),
                "cavg": float(bundle.record["cavg"]),
                "lambda_star": 0.0,
                "coverage_lcb": 0.0,
                "worst_position_coverage_lcb": 0.0,
                "outlier_rate_ucb": 1.0,
            }
            failure_certificate = {
                "level_0_realizable": bool(audit.exact_token_roundtrip and audit.inserted_once),
                "level_1_attractor": False,
                "level_2_low_occupancy": False,
                "level_1_gates": {"fit": False},
                "level_2_gates": {"fit": False},
            }
            write_json(evaluation_path, bundle.record)
            write_json(realizability_path, audit_to_dict(audit))
            write_json(summary_path, failure_summary)
            write_json(certificate_path, failure_certificate)
            local_artifacts = [evaluation_path, realizability_path, summary_path, certificate_path]
            write_completion(
                candidate_dir,
                local_artifacts,
                {"task": task, "length": length, "candidate_rank": rank, "candidate_key": candidate.key},
            )
            candidate_artifacts.extend([*local_artifacts, candidate_dir / "COMPLETE.json"])
            candidate_summaries.append(
                {
                    "candidate_key": candidate.key,
                    "token_ids": candidate.key,
                    "trigger": candidate.trigger,
                    "actual_token_length": candidate.actual_token_length,
                    "task": task,
                    "length": length,
                    "candidate_rank": rank,
                    **failure_summary,
                    **failure_certificate,
                    "candidate_dir": candidate_dir.relative_to(output).as_posix(),
                }
            )
            continue
        frozen, summary = freeze_validation_bundle(
            bundle,
            candidate_key=candidate.key,
            token_ids=candidate.token_ids,
            trigger=candidate.trigger,
            task=task,
            benign_probe=_role_embeddings(output, "validation_benign_probe"),
            config=config,
            seed=int(config["seed"]) + length * 1009 + rank * 100003 + sum(map(ord, task)),
            group_ids=validation_frame["source_group"].astype(str).to_numpy(),
        )
        certificate = certification(summary, audit_to_dict(audit), thresholds, config)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        evaluation_path = candidate_dir / "evaluation_record.json"
        realizability_path = candidate_dir / "realizability.json"
        summary_path = candidate_dir / "validation_summary.json"
        certificate_path = candidate_dir / "certification.json"
        write_json(evaluation_path, bundle.record)
        write_json(realizability_path, audit_to_dict(audit))
        write_json(summary_path, summary)
        write_json(certificate_path, certificate)
        frozen_artifacts = save_frozen_candidate(candidate_dir / "frozen", frozen, certificate)
        snapshot_artifacts = save_snapshot(
            bundle,
            task=task,
            benign=_role_embeddings(output, "validation_benign_probe"),
            projection=projection,
            output=candidate_dir / "projection",
            metadata={
                "title": f"V5 validation {task} L={length} candidate={rank}",
                "task": task,
                "length": length,
                "candidate_rank": rank,
                "candidate_key": candidate.key,
                "trigger": candidate.trigger,
            },
            sample_count=int(config["plot"]["projection_sample_count"]),
            dpi=int(config["plot"]["dpi"]),
        )
        candidate_metadata = {
            "candidate_key": candidate.key,
            "token_ids": candidate.key,
            "trigger": candidate.trigger,
            "actual_token_length": candidate.actual_token_length,
            "task": task,
            "length": length,
            "candidate_rank": rank,
            "occupancy_auc": summary["occupancy_auc"],
            "cmax": summary["cmax"],
            "cavg": summary["cavg"],
            "lambda_star": summary["lambda_star"],
            "coverage_lcb": summary["coverage_lcb"],
            "worst_position_coverage_lcb": summary["worst_position_coverage_lcb"],
            "outlier_rate_ucb": summary["outlier_rate_ucb"],
            **certificate,
            "candidate_dir": candidate_dir.relative_to(output).as_posix(),
        }
        candidate_summaries.append(candidate_metadata)
        local_artifacts = [
            evaluation_path,
            realizability_path,
            summary_path,
            certificate_path,
            *frozen_artifacts,
            *snapshot_artifacts,
        ]
        write_completion(
            candidate_dir,
            [path for path in local_artifacts if path.parent == candidate_dir],
            {"task": task, "length": length, "candidate_rank": rank, "candidate_key": candidate.key},
        )
        candidate_artifacts.extend(local_artifacts)

    rng = np.random.default_rng(int(config["seed"]) + length * 65537 + sum(map(ord, task)))
    space = _space(output, tokenizer)
    random_candidates = space.sample_valid(
        length,
        int(config["validation"]["matched_random_count"]),
        rng=rng,
        maximum_attempts=int(config["search"]["maximum_materialization_attempts"]),
    )
    random_records = []
    for candidate, _ in random_candidates:
        random_records.append(evaluator.evaluate(candidate).record)
    target.mkdir(parents=True, exist_ok=True)
    summary_file = target / "summary.json"
    candidates_file = target / "candidates.csv"
    random_file = target / "matched_random_diagnostic.json"
    write_json(
        summary_file,
        {
            "schema_version": "mode3-v5-validation-summary-v1",
            "task": task,
            "length": length,
            "candidate_count": len(candidate_summaries),
            "candidates": candidate_summaries,
            "random_baseline_is_gate": False,
        },
    )
    write_csv(candidates_file, [flatten_record(record) for record in candidate_summaries])
    write_json(
        random_file,
        {
            "search_feedback": False,
            "gate": False,
            "count": len(random_records),
            "cmax": [record["cmax"] for record in random_records],
            "occupancy_auc": [record["occupancy_auc"] for record in random_records],
            "candidate_percentiles": [
                {
                    "candidate_key": record["candidate_key"],
                    "cmax_percentile": float(np.mean([random["cmax"] <= record["cmax"] for random in random_records])),
                    "occupancy_auc_percentile": float(
                        np.mean([random["occupancy_auc"] <= record["occupancy_auc"] for random in random_records])
                    ),
                }
                for record in candidate_summaries
            ],
        },
    )
    ledger = _save_ledger(
        output, oracle, f"validation_{task}_length_{length:02d}", phase="validation", task=task, length=length
    )
    _write_phase_completion(
        output,
        target,
        [summary_file, candidates_file, random_file, ledger, *candidate_artifacts],
        metadata,
    )


def command_freeze(config: Mapping[str, Any], output: Path) -> None:
    target = output / "frozen"
    metadata = {"phase": "freeze", "run_id": config["run_id"]}
    if _phase_valid(output, target, metadata):
        return
    candidates = []
    maximum = int(config["lengths"]["maximum"])
    for task in TASKS:
        for length in range(int(config["lengths"]["minimum"]), maximum + 1):
            validation_dir = output / "validation" / task / f"length_{length:02d}"
            if not _phase_valid(output, validation_dir, {"phase": "validation", "task": task, "length": length}):
                raise RuntimeError(f"cannot freeze before all validations complete: {task}/{length}")
            payload = json.loads((validation_dir / "summary.json").read_text(encoding="utf-8"))
            candidates.extend(payload["candidates"])
    selections = []
    for task in TASKS:
        for level_key, level_name in (
            ("level_1_attractor", "A"),
            ("level_2_low_occupancy", "LO"),
        ):
            passing = [record for record in candidates if record["task"] == task and bool(record[level_key])]
            if not passing:
                continue
            selected = min(
                passing,
                key=lambda record: (
                    int(record["actual_token_length"]),
                    float(record["occupancy_auc"]),
                    float(record["cmax"]),
                    str(record["candidate_key"]),
                ),
            )
            selections.append(
                {
                    **selected,
                    "certificate_level": level_name,
                    "selection_id": f"{task}_{level_name}_L{int(selected['actual_token_length']):02d}_{str(selected['candidate_key']).replace(',', '-')[:48]}",
                    "frozen_dir": f"{selected['candidate_dir']}/frozen",
                }
            )
    gate_open = bool(selections)
    target.mkdir(parents=True, exist_ok=True)
    selection_path = target / "selection.json"
    write_json(
        selection_path,
        {
            "schema_version": "mode3-v5-frozen-selection-v1",
            "gate_open": gate_open,
            "selection_count": len(selections),
            "selections": selections,
            "test_and_ood_refit_prohibited": True,
        },
    )
    gate_state = target / "gate_state.json"
    write_json(
        gate_state,
        {
            "gate_open": gate_open,
            "validation_selection_sha256": sha256_file(selection_path),
            "initial_seal_sha256": sha256_file(output / "sealed_state.json"),
        },
    )
    _write_phase_completion(output, target, [selection_path, gate_state], metadata)


def _selected_unique(output: Path) -> list[dict[str, Any]]:
    payload = json.loads((output / "frozen" / "selection.json").read_text(encoding="utf-8"))
    unique = {}
    for record in payload["selections"]:
        key = (record["task"], record["candidate_key"])
        unique.setdefault(key, record)
        unique[key].setdefault("selected_levels", []).append(record["certificate_level"])
    return sorted(
        unique.values(), key=lambda record: (int(record["actual_token_length"]), TASKS.index(record["task"]), record["candidate_key"])
    )


def _encode_sealed_phase(
    config: Mapping[str, Any],
    output: Path,
    device: str | None,
    *,
    phase: str,
    trigger_role: str,
    benign_role: str,
) -> None:
    target = output / phase
    metadata = {"phase": phase, "run_id": config["run_id"]}
    if _phase_valid(output, target, metadata):
        return
    frozen_dir = output / "frozen"
    if not _phase_valid(output, frozen_dir, {"phase": "freeze", "run_id": config["run_id"]}):
        raise RuntimeError("validation freeze must complete before test/OOD")
    selection_payload = json.loads((frozen_dir / "selection.json").read_text(encoding="utf-8"))
    if not selection_payload["gate_open"]:
        target.mkdir(parents=True, exist_ok=True)
        status = target / "not_opened.json"
        write_json(status, {"gate_open": False, "reason": "no validation-certified V5 attractor"})
        _write_phase_completion(output, target, [status], metadata)
        return
    oracle = _oracle(config, device)
    frame = _role_frame(output, trigger_role)
    benign_frame = _role_frame(output, benign_role)
    clean = oracle.encode(frame["text"].astype(str).tolist())
    benign = oracle.encode(benign_frame["text"].astype(str).tolist())
    embedding_dir = output / "embeddings"
    clean_path = embedding_dir / f"{trigger_role}.npz"
    benign_path = embedding_dir / f"{benign_role}.npz"
    write_npz(clean_path, values=clean)
    write_npz(benign_path, values=benign)
    artifacts = [clean_path, benign_path]
    tokenizer = _tokenizer(config)
    manifest = _manifest(output)
    summaries = []
    for record in _selected_unique(output):
        candidate = _candidate_from_ids(tokenizer, str(record["token_ids"]))
        frozen = load_frozen_candidate(output / record["frozen_dir"])
        texts = materialize_views(
            frame,
            candidate.trigger,
            candidate.task,
            role=trigger_role,
            manifest=manifest,
            random_replicates=int(config["insertion"]["random_replicates"]),
            separator=str(config["insertion"]["separator"]),
        )
        views = {view: oracle.encode(values) for view, values in texts.items()}
        candidate_target = target / record["selection_id"]
        candidate_target.mkdir(parents=True, exist_ok=True)
        view_paths = []
        for view, values in views.items():
            path = candidate_target / f"view_{view}.npz"
            write_npz(path, values=values)
            view_paths.append(path)
        summary = evaluate_frozen_no_refit(frozen, views, benign, config)
        summary.update(
            {
                "selection_id": record["selection_id"],
                "candidate_key": candidate.key,
                "trigger": candidate.trigger,
                "task": candidate.task,
                "actual_token_length": candidate.actual_token_length,
                "selected_levels": record["selected_levels"],
            }
        )
        summary_path = candidate_target / "summary.json"
        write_json(summary_path, summary)
        write_completion(
            candidate_target,
            [summary_path, *view_paths],
            {"phase": phase, "selection_id": record["selection_id"]},
        )
        summaries.append(summary)
        artifacts.extend([summary_path, *view_paths, candidate_target / "COMPLETE.json"])
    overall = target / "summary.json"
    write_json(overall, {"phase": phase, "refit_performed": False, "candidates": summaries})
    ledger = _save_ledger(output, oracle, phase, phase=phase)
    artifacts.extend([overall, ledger])
    seal_audit = target / "seal_audit.json"
    write_json(
        seal_audit,
        {
            "phase": phase,
            "encoded_after_validation_freeze": True,
            "query_ledger": ledger.relative_to(output).as_posix(),
            "initial_seal_sha256": sha256_file(output / "sealed_state.json"),
            "validation_selection_sha256": sha256_file(output / "frozen" / "selection.json"),
            "refit_performed": False,
        },
    )
    artifacts.append(seal_audit)
    _write_phase_completion(output, target, artifacts, metadata)


def command_retrieval(config: Mapping[str, Any], output: Path, device: str | None) -> None:
    target = output / "downstream"
    metadata = {"phase": "retrieval", "run_id": config["run_id"]}
    if _phase_valid(output, target, metadata):
        return
    if not _phase_valid(output, output / "test", {"phase": "test", "run_id": config["run_id"]}):
        raise RuntimeError("one-time test must complete before retrieval")
    if (output / "test" / "not_opened.json").is_file():
        target.mkdir(parents=True, exist_ok=True)
        status = target / "not_opened.json"
        write_json(status, {"reason": "validation gate did not open the one-time test"})
        _write_phase_completion(output, target, [status], metadata)
        return
    test_payload = json.loads((output / "test" / "summary.json").read_text(encoding="utf-8"))
    passing = [record for record in test_payload["candidates"] if record.get("level_2_test_pass")]
    if not passing:
        target.mkdir(parents=True, exist_ok=True)
        status = target / "not_opened.json"
        write_json(status, {"reason": "no frozen candidate passed the low-occupancy one-time test"})
        _write_phase_completion(output, target, [status], metadata)
        return
    selected = min(
        passing,
        key=lambda record: (int(record["actual_token_length"]), TASKS.index(record["task"]), record["candidate_key"]),
    )
    selection = next(record for record in _selected_unique(output) if record["selection_id"] == selected["selection_id"])
    tokenizer = _tokenizer(config)
    candidate = _candidate_from_ids(tokenizer, str(selection["token_ids"]))
    frozen = load_frozen_candidate(output / selection["frozen_dir"])
    oracle = _oracle(config, device)
    validation_frame = _role_frame(output, "validation_trigger")
    validation_texts = materialize_views(
        validation_frame,
        candidate.trigger,
        candidate.task,
        role="validation_trigger",
        manifest=_manifest(output),
        random_replicates=int(config["insertion"]["random_replicates"]),
        separator=str(config["insertion"]["separator"]),
    )
    validation_views = {view: oracle.encode(texts) for view, texts in validation_texts.items()}
    anchor = choose_single_real_text_anchor(
        validation_texts,
        validation_views,
        [structure.centers for structure in frozen.structures.values()],
        [structure.radii for structure in frozen.structures.values()],
    )
    candidate_test = output / "test" / selection["selection_id"]
    triggered = np.concatenate(
        [np.load(path)["values"] for path in sorted(candidate_test.glob("view_*.npz"))], axis=0
    )
    clean = _role_embeddings(output, "test_trigger")
    benign = _role_embeddings(output, "test_benign_probe")
    result = controlled_single_poison_retrieval(
        triggered,
        clean,
        benign,
        anchor,
        top_k=config["retrieval"]["top_k"],
    )
    target.mkdir(parents=True, exist_ok=True)
    result_path = target / "single_poison_retrieval.json"
    vector_path = target / "poison_anchor.npz"
    write_json(result_path, result)
    write_npz(vector_path, vector=anchor.vector)
    ledger = _save_ledger(output, oracle, "retrieval", phase="retrieval")
    _write_phase_completion(output, target, [result_path, vector_path, ledger], metadata)


def _query_budget(output: Path) -> dict[str, Any]:
    rows = []
    totals = {"encode_calls": 0, "requested_texts": 0, "cache_hits": 0, "submitted_texts": 0}
    for path in sorted((output / "query_ledgers").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = {"path": path.relative_to(output).as_posix(), **payload}
        rows.append(row)
        for key in totals:
            totals[key] += int(payload.get(key, 0))
    return {"schema_version": "mode3-v5-query-budget-v1", "ledger_count": len(rows), "totals": totals, "ledgers": rows}


def _write_result_manifest(output: Path) -> Path:
    manifest = output / "sha256_manifest.csv"
    rows = []
    for path in sorted((value for value in output.rglob("*") if value.is_file()), key=lambda value: value.as_posix()):
        if path == manifest or path.name.endswith(".tmp"):
            continue
        rows.append(
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_csv(manifest, rows)
    return manifest


def command_finalize(config: Mapping[str, Any], output: Path) -> None:
    target = output / "finalize"
    metadata = {"phase": "finalize", "run_id": config["run_id"]}
    if _phase_valid(output, target, metadata):
        return
    for phase in ("frozen", "test", "ood", "downstream"):
        expected_phase = "freeze" if phase == "frozen" else "retrieval" if phase == "downstream" else phase
        if not _phase_valid(output, output / phase, {"phase": expected_phase, "run_id": config["run_id"]}):
            raise RuntimeError(f"cannot finalize before V5 {phase} completes")
    rows = []
    for task in TASKS:
        for length in range(int(config["lengths"]["minimum"]), int(config["lengths"]["maximum"]) + 1):
            payload = json.loads(
                (output / "validation" / task / f"length_{length:02d}" / "summary.json").read_text(encoding="utf-8")
            )
            candidates = payload["candidates"]
            best = min(
                candidates,
                key=lambda record: (
                    not bool(record["level_2_low_occupancy"]),
                    not bool(record["level_1_attractor"]),
                    float(record["occupancy_auc"]),
                    float(record["cmax"]),
                    str(record["candidate_key"]),
                ),
            )
            rows.append(best)
    frontier = pd.DataFrame.from_records(rows)
    frontier_path = output / "length_frontier.csv"
    _write_frame(frontier_path, frontier)
    minimum_lengths = {}
    for task in TASKS:
        subset = frontier[frontier["task"] == task]
        minimum_lengths[task] = {
            "level_1_attractor": (
                int(subset.loc[subset["level_1_attractor"].astype(bool), "actual_token_length"].min())
                if subset["level_1_attractor"].astype(bool).any()
                else None
            ),
            "level_2_low_occupancy": (
                int(subset.loc[subset["level_2_low_occupancy"].astype(bool), "actual_token_length"].min())
                if subset["level_2_low_occupancy"].astype(bool).any()
                else None
            ),
        }
    import matplotlib.pyplot as plt

    figure_path = output / "length_frontier.png"
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    colors = {task: plt.cm.tab10(index) for index, task in enumerate(TASKS)}
    for task in TASKS:
        subset = frontier[frontier["task"] == task].sort_values("actual_token_length")
        axes[0].plot(subset["actual_token_length"], subset["cmax"], marker="o", ms=3, label=task, color=colors[task])
        axes[1].plot(
            subset["actual_token_length"], subset["occupancy_auc"], marker="o", ms=3, label=task, color=colors[task]
        )
        certified = subset[subset["level_2_low_occupancy"].astype(bool)]
        axes[0].scatter(certified["actual_token_length"], certified["cmax"], marker="*", s=90, color=colors[task])
        axes[1].scatter(
            certified["actual_token_length"], certified["occupancy_auc"], marker="*", s=90, color=colors[task]
        )
    axes[0].set_ylabel("Frozen robust Cmax")
    axes[1].set_ylabel("Multi-scale occupancy UCB AUC")
    axes[1].set_xlabel("Actual tokenizer length")
    axes[0].grid(alpha=0.2)
    axes[1].grid(alpha=0.2)
    axes[0].legend(ncol=3, fontsize=8)
    fig.savefig(figure_path, dpi=int(config["plot"]["dpi"]))
    plt.close(fig)

    selection = json.loads((output / "frozen" / "selection.json").read_text(encoding="utf-8"))
    test_status = (
        json.loads((output / "test" / "summary.json").read_text(encoding="utf-8"))
        if (output / "test" / "summary.json").is_file()
        else {"not_opened": True}
    )
    ood_status = (
        json.loads((output / "ood" / "summary.json").read_text(encoding="utf-8"))
        if (output / "ood" / "summary.json").is_file()
        else {"not_opened": True}
    )
    retrieval_status = (
        json.loads((output / "downstream" / "single_poison_retrieval.json").read_text(encoding="utf-8"))
        if (output / "downstream" / "single_poison_retrieval.json").is_file()
        else json.loads((output / "downstream" / "not_opened.json").read_text(encoding="utf-8"))
    )
    budget = _query_budget(output)
    if int(budget["totals"]["submitted_texts"]) > int(config["runtime"]["maximum_submitted_text_queries"]):
        raise RuntimeError("V5 submitted-text query budget exceeded")
    budget_path = output / "query_budget.json"
    write_json(budget_path, budget)
    final_status = {
        "schema_version": "mode3-v5-final-status-v1",
        "run_id": config["run_id"],
        "run_code_commit": _git_commit(),
        "minimum_lengths": minimum_lengths,
        "p1_any_level_1": any(minimum_lengths[task]["level_1_attractor"] is not None for task in ("prefix", "suffix", "random")),
        "p2_level_1": minimum_lengths["conditional"]["level_1_attractor"] is not None,
        "p3_level_1": minimum_lengths["shared"]["level_1_attractor"] is not None,
        "gate_open": bool(selection["gate_open"]),
        "test": test_status,
        "ood": ood_status,
        "retrieval": retrieval_status,
    }
    final_status_path = output / "final_status.json"
    write_json(final_status_path, final_status)
    summary_path = output / "summary.json"
    write_json(
        summary_path,
        {
            "run_contract": _contract(output),
            "thresholds": _formal_thresholds(output),
            "minimum_lengths": minimum_lengths,
            "frozen_selection": selection,
            "query_budget_totals": budget["totals"],
        },
    )
    report_path = output / "report.md"
    lines = [
        "# Mode 3 V5 audited result",
        "",
        f"- Run ID: `{config['run_id']}`",
        f"- Code commit: `{_git_commit()}`",
        f"- Validation gate opened: `{bool(selection['gate_open'])}`",
        f"- Submitted embedding texts: `{budget['totals']['submitted_texts']}`",
        "",
        "## Minimum validation-certified actual token lengths",
        "",
        "| Protocol | Level 1 attractor | Level 2 low occupancy |",
        "|---|---:|---:|",
    ]
    for task in TASKS:
        values = minimum_lengths[task]
        lines.append(f"| {task} | {values['level_1_attractor']} | {values['level_2_low_occupancy']} |")
    lines.extend(
        [
            "",
            "P1, P2, and P3 are reported independently. Shared-attractor failure does not negate a position-specific or position-conditional certificate.",
            "Test and OOD used validation-frozen centers and radii without refitting. Retrieval, when opened, used one real poison text and did not feed search.",
        ]
    )
    write_text(report_path, "\n".join(lines) + "\n")
    manifest_path = _write_result_manifest(output)
    _write_phase_completion(
        output,
        target,
        [frontier_path, figure_path, budget_path, final_status_path, summary_path, report_path, manifest_path],
        metadata,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v5_mode3.yaml")
    parser.add_argument("--output")
    parser.add_argument("--device")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--shard", type=int, required=True)
    calibrate.add_argument("--shards", type=int, required=True)
    merge_calibration = subparsers.add_parser("merge-calibration")
    merge_calibration.add_argument("--shards", type=int, required=True)
    subparsers.add_parser("register")
    screen = subparsers.add_parser("screen")
    screen.add_argument("--task", choices=TASKS, required=True)
    screen.add_argument("--shard", type=int, required=True)
    screen.add_argument("--shards", type=int, required=True)
    formalize = subparsers.add_parser("formalize-screen")
    formalize.add_argument("--task", choices=TASKS, required=True)
    search = subparsers.add_parser("search")
    search.add_argument("--task", choices=TASKS, required=True)
    search.add_argument("--length", type=int, required=True)
    search.add_argument("--restart", type=int, required=True)
    merge = subparsers.add_parser("merge-search")
    merge.add_argument("--task", choices=TASKS, required=True)
    merge.add_argument("--length", type=int, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--task", choices=TASKS, required=True)
    validate.add_argument("--length", type=int, required=True)
    subparsers.add_parser("freeze")
    subparsers.add_parser("test")
    subparsers.add_parser("ood")
    subparsers.add_parser("retrieval")
    subparsers.add_parser("finalize")
    args = parser.parse_args(argv)
    config_path = _resolve(args.config)
    assert config_path is not None
    config = _read_config(config_path)
    output = _output(config, args.output)
    if args.command not in {"prepare", "calibrate", "merge-calibration", "register"}:
        _assert_contract(config_path, config, output)
    if args.command == "prepare":
        command_prepare(config_path, config, output, args.device)
    elif args.command == "calibrate":
        command_calibrate(
            config_path,
            config,
            output,
            args.device,
            shard=args.shard,
            shards=args.shards,
        )
    elif args.command == "merge-calibration":
        command_merge_calibration(config, output, shards=args.shards)
    elif args.command == "register":
        command_register(config_path, config, output)
    elif args.command == "screen":
        command_screen(config, output, args.device, task=args.task, shard=args.shard, shards=args.shards)
    elif args.command == "formalize-screen":
        command_formalize_screen(config, output, args.device, task=args.task)
    elif args.command == "search":
        command_search(config, output, args.device, task=args.task, length=args.length, restart=args.restart)
    elif args.command == "merge-search":
        command_merge_search(config, output, args.device, task=args.task, length=args.length)
    elif args.command == "validate":
        command_validate(config, output, args.device, task=args.task, length=args.length)
    elif args.command == "freeze":
        command_freeze(config, output)
    elif args.command == "test":
        _encode_sealed_phase(
            config,
            output,
            args.device,
            phase="test",
            trigger_role="test_trigger",
            benign_role="test_benign_probe",
        )
    elif args.command == "ood":
        _encode_sealed_phase(
            config,
            output,
            args.device,
            phase="ood",
            trigger_role="ood_trigger",
            benign_role="ood_benign_probe",
        )
    elif args.command == "retrieval":
        command_retrieval(config, output, args.device)
    elif args.command == "finalize":
        command_finalize(config, output)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
