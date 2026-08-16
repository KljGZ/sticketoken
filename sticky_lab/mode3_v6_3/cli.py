"""Fail-closed command line for the V6.3 light experiment."""

from __future__ import annotations

import argparse
import csv
import copy
import json
from pathlib import Path
import shutil
import subprocess
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

from .budget import BudgetLedger, registered_budget
from .cache import CallRegistry, CallSpace, EmbeddingCache
from .config import (
    assert_output_leaf,
    canonical_sha256,
    config_for_profile,
    load_config,
    resolved_config,
    sha256_file,
)
from .confirm import confirm_fixed_cap, paired_position_audit
from .data_contract import register_v63_roles, required_unique_capacity
from .encoding import (
    CachedEncoder,
    EncodingRequest,
    FinalEmbeddingOracle,
    INSERTION_PROTOCOL,
    build_call_space,
)
from .errors import CandidateRejected, ProtocolViolation
from .freeze import (
    load_freeze,
    select_primary_and_secondaries,
    write_freeze,
)
from .followups import single_poison_retrieval, summarize_replications
from .funnel import (
    EVALUATION_STAGES,
    STAGES,
    assigned_positions,
    clean_precompute_requests,
    fit_and_score_candidate,
    position_manifest,
)
from .geometry import FrozenCap
from .ranking import select_stage
from .report import (
    atomic_json,
    final_status,
    read_jsonl,
    write_jsonl,
    write_parquet,
)
from .roles import (
    bind_role,
    build_role_manifest,
    records_sha256,
    validate_nested_search_chains,
)
from .sealing import (
    assert_still_sealed,
    build_sealed_inventory,
    grant_access,
    physically_seal,
)
from .semantic_controls import (
    evaluate_semantic_controls,
    select_matched_controls,
    token_frequency_statistics,
)
from .status import summarize
from .tokenizer_audit import (
    audit_candidate,
    prepare_contexts,
    shard_candidates,
    standalone_candidates,
    tokenizer_sha256,
)


DEFAULT_CONFIG = "configs/v6_3_mode3_light.yaml"
DEFAULT_OUTPUT = "results/sticky_lab/sentence_t5_base/mode3_v6_3_light"


def _config(args: argparse.Namespace) -> dict[str, Any]:
    return config_for_profile(load_config(Path(args.config)), str(args.profile))


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_clean() -> bool:
    return subprocess.check_output(["git", "status", "--porcelain"], text=True).strip() == ""


def _tokenizer(config: Mapping[str, Any]) -> Any:
    from transformers import AutoTokenizer

    model = config["model"]
    local = Path(str(model["local_path"]))
    source = str(local) if local.is_dir() else str(model["id"])
    return AutoTokenizer.from_pretrained(
        source,
        revision=None if local.is_dir() else str(model["revision"]),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        use_fast=True,
    )


def _verify_bound_file(path: Path, expected: str) -> None:
    if not path.is_file() or sha256_file(path) != str(expected):
        raise ProtocolViolation(f"bound resource mismatch: {path}")


def _write_role(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_jsonl(path, [dict(row) for row in rows])


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    if not values:
        raise ProtocolViolation(f"refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
        handle.flush()
    temporary.replace(path)


def _role_path(output: Path, role: str) -> Path:
    return output / "registration" / "roles" / f"{role}.jsonl"


def _view_path(output: Path, stage: str, chain: str) -> Path:
    return output / "registration" / "views" / f"{stage}_{chain}.jsonl"


def _load_role(output: Path, role: str) -> list[dict[str, Any]]:
    return read_jsonl(_role_path(output, role))


def _load_view(output: Path, stage: str, chain: str) -> list[dict[str, Any]]:
    return read_jsonl(_view_path(output, stage, chain))


def _sealed_inventory(output: Path) -> dict[str, Any]:
    return json.loads((output / "sealed" / "SEALED_INVENTORY.json").read_text(encoding="utf-8"))


def _load_sealed_role(output: Path, role: str) -> list[dict[str, Any]]:
    inventory = _sealed_inventory(output)
    path = Path(inventory["files"][role]["path"])
    return read_jsonl(path)


def _profile_seed(config: Mapping[str, Any]) -> int:
    profile = str(config.get("run_profile", "formal"))
    return int(config["positions"]["seed"]) + {"formal": 0, "dry_run": 100_000, "pilot": 200_000}[profile]


def command_prepare(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    from sticky_lab.mode3_v6.data import (
        audit_csv_corpus,
        load_registered_records,
        require_formal_capacity,
    )

    output = Path(args.output).resolve()
    assert_output_leaf(output)
    complete = output / "registration" / "COMPLETE.json"
    if complete.is_file():
        registered = json.loads(complete.read_text(encoding="utf-8"))
        if registered.get("profile") != config["run_profile"]:
            raise ProtocolViolation("existing registration belongs to another profile")
        return
    output.mkdir(parents=True, exist_ok=True)
    _verify_bound_file(Path(config["data"]["corpus_manifest"]), str(config["data"]["corpus_manifest_sha256"]))
    _verify_bound_file(Path(config["data"]["independent_corpus_audit"]), str(config["data"]["independent_corpus_audit_sha256"]))
    _verify_bound_file(Path(config["model"]["checksum_manifest"]), str(config["model"]["checksum_manifest_sha256"]))
    disk = shutil.disk_usage(output.parent)
    required_disk = int(float(config["resources"]["minimum_free_disk_peak_multiplier"]) * int(config["resources"]["estimated_peak_cache_bytes"]))
    if disk.free < required_disk:
        atomic_json(output / "FINAL_STATUS.json", {
            "status": "BLOCKED_STORAGE_PREFLIGHT", "free_bytes": disk.free,
            "required_bytes": required_disk,
        })
        raise ProtocolViolation(f"storage preflight: {disk.free} < {required_disk}")
    capacity = audit_csv_corpus(
        str(config["data"]["input_glob"]), list(config["data"]["required_columns"]),
        required_unique_capacity(config),
        int(config["data"]["minimum_iid_sources"]) + int(config["data"]["minimum_ood_sources"]),
    )
    atomic_json(output / "registration" / "data_capacity_audit.json", capacity.to_dict())
    require_formal_capacity(capacity)
    records = load_registered_records(
        str(config["data"]["input_glob"]), list(config["data"]["required_columns"])
    )
    roles, views, allocation = register_v63_roles(
        records, config, seed=_profile_seed(config)
    )
    validate_nested_search_chains(views, config["data"]["search_chain_sizes"])
    discovery_roles = ("fit", "radius", "score", "discovery_benign")
    for role in discovery_roles:
        _write_role(_role_path(output, role), roles[role])
    for stage in STAGES:
        for chain in ("fit", "radius", "score"):
            _write_role(_view_path(output, stage, chain), views[stage][chain])
    sealed_dir = output.parent / f".{output.name}-{config['run_profile']}-sealed-roles"
    if sealed_dir.exists() and any(sealed_dir.iterdir()):
        raise ProtocolViolation(f"sealed target already contains files: {sealed_dir}")
    sealed_dir.mkdir(parents=True, exist_ok=True)
    sealed_names = [role for role in roles if role not in discovery_roles]
    sealed_paths: dict[str, Path] = {}
    for role in sorted(sealed_names):
        path = sealed_dir / f"{role}.jsonl"
        _write_role(path, roles[role])
        sealed_paths[role] = path
    nested = {
        stage: {
            chain: {
                "count": len(views[stage][chain]),
                "records_sha256": records_sha256(views[stage][chain]),
            }
            for chain in ("fit", "radius", "score")
        }
        for stage in STAGES
    }
    bindings = [bind_role(role, "discovery", roles[role]) for role in discovery_roles]
    bindings.extend(bind_role(role, "sealed", roles[role]) for role in sealed_names)
    role_manifest = build_role_manifest(bindings, nested)
    atomic_json(output / "data_role_manifest.json", role_manifest)
    atomic_json(output / "registration" / "allocation_audit.json", allocation)
    inventory = build_sealed_inventory(
        output, sealed_paths, role_manifest_sha256=str(role_manifest["manifest_sha256"])
    )
    tokenizer = _tokenizer(config)
    observed_tokenizer_hash = tokenizer_sha256(tokenizer)
    if observed_tokenizer_hash != str(config["model"]["tokenizer_sha256"]):
        raise ProtocolViolation("tokenizer hash differs from frozen config")
    call_space = build_call_space(
        tokenizer,
        {role: roles[role] for role in discovery_roles},
        config,
        trigger_roles=("fit", "radius", "score"),
    )
    call_space.write(output / "registration" / "call_space.jsonl")
    positions = position_manifest(views, seed=int(config["positions"]["seed"]))
    atomic_json(output / "position_manifest.json", positions)
    random_rows = [
        {
            "ordinal": entry.ordinal, "role": entry.key.role,
            "text_id": entry.key.text_id, "boundary_id": entry.key.random_boundary_id,
        }
        for entry in call_space.entries if entry.key.position == "random"
    ]
    write_jsonl(output / "random_boundary_manifest.jsonl", random_rows)
    plan = registered_budget(config, int(config["tokenizer"]["expected_legal_vocab_for_budget"]))
    atomic_json(output / "budget" / "planned.json", plan)
    resolved = resolved_config(config, source_path=Path(args.config))
    atomic_json(output / "resolved_config.json", resolved)
    run_manifest = {
        "schema_version": "mode3-v6-3-run-manifest-v1",
        "experiment_name": config["experiment_name"],
        "profile": config["run_profile"],
        "scientific_claims_allowed": config["run_profile"] == "formal",
        "code_commit": _git_commit(), "worktree_clean_at_registration": _git_clean(),
        "config_sha256": canonical_sha256(config),
        "source_config_file_sha256": sha256_file(Path(args.config)),
        "model_revision": config["model"]["revision"],
        "tokenizer_sha256": observed_tokenizer_hash,
        "role_manifest_sha256": role_manifest["manifest_sha256"],
        "call_space_sha256": call_space.manifest_sha256,
        "sealed_inventory_sha256": inventory["inventory_sha256"],
        "allowed_physical_gpus": list(config["resources"]["allowed_physical_gpus"]),
        "forbidden_physical_gpus": list(config["resources"]["forbidden_physical_gpus"]),
        "state": "PROTOCOL_LOCKED",
    }
    atomic_json(output / "run_manifest.json", run_manifest)
    physically_seal(list(sealed_paths.values()))
    atomic_json(complete, {
        "schema_version": "mode3-v6-3-registration-complete-v1",
        "status": "DATA_PREFLIGHT_PASSED", "profile": config["run_profile"],
        "role_manifest_sha256": role_manifest["manifest_sha256"],
        "call_space_sha256": call_space.manifest_sha256,
        "sealed_roles_encoded": False, "confirm_tokenizer_accessed": False,
    })


def _audit_contexts(output: Path, tokenizer: Any, config: Mapping[str, Any]) -> tuple[Any, ...]:
    rows: list[dict[str, Any]] = []
    for chain in ("fit", "radius", "score"):
        rows.extend(_load_view(output, "s0", chain))
    return prepare_contexts(
        tokenizer, rows,
        maximum_length=int(config["model"]["maximum_sequence_length"]),
        required=int(config["tokenizer"]["contextual_audit_samples"]),
    )


def command_enumerate(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output).resolve()
    tokenizer = _tokenizer(config)
    contexts = _audit_contexts(output, tokenizer, config)
    candidates = standalone_candidates(tokenizer)
    if config["run_profile"] != "formal" and int(args.shards) != 1:
        raise ProtocolViolation("engineering enumeration uses one shard for an exact legal-token limit")
    candidates = shard_candidates(candidates, int(args.shard), int(args.shards))
    legal = []
    audits = []
    limit = None if config["run_profile"] == "formal" else int(config["tokenizer"]["engineering_legal_token_limit"])
    for token_id, token_text in candidates:
        item, audit = audit_candidate(
            tokenizer, token_id, token_text, contexts,
            seed=int(config["positions"]["seed"]),
            batch_size=int(config["model"]["batch_size"]) * 4,
        )
        audits.append(audit)
        if item is not None:
            legal.append(item.to_dict())
            if limit is not None and len(legal) >= limit:
                break
    target = output / "enumeration" / f"shard_{int(args.shard):02d}"
    write_jsonl(target / "legal_tokens.jsonl", legal)
    write_jsonl(target / "tokenizer_audit.jsonl", audits)
    atomic_json(target / "COMPLETE.json", {
        "schema_version": "mode3-v6-3-enumeration-shard-v1",
        "shard": int(args.shard), "shards": int(args.shards),
        "standalone_candidates_examined": len(audits), "legal_tokens": len(legal),
        "formal_exhaustive_shard": config["run_profile"] == "formal",
        "tokenizer_sha256": tokenizer_sha256(tokenizer),
    })


def command_merge_enumeration(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output).resolve()
    legal: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    completes = []
    for shard in range(int(args.shards)):
        target = output / "enumeration" / f"shard_{shard:02d}"
        completes.append(json.loads((target / "COMPLETE.json").read_text(encoding="utf-8")))
        legal.extend(read_jsonl(target / "legal_tokens.jsonl"))
        audits.extend(read_jsonl(target / "tokenizer_audit.jsonl"))
    if any(row["shards"] != int(args.shards) or row["tokenizer_sha256"] != completes[0]["tokenizer_sha256"] for row in completes):
        raise ProtocolViolation("enumeration shard mismatch")
    legal.sort(key=lambda row: int(row["token_id"]))
    if len({int(row["token_id"]) for row in legal}) != len(legal):
        raise ProtocolViolation("enumeration merged duplicate legal tokens")
    if config["run_profile"] == "formal":
        tokenizer = _tokenizer(config)
        expected = {token_id for token_id, _ in standalone_candidates(tokenizer)}
        observed = {int(row["token_id"]) for row in audits}
        if observed != expected:
            raise ProtocolViolation(f"formal tokenizer enumeration incomplete: {len(observed)}/{len(expected)}")
    required = int(config["funnel"]["s0_keep"])
    if len(legal) < required:
        raise ProtocolViolation(f"legal vocabulary {len(legal)} is below registered S0 retention {required}")
    write_jsonl(output / "legal_tokens.jsonl", legal)
    write_jsonl(output / "tokenizer_audit.jsonl", sorted(audits, key=lambda row: int(row["token_id"])))
    atomic_json(output / "enumeration" / "COMPLETE.json", {
        "schema_version": "mode3-v6-3-enumeration-complete-v1",
        "legal_tokens": len(legal), "actual_tokenizer_length": 1,
        "formal_exhaustive": config["run_profile"] == "formal",
        "all_legal_tokens_enter_s0": True,
        "tokenizer_sha256": completes[0]["tokenizer_sha256"],
    })
    atomic_json(output / "budget" / "planned_actual_vocab.json", registered_budget(config, len(legal)))


def _runtime(
    output: Path, config: Mapping[str, Any], physical_gpu: int,
    *, root: Path | None = None, call_space_path: Path | None = None,
) -> tuple[CachedEncoder, CallSpace]:
    call_space = CallSpace.read(call_space_path or (output / "registration" / "call_space.jsonl"))
    runtime_root = root or output
    ledger = BudgetLedger(output, config["budget"])
    registry = CallRegistry(runtime_root, call_space, ledger)
    cache = EmbeddingCache(runtime_root, call_space)
    oracle = FinalEmbeddingOracle(
        config, physical_gpu=int(physical_gpu),
        expected_tokenizer_hash=str(config["model"]["tokenizer_sha256"]),
    )
    return CachedEncoder(config, call_space, registry, cache, oracle), call_space


def command_precompute_clean(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output).resolve()
    marker = output / "registration" / "CLEAN_BASE_COMPLETE.json"
    if marker.is_file():
        return
    encoder, _ = _runtime(output, config, int(args.physical_gpu))
    score = _load_view(output, "full", "score")
    benign = _load_role(output, "discovery_benign")
    requests = clean_precompute_requests(score, benign)
    _, _, cache = encoder.encode_requests(
        token_id=-2, token_text="", requests=requests, phase="discovery:clean_base"
    )
    atomic_json(marker, {
        "schema_version": "mode3-v6-3-clean-base-v1",
        "score_records": len(score), "benign_records": len(benign),
        "cache": cache, "sealed_roles_encoded": False,
    })


def _candidate_tokens(output: Path, stage: str) -> list[dict[str, Any]]:
    legal = {int(row["token_id"]): row for row in read_jsonl(output / "legal_tokens.jsonl")}
    if stage == "s0":
        ids = sorted(legal)
    else:
        previous = {"s1": "s0", "s2": "s1", "full": "s2", "top100": "full"}[stage]
        ids = list(map(int, json.loads((output / "stages" / previous / "selected.json").read_text(encoding="utf-8"))["token_ids"]))
    return [legal[token_id] for token_id in ids]


def _previous_caps(output: Path, stage: str) -> dict[int, FrozenCap]:
    if stage == "s0":
        return {}
    previous = {"s1": "s0", "s2": "s1", "full": "s2", "top100": "full"}[stage]
    path = output / "stages" / previous / "selected_models.jsonl"
    return {int(row["token_id"]): FrozenCap.from_dict(row["cap"]) for row in read_jsonl(path)}


def _compact_audit(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    center = result["fit"]["center_fit"]
    indices = center.pop("inlier_indices", {})
    center["inlier_counts"] = {key: len(item) for key, item in indices.items()}
    center["inlier_indices_sha256"] = canonical_sha256(indices)
    return result


def command_stage_shard(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output).resolve()
    stage = str(args.stage)
    target = output / "stages" / stage / f"shard_{int(args.shard):02d}"
    if (target / "COMPLETE.json").is_file():
        return
    target.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_tokens(output, stage)
    candidates = [row for index, row in enumerate(candidates) if index % int(args.shards) == int(args.shard)]
    data_stage = "full" if stage == "top100" else stage
    fit = _load_view(output, data_stage, "fit")
    radius = _load_view(output, data_stage, "radius")
    score = _load_view(output, data_stage, "score")
    benign = _load_role(output, "discovery_benign")
    role_hashes = {"fit": records_sha256(fit), "radius": records_sha256(radius), "score": records_sha256(score)}
    previous = _previous_caps(output, stage)
    metrics: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    try:
        encoder, _ = _runtime(output, config, int(args.physical_gpu))
        for candidate in candidates:
            token_id = int(candidate["token_id"])
            try:
                cap, metric, audit = fit_and_score_candidate(
                    encoder, token_id=token_id, token_text=str(candidate["token_text"]),
                    stage=stage, fit_records=fit, radius_records=radius,
                    score_records=score, benign_records=benign,
                    role_hashes=role_hashes, config=config,
                    previous_cap=previous.get(token_id),
                )
            except CandidateRejected as error:
                rejections.append({
                    "token_id": token_id, "token_text": str(candidate["token_text"]),
                    "stage": stage, "status": "candidate_rejected",
                    "reason": type(error).__name__, "detail": str(error),
                })
                continue
            metrics.append(metric.to_dict())
            models.append({"token_id": token_id, "stage": stage, "cap": cap.to_dict()})
            audits.append(_compact_audit(audit))
        write_jsonl(target / "metrics.jsonl", metrics)
        write_jsonl(target / "models.jsonl", models)
        write_jsonl(target / "audits.jsonl", audits)
        write_jsonl(target / "rejections.jsonl", rejections)
        atomic_json(target / "COMPLETE.json", {
            "schema_version": "mode3-v6-3-stage-shard-v1",
            "stage": stage, "shard": int(args.shard), "shards": int(args.shards),
            "candidate_tokens": [int(row["token_id"]) for row in candidates],
            "valid": len(metrics), "rejected": len(rejections),
            "from_scratch_refit": True, "single_cap_only": True,
            "physical_gpu": int(args.physical_gpu),
        })
    except Exception as error:
        atomic_json(target / "FAILED.json", {
            "schema_version": "mode3-v6-3-stage-failed-v1",
            "stage": stage, "shard": int(args.shard),
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(), "partial_merge_allowed": False,
        })
        raise


def command_merge_stage(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output).resolve()
    stage = str(args.stage)
    target = output / "stages" / stage
    if (target / "COMPLETE.json").is_file():
        return
    metrics: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    candidate_ids: list[int] = []
    for shard in range(int(args.shards)):
        root = target / f"shard_{shard:02d}"
        if (root / "FAILED.json").exists() or not (root / "COMPLETE.json").is_file():
            raise ProtocolViolation(f"cannot merge incomplete/failed {stage} shard {shard}")
        complete = json.loads((root / "COMPLETE.json").read_text(encoding="utf-8"))
        if complete["stage"] != stage or int(complete["shards"]) != int(args.shards):
            raise ProtocolViolation(f"shard identity mismatch {stage}/{shard}")
        candidate_ids.extend(map(int, complete["candidate_tokens"]))
        metrics.extend(read_jsonl(root / "metrics.jsonl"))
        models.extend(read_jsonl(root / "models.jsonl"))
        rejections.extend(read_jsonl(root / "rejections.jsonl"))
    expected = {int(row["token_id"]) for row in _candidate_tokens(output, stage)}
    if set(candidate_ids) != expected or len(candidate_ids) != len(expected):
        raise ProtocolViolation(f"{stage} shard coverage mismatch")
    observed = {int(row["token_id"]) for row in metrics}.union(int(row["token_id"]) for row in rejections)
    if observed != expected:
        raise ProtocolViolation(f"{stage} result coverage mismatch")
    if stage == "top100":
        selected = sorted(int(row["token_id"]) for row in metrics)
        selection_audit = {"method": "complete_position_freeze_input", "selected": selected}
        if rejections:
            raise ProtocolViolation("a top100 complete-position candidate was rejected")
    else:
        keep = {
            "s0": int(config["funnel"]["s0_keep"]),
            "s1": int(config["funnel"]["s1_keep"]),
            "s2": int(config["funnel"]["s2_keep"]),
            "full": int(config["funnel"]["full_top"]),
        }[stage]
        selected, selection_audit = select_stage(
            metrics, keep, seed=int(config["positions"]["seed"]) + EVALUATION_STAGES.index(stage)
        )
    model_index = {int(row["token_id"]): row for row in models}
    selected_models = [model_index[token_id] for token_id in selected]
    write_jsonl(target / "all_metrics.jsonl", metrics)
    write_jsonl(target / "all_rejections.jsonl", rejections)
    write_jsonl(target / "selected_models.jsonl", selected_models)
    atomic_json(target / "selection_audit.json", selection_audit)
    atomic_json(target / "selected.json", {"token_ids": selected})
    required_name = {
        "s0": "stage_s0_all_metrics.parquet",
        "s1": "stage_s1_all_metrics.parquet",
        "s2": "stage_s2_all_metrics.parquet",
        "full": "full_5000_all_metrics.parquet",
        "top100": "top100_complete_metrics.parquet",
    }[stage]
    write_parquet(output / required_name, metrics)
    atomic_json(target / "COMPLETE.json", {
        "schema_version": "mode3-v6-3-stage-complete-v1",
        "stage": stage, "candidate_tokens": len(expected),
        "valid": len(metrics), "rejected": len(rejections),
        "selected_tokens": len(selected), "from_scratch_refit": True,
        "single_cap_only": True,
    })


def command_freeze(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output).resolve()
    assert_still_sealed(output)
    metrics = read_jsonl(output / "stages" / "top100" / "all_metrics.jsonl")
    models = read_jsonl(output / "stages" / "top100" / "selected_models.jsonl")
    expected = int(config["funnel"]["full_top"])
    primary, secondaries = select_primary_and_secondaries(
        metrics, maximum_radius_degrees=float(config["geometry"]["maximum_radius_degrees"]),
        expected_count=expected,
    )
    caps = {int(row["token_id"]): FrozenCap.from_dict(row["cap"]) for row in models}
    role_manifest = json.loads((output / "data_role_manifest.json").read_text(encoding="utf-8"))
    run = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    discovery = {
        role: str(role_manifest["bindings"][role]["records_sha256"])
        for role in ("fit", "radius", "score", "discovery_benign")
    }
    confirm = {
        role: str(binding["records_sha256"])
        for role, binding in role_manifest["bindings"].items()
        if binding["role_class"] == "sealed"
    }
    thresholds = {
        **{key: float(config["certification"][key]) for key in (
            "balanced_coverage_lcb", "worst_position_lcb", "worst_source_lcb",
            "independent_benign_core_ucb", "outside_to_inside_lcb",
            "conditional_outside_origin_lcb", "moat_occupancy_1_10_ucb",
            "basin_lambda_star", "basin_occupancy_auc_1_1_5",
            "central_collapse_median_depth",
        )},
        "maximum_radius_degrees": float(config["geometry"]["maximum_radius_degrees"]),
    }
    fit_rows = _load_view(output, "full", "fit")
    sources = sorted({str(row["source_id"]) for row in fit_rows})
    source_weights = {source: 1.0 / len(sources) for source in sources}
    position_weights = {position: 1.0 / 3.0 for position in ("prefix", "suffix", "random")}
    primary_path, digest = write_freeze(
        output, primary_metric=primary, secondary_metrics=secondaries, caps=caps,
        metadata={
            "maximum_radius_degrees": config["geometry"]["maximum_radius_degrees"],
            "code_commit": run["code_commit"], "config_sha256": run["config_sha256"],
            "role_manifest_sha256": run["role_manifest_sha256"],
            "discovery_role_hashes": discovery, "confirm_role_hashes": confirm,
            "tokenizer_sha256": run["tokenizer_sha256"],
            "model_revision": run["model_revision"],
            "call_space_sha256": run["call_space_sha256"],
            "certification_thresholds": thresholds,
            "source_weights": source_weights,
            "position_weights": position_weights,
            "random_boundary_manifest_sha256": sha256_file(
                output / "random_boundary_manifest.jsonl"
            ),
            "pretruncation_protocol_sha256": canonical_sha256(INSERTION_PROTOCOL),
        },
    )
    shutil.copyfile(primary_path, output / "primary_freeze.json")
    shutil.copyfile(output / "freeze" / "secondary.jsonl", output / "secondary_freeze.jsonl")
    atomic_json(output / "orchestration_logs" / "freeze_status.json", {
        "status": "PRIMARY_FROZEN", "freeze_sha256": digest,
        "confirm_still_sealed": True,
    })


def command_grant_confirm(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output).resolve()
    run = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    grant_access(
        output, role_manifest_sha256=str(run["role_manifest_sha256"]),
        sealed_inventory_path=output / "sealed" / "SEALED_INVENTORY.json",
    )


def command_prepare_confirm(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output).resolve()
    marker = output / "confirm_runtime" / "PREPARED.json"
    if marker.is_file():
        return
    run = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    if not (output / "sealed" / "SEALED_ACCESS_GRANT.json").is_file():
        raise ProtocolViolation("confirm preparation requires sealed access grant")
    inventory = _sealed_inventory(output)
    roles = {role: _load_sealed_role(output, role) for role in inventory["files"]}
    tokenizer = _tokenizer(config)
    trigger_roles = [
        role for role in roles
        if role.endswith("_trigger") or role in {
            "confirm_trigger", "paired_position_audit", "semantic_control",
            "iid_replication_0", "iid_replication_1", "iid_replication_2",
            "retrieval_probe",
        }
    ]
    call_space = build_call_space(
        tokenizer, roles, config, trigger_roles=tuple(trigger_roles)
    )
    path = output / "confirm_runtime" / "call_space.jsonl"
    call_space.write(path)
    atomic_json(marker, {
        "schema_version": "mode3-v6-3-confirm-runtime-v1",
        "freeze_sha256": (output / "freeze" / "FREEZE.sha256").read_text(encoding="utf-8").split()[0],
        "role_manifest_sha256": run["role_manifest_sha256"],
        "call_space_sha256": call_space.manifest_sha256,
        "refit_functions_imported": False,
    })


def _confirm_runtime(output: Path, config: Mapping[str, Any], physical_gpu: int) -> CachedEncoder:
    encoder, _ = _runtime(
        output, config, physical_gpu,
        root=output / "confirm_runtime",
        call_space_path=output / "confirm_runtime" / "call_space.jsonl",
    )
    return encoder


def command_precompute_confirm_clean(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output).resolve()
    marker = output / "confirm_runtime" / "CLEAN_BASE_COMPLETE.json"
    if marker.is_file():
        return
    encoder = _confirm_runtime(output, config, int(args.physical_gpu))
    trigger = _load_sealed_role(output, "confirm_trigger")
    benign = _load_sealed_role(output, "confirm_benign")
    requests = [EncodingRequest("confirm_trigger", row, "clean", 0) for row in trigger]
    requests += [EncodingRequest("confirm_benign", row, "clean", 0) for row in benign]
    _, _, cache = encoder.encode_requests(
        token_id=-2, token_text="", requests=requests, phase="confirm:clean_base"
    )
    atomic_json(marker, {"trigger": len(trigger), "benign": len(benign), "cache": cache})


def _cached_role_clean(
    encoder: CachedEncoder, role: str, rows: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    entries = [encoder.call_space.lookup_request(role, str(row["text_id"]), "clean") for row in rows]
    found, missing = encoder.cache.fetch(-2, [entry.ordinal for entry in entries])
    if missing:
        raise ProtocolViolation(f"confirm clean cache missing {len(missing)} calls for {role}")
    return np.stack([found[entry.ordinal] for entry in entries])


def command_confirm(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output).resolve()
    marker = output / "confirm" / "COMPLETE.json"
    if marker.is_file():
        return
    freeze_sha = (output / "freeze" / "FREEZE.sha256").read_text(encoding="utf-8").split()[0]
    artifact = load_freeze(output / "freeze" / "primary.json", freeze_sha)
    encoder = _confirm_runtime(output, config, int(args.physical_gpu))
    trigger = _load_sealed_role(output, "confirm_trigger")
    benign = _load_sealed_role(output, "confirm_benign")
    seed = int(config["positions"]["seed"])
    trigger_rows: list[dict[str, Any]] = []
    requests: list[EncodingRequest] = []
    for row in trigger:
        position = assigned_positions(row, "s0", seed=seed)[0]
        registered = dict(row)
        registered["position"] = position
        trigger_rows.append(registered)
        requests.append(EncodingRequest("confirm_trigger", row, position, 0))
    triggered, audits, cache = encoder.encode_requests(
        token_id=artifact.token_id, token_text=str(artifact.cap["token_text"]),
        requests=requests, phase="confirm:primary_trigger",
    )
    clean = _cached_role_clean(encoder, "confirm_trigger", trigger)
    benign_vectors = _cached_role_clean(encoder, "confirm_benign", benign)
    result = confirm_fixed_cap(
        artifact, trigger_rows=trigger_rows, triggered_vectors=triggered,
        paired_clean_vectors=clean, benign_rows=benign, benign_vectors=benign_vectors,
        observed_role_hashes={
            "confirm_trigger": records_sha256(trigger),
            "confirm_benign": records_sha256(benign),
        },
        freeze_sha256=freeze_sha,
        radial_multipliers=config["certification"]["radial_multipliers"],
        familywise_alpha=float(config["certification"]["familywise_alpha"]),
    )
    atomic_json(output / "confirm" / "primary_certificate.json", {
        key: value for key, value in result.items()
        if key not in {"observations", "source_position_intervals", "benign_depth_by_source"}
    })
    atomic_json(output / "confirm_certificate.json", {
        key: value for key, value in result.items()
        if key not in {"observations", "source_position_intervals", "benign_depth_by_source"}
    })
    _write_csv(output / "source_position_intervals.csv", result["source_position_intervals"])
    write_parquet(output / "confirm" / "migration.parquet", result["observations"])
    _write_csv(output / "radial_occupancy_curve.csv", result["radial_occupancy"]["curve"])
    write_jsonl(output / "confirm" / "runtime_realization_audit.jsonl", [audit.to_dict() for audit in audits])

    paired_rows = _load_sealed_role(output, "paired_position_audit")
    vectors_by_position: dict[str, np.ndarray] = {}
    paired_observations: list[dict[str, Any]] = []
    for position in ("prefix", "suffix", "random"):
        values, paired_audits, _ = encoder.encode_requests(
            token_id=artifact.token_id, token_text=str(artifact.cap["token_text"]),
            requests=[EncodingRequest("paired_position_audit", row, position, 0) for row in paired_rows],
            phase=f"confirm:paired:{position}",
        )
        vectors_by_position[position] = values
        inside = artifact.frozen_cap().contains(values)
        paired_observations.extend({
            "text_id": str(row["text_id"]), "source_id": str(row["source_id"]),
            "position": position, "inside": bool(member),
        } for row, member in zip(paired_rows, inside))
        write_jsonl(output / "confirm" / f"paired_{position}_realization.jsonl", [audit.to_dict() for audit in paired_audits])
    paired_summary = paired_position_audit(artifact, paired_rows, vectors_by_position)
    atomic_json(output / "confirm" / "paired_position_audit.json", paired_summary)
    write_parquet(output / "paired_position_audit.parquet", paired_observations)
    atomic_json(marker, {
        "schema_version": "mode3-v6-3-confirm-complete-v1",
        "status": "CERTIFIED_ST_FCA_CORE" if result["levels"]["B_ST_FCA_CORE"] else "VALID_PRIMARY_NOT_CERTIFIED",
        "freeze_sha256": freeze_sha, "refit_performed": False,
        "independent_text_units": len(trigger), "cache": cache,
    })
    if not result["levels"]["B_ST_FCA_CORE"]:
        for name in ("semantic_controls", "iid_summary", "ood_summary", "retrieval_summary"):
            atomic_json(output / f"{name}.json", {
                "status": "SKIPPED", "reason": "independent Core gate closed",
                "search_feedback": False,
            })
        atomic_json(output / "FINAL_STATUS.json", final_status(result, profile=str(config["run_profile"])))


def _fixed_role_confirmation(
    output: Path,
    config: Mapping[str, Any],
    encoder: CachedEncoder,
    artifact: Any,
    freeze_sha: str,
    *,
    trigger_role: str,
    benign_role: str,
    phase: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trigger = _load_sealed_role(output, trigger_role)
    benign = _load_sealed_role(output, benign_role)
    seed = int(config["positions"]["seed"])
    trigger_rows: list[dict[str, Any]] = []
    trigger_requests: list[EncodingRequest] = []
    for row in trigger:
        position = assigned_positions(row, "s0", seed=seed)[0]
        registered = dict(row)
        registered["position"] = position
        trigger_rows.append(registered)
        trigger_requests.append(EncodingRequest(trigger_role, row, position, 0))
    triggered, trigger_audits, _ = encoder.encode_requests(
        token_id=artifact.token_id,
        token_text=str(artifact.cap["token_text"]),
        requests=trigger_requests,
        phase=f"{phase}:trigger",
    )
    clean, _, _ = encoder.encode_requests(
        token_id=-2,
        token_text="",
        requests=[EncodingRequest(trigger_role, row, "clean", 0) for row in trigger],
        phase=f"{phase}:paired_clean",
    )
    benign_vectors, _, _ = encoder.encode_requests(
        token_id=-2,
        token_text="",
        requests=[EncodingRequest(benign_role, row, "clean", 0) for row in benign],
        phase=f"{phase}:benign",
    )
    result = confirm_fixed_cap(
        artifact,
        trigger_rows=trigger_rows,
        triggered_vectors=triggered,
        paired_clean_vectors=clean,
        benign_rows=benign,
        benign_vectors=benign_vectors,
        observed_role_hashes={
            trigger_role: records_sha256(trigger),
            benign_role: records_sha256(benign),
        },
        freeze_sha256=freeze_sha,
        radial_multipliers=config["certification"]["radial_multipliers"],
        familywise_alpha=float(config["certification"]["familywise_alpha"]),
    )
    return result, [audit.to_dict() for audit in trigger_audits]


def _certificate_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in result.items()
        if key not in {"observations", "source_position_intervals", "benign_depth_by_source"}
    }


def _semantic_followup(
    output: Path,
    config: Mapping[str, Any],
    encoder: CachedEncoder,
    artifact: Any,
    confirmation: Mapping[str, Any],
) -> dict[str, Any]:
    rows = _load_sealed_role(output, "semantic_control")
    legal = read_jsonl(output / "legal_tokens.jsonl")
    tokenizer = encoder.oracle.tokenizer
    discovery = _load_view(output, "full", "fit")
    frequency = token_frequency_statistics(
        tokenizer, discovery, [int(row["token_id"]) for row in legal]
    )
    needed = min(
        int(config["followups"]["semantic_matched_controls"]),
        len(legal) - 1,
    )
    if needed < 20:
        raise ProtocolViolation("semantic follow-up requires at least 20 legal matched controls")
    pool_size = min(len(legal) - 1, max(needed * 3, needed))
    pool, matching_audit = select_matched_controls(
        candidate_id=artifact.token_id,
        candidate_text=str(artifact.cap["token_text"]),
        legal_tokens=legal,
        frequency=frequency,
        count=pool_size,
    )
    seed = int(config["positions"]["seed"])
    assigned = [assigned_positions(row, "s0", seed=seed)[0] for row in rows]
    candidate, candidate_audits, _ = encoder.encode_requests(
        token_id=artifact.token_id,
        token_text=str(artifact.cap["token_text"]),
        requests=[
            EncodingRequest("semantic_control", row, position, 0)
            for row, position in zip(rows, assigned)
        ],
        phase="semantic:candidate",
    )
    cap = artifact.frozen_cap()
    candidate_coverage = float(np.mean(cap.contains(candidate)))
    control_coverages: list[float] = []
    accepted_controls: list[dict[str, Any]] = []
    rejected_controls: list[dict[str, Any]] = []
    for control in pool:
        try:
            vectors, _, _ = encoder.encode_requests(
                token_id=int(control["token_id"]),
                token_text=str(control["token_text"]),
                requests=[
                    EncodingRequest("semantic_control", row, position, 0)
                    for row, position in zip(rows, assigned)
                ],
                phase=f"semantic:control:{int(control['token_id'])}",
            )
        except CandidateRejected as error:
            rejected_controls.append({
                "token_id": int(control["token_id"]),
                "token_text": str(control["token_text"]),
                "reason": type(error).__name__,
                "detail": str(error),
            })
            continue
        coverage = float(np.mean(cap.contains(vectors)))
        control_coverages.append(coverage)
        accepted_controls.append({
            "token_id": int(control["token_id"]),
            "token_text": str(control["token_text"]),
            "coverage": coverage,
        })
        if len(accepted_controls) >= needed:
            break
    if len(accepted_controls) != needed:
        raise ProtocolViolation(
            f"runtime-realizable semantic controls {len(accepted_controls)}/{needed}"
        )
    wrapper_coverages: dict[str, float] = {}
    wrapper_audits: list[dict[str, Any]] = []
    for position in ("prefix", "suffix", "random"):
        vectors, audits, _ = encoder.encode_requests(
            token_id=artifact.token_id,
            token_text=str(artifact.cap["token_text"]),
            requests=[EncodingRequest("semantic_control", row, position, 0) for row in rows],
            phase=f"semantic:wrapper:{position}",
        )
        wrapper_coverages[f"one_insertion_{position}"] = float(np.mean(cap.contains(vectors)))
        wrapper_audits.extend(audit.to_dict() for audit in audits)
    result = evaluate_semantic_controls(
        confirmation,
        candidate_coverage=candidate_coverage,
        matched_control_coverages=control_coverages,
        wrapper_coverages=wrapper_coverages,
        minimum_margin=float(config["followups"]["semantic_minimum_margin"]),
        minimum_wrapper_coverage=float(
            config["followups"]["semantic_minimum_wrapper_coverage"]
        ),
    )
    result.update({
        "matching_audit": matching_audit,
        "accepted_control_results": accepted_controls,
        "rejected_control_results": rejected_controls,
        "naturalness_proxy": "discovery document frequency and IDF",
        "pos_semantic_category_proxy": "lexical and Unicode category; no gradients",
        "input_embedding_norm_used": False,
        "input_embedding_norm_reason": "post-Core controls remain pure black-box",
    })
    write_jsonl(
        output / "followups" / "semantic_runtime_realization.jsonl",
        [audit.to_dict() for audit in candidate_audits] + wrapper_audits,
    )
    return result


def _iid_followup(
    output: Path,
    config: Mapping[str, Any],
    encoder: CachedEncoder,
    artifact: Any,
    freeze_sha: str,
    confirmation: Mapping[str, Any],
) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    for index in range(3):
        role = f"iid_replication_{index}"
        result, audits = _fixed_role_confirmation(
            output, config, encoder, artifact, freeze_sha,
            trigger_role=role, benign_role="confirm_benign", phase=f"iid:{index}",
        )
        results[role] = result
        atomic_json(
            output / "followups" / "iid" / f"{role}.json",
            _certificate_summary(result),
        )
        write_jsonl(
            output / "followups" / "iid" / f"{role}_realization.jsonl", audits
        )
    summary = summarize_replications(confirmation, results)
    summary["certificates"] = {
        role: _certificate_summary(result) for role, result in results.items()
    }
    return summary


def _ood_followup(
    output: Path,
    config: Mapping[str, Any],
    encoder: CachedEncoder,
    artifact: Any,
    freeze_sha: str,
) -> dict[str, Any]:
    domains = list(map(str, config["data"]["ood_domains_allowlist"]))
    certificates: dict[str, dict[str, Any]] = {}
    for index, domain in enumerate(domains):
        trigger_role = f"ood_{index}_trigger"
        benign_role = f"ood_{index}_benign"
        result, audits = _fixed_role_confirmation(
            output, config, encoder, artifact, freeze_sha,
            trigger_role=trigger_role, benign_role=benign_role,
            phase=f"ood:{index}:{domain}",
        )
        certificates[domain] = result
        atomic_json(
            output / "followups" / "ood" / f"ood_{index}.json",
            {"domain": domain, **_certificate_summary(result)},
        )
        write_jsonl(
            output / "followups" / "ood" / f"ood_{index}_realization.jsonl", audits
        )
    core = {
        domain: bool(result["levels"]["B_ST_FCA_CORE"])
        for domain, result in certificates.items()
    }
    worst_domain = min(
        domains,
        key=lambda domain: float(certificates[domain]["coverage"]["balanced_lower"]),
    )
    return {
        "schema_version": "mode3-v6-3-ood-summary-v1",
        "domain_core_certified": core,
        "all_ood_core_certified": all(core.values()),
        "worst_domain": worst_domain,
        "certificates": {
            domain: _certificate_summary(result)
            for domain, result in certificates.items()
        },
        "source_isolated": True,
        "search_feedback": False,
        "refit_performed": False,
    }


def _retrieval_followup(
    output: Path,
    config: Mapping[str, Any],
    encoder: CachedEncoder,
    artifact: Any,
    confirmation: Mapping[str, Any],
) -> dict[str, Any]:
    if not bool(confirmation["levels"]["B_ST_FCA_CORE"]):
        raise ProtocolViolation("retrieval requires independent Core")
    confirm_rows = _load_sealed_role(output, "confirm_trigger")
    query_count = min(int(config["followups"]["retrieval_query_count"]), len(confirm_rows))
    query_rows = confirm_rows[:query_count]
    seed = int(config["positions"]["seed"])
    positions = [assigned_positions(row, "s0", seed=seed)[0] for row in query_rows]
    triggered_queries, _, _ = encoder.encode_requests(
        token_id=artifact.token_id,
        token_text=str(artifact.cap["token_text"]),
        requests=[
            EncodingRequest("confirm_trigger", row, position, 0)
            for row, position in zip(query_rows, positions)
        ],
        phase="retrieval:triggered_queries",
    )
    clean_queries, _, _ = encoder.encode_requests(
        token_id=-2,
        token_text="",
        requests=[EncodingRequest("confirm_trigger", row, "clean", 0) for row in query_rows],
        phase="retrieval:clean_queries",
    )
    key_rows = _load_sealed_role(output, "retrieval_probe")
    benign_keys, _, _ = encoder.encode_requests(
        token_id=-2,
        token_text="",
        requests=[EncodingRequest("retrieval_probe", row, "clean", 0) for row in key_rows],
        phase="retrieval:benign_keys",
    )
    anchor = key_rows[0]
    poison_by_position: dict[str, np.ndarray] = {}
    realization: list[dict[str, Any]] = []
    for position in ("prefix", "suffix", "random"):
        vector, audits, _ = encoder.encode_requests(
            token_id=artifact.token_id,
            token_text=str(artifact.cap["token_text"]),
            requests=[EncodingRequest("retrieval_probe", anchor, position, 0)],
            phase=f"retrieval:single_anchor:{position}",
        )
        poison_by_position[position] = vector[0]
        realization.extend(audit.to_dict() for audit in audits)
    sizes = sorted({
        min(int(size), len(benign_keys))
        for size in config["followups"]["retrieval_index_sizes"]
        if min(int(size), len(benign_keys)) > 0
    })
    experiments: dict[str, Any] = {}
    for position, poison in poison_by_position.items():
        experiments[position] = {}
        for size in sizes:
            experiments[position][str(size)] = {
                "triggered_query": single_poison_retrieval(
                    confirmation, triggered_queries, benign_keys[:size], poison
                ),
                "clean_query_control": single_poison_retrieval(
                    confirmation, clean_queries, benign_keys[:size], poison
                ),
            }
    write_jsonl(output / "followups" / "retrieval_realization.jsonl", realization)
    return {
        "schema_version": "mode3-v6-3-retrieval-summary-v1",
        "status": "COMPLETED_DESCRIPTIVE_NO_REGISTERED_BINARY_GATE",
        "queries": query_count,
        "real_text_anchor_id": str(anchor["text_id"]),
        "single_anchor_primary": True,
        "mathematical_center_written_to_index": False,
        "positions": ["prefix", "suffix", "random"],
        "index_sizes": sizes,
        "experiments": experiments,
        "certified_retrieval": False,
        "search_feedback": False,
        "refit_performed": False,
    }


def command_followups(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output).resolve()
    marker = output / "followups" / "COMPLETE.json"
    if marker.is_file():
        return
    confirmation = json.loads(
        (output / "confirm" / "primary_certificate.json").read_text(encoding="utf-8")
    )
    if not bool(confirmation["levels"]["B_ST_FCA_CORE"]):
        raise ProtocolViolation("follow-ups remain closed because independent Core failed")
    freeze_sha = (output / "freeze" / "FREEZE.sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    artifact = load_freeze(output / "freeze" / "primary.json", freeze_sha)
    encoder = _confirm_runtime(output, config, int(args.physical_gpu))
    semantic = _semantic_followup(output, config, encoder, artifact, confirmation)
    atomic_json(output / "semantic_controls.json", semantic)
    iid = _iid_followup(
        output, config, encoder, artifact, freeze_sha, confirmation
    )
    atomic_json(output / "iid_summary.json", iid)
    ood = _ood_followup(output, config, encoder, artifact, freeze_sha)
    atomic_json(output / "ood_summary.json", ood)
    retrieval = _retrieval_followup(
        output, config, encoder, artifact, confirmation
    )
    atomic_json(output / "retrieval_summary.json", retrieval)
    followup_summary = {
        "ST_FCA_ANOMALY": bool(semantic["anomaly_supported"]),
        "IID_REPLICATION": bool(iid["all_core_certified"]),
        "OOD": bool(ood["all_ood_core_certified"]),
        "RETRIEVAL_DESCRIPTIVE_COMPLETE": True,
        "RETRIEVAL_CERTIFIED": bool(retrieval["certified_retrieval"]),
    }
    atomic_json(marker, {
        "schema_version": "mode3-v6-3-followups-complete-v1",
        "status": "COMPLETE",
        "freeze_sha256": freeze_sha,
        "refit_performed": False,
        "search_feedback": False,
        "summary": followup_summary,
    })
    atomic_json(
        output / "FINAL_STATUS.json",
        final_status(
            confirmation,
            profile=str(config["run_profile"]),
            followups=followup_summary,
        ),
    )


def command_status(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    del config
    print(json.dumps(summarize(Path(args.output)), indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--profile", choices=("formal", "dry_run", "pilot"), default="formal")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    enum = sub.add_parser("enumerate")
    enum.add_argument("--shard", type=int, required=True)
    enum.add_argument("--shards", type=int, required=True)
    merge_enum = sub.add_parser("merge-enumeration")
    merge_enum.add_argument("--shards", type=int, required=True)
    for name in ("precompute-clean", "precompute-confirm-clean", "confirm", "followups"):
        command = sub.add_parser(name)
        command.add_argument("--physical-gpu", type=int, required=True)
    stage = sub.add_parser("stage-shard")
    stage.add_argument("--stage", choices=EVALUATION_STAGES, required=True)
    stage.add_argument("--shard", type=int, required=True)
    stage.add_argument("--shards", type=int, required=True)
    stage.add_argument("--physical-gpu", type=int, required=True)
    merge = sub.add_parser("merge-stage")
    merge.add_argument("--stage", choices=EVALUATION_STAGES, required=True)
    merge.add_argument("--shards", type=int, required=True)
    sub.add_parser("freeze")
    sub.add_parser("grant-confirm")
    sub.add_parser("prepare-confirm")
    sub.add_parser("status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _config(args)
    commands = {
        "prepare": command_prepare,
        "enumerate": command_enumerate,
        "merge-enumeration": command_merge_enumeration,
        "precompute-clean": command_precompute_clean,
        "stage-shard": command_stage_shard,
        "merge-stage": command_merge_stage,
        "freeze": command_freeze,
        "grant-confirm": command_grant_confirm,
        "prepare-confirm": command_prepare_confirm,
        "precompute-confirm-clean": command_precompute_confirm_clean,
        "confirm": command_confirm,
        "followups": command_followups,
        "status": command_status,
    }
    commands[args.command](args, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
