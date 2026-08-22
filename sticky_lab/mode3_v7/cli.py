"""Fail-closed command line for the complete V7 occupancy-frontier flow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

from sticky_lab.mode3_v6.data import (
    audit_csv_corpus,
    load_registered_records,
    require_formal_capacity,
)
from sticky_lab.mode3_v6_3.errors import CandidateRejected, ProtocolViolation
from sticky_lab.mode3_v6_3.config import verify_environment_lock
from sticky_lab.mode3_v6_3.report import (
    atomic_json,
    read_jsonl,
    result_inventory,
    sha256_file,
    write_jsonl,
    write_parquet,
)

from .budget import BudgetLedger, registered_budget
from .candidate_ranking import select_top_token_beta_pairs
from .config import (
    assert_output_leaf,
    canonical_sha256,
    config_for_profile,
    load_config,
    resolved_config,
)
from .confirm import confirm_frozen_operating_point, paired_position_audit
from .diagnostics import diagnose_selected_frontier
from .encoding import (
    CallRegistry,
    CallSpace,
    CachedEncoder,
    EmbeddingCache,
    EncodingRequest,
    FinalEmbeddingOracle,
    build_confirm_call_space,
    build_discovery_call_space,
)
from .freeze import load_freeze, write_freeze
from .funnel import (
    cached_clean_matrix,
    fit_and_score_candidate,
    precompute_discovery_clean,
)
from .reuse import aggregate_fallback_candidates, audit_r5_s0_reuse
from .roles import (
    CHAINS,
    DISCOVERY_ROLES,
    SEALED_ROLES,
    STAGES,
    bind_role,
    build_role_manifest,
    records_sha256,
    register_v7_roles,
    required_unique_capacity,
)
from .sealing import (
    assert_still_sealed,
    build_sealed_inventory,
    grant_access,
    physically_seal,
    read_sealed_jsonl,
)
from .tokenizer_audit import tokenizer_backend_sha256, tokenizer_sha256


DEFAULT_CONFIG = "configs/v7_mode3_occupancy_frontier.yaml"
DEFAULT_OUTPUT = (
    "results/sticky_lab/sentence_t5_base/mode3_v7_occupancy_frontier_r3_priority"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


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


def _verify_file(path: Path, expected: str) -> None:
    if not path.is_file() or sha256_file(path) != str(expected):
        raise ProtocolViolation(f"V7 bound resource mismatch: {path}")


def _verify_checksum_manifest(path: Path) -> None:
    observed = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ProtocolViolation(f"invalid checksum row: {line}")
        expected, filename = fields
        target = Path(filename.lstrip(" *"))
        if not target.is_file() or sha256_file(target) != expected:
            raise ProtocolViolation(f"model resource checksum mismatch: {target}")
        observed += 1
    if observed == 0:
        raise ProtocolViolation("empty model checksum manifest")


def _verify_static_identity(config: Mapping[str, Any]) -> tuple[Any, str, str | None]:
    _verify_file(Path(config["data"]["corpus_manifest"]), config["data"]["corpus_manifest_sha256"])
    _verify_file(
        Path(config["data"]["independent_corpus_audit"]),
        config["data"]["independent_corpus_audit_sha256"],
    )
    _verify_file(
        Path(config["model"]["checksum_manifest"]),
        config["model"]["checksum_manifest_sha256"],
    )
    _verify_checksum_manifest(Path(config["model"]["checksum_manifest"]))
    _verify_file(
        Path(config["resources"]["environment_lock"]),
        config["resources"]["environment_lock_sha256"],
    )
    verify_environment_lock(Path(config["resources"]["environment_lock"]))
    tokenizer = _tokenizer(config)
    observed = tokenizer_sha256(
        tokenizer, algorithm=str(config["model"]["tokenizer_hash_algorithm"])
    )
    if observed != str(config["model"]["tokenizer_sha256"]):
        raise ProtocolViolation("V7 tokenizer identity drift")
    return tokenizer, observed, tokenizer_backend_sha256(tokenizer)


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
    run = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    inventory = _sealed_inventory(output)
    return read_sealed_jsonl(
        output,
        role,
        Path(inventory["files"][role]["path"]),
        role_manifest_sha256=str(run["role_manifest_sha256"]),
    )


def _protocol_lock(config: Mapping[str, Any], source_config: Path) -> dict[str, Any]:
    protocol_document = REPOSITORY_ROOT / "docs" / "V7_PROTOCOL.md"
    payload = {
        "schema_version": "mode3-v7-protocol-lock-v1",
        "run_id": str(config["run_id"]),
        "code_commit": _git_commit(),
        "source_config_sha256": sha256_file(source_config),
        "protocol_document_sha256": sha256_file(protocol_document),
        "positions": ["prefix", "suffix"],
        "random_position_enabled": False,
        "one_token_one_insertion": True,
        "shared_center": True,
        "shared_radius": True,
        "center_fit_triggered_only": True,
        "center_trim_fraction": 0.10,
        "source_equal_weight": True,
        "position_equal_weight": True,
        "occupancy_grid": list(map(float, config["radius"]["occupancy_grid"])),
        "radius_rule": "largest source-balanced-UCB-feasible radius",
        "maximum_radius_degrees": 35.0,
        "prefix_coverage_lcb": 0.80,
        "suffix_coverage_lcb": 0.80,
        "migration_metrics_are_gates": False,
        "q92_diagnostic_only": True,
        "confirm_refit_allowed": False,
        "confirm_primary_beta_count": 1,
        "v6_outputs_read_only": True,
        "authorized_physical_gpus": [4, 5, 6, 7],
        "forbidden_physical_gpus": [0, 1, 2, 3],
    }
    payload["protocol_lock_sha256"] = canonical_sha256(payload)
    return payload


def _assert_registered_identity(
    output: Path,
    config: Mapping[str, Any],
    source_config: Path,
) -> None:
    """Bind every resumed command to the original formal registration."""

    marker = output / "registration" / "COMPLETE.json"
    run_path = output / "run_manifest.json"
    lock_path = output / "V7_PROTOCOL_LOCK.json"
    if not marker.is_file() or not run_path.is_file() or not lock_path.is_file():
        raise ProtocolViolation("V7 command requires a complete registration")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    expected = {
        "run_id": str(config["run_id"]),
        "profile": str(config["run_profile"]),
        "code_commit": _git_commit(),
        "config_sha256": canonical_sha256(config),
        "source_config_file_sha256": sha256_file(source_config),
        "protocol_lock_sha256": sha256_file(lock_path),
    }
    drift = {
        key: {"expected": value, "observed": run.get(key)}
        for key, value in expected.items()
        if run.get(key) != value
    }
    if drift:
        raise ProtocolViolation(f"V7 registered identity drift: {drift}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    unsigned_lock = {key: value for key, value in lock.items() if key != "protocol_lock_sha256"}
    if lock.get("protocol_lock_sha256") != canonical_sha256(unsigned_lock):
        raise ProtocolViolation("V7 protocol-lock content hash mismatch")
    protocol_document = REPOSITORY_ROOT / "docs" / "V7_PROTOCOL.md"
    if lock.get("protocol_document_sha256") != sha256_file(protocol_document):
        raise ProtocolViolation("V7 protocol document drift")
    if sha256_file(output / "V7_PROTOCOL.md") != sha256_file(protocol_document):
        raise ProtocolViolation("V7 output protocol copy drift")

    role_manifest = json.loads(
        (output / "data_role_manifest.json").read_text(encoding="utf-8")
    )
    unsigned_roles = {
        key: value for key, value in role_manifest.items() if key != "manifest_sha256"
    }
    if role_manifest.get("manifest_sha256") != canonical_sha256(unsigned_roles):
        raise ProtocolViolation("V7 role-manifest content hash mismatch")
    if role_manifest.get("manifest_sha256") != run.get("role_manifest_sha256"):
        raise ProtocolViolation("V7 role-manifest registration mismatch")
    for role in DISCOVERY_ROLES:
        observed = records_sha256(_load_role(output, role))
        expected_role = role_manifest["bindings"][role]["records_sha256"]
        if observed != expected_role:
            raise ProtocolViolation(f"V7 discovery role drift: {role}")
    for stage in STAGES:
        for chain in CHAINS:
            observed = records_sha256(_load_view(output, stage, chain))
            expected_view = role_manifest["nested_search_views"][stage][chain][
                "records_sha256"
            ]
            if observed != expected_view:
                raise ProtocolViolation(f"V7 nested view drift: {stage}/{chain}")

    position_manifest = json.loads(
        (output / "position_manifest.json").read_text(encoding="utf-8")
    )
    unsigned_positions = {
        key: value for key, value in position_manifest.items() if key != "manifest_sha256"
    }
    if position_manifest.get("manifest_sha256") != canonical_sha256(unsigned_positions):
        raise ProtocolViolation("V7 position-manifest content hash mismatch")
    if sha256_file(output / "position_manifest.json") != run.get(
        "position_manifest_file_sha256"
    ):
        raise ProtocolViolation("V7 position-manifest file hash mismatch")
    call_space = CallSpace.read(output / "registration" / "call_space.jsonl")
    if call_space.manifest_sha256 != run.get("call_space_sha256"):
        raise ProtocolViolation("V7 discovery call-space registration mismatch")

    inventory = json.loads(
        (output / "sealed" / "SEALED_INVENTORY.json").read_text(encoding="utf-8")
    )
    unsigned_inventory = {
        key: value for key, value in inventory.items() if key != "inventory_sha256"
    }
    if inventory.get("inventory_sha256") != canonical_sha256(unsigned_inventory):
        raise ProtocolViolation("V7 sealed-inventory content hash mismatch")
    if inventory.get("inventory_sha256") != run.get("sealed_inventory_sha256"):
        raise ProtocolViolation("V7 sealed-inventory registration mismatch")
    if str(config["run_profile"]) == "formal":
        if run.get("worktree_clean_at_registration") is not True or not _git_clean():
            raise ProtocolViolation("formal V7 requires the registered clean worktree")


def command_prepare(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output).resolve()
    assert_output_leaf(output, config)
    marker = output / "registration" / "COMPLETE.json"
    if marker.is_file():
        _assert_registered_identity(output, config, Path(args.config).resolve())
        return
    if str(config["run_profile"]) == "formal" and not _git_clean():
        raise ProtocolViolation("formal V7 registration requires a clean worktree")
    tokenizer, tokenizer_hash, backend_hash = _verify_static_identity(config)
    output.mkdir(parents=True, exist_ok=True)
    required_disk = int(config["resources"]["registration_minimum_free_bytes"])
    free = shutil.disk_usage(output.parent).free
    if free < required_disk:
        atomic_json(
            output / "V7_FINAL_STATUS.json",
            {
                "status": "BLOCKED_REGISTRATION_STORAGE_PREFLIGHT",
                "free_bytes": free,
                "required_bytes": required_disk,
            },
        )
        raise ProtocolViolation(f"V7 storage preflight failed: {free} < {required_disk}")
    capacity = audit_csv_corpus(
        str(config["data"]["input_glob"]),
        list(config["data"]["required_columns"]),
        required_unique_capacity(config),
        int(config["data"]["minimum_iid_sources"]),
    )
    atomic_json(output / "registration" / "data_capacity_audit.json", capacity.to_dict())
    require_formal_capacity(capacity)
    records = load_registered_records(
        str(config["data"]["input_glob"]), list(config["data"]["required_columns"])
    )
    profile_offset = {"formal": 0, "dry_run": 100_000, "pilot": 200_000}[config["run_profile"]]
    roles, views, allocation = register_v7_roles(
        records, config, seed=int(config["positions"]["seed"]) + profile_offset
    )
    for role in DISCOVERY_ROLES:
        write_jsonl(_role_path(output, role), roles[role])
    for stage in STAGES:
        for chain in CHAINS:
            write_jsonl(_view_path(output, stage, chain), views[stage][chain])
    sealed_dir = output.parent / f".{output.name}-{config['run_profile']}-sealed-roles"
    sealed_dir.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        os.chmod(sealed_dir, 0o700)
    sealed_paths: dict[str, Path] = {}
    for role in SEALED_ROLES:
        path = sealed_dir / f"{role}.jsonl"
        if path.is_file():
            if os.name != "nt":
                os.chmod(path, 0o600)
            existing = read_jsonl(path)
            if records_sha256(existing) != records_sha256(roles[role]):
                raise ProtocolViolation(f"partial sealed V7 role differs on resume: {role}")
        else:
            write_jsonl(path, roles[role])
        sealed_paths[role] = path
    nested = {
        stage: {
            chain: {
                "count": len(views[stage][chain]),
                "records_sha256": records_sha256(views[stage][chain]),
            }
            for chain in CHAINS
        }
        for stage in STAGES
    }
    bindings = [bind_role(role, "discovery", roles[role]) for role in DISCOVERY_ROLES]
    bindings.extend(bind_role(role, "sealed", roles[role]) for role in SEALED_ROLES)
    role_manifest = build_role_manifest(bindings, nested)
    atomic_json(output / "data_role_manifest.json", role_manifest)
    atomic_json(output / "registration" / "allocation_audit.json", allocation)
    inventory = build_sealed_inventory(
        output, sealed_paths, role_manifest_sha256=role_manifest["manifest_sha256"]
    )
    call_space = build_discovery_call_space(
        tokenizer, {role: roles[role] for role in DISCOVERY_ROLES}, config
    )
    call_space.write(output / "registration" / "call_space.jsonl")
    position_manifest = {
        "schema_version": "mode3-v7-position-manifest-v1",
        "positions": ["prefix", "suffix"],
        "random_position_enabled": False,
        "entries": [
            {
                "stage": stage,
                "role": role,
                "text_id": str(row["text_id"]),
                "source_id": str(row["source_id"]),
                "positions": ["prefix", "suffix"],
            }
            for stage in STAGES
            for role in ("fit", "select")
            for row in views[stage][role]
        ],
    }
    position_manifest["manifest_sha256"] = canonical_sha256(position_manifest)
    atomic_json(output / "position_manifest.json", position_manifest)
    lock = _protocol_lock(config, Path(args.config))
    atomic_json(output / "V7_PROTOCOL_LOCK.json", lock)
    (output / "V7_PROTOCOL_LOCK.sha256").write_text(
        f"{sha256_file(output / 'V7_PROTOCOL_LOCK.json')}  V7_PROTOCOL_LOCK.json\n",
        encoding="utf-8",
    )
    atomic_json(output / "resolved_config.json", resolved_config(config, source_path=Path(args.config)))
    protocol_document = REPOSITORY_ROOT / "docs" / "V7_PROTOCOL.md"
    if not protocol_document.is_file():
        raise ProtocolViolation(f"registered V7 protocol document is missing: {protocol_document}")
    shutil.copyfile(protocol_document, output / "V7_PROTOCOL.md")
    atomic_json(
        output / "V7_OCCUPANCY_GRID.json",
        {
            "schema_version": "mode3-v7-occupancy-grid-v1",
            "occupancy_grid": list(map(float, config["radius"]["occupancy_grid"])),
            "radius_rule": "largest source-balanced one-sided UCB feasible radius",
            "maximum_radius_degrees": float(config["geometry"]["maximum_radius_degrees"]),
            "center_refit_per_beta": False,
            "prefix_and_suffix_coverage_lcbs_are_separate": True,
        },
    )
    run_manifest = {
        "schema_version": "mode3-v7-run-manifest-v1",
        "run_id": config["run_id"],
        "protocol_revision": int(config["protocol_revision"]),
        "profile": config["run_profile"],
        "scientific_claims_allowed": config["run_profile"] == "formal",
        "code_commit": _git_commit(),
        "worktree_clean_at_registration": _git_clean(),
        "config_sha256": canonical_sha256(config),
        "source_config_file_sha256": sha256_file(Path(args.config)),
        "protocol_lock_sha256": sha256_file(output / "V7_PROTOCOL_LOCK.json"),
        "model_revision": config["model"]["revision"],
        "tokenizer_sha256": tokenizer_hash,
        "tokenizer_backend_sha256_diagnostic": backend_hash,
        "role_manifest_sha256": role_manifest["manifest_sha256"],
        "position_manifest_file_sha256": sha256_file(output / "position_manifest.json"),
        "call_space_sha256": call_space.manifest_sha256,
        "sealed_inventory_sha256": inventory["inventory_sha256"],
        "allowed_physical_gpus": list(config["resources"]["allowed_physical_gpus"]),
        "forbidden_physical_gpus": list(config["resources"]["forbidden_physical_gpus"]),
        "state": "V7_PROTOCOL_LOCKED",
    }
    atomic_json(output / "run_manifest.json", run_manifest)
    physically_seal(list(sealed_paths.values()))
    atomic_json(
        marker,
        {
            "schema_version": "mode3-v7-registration-complete-v1",
            "status": "V7_PROTOCOL_LOCKED",
            "run_id": config["run_id"],
            "profile": config["run_profile"],
            "code_commit": _git_commit(),
            "role_manifest_sha256": role_manifest["manifest_sha256"],
            "call_space_sha256": call_space.manifest_sha256,
            "confirm_accessed": False,
        },
    )


def command_reuse_s0(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output).resolve()
    target = output / "stages" / "s0"
    if (target / "COMPLETE.json").is_file():
        return
    source = Path(args.source_output or config["reuse"]["source_output"])
    if str(config["run_profile"]) == "formal" and bool(args.skip_exact_cache_audit):
        raise ProtocolViolation("formal V7 cannot skip the exact r5 cache audit")
    audit, legal, metrics = audit_r5_s0_reuse(
        source, config, exact_cache_audit=not bool(args.skip_exact_cache_audit)
    )
    keep = min(
        int(audit["full_candidate_count"]),
        int(config["funnel"]["full_candidates"]),
    )
    selected, selection = aggregate_fallback_candidates(
        metrics,
        keep=keep,
        deterministic_audit=int(config["search"]["deterministic_random_audit"]),
        seed=int(config["positions"]["seed"]),
    )
    legal_by_id = {int(row["token_id"]): row for row in legal}
    target.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "legal_tokens.jsonl", legal)
    tokenizer_audit_source = source / "tokenizer_audit.jsonl"
    if tokenizer_audit_source.is_file():
        shutil.copyfile(tokenizer_audit_source, output / "tokenizer_audit_v6_stricter_baseline.jsonl")
    atomic_json(output / "V7_S0_REUSE_AUDIT.json", audit)
    write_parquet(
        output / "v7_s0_frontier_all_tokens.parquet",
        [
            {
                **dict(row),
                "formal_v7_frontier": False,
                "proposal_only": True,
                "v7_reuse_level": str(audit["reuse_level"]),
                "source_geometry": "v6_3_q92_prefix_suffix_random",
                "v7_center_or_radius_reused": False,
            }
            for row in metrics
        ],
    )
    atomic_json(target / "selection_audit.json", selection)
    atomic_json(target / "selected.json", {"token_ids": selected})
    write_jsonl(
        target / "selected_tokens.jsonl",
        [
            {
                "token_id": token_id,
                "token_text": str(legal_by_id[token_id]["token_text"]),
                "proposal_only": True,
            }
            for token_id in selected
        ],
    )
    atomic_json(
        target / "COMPLETE.json",
        {
            "schema_version": "mode3-v7-s0-reuse-complete-v1",
            "status": "V7_S0_REUSE_COMPLETE",
            "legal_tokens": len(legal),
            "selected_tokens": len(selected),
            "reuse_level": audit["reuse_level"],
            "old_q92_caps_formal": False,
            "full_refit_required": True,
        },
    )
    atomic_json(
        output / "budget" / "planned.json",
        registered_budget(
            config,
            full_candidates=len(selected),
            s0_raw_reused=False,
        ),
    )


def _runtime(
    output: Path,
    config: Mapping[str, Any],
    physical_gpu: int,
    *,
    confirm: bool = False,
) -> tuple[CachedEncoder, CallSpace]:
    root = output / "confirm_runtime" if confirm else output
    call_path = (
        output / "confirm_runtime" / "call_space.jsonl"
        if confirm
        else output / "registration" / "call_space.jsonl"
    )
    call_space = CallSpace.read(call_path)
    ledger = BudgetLedger(output, config["budget"])
    registry = CallRegistry(root, call_space, ledger)
    cache = EmbeddingCache(root, call_space)
    oracle = FinalEmbeddingOracle(
        config,
        physical_gpu=int(physical_gpu),
        expected_tokenizer_hash=str(config["model"]["tokenizer_sha256"]),
    )
    return CachedEncoder(config, call_space, registry, cache, oracle), call_space


def command_precompute_clean(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output).resolve()
    marker = output / "registration" / "DISCOVERY_CLEAN_COMPLETE.json"
    if marker.is_file():
        return
    encoder, _ = _runtime(output, config, int(args.physical_gpu))
    calibration = _load_view(output, "full", "calibration")
    select = _load_view(output, "full", "select")
    axis = _load_role(output, "axis_fit_benign")
    cache_audit = precompute_discovery_clean(
        encoder,
        calibration_records=calibration,
        select_records=select,
        axis_records=axis,
    )
    axis_vectors = cached_clean_matrix(encoder, axis, "axis_fit_benign").astype(np.float64)
    e_star = axis_vectors.mean(axis=0)
    e_star /= np.linalg.norm(e_star)
    axis_dir = output / "axis"
    axis_dir.mkdir(parents=True, exist_ok=True)
    np.save(axis_dir / "e_star.npy", e_star.astype(np.float64), allow_pickle=False)
    atomic_json(
        axis_dir / "e_star.json",
        {
            "schema_version": "mode3-v7-e-star-v1",
            "role_sha256": records_sha256(axis),
            "observations": len(axis),
            "dimension": len(e_star),
            "vector_sha256": sha256_file(axis_dir / "e_star.npy"),
            "confirm_data_used": False,
        },
    )
    atomic_json(marker, {"status": "DISCOVERY_CLEAN_COMPLETE", "cache": cache_audit})


def _full_candidates(output: Path) -> list[dict[str, Any]]:
    return read_jsonl(output / "stages" / "s0" / "selected_tokens.jsonl")


def command_stage_shard(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    if str(args.stage) != "full":
        raise ProtocolViolation("V7 formal funnel has one from-scratch FULL stage")
    registered_shards = int(config["funnel"]["shards"])
    if int(args.shards) != registered_shards:
        raise ProtocolViolation(
            f"V7 FULL shard-count drift: {args.shards} != {registered_shards}"
        )
    if int(args.shard) < 0 or int(args.shard) >= registered_shards:
        raise ProtocolViolation(f"V7 FULL shard index is out of range: {args.shard}")
    output = Path(args.output).resolve()
    target = output / "stages" / "full" / f"shard_{int(args.shard):02d}"
    if (target / "COMPLETE.json").is_file():
        return
    target.mkdir(parents=True, exist_ok=True)
    all_candidates = _full_candidates(output)
    candidates = [
        row
        for index, row in enumerate(all_candidates)
        if index % int(args.shards) == int(args.shard)
    ]
    fit = _load_view(output, "full", "fit")
    calibration = _load_view(output, "full", "calibration")
    select = _load_view(output, "full", "select")
    axis = _load_role(output, "axis_fit_benign")
    role_hashes = {
        "fit": records_sha256(fit),
        "calibration": records_sha256(calibration),
        "select": records_sha256(select),
        "axis_fit_benign": records_sha256(axis),
    }
    frontiers: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    try:
        encoder, _ = _runtime(output, config, int(args.physical_gpu))
        for candidate in candidates:
            token_id = int(candidate["token_id"])
            try:
                frontier, audit = fit_and_score_candidate(
                    encoder,
                    token_id=token_id,
                    token_text=str(candidate["token_text"]),
                    stage="full",
                    fit_records=fit,
                    calibration_records=calibration,
                    select_records=select,
                    axis_records=axis,
                    role_hashes=role_hashes,
                    config=config,
                )
            except CandidateRejected as error:
                rejections.append(
                    {
                        "token_id": token_id,
                        "token_text": str(candidate["token_text"]),
                        "reason": type(error).__name__,
                        "detail": str(error),
                    }
                )
            else:
                long_rows.extend(frontier.pop("long_rows"))
                frontiers.append(frontier)
                audits.append(audit)
        write_jsonl(target / "frontiers.jsonl", frontiers)
        write_jsonl(target / "long_rows.jsonl", long_rows)
        write_jsonl(target / "audits.jsonl", audits)
        write_jsonl(target / "rejections.jsonl", rejections)
        atomic_json(
            target / "COMPLETE.json",
            {
                "schema_version": "mode3-v7-full-shard-v1",
                "stage": "full",
                "shard": int(args.shard),
                "shards": int(args.shards),
                "candidate_tokens": [int(row["token_id"]) for row in candidates],
                "valid": len(frontiers),
                "rejected": len(rejections),
                "physical_gpu": int(args.physical_gpu),
                "positions": ["prefix", "suffix"],
                "frontiers_sha256": sha256_file(target / "frontiers.jsonl"),
                "long_rows_sha256": sha256_file(target / "long_rows.jsonl"),
                "audits_sha256": sha256_file(target / "audits.jsonl"),
                "rejections_sha256": sha256_file(target / "rejections.jsonl"),
            },
        )
    except Exception as error:
        atomic_json(
            target / "FAILED.json",
            {
                "schema_version": "mode3-v7-stage-failed-v1",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "partial_merge_allowed": False,
            },
        )
        raise


def command_merge_stage(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    registered_shards = int(config["funnel"]["shards"])
    if int(args.shards) != registered_shards:
        raise ProtocolViolation(
            f"V7 FULL merge shard-count drift: {args.shards} != {registered_shards}"
        )
    output = Path(args.output).resolve()
    target = output / "stages" / "full"
    if (target / "COMPLETE.json").is_file():
        return
    frontiers: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    candidate_ids: list[int] = []
    for shard in range(int(args.shards)):
        root = target / f"shard_{shard:02d}"
        if (root / "FAILED.json").exists() or not (root / "COMPLETE.json").is_file():
            raise ProtocolViolation(f"cannot merge incomplete V7 FULL shard {shard}")
        complete = json.loads((root / "COMPLETE.json").read_text(encoding="utf-8"))
        for filename, field in (
            ("frontiers.jsonl", "frontiers_sha256"),
            ("long_rows.jsonl", "long_rows_sha256"),
            ("audits.jsonl", "audits_sha256"),
            ("rejections.jsonl", "rejections_sha256"),
        ):
            if complete.get(field) != sha256_file(root / filename):
                raise ProtocolViolation(
                    f"V7 FULL shard {shard} artifact hash mismatch: {filename}"
                )
        candidate_ids.extend(map(int, complete["candidate_tokens"]))
        frontiers.extend(read_jsonl(root / "frontiers.jsonl"))
        long_rows.extend(read_jsonl(root / "long_rows.jsonl"))
        rejections.extend(read_jsonl(root / "rejections.jsonl"))
    expected = {int(row["token_id"]) for row in _full_candidates(output)}
    if set(candidate_ids) != expected or len(candidate_ids) != len(expected):
        raise ProtocolViolation("V7 FULL shard candidate coverage mismatch")
    valid_ids = [int(row["token_id"]) for row in frontiers]
    rejected_ids = [int(row["token_id"]) for row in rejections]
    if len(valid_ids) != len(set(valid_ids)) or len(rejected_ids) != len(set(rejected_ids)):
        raise ProtocolViolation("V7 FULL result repeats a candidate token")
    if set(valid_ids).intersection(rejected_ids):
        raise ProtocolViolation("V7 FULL candidate is both valid and rejected")
    observed = set(valid_ids).union(rejected_ids)
    if observed != expected or len(valid_ids) + len(rejected_ids) != len(expected):
        raise ProtocolViolation("V7 FULL result coverage mismatch")
    expected_curve_rows = len(frontiers) * len(config["radius"]["occupancy_grid"]) * 2
    if len(long_rows) != expected_curve_rows:
        raise ProtocolViolation(
            f"V7 FULL curve row-count mismatch: {len(long_rows)}/{expected_curve_rows}"
        )
    curve_counts: dict[tuple[int, float, str], int] = {}
    for row in long_rows:
        key = (
            int(row["token_id"]),
            float(row["beta_target"]),
            str(row["position"]),
        )
        curve_counts[key] = curve_counts.get(key, 0) + 1
    expected_curve_keys = {
        (token_id, float(beta), position)
        for token_id in valid_ids
        for beta in config["radius"]["occupancy_grid"]
        for position in ("prefix", "suffix")
    }
    if set(curve_counts) != expected_curve_keys or any(
        count != 1 for count in curve_counts.values()
    ):
        raise ProtocolViolation("V7 FULL frontier grid is incomplete or duplicated")
    top_pairs, selection_audit = select_top_token_beta_pairs(
        frontiers, keep=int(config["funnel"]["full_top_pairs"])
    )
    selected_ids = {int(row["token_id"]) for row in top_pairs}
    write_jsonl(target / "all_frontiers.jsonl", frontiers)
    write_jsonl(target / "all_rejections.jsonl", rejections)
    write_jsonl(
        target / "selected_frontiers.jsonl",
        [row for row in frontiers if int(row["token_id"]) in selected_ids],
    )
    write_parquet(output / "v7_full_frontier_all_candidates.parquet", long_rows)
    write_parquet(output / "v7_occupancy_curves.parquet", long_rows)
    redundant = [
        {
            "token_id": row["token_id"],
            "token_text": row["token_text"],
            "beta_target": row["beta_target"],
            "position": row["position"],
            "capture_given_clean_outside": row["capture_outside_point"],
            "outside_to_inside": row["outside_to_inside"],
            "conditional_origin_outside": row["conditional_origin_outside"],
            "inside_retention": row["inside_retention"],
            "net_gain": row["net_gain"],
            "hard_gate": False,
        }
        for row in long_rows
    ]
    write_parquet(output / "v7_redundant_metrics.parquet", redundant)
    write_jsonl(
        output / "v7_e_star_frontier.jsonl",
        [
            {
                "token_id": int(row["token_id"]),
                "token_text": str(row["token_text"]),
                "beta80_ps": row.get("beta80_ps"),
                "beta_axis": row.get("beta_axis"),
                "beta80_precedes_beta_axis": bool(row.get("beta80_precedes_beta_axis")),
                "axis_geometry": row["axis_geometry"],
                "used_for_selection": False,
            }
            for row in frontiers
        ],
    )
    atomic_json(output / "v7_top20_token_beta_pairs.json", {"pairs": top_pairs})
    atomic_json(target / "selection_audit.json", selection_audit)
    terminal = len(top_pairs) < 5
    atomic_json(
        target / "COMPLETE.json",
        {
            "schema_version": "mode3-v7-full-complete-v1",
            "status": "VALID_NO_OCCUPANCY_FEASIBLE_CANDIDATE" if terminal else "V7_SEARCH_COMPLETE",
            "candidate_tokens": len(expected),
            "valid": len(frontiers),
            "rejected": len(rejections),
            "ps80_positive_candidates": selection_audit["ps80_positive_candidates"],
            "selected_pairs": len(top_pairs),
            "all_frontiers_sha256": sha256_file(target / "all_frontiers.jsonl"),
            "all_rejections_sha256": sha256_file(target / "all_rejections.jsonl"),
            "selected_frontiers_sha256": sha256_file(
                target / "selected_frontiers.jsonl"
            ),
            "top_pairs_sha256": sha256_file(
                output / "v7_top20_token_beta_pairs.json"
            ),
            "full_frontier_parquet_sha256": sha256_file(
                output / "v7_full_frontier_all_candidates.parquet"
            ),
        },
    )
    if terminal:
        final = {
            "schema_version": "mode3-v7-final-status-v1",
            "status": "VALID_NO_OCCUPANCY_FEASIBLE_CANDIDATE",
            "terminal": True,
            "confirm_accessed": False,
            "reason": "fewer than five candidates reached separate prefix/suffix 80% LCB within beta<=15%",
        }
        atomic_json(output / "V7_FINAL_STATUS.json", final)
        atomic_json(output / "FINAL_STATUS.json", final)
        atomic_json(output / "result_inventory.json", result_inventory(output))


def _validated_full_selection(
    output: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    marker_path = output / "stages" / "full" / "COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    bound_files = (
        (
            output / "stages" / "full" / "all_frontiers.jsonl",
            "all_frontiers_sha256",
        ),
        (
            output / "stages" / "full" / "all_rejections.jsonl",
            "all_rejections_sha256",
        ),
        (
            output / "stages" / "full" / "selected_frontiers.jsonl",
            "selected_frontiers_sha256",
        ),
        (output / "v7_top20_token_beta_pairs.json", "top_pairs_sha256"),
        (
            output / "v7_full_frontier_all_candidates.parquet",
            "full_frontier_parquet_sha256",
        ),
    )
    for path, field in bound_files:
        if marker.get(field) != sha256_file(path):
            raise ProtocolViolation(f"V7 FULL merged artifact hash mismatch: {path.name}")
    frontiers = read_jsonl(output / "stages" / "full" / "all_frontiers.jsonl")
    expected_pairs, _ = select_top_token_beta_pairs(
        frontiers, keep=int(config["funnel"]["full_top_pairs"])
    )
    observed = json.loads(
        (output / "v7_top20_token_beta_pairs.json").read_text(encoding="utf-8")
    ).get("pairs", [])
    if canonical_sha256(observed) != canonical_sha256(expected_pairs):
        raise ProtocolViolation("V7 top-20 selection differs from the registered ranking")
    return frontiers, expected_pairs


def command_diagnostics(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    """Run report-only center/e* diagnostics after the top-20 are fixed."""

    output = Path(args.output).resolve()
    target = output / "diagnostics" / "post_selection"
    marker = target / "COMPLETE.json"
    if marker.is_file():
        return
    all_frontiers, selected_pairs = _validated_full_selection(output, config)
    selected_ids = [int(row["token_id"]) for row in selected_pairs]
    frontiers = {int(row["token_id"]): row for row in all_frontiers}
    if any(token_id not in frontiers for token_id in selected_ids):
        raise ProtocolViolation("post-selection diagnostics cannot resolve every selected token")
    fit = _load_view(output, "full", "fit")
    select = _load_view(output, "full", "select")
    call_space = CallSpace.read(output / "registration" / "call_space.jsonl")
    cache = EmbeddingCache(output, call_space)
    e_star = np.load(output / "axis" / "e_star.npy", allow_pickle=False)
    results: list[dict[str, Any]] = []
    for token_id in selected_ids:
        token_target = target / f"token_{token_id}" / "COMPLETE.json"
        if token_target.is_file():
            result = json.loads(token_target.read_text(encoding="utf-8"))
            if str(result.get("center_hash")) != str(frontiers[token_id]["center_hash"]):
                raise ProtocolViolation(f"post-selection diagnostic center drift for token {token_id}")
        else:
            result = diagnose_selected_frontier(
                frontiers[token_id],
                fit_records=fit,
                select_records=select,
                e_star=e_star,
                cache=cache,
                call_space=call_space,
                config=config,
            )
            atomic_json(token_target, result)
        results.append(result)
    write_jsonl(output / "v7_center_drift.jsonl", results)
    write_jsonl(output / "v7_e_star_analysis.jsonl", results)
    atomic_json(
        marker,
        {
            "schema_version": "mode3-v7-post-selection-diagnostics-complete-v1",
            "status": "V7_POST_SELECTION_DIAGNOSTICS_COMPLETE",
            "selected_tokens": selected_ids,
            "bootstrap_samples_per_token": int(
                config["diagnostics"]["post_selection_center_bootstrap_samples"]
            ),
            "used_for_selection": False,
            "confirm_data_used": False,
        },
    )


def command_compact_cache(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    """Delete only non-selected V7 embedding caches after durable FULL outputs."""

    output = Path(args.output).resolve()
    target = output / "cache_compaction"
    marker = target / "COMPLETE.json"
    if marker.is_file():
        return
    if not (output / "stages" / "full" / "COMPLETE.json").is_file():
        raise ProtocolViolation("V7 cache compaction requires a complete FULL merge")
    if not (
        output / "diagnostics" / "post_selection" / "COMPLETE.json"
    ).is_file():
        raise ProtocolViolation("V7 cache compaction requires post-selection diagnostics")
    _, selected_pairs = _validated_full_selection(output, config)
    keep = {-2}.union(int(row["token_id"]) for row in selected_pairs)
    cache_root = (output / "embedding_cache").resolve()
    if cache_root.parent != output or cache_root.name != "embedding_cache":
        raise ProtocolViolation(f"unsafe V7 cache root: {cache_root}")
    cache_root.mkdir(parents=True, exist_ok=True)
    plan_path = target / "PLAN.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if set(map(int, plan.get("kept_token_ids", []))) != keep:
            raise ProtocolViolation("V7 cache-compaction keep set drift")
    else:
        removals: list[dict[str, Any]] = []
        for path in sorted(cache_root.glob("token_*")):
            if not path.is_dir() or path.is_symlink():
                raise ProtocolViolation(f"unsafe V7 cache entry: {path}")
            try:
                token_id = int(path.name[len("token_") :])
            except ValueError as error:
                raise ProtocolViolation(f"unrecognized V7 cache entry: {path}") from error
            if token_id in keep:
                continue
            files = [child for child in path.rglob("*") if child.is_file()]
            removals.append(
                {
                    "token_id": token_id,
                    "directory": path.name,
                    "files": len(files),
                    "bytes": sum(child.stat().st_size for child in files),
                }
            )
        plan = {
            "schema_version": "mode3-v7-cache-compaction-plan-v1",
            "scope": "v7_nonselected_embedding_cache_only",
            "cache_root": str(cache_root),
            "kept_token_ids": sorted(keep),
            "removals": removals,
            "full_frontier_retained": True,
            "v6_paths_touched": False,
        }
        atomic_json(plan_path, plan)
    removed_directories = 0
    removed_bytes = 0
    for row in plan["removals"]:
        path = (cache_root / str(row["directory"])).resolve()
        if path.parent != cache_root or path.name != str(row["directory"]):
            raise ProtocolViolation(f"unsafe V7 cache-compaction target: {path}")
        if path.exists():
            if path.is_symlink() or not path.is_dir():
                raise ProtocolViolation(f"unsafe V7 cache-compaction target type: {path}")
            shutil.rmtree(path)
        removed_directories += 1
        removed_bytes += int(row["bytes"])
    atomic_json(
        marker,
        {
            "schema_version": "mode3-v7-cache-compaction-complete-v1",
            "status": "V7_NONSELECTED_CACHE_COMPACTED",
            "removed_directories": removed_directories,
            "removed_bytes": removed_bytes,
            "kept_token_ids": sorted(keep),
            "full_frontier_retained": True,
            "v6_paths_touched": False,
        },
    )


def command_freeze(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output).resolve()
    if (output / "freeze" / "COMPLETE.json").is_file():
        return
    assert_still_sealed(output)
    frontiers, _ = _validated_full_selection(output, config)
    run = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    role_manifest = json.loads((output / "data_role_manifest.json").read_text(encoding="utf-8"))
    discovery = {
        role: str(role_manifest["bindings"][role]["records_sha256"])
        for role in DISCOVERY_ROLES
    }
    confirm = {
        role: str(role_manifest["bindings"][role]["records_sha256"])
        for role in SEALED_ROLES
    }
    metadata = {
        "maximum_radius_degrees": float(config["geometry"]["maximum_radius_degrees"]),
        "code_commit": run["code_commit"],
        "config_sha256": run["config_sha256"],
        "protocol_lock_sha256": run["protocol_lock_sha256"],
        "role_manifest_sha256": run["role_manifest_sha256"],
        "discovery_role_hashes": discovery,
        "confirm_role_hashes": confirm,
        "tokenizer_sha256": run["tokenizer_sha256"],
        "model_revision": run["model_revision"],
        "call_space_sha256": run["call_space_sha256"],
        "e_star_sha256": sha256_file(output / "axis" / "e_star.npy"),
        "full_frontier_sha256": sha256_file(
            output / "stages" / "full" / "all_frontiers.jsonl"
        ),
        "certification_thresholds": {
            "prefix_coverage_lcb": float(config["certification"]["prefix_coverage_lcb"]),
            "suffix_coverage_lcb": float(config["certification"]["suffix_coverage_lcb"]),
            "maximum_radius_degrees": float(config["geometry"]["maximum_radius_degrees"]),
            "actual_trigger_token_length": 1,
            "one_insertion_only": True,
        },
    }
    primary_path, digest = write_freeze(output, frontiers=frontiers, metadata=metadata)
    shutil.copyfile(primary_path, output / "v7_primary_freeze.json")
    shutil.copyfile(output / "freeze" / "secondary.jsonl", output / "v7_secondary_freeze.jsonl")
    atomic_json(
        output / "orchestration_logs" / "freeze_status.json",
        {"status": "V7_PRIMARY_FROZEN", "freeze_sha256": digest, "confirm_still_sealed": True},
    )


def command_grant_confirm(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output).resolve()
    marker = output / "confirm_runtime" / "REGISTRATION_COMPLETE.json"
    if marker.is_file():
        return
    run = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    grant = grant_access(
        output,
        role_manifest_sha256=str(run["role_manifest_sha256"]),
        sealed_inventory_path=output / "sealed" / "SEALED_INVENTORY.json",
    )
    roles = {role: _load_sealed_role(output, role) for role in SEALED_ROLES}
    tokenizer = _tokenizer(config)
    call_space = build_confirm_call_space(tokenizer, roles, config)
    call_space.write(output / "confirm_runtime" / "call_space.jsonl")
    atomic_json(
        marker,
        {
            "schema_version": "mode3-v7-confirm-runtime-registration-v1",
            "status": "V7_CONFIRM_GRANTED",
            "freeze_sha256": grant["freeze_sha256"],
            "call_space_sha256": call_space.manifest_sha256,
            "roles": {role: len(rows) for role, rows in roles.items()},
        },
    )


def command_confirm(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output).resolve()
    marker = output / "confirm" / "COMPLETE.json"
    if marker.is_file():
        return
    freeze_sha = (output / "freeze" / "FREEZE.sha256").read_text(encoding="utf-8").split()[0]
    artifact = load_freeze(output / "freeze" / "primary.json", freeze_sha)
    encoder, confirm_call_space = _runtime(
        output, config, int(args.physical_gpu), confirm=True
    )
    confirm_registration = json.loads(
        (output / "confirm_runtime" / "REGISTRATION_COMPLETE.json").read_text(
            encoding="utf-8"
        )
    )
    if confirm_registration.get("freeze_sha256") != freeze_sha:
        raise ProtocolViolation("V7 confirm runtime is bound to a different freeze")
    if confirm_registration.get("call_space_sha256") != confirm_call_space.manifest_sha256:
        raise ProtocolViolation("V7 confirm call-space hash drift")
    prefix = _load_sealed_role(output, "confirm_prefix")
    suffix = _load_sealed_role(output, "confirm_suffix")
    benign = _load_sealed_role(output, "confirm_benign")
    paired = _load_sealed_role(output, "confirm_paired")
    clean_requests = (
        [EncodingRequest("confirm_prefix", row, "clean", 0) for row in prefix]
        + [EncodingRequest("confirm_suffix", row, "clean", 0) for row in suffix]
        + [EncodingRequest("confirm_benign", row, "clean", 0) for row in benign]
        + [EncodingRequest("confirm_paired", row, "clean", 0) for row in paired]
    )
    _, _, clean_cache = encoder.encode_requests(
        token_id=-2, token_text="", requests=clean_requests, phase="v7:confirm:clean"
    )
    token_id = artifact.token_id
    token_text = str(artifact.operating_point["token_text"])
    prefix_vectors, prefix_audits, _ = encoder.encode_requests(
        token_id=token_id,
        token_text=token_text,
        requests=[EncodingRequest("confirm_prefix", row, "prefix", 0) for row in prefix],
        phase="v7:confirm:prefix",
    )
    suffix_vectors, suffix_audits, _ = encoder.encode_requests(
        token_id=token_id,
        token_text=token_text,
        requests=[EncodingRequest("confirm_suffix", row, "suffix", 0) for row in suffix],
        phase="v7:confirm:suffix",
    )
    paired_prefix, paired_prefix_audits, _ = encoder.encode_requests(
        token_id=token_id,
        token_text=token_text,
        requests=[EncodingRequest("confirm_paired", row, "prefix", 0) for row in paired],
        phase="v7:confirm:paired_prefix",
    )
    paired_suffix, paired_suffix_audits, _ = encoder.encode_requests(
        token_id=token_id,
        token_text=token_text,
        requests=[EncodingRequest("confirm_paired", row, "suffix", 0) for row in paired],
        phase="v7:confirm:paired_suffix",
    )
    prefix_clean = cached_clean_matrix(encoder, prefix, "confirm_prefix")
    suffix_clean = cached_clean_matrix(encoder, suffix, "confirm_suffix")
    benign_vectors = cached_clean_matrix(encoder, benign, "confirm_benign")
    result = confirm_frozen_operating_point(
        artifact,
        prefix_rows=prefix,
        prefix_triggered_vectors=prefix_vectors,
        prefix_clean_vectors=prefix_clean,
        suffix_rows=suffix,
        suffix_triggered_vectors=suffix_vectors,
        suffix_clean_vectors=suffix_clean,
        benign_rows=benign,
        benign_vectors=benign_vectors,
        observed_role_hashes={
            "confirm_prefix": records_sha256(prefix),
            "confirm_suffix": records_sha256(suffix),
            "confirm_benign": records_sha256(benign),
        },
        freeze_sha256=freeze_sha,
        familywise_alpha=float(config["certification"]["familywise_alpha"]),
    )
    paired_summary = paired_position_audit(
        artifact, paired, paired_prefix, paired_suffix
    )
    certificate = {
        key: value
        for key, value in result.items()
        if key not in {"observations", "benign_observations"}
    }
    atomic_json(output / "confirm" / "v7_confirm_certificate.json", certificate)
    atomic_json(output / "v7_confirm_certificate.json", certificate)
    atomic_json(output / "confirm" / "paired_position_audit.json", paired_summary)
    write_parquet(output / "confirm" / "migration.parquet", result["observations"])
    write_parquet(output / "confirm" / "benign_occupancy.parquet", result["benign_observations"])
    write_jsonl(
        output / "confirm" / "runtime_realization_audit.jsonl",
        [
            audit.to_dict()
            for audit in prefix_audits
            + suffix_audits
            + paired_prefix_audits
            + paired_suffix_audits
        ],
    )
    atomic_json(
        marker,
        {
            "schema_version": "mode3-v7-confirm-complete-v1",
            "status": result["status"],
            "freeze_sha256": freeze_sha,
            "refit_performed": False,
            "cache": clean_cache,
        },
    )
    final = {
        "schema_version": "mode3-v7-final-status-v1",
        "status": result["status"],
        "terminal": True,
        "certified": result["certified"],
        "evidence_grade": result["evidence_grade"],
        "token_id": result["token_id"],
        "token_text": result["token_text"],
        "beta": result["beta_frozen"],
        "radius_degrees": result["radius_degrees"],
        "gates": result["gates"],
        "refit_performed": False,
    }
    atomic_json(output / "V7_FINAL_STATUS.json", final)
    atomic_json(output / "FINAL_STATUS.json", final)
    atomic_json(output / "result_inventory.json", result_inventory(output))


def command_status(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output).resolve()
    stages = {
        "registration": (output / "registration" / "COMPLETE.json").is_file(),
        "s0_reuse": (output / "stages" / "s0" / "COMPLETE.json").is_file(),
        "discovery_clean": (output / "registration" / "DISCOVERY_CLEAN_COMPLETE.json").is_file(),
        "full": (output / "stages" / "full" / "COMPLETE.json").is_file(),
        "freeze": (output / "freeze" / "COMPLETE.json").is_file(),
        "confirm": (output / "confirm" / "COMPLETE.json").is_file(),
    }
    final = None
    if (output / "V7_FINAL_STATUS.json").is_file():
        final = json.loads((output / "V7_FINAL_STATUS.json").read_text(encoding="utf-8"))
    print(json.dumps({"stages": stages, "final": final}, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--profile", choices=("formal", "pilot", "dry_run"), default="formal")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    reuse = commands.add_parser("reuse-s0")
    reuse.add_argument("--source-output")
    reuse.add_argument("--skip-exact-cache-audit", action="store_true")
    precompute = commands.add_parser("precompute-clean")
    precompute.add_argument("--physical-gpu", type=int, required=True)
    shard = commands.add_parser("stage-shard")
    shard.add_argument("--stage", choices=("full",), required=True)
    shard.add_argument("--shard", type=int, required=True)
    shard.add_argument("--shards", type=int, required=True)
    shard.add_argument("--physical-gpu", type=int, required=True)
    merge = commands.add_parser("merge-stage")
    merge.add_argument("--stage", choices=("full",), required=True)
    merge.add_argument("--shards", type=int, required=True)
    commands.add_parser("freeze")
    commands.add_parser("diagnostics")
    commands.add_parser("compact-cache")
    commands.add_parser("grant-confirm")
    confirm = commands.add_parser("confirm")
    confirm.add_argument("--physical-gpu", type=int, required=True)
    commands.add_parser("status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = _config(args)
    if args.command != "prepare":
        _assert_registered_identity(
            Path(args.output).resolve(), config, Path(args.config).resolve()
        )
    handlers = {
        "prepare": command_prepare,
        "reuse-s0": command_reuse_s0,
        "precompute-clean": command_precompute_clean,
        "stage-shard": command_stage_shard,
        "merge-stage": command_merge_stage,
        "freeze": command_freeze,
        "diagnostics": command_diagnostics,
        "compact-cache": command_compact_cache,
        "grant-confirm": command_grant_confirm,
        "confirm": command_confirm,
        "status": command_status,
    }
    handlers[args.command](args, config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
