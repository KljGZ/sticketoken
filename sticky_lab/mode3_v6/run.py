"""Fail-closed V6 command line entry point.

The formal runner exposes separate exhaustive, black-box, white-box,
validation, sealed-test, OOD, semantic-control, retrieval and publication
phases.  Test/OOD commands require a validation freeze artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping

import numpy as np
import yaml

from .atomic_io import write_json, write_jsonl
from .data import (
    DataGap, audit_csv_corpus, build_all_role_sizes, load_registered_records, register_v6_roles,
    require_formal_capacity, required_unique_capacity, write_capacity_audit,
)
from .deduplication import audit_role_leakage
from .fingerprint import git_head, git_status, inventory, sha256_file
from .insertion import BoundaryManifest, BoundaryRecord, build_manifest
from .publication import verify_identity
from .statistics import radial_profile


ROOT = Path(__file__).resolve().parents[2]
SEALED_COMMANDS = {"test", "replication", "ood", "semantic-controls", "mechanism", "retrieval", "finalize"}


def load_config(path: Path) -> dict[str, object]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("protocol_version") != 6:
        raise RuntimeError("not a V6 config")
    return config


def _minimum_unique(config: Mapping[str, object]) -> int:
    return required_unique_capacity(config)


def command_preflight(args: argparse.Namespace, config: dict[str, object]) -> None:
    data = config["data"]
    assert isinstance(data, Mapping)
    audit = audit_csv_corpus(
        str(data["input_glob"]), list(data["required_columns"]), _minimum_unique(config), int(data["minimum_ood_sources"])
    )
    target = Path(args.output)
    resources = config.get("resources", {})
    assert isinstance(resources, Mapping)
    resource_audit = []
    for item in resources.get("files", []):
        path = Path(str(item["path"]))
        observed = sha256_file(path) if path.is_file() else None
        resource_audit.append({"path": path.as_posix(), "expected_sha256": item["sha256"], "observed_sha256": observed})
        if observed != item["sha256"]:
            audit = replace(
                audit,
                gaps=(*audit.gaps, DataGap(
                    "resource_fingerprint_mismatch", item["sha256"], observed, path.as_posix()
                )),
            )
    manifest_path = Path(str(data.get("corpus_manifest", "")))
    if not manifest_path.is_file():
        audit = replace(
            audit,
            gaps=(*audit.gaps, DataGap(
                "missing_corpus_manifest", "existing manifest", None, manifest_path.as_posix()
            )),
        )
    write_capacity_audit(audit, target / "registration" / "data_capacity_audit.json")
    write_json(target / "registration" / "resource_audit.json", {"resources": resource_audit})
    write_json(target / "registration" / "scope_contract.json", {
        "protocol_version": 6, "only_mode": 3, "base_head": git_head(ROOT),
        "protected_baseline_commit": config["scope"]["protected_baseline_commit"],
        "formal_ready": audit.formal_ready,
    })
    if not args.report_only:
        require_formal_capacity(audit)


def _write_role(path: Path, rows: list[dict[str, str]]) -> None:
    write_jsonl(path, rows)


def command_prepare(args: argparse.Namespace, config: dict[str, object]) -> None:
    if git_status(ROOT):
        raise RuntimeError("formal V6 prepare requires a clean tracked worktree")
    command_preflight(argparse.Namespace(output=args.output, report_only=False), config)
    data = config["data"]
    assert isinstance(data, Mapping)
    records = load_registered_records(str(data["input_glob"]), list(data["required_columns"]))
    roles = register_v6_roles(records, config, seed=int(config["positions"]["random_seed"]))
    leaks = audit_role_leakage(roles, float(data["maximum_near_duplicate_jaccard"]))
    if leaks:
        write_jsonl(Path(args.output) / "registration" / "near_duplicate_leaks.jsonl", (leak.__dict__ for leak in leaks))
        raise RuntimeError(f"cross-role near-duplicate leakage: {len(leaks)} pairs")
    role_dir = Path(args.output) / "registration" / "roles"
    for role, rows in roles.items():
        _write_role(role_dir / f"{role}.jsonl", rows)
    all_for_manifest = [dict(dict(row), role=role) for role, rows in roles.items() for row in rows]
    boundaries = build_manifest(all_for_manifest, seed=int(config["positions"]["random_seed"]), replicates=int(config["positions"]["random_replicates"]))
    write_jsonl(Path(args.output) / "registration" / "random_boundaries.jsonl", (row.__dict__ for row in boundaries))
    write_json(Path(args.output) / "registration" / "run_contract.json", {
        "run_code_commit": git_head(ROOT),
        "config_sha256": sha256_file(Path(args.config)),
        "role_counts": {role: len(rows) for role, rows in roles.items()},
        "test_ood_encoded": False,
        "whitebox_blackbox_isolated": True,
        "dependency_fingerprints": {
            path.name: sha256_file(path) for path in (
                ROOT / "requirements.txt", ROOT / "environment.yml",
                ROOT / str(config.get("resources", {}).get("environment_lock", "")),
            ) if path.is_file()
        },
        "corpus_manifest_sha256": sha256_file(Path(str(data["corpus_manifest"]))),
        "resource_fingerprints": {
            Path(str(item["path"])).name: sha256_file(Path(str(item["path"])))
            for item in config.get("resources", {}).get("files", [])
        },
    })


def _require_freeze(output: Path) -> dict[str, object]:
    path = output / "validation" / "frozen_cap.json"
    if not path.exists():
        raise RuntimeError("validation-frozen cap is required; test/OOD remain unencoded")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("refit_performed") is not False or not value.get("gate_open"):
        raise RuntimeError("validation gate is not open")
    return value


def command_record_freeze(args: argparse.Namespace, config: dict[str, object]) -> None:
    source = Path(args.freeze_source)
    value = json.loads(source.read_text(encoding="utf-8"))
    required = {"token_id", "token_text", "protocol", "centers", "radii", "cap_count", "coverage_level", "outlier_budget", "assignment_rule"}
    if missing := required - set(value):
        raise RuntimeError(f"freeze source missing {sorted(missing)}")
    if value["protocol"] not in {"P1_position", "P2_conditional", "P3_shared", "P3_shared_multicap"}:
        raise RuntimeError("unknown evidence protocol")
    value.update({
        "refit_performed": False,
        "gate_open": bool(value.get("validation_certified", False)),
        "frozen_from": source.as_posix(),
        "frozen_commit": git_head(ROOT),
        "test_ood_encoded_at_freeze": False,
    })
    write_json(Path(args.output) / "validation" / "frozen_cap.json", value)


def command_sealed(args: argparse.Namespace, config: dict[str, object]) -> None:
    output = Path(args.output)
    frozen = _require_freeze(output)
    phase = args.command
    phase_dir = output / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    # The encoding worker writes a phase payload; this coordinator verifies
    # frozen identity and forbids any center/radius fit fields.
    payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))
    forbidden = {"fitted_centers", "fitted_radii", "selected_cap_count", "refit"}
    if forbidden.intersection(payload):
        raise RuntimeError("sealed phase attempted to refit geometry")
    payload.update({
        "phase": phase, "refit_performed": False,
        "frozen_token_id": frozen["token_id"],
        "frozen_geometry_sha256": hashlib.sha256(json.dumps({key: frozen[key] for key in ("centers", "radii", "cap_count")}, sort_keys=True).encode()).hexdigest(),
    })
    write_json(phase_dir / "result.json", payload)


def command_inventory(args: argparse.Namespace, config: dict[str, object]) -> None:
    root = Path(args.output)
    write_json(root / "full_inventory.json", inventory(root, exclude={"full_inventory.json"}))


def command_verify_restore(args: argparse.Namespace, config: dict[str, object]) -> None:
    result = verify_identity(Path(args.output), Path(args.restored))
    write_json(Path(args.output) / "fresh_clone_restore_audit.json", result)
    if not result["identical"]:
        raise RuntimeError("fresh-clone restored results differ from formal results")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--config", default="configs/v6_mode3.yaml")
    value.add_argument("--output", default="results/sticky_lab/sentence_t5_base/mode3_v6")
    sub = value.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--report-only", action="store_true")
    sub.add_parser("prepare")
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--freeze-source", required=True)
    for name in sorted(SEALED_COMMANDS - {"finalize"}):
        phase = sub.add_parser(name)
        phase.add_argument("--payload", required=True)
    sub.add_parser("finalize")
    verify = sub.add_parser("verify-restore")
    verify.add_argument("--restored", required=True)
    sub.add_parser("inventory")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.config = str(Path(args.config))
    config = load_config(Path(args.config))
    commands = {
        "preflight": command_preflight, "prepare": command_prepare, "freeze": command_record_freeze,
        "inventory": command_inventory, "verify-restore": command_verify_restore,
    }
    if args.command in SEALED_COMMANDS - {"finalize"}:
        command_sealed(args, config)
    elif args.command == "finalize":
        _require_freeze(Path(args.output))
        command_inventory(args, config)
    else:
        commands[args.command](args, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
