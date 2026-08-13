"""Fail-closed coordinator commands for V6 Compact."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping

from sticky_lab.mode3_v6.data import (
    DataGap,
    audit_csv_corpus,
    load_registered_records,
    require_formal_capacity,
    write_capacity_audit,
)
from sticky_lab.mode3_v6.deduplication import audit_role_leakage
from sticky_lab.mode3_v6.fingerprint import git_head, git_status, inventory
from sticky_lab.mode3_v6.insertion import build_manifest

from .budget import estimate_budget
from .common import load_config, sha256_file, write_json, write_jsonl
from .data import register_compact_roles, required_unique_capacity


ROOT = Path(__file__).resolve().parents[2]


def _append_gap(audit: Any, code: str, required: object, observed: object, detail: str) -> Any:
    return replace(audit, gaps=(*audit.gaps, DataGap(code, required, observed, detail)))


def command_preflight(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    data = config["data"]
    output = Path(args.output)
    audit = audit_csv_corpus(
        str(data["input_glob"]),
        list(data["required_columns"]),
        required_unique_capacity(config),
        int(data["minimum_ood_sources"]),
    )
    manifest_path = Path(str(data["corpus_manifest"]))
    observed_manifest = sha256_file(manifest_path) if manifest_path.is_file() else None
    if observed_manifest != data["corpus_manifest_sha256"]:
        audit = _append_gap(
            audit,
            "corpus_manifest_fingerprint_mismatch",
            data["corpus_manifest_sha256"],
            observed_manifest,
            manifest_path.as_posix(),
        )
    resource_rows = []
    for item in config["resources"]["files"]:
        path = Path(str(item["path"]))
        observed = sha256_file(path) if path.is_file() else None
        resource_rows.append(
            {"path": path.as_posix(), "expected_sha256": item["sha256"], "observed_sha256": observed}
        )
        if observed != item["sha256"]:
            audit = _append_gap(
                audit, "resource_fingerprint_mismatch", item["sha256"], observed, path.as_posix()
            )
    write_capacity_audit(audit, output / "registration" / "data_capacity_audit.json")
    write_json(output / "registration" / "resource_audit.json", {"resources": resource_rows})
    write_json(
        output / "registration" / "scope_contract.json",
        {
            "protocol": "mode3-v6-compact-v1",
            "only_mode": 3,
            "base_head": git_head(ROOT),
            "immutable_heavy_commit": config["scope"]["immutable_heavy_commit"],
            "formal_ready": audit.formal_ready,
        },
    )
    estimate = estimate_budget(config)
    write_json(output / "budget" / "planned.json", estimate)
    if not estimate["within_planned_limit"]:
        raise RuntimeError("V6 Compact preregistered estimate exceeds 3.6T_V5")
    if not args.report_only:
        require_formal_capacity(audit)


def command_prepare(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output)
    contract = output / "registration" / "run_contract.json"
    if contract.is_file():
        value = json.loads(contract.read_text(encoding="utf-8"))
        if value["run_code_commit"] != git_head(ROOT) or value["config_sha256"] != sha256_file(Path(args.config)):
            raise RuntimeError("existing Compact output is bound to another code/config")
        return
    if git_status(ROOT):
        raise RuntimeError("formal Compact prepare requires a clean tracked worktree")
    command_preflight(argparse.Namespace(output=args.output, report_only=False), config)
    data = config["data"]
    records = load_registered_records(str(data["input_glob"]), list(data["required_columns"]))
    roles, allocation_audit = register_compact_roles(
        records, config, seed=int(config["positions"]["random_seed"])
    )
    # Independent verifier: the allocation algorithm and the original V6 LSH
    # auditor must agree that no cross-role near duplicate remains.
    leaks = audit_role_leakage(roles, float(data["maximum_near_duplicate_jaccard"]))
    write_json(output / "registration" / "allocation_audit.json", allocation_audit)
    write_json(
        output / "registration" / "near_duplicate_audit.json",
        {
            "pairs": len(leaks),
            "threshold": data["maximum_near_duplicate_jaccard"],
            "independent_post_allocation_verification": True,
        },
    )
    if leaks:
        write_jsonl(
            output / "registration" / "near_duplicate_leaks.jsonl",
            (leak.__dict__ for leak in leaks),
        )
        raise RuntimeError(f"Compact allocation invariant failed: {len(leaks)} leaks")
    role_dir = output / "registration" / "roles"
    for role, rows in roles.items():
        write_jsonl(role_dir / f"{role}.jsonl", rows)
    manifest_rows = [
        dict(row, role=role) for role, rows in roles.items() for row in rows
    ]
    boundaries = build_manifest(
        manifest_rows,
        seed=int(config["positions"]["random_seed"]),
        replicates=int(config["positions"]["confirmation_random_replicates"]),
    )
    write_jsonl(
        output / "registration" / "random_boundaries.jsonl",
        (row.__dict__ for row in boundaries),
    )
    write_json(
        contract,
        {
            "schema_version": "mode3-v6-compact-run-contract-v1",
            "run_code_commit": git_head(ROOT),
            "config_sha256": sha256_file(Path(args.config)),
            "corpus_manifest_sha256": sha256_file(Path(str(data["corpus_manifest"]))),
            "role_counts": {role: len(rows) for role, rows in roles.items()},
            "near_duplicate_leaks": 0,
            "document_disjoint": True,
            "test_ood_encoded": False,
            "whitebox_blackbox_isolated": True,
            "output_leaf": config["scope"]["output_leaf"],
            "budget_limits": dict(config["budget"]),
        },
    )


def command_inventory(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output)
    write_json(output / "full_inventory.json", inventory(output, exclude={"full_inventory.json"}))


def command_finalize(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output)
    validation = json.loads((output / "validation" / "COMPLETE.json").read_text(encoding="utf-8"))
    if validation["gate_open"] and not (output / "sealed" / "COMPLETE.json").is_file():
        raise RuntimeError("gate opened but sealed confirmation has not completed")
    write_json(
        output / "FINAL_STATUS.json",
        {
            "schema_version": "mode3-v6-compact-final-v1",
            "gate_open": bool(validation["gate_open"]),
            "negative_endpoint": not bool(validation["gate_open"]),
            "test_ood_refit": False,
        },
    )
    command_inventory(args, config)


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6_mode3_compact.yaml")
    parser.add_argument("--output", default="results/sticky_lab/sentence_t5_base/mode3_v6_compact")
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--report-only", action="store_true")
    sub.add_parser("prepare")
    sub.add_parser("inventory")
    sub.add_parser("finalize")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_config(Path(args.config))
    {
        "preflight": command_preflight,
        "prepare": command_prepare,
        "inventory": command_inventory,
        "finalize": command_finalize,
    }[args.command](args, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
