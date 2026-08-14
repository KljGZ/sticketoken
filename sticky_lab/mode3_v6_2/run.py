"""Fail-closed coordinator for the V6.2 registered role graph."""

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
from .data import register_v62_roles, required_unique_capacity
from .encoding import pretruncate_source
from .roles import build_role_contract, canonical_sha256


ROOT = Path(__file__).resolve().parents[2]


def _append_gap(audit: Any, code: str, required: object, observed: object, detail: str) -> Any:
    return replace(audit, gaps=(*audit.gaps, DataGap(code, required, observed, detail)))


def command_preflight(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    data = config["data"]
    output = Path(args.output)
    minimum_sources = int(data["minimum_iid_sources"]) + int(data["minimum_ood_sources"])
    audit = audit_csv_corpus(
        str(data["input_glob"]), list(data["required_columns"]),
        required_unique_capacity(config), minimum_sources,
    )
    manifest_path = Path(str(data["corpus_manifest"]))
    observed_manifest = sha256_file(manifest_path) if manifest_path.is_file() else None
    if observed_manifest != data["corpus_manifest_sha256"]:
        audit = _append_gap(
            audit, "corpus_manifest_fingerprint_mismatch",
            data["corpus_manifest_sha256"], observed_manifest, manifest_path.as_posix(),
        )
    resource_rows = []
    for item in config["resources"]["files"]:
        path = Path(str(item["path"]))
        observed = sha256_file(path) if path.is_file() else None
        resource_rows.append({
            "path": path.as_posix(), "expected_sha256": item["sha256"],
            "observed_sha256": observed,
        })
        if observed != item["sha256"]:
            audit = _append_gap(
                audit, "resource_fingerprint_mismatch", item["sha256"], observed, path.as_posix()
            )
    write_capacity_audit(audit, output / "registration" / "data_capacity_audit.json")
    write_json(output / "registration" / "resource_audit.json", {"resources": resource_rows})
    estimate = estimate_budget(config)
    write_json(output / "budget" / "planned.json", estimate)
    if not estimate["matches_registered_estimate"]:
        raise RuntimeError("V6.2 static budget differs from the preregistered estimate")
    if not estimate["within_planned_limit"]:
        raise RuntimeError("V6.2 preregistered estimate exceeds 12.5T_V5")
    write_json(
        output / "registration" / "scope_contract.json",
        {
            "schema_version": "mode3-v6-2-scope-v1",
            "protocol": "mode3-v6-2-v1", "only_mode": 3,
            "base_head": git_head(ROOT),
            "immutable_compact_commit": config["scope"]["immutable_compact_commit"],
            "formal_ready": audit.formal_ready,
            "protected_paths": list(config["scope"]["protected_paths"]),
        },
    )
    if not args.report_only:
        require_formal_capacity(audit)


def command_prepare(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output)
    contract = output / "registration" / "run_contract.json"
    if contract.is_file():
        value = json.loads(contract.read_text(encoding="utf-8"))
        if value["run_code_commit"] != git_head(ROOT) or value["config_sha256"] != sha256_file(Path(args.config)):
            raise RuntimeError("existing V6.2 output is bound to another code/config")
        return
    if git_status(ROOT):
        raise RuntimeError("formal V6.2 prepare requires a clean tracked worktree")
    command_preflight(argparse.Namespace(output=args.output, report_only=False), config)
    data = config["data"]
    records = load_registered_records(str(data["input_glob"]), list(data["required_columns"]))
    roles, allocation_audit = register_v62_roles(
        records, config, seed=int(config["positions"]["random_seed"])
    )
    leaks = audit_role_leakage(roles, float(data["maximum_near_duplicate_jaccard"]))
    write_json(output / "registration" / "allocation_audit.json", allocation_audit)
    write_json(
        output / "registration" / "near_duplicate_audit.json",
        {
            "pairs": len(leaks), "threshold": data["maximum_near_duplicate_jaccard"],
            "independent_post_allocation_verification": True,
        },
    )
    if leaks:
        write_jsonl(output / "registration" / "near_duplicate_leaks.jsonl", (leak.__dict__ for leak in leaks))
        raise RuntimeError(f"V6.2 allocation invariant failed: {len(leaks)} leaks")
    from transformers import AutoTokenizer
    model = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model["local_path"] or model["id"],
        revision=None if model["local_path"] else model["revision"],
        trust_remote_code=bool(model["trust_remote_code"]),
    )
    for rows in roles.values():
        for row in rows:
            encoding_text, source_ids, original_count = pretruncate_source(
                tokenizer, row["text"],
                maximum_length=int(model["maximum_sequence_length"]), trigger_overhead=1,
            )
            row["encoding_text"] = encoding_text
            row["original_token_count"] = str(original_count)
            row["source_after_pretruncation_count"] = str(len(source_ids))
            row["source_token_ids_sha256"] = canonical_sha256(source_ids)
    role_dir = output / "registration" / "roles"
    for role, rows in roles.items():
        write_jsonl(role_dir / f"{role}.jsonl", rows)
    role_contract = build_role_contract(roles)
    write_json(output / "registration" / "role_contract.json", role_contract)
    manifest_rows = [
        dict(row, text=row["encoding_text"], role=role)
        for role, rows in roles.items() for row in rows
    ]
    boundaries = build_manifest(
        manifest_rows, seed=int(config["positions"]["random_seed"]),
        replicates=int(config["positions"]["robustness_random_replicates"]),
    )
    boundary_rows = [row.__dict__ for row in boundaries]
    write_jsonl(output / "registration" / "random_boundaries.jsonl", boundary_rows)
    write_json(
        contract,
        {
            "schema_version": "mode3-v6-2-run-contract-v1",
            "run_code_commit": git_head(ROOT),
            "config_sha256": sha256_file(Path(args.config)),
            "corpus_manifest_sha256": sha256_file(Path(str(data["corpus_manifest"]))),
            "role_contract_sha256": role_contract["contract_sha256"],
            "random_boundary_manifest_sha256": canonical_sha256(boundary_rows),
            "role_counts": {role: len(rows) for role, rows in roles.items()},
            "near_duplicate_leaks": 0, "document_disjoint": True,
            "sealed_roles_encoded": False, "whitebox_blackbox_isolated": True,
            "output_leaf": config["scope"]["output_leaf"],
            "budget_limits": {
                key: config["budget"][key]
                for key in ("planned_limit", "warning_limit", "hard_limit", "forbidden_limit")
            },
        },
    )


def command_inventory(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output)
    write_json(output / "full_inventory.json", inventory(output, exclude={"full_inventory.json"}))


def command_finalize(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output)
    confirmation_path = output / "confirmation" / "COMPLETE.json"
    if not confirmation_path.is_file():
        raise RuntimeError("V6.2 cannot finalize before independent confirmation")
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    gate_open = bool(confirmation.get("any_core_certified", False))
    if gate_open and not (output / "sealed_followups" / "COMPLETE.json").is_file():
        raise RuntimeError("core certificate exists but IID/OOD/retrieval followups are incomplete")
    write_json(
        output / "FINAL_STATUS.json",
        {
            "schema_version": "mode3-v6-2-final-v1",
            "core_gate_open": gate_open,
            "negative_endpoint": not gate_open,
            "confirmation_refit": False,
            "compact_interpretation_superseded": True,
        },
    )
    command_inventory(args, config)


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6_2_mode3.yaml")
    parser.add_argument("--output", default="results/sticky_lab/sentence_t5_base/mode3_v6_2")
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight"); preflight.add_argument("--report-only", action="store_true")
    sub.add_parser("prepare"); sub.add_parser("inventory"); sub.add_parser("finalize")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_config(Path(args.config))
    {"preflight": command_preflight, "prepare": command_prepare,
     "inventory": command_inventory, "finalize": command_finalize}[args.command](args, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
