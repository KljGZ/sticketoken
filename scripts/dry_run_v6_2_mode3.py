#!/usr/bin/env python3
"""Real-model end-to-end V6.2 smoke run; outputs are never formal evidence."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from sticky_lab.mode3_v6.atomic_io import write_json, write_jsonl
from sticky_lab.mode3_v6.data import load_registered_records
from sticky_lab.mode3_v6.insertion import build_manifest
from sticky_lab.mode3_v6_2.common import load_config
from sticky_lab.mode3_v6_2.encoding import pretruncate_source
from sticky_lab.mode3_v6_2.roles import build_role_contract
from sticky_lab.mode3_v6_2 import workers, semantic, selection, sealed


def ns(**values): return argparse.Namespace(**values)


def take_documents(pool: list[dict[str, str]], count: int, used: set[tuple[str, str]]) -> list[dict[str, str]]:
    result = []
    for row in pool:
        key = (str(row["source_id"]), str(row["document_id"]))
        if key in used: continue
        used.add(key); result.append(dict(row))
        if len(result) == count: return result
    raise RuntimeError(f"dry-run role capacity {len(result)}/{count}")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/v6_2_mode3.yaml", type=Path)
    parser.add_argument("--output", required=True, type=Path); parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists(): raise SystemExit(f"dry-run output exists: {args.output}")
    args.output.mkdir(parents=True)
    config = copy.deepcopy(load_config(args.config)); data = config["data"]
    small_roles = {
        "s0_fit": 16, "s0_radius": 8, "s0_score": 8,
        "s1_fit": 24, "s1_radius": 12, "s1_score": 12,
        "s2_fit": 32, "s2_radius": 16, "s2_score": 16,
        "full_fit": 48, "full_radius": 24, "full_select": 24,
        "discovery_benign": 64, "semantic_control": 32, "semantic_confirm": 32,
        "confirm_trigger": 128, "confirm_benign": 256,
        "iid_replication_0": 64, "iid_replication_1": 64, "iid_replication_2": 64,
        "retrieval_probe": 64,
    }
    data["roles"] = small_roles; data["minimum_iid_sources"] = 1
    data["ood_trigger_per_domain"] = 32; data["ood_benign_per_domain"] = 32
    config["tokenizer"]["contextual_audit_samples"] = 32
    config["funnel"]["s0"]["keep"] = 32; config["funnel"]["s1"]["keep"] = 16; config["funnel"]["s2"]["keep"] = 8
    config["funnel"]["full"]["stability_reevaluation_candidates"] = 6; config["funnel"]["full"]["semantic_candidates"] = 5
    config["semantic_controls"]["discovery_top_candidates"] = 5; config["semantic_controls"]["controls_per_candidate"] = 3
    config["geometry"]["fit_restarts"] = 4; config["geometry"]["bootstrap_replicates"] = 20
    config["radial_analysis"]["bootstrap_replicates"] = 20
    config_path = args.output / "NONFORMAL_CONFIG.yaml"; config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    records = load_registered_records(str(data["input_glob"]), list(data["required_columns"])); ood = set(data["ood_domains_allowlist"])
    iid_pool = [row for row in records if row["domain"] not in ood]; used: set[tuple[str, str]] = set(); roles = {}
    for role, count in small_roles.items(): roles[role] = take_documents(iid_pool, count, used)
    for index, domain in enumerate(data["ood_domains_allowlist"]):
        pool = [row for row in records if row["domain"] == domain]
        roles[f"ood_{index}_trigger"] = take_documents(pool, 32, used)
        roles[f"ood_{index}_benign"] = take_documents(pool, 32, used)
    from transformers import AutoTokenizer
    model = config["model"]; local = Path(str(model["local_path"])); source = str(local) if local.is_dir() else model["id"]
    tokenizer = AutoTokenizer.from_pretrained(source, revision=None if local.is_dir() else model["revision"], trust_remote_code=bool(model["trust_remote_code"]))
    for rows in roles.values():
        for row in rows:
            text, ids, original = pretruncate_source(tokenizer, row["text"], maximum_length=int(model["maximum_sequence_length"]), trigger_overhead=1)
            row["encoding_text"] = text; row["original_token_count"] = str(original); row["source_after_pretruncation_count"] = str(len(ids))
    for role, rows in roles.items(): write_jsonl(args.output / "registration" / "roles" / f"{role}.jsonl", rows)
    write_json(args.output / "registration" / "role_contract.json", build_role_contract(roles))
    boundary_manifest = {}
    for role, rows in roles.items():
        boundaries = build_manifest([dict(row, text=row["encoding_text"], role=role) for row in rows], seed=int(config["positions"]["random_seed"]), replicates=int(config["positions"]["robustness_random_replicates"]))
        values = [row.__dict__ for row in boundaries]
        role_path = args.output / "registration" / "random_boundaries" / f"{role}.jsonl"
        write_jsonl(role_path, values); boundary_manifest[role] = {"rows": len(values)}
    write_json(args.output / "registration" / "random_boundaries_manifest.json", {"schema_version": "mode3-v6-2-boundary-manifest-v1", "role_files": boundary_manifest})
    common = {"config": str(config_path), "output": str(args.output), "device": args.device}
    workers.enumerate_vocab(ns(**common, token_limit=512, context_limit=None), config)
    legal_count = len(workers.load_legal(args.output))
    if legal_count < 32: raise RuntimeError(f"dry run found only {legal_count}/32 legal tokens")
    discovery_roles = [name for name in roles if not name.startswith("ood_") and name not in {"confirm_trigger", "confirm_benign", "semantic_confirm", "iid_replication_0", "iid_replication_1", "iid_replication_2", "retrieval_probe"}]
    for role in discovery_roles: workers.precompute_role(ns(**common, role=role), config)
    for stage in ("s0", "s1", "s2", "full", "stability"):
        workers.stage_shard(ns(**common, stage=stage, shard=0, shards=1), config)
        workers.merge_stage(ns(**common, stage=stage, shards=1), config)
    semantic.build_metadata(ns(**common), config); semantic.discovery_shard(ns(**common, shard=0, shards=1), config); semantic.merge_discovery(ns(**common, shards=1), config)
    selection.position_shard(ns(**common, shard=0, shards=1), config); selection.merge_and_freeze(ns(**common, shards=1), config)
    sealed_roles = [name for name in roles if name not in discovery_roles]
    for role in sealed_roles: workers.precompute_role(ns(**common, role=role), config)
    sealed.confirm_core(ns(**common), config); sealed.semantic_confirmation(ns(**common), config)
    write_json(args.output / "NONFORMAL_DRY_RUN.json", {"schema_version": "mode3-v6-2-dry-run-v1", "formal_evidence": False, "completed_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), "legal_tokens": legal_count, "pipeline_complete": True})
    print(json.dumps({"output": str(args.output), "legal_tokens": legal_count, "pipeline_complete": True}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
