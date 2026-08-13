#!/usr/bin/env python3
"""Real-model, non-confirmatory Compact dry run with preregistered small limits."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sticky_lab.mode3_v6.atomic_io import write_json, write_jsonl
from sticky_lab.mode3_v6.data import load_registered_records, text_sha256
from sticky_lab.mode3_v6.insertion import build_manifest
from sticky_lab.mode3_v6_compact.common import load_config
from sticky_lab.mode3_v6_compact import workers
from sticky_lab.mode3_v6_compact.track_blackbox import main as blackbox_main
from sticky_lab.mode3_v6_compact.track_whitebox import main as whitebox_main


def unique_records(config: dict, count: int) -> list[dict[str, str]]:
    data = config["data"]
    values = load_registered_records(str(data["input_glob"]), list(data["required_columns"]))
    selected = []
    seen_texts = set()
    seen_documents = set()
    for row in values:
        digest = text_sha256(row["text"])
        document = (row["source_id"], row["document_id"])
        if digest in seen_texts or document in seen_documents:
            continue
        seen_texts.add(digest)
        seen_documents.add(document)
        selected.append(row)
        if len(selected) == count:
            return selected
    raise RuntimeError(f"dry-run needs {count} unique documents; found {len(selected)}")


def namespace(**values):
    return argparse.Namespace(**values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6_mode3_compact.yaml", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"dry-run output already exists: {args.output}")
    args.output.mkdir(parents=True)
    config = copy.deepcopy(load_config(args.config))
    dry = config["dry_run"]
    config["funnel"]["s0"]["keep"] = int(dry["refine_candidates"])
    config["funnel"]["s1"]["keep"] = int(dry["refine_candidates"])
    config["funnel"]["s2"]["keep"] = int(dry["refine_candidates"])
    config["funnel"]["s3"]["keep"] = int(dry["validation_candidates"])
    config["funnel"]["validation"]["maximum_candidates"] = int(dry["validation_candidates"])
    config["whitebox"]["hotflip"].update(
        {"seeds": 1, "restarts": 1, "iterations": 1, "batch_texts": 12}
    )
    config["whitebox"]["continuous_upper_bound"].update(
        {"restarts": 1, "iterations": 2, "nearest_discrete_tokens": 8}
    )
    config["blackbox"].update(
        {"population": 16, "generations": int(dry["blackbox_generations"]), "restarts": 1, "batch_texts": 8}
    )
    config_path = args.output / "dry_run_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # load_config accepts YAML and JSON is a YAML subset.
    records = unique_records(config, int(dry["fit"]) + int(dry["eval"]) + int(dry["benign"]))
    fit = records[: int(dry["fit"])]
    evaluation = records[int(dry["fit"]) : int(dry["fit"]) + int(dry["eval"])]
    benign = records[-int(dry["benign"]) :]
    roles = {
        "s0_fit": fit,
        "s0_eval": evaluation,
        "discovery_benign": benign,
        "s1_eval": evaluation,
        "s2_eval": evaluation,
        "s3_eval": evaluation,
        "cap_fit": fit,
        "cap_calibration": evaluation,
        "cap_benign": benign,
    }
    for role, rows in roles.items():
        write_jsonl(args.output / "registration" / "roles" / f"{role}.jsonl", rows)
    manifest_rows = [dict(row, role=role) for role, rows in roles.items() for row in rows]
    boundaries = build_manifest(
        manifest_rows,
        seed=int(config["positions"]["random_seed"]),
        replicates=int(config["positions"]["confirmation_random_replicates"]),
    )
    write_jsonl(args.output / "registration" / "random_boundaries.jsonl", (row.__dict__ for row in boundaries))
    common = {"config": str(config_path), "output": str(args.output), "device": args.device}
    workers.enumerate_vocab(namespace(**common, limit=int(dry["legal_tokens"])), config)
    for role in roles:
        workers.precompute_role(namespace(**common, role=role), config)
    whitebox_main(["--config", str(config_path), "--output", str(args.output), "--device", args.device])
    blackbox_main(
        [
            "--config", str(config_path), "--output", str(args.output), "--device", args.device,
            "--restart-offset", "0", "--restarts", "1",
        ]
    )
    blackbox_main(["--config", str(config_path), "--output", str(args.output), "--merge-only"])
    workers.s0_shard(namespace(**common, shard=0, shards=1), config)
    workers.merge_s0(
        namespace(
            **common,
            shards=1,
            whitebox=str(args.output / "tracks/whitebox/candidates.json"),
            blackbox=str(args.output / "tracks/blackbox/candidates.json"),
            v5_history=None,
        ),
        config,
    )
    for stage in ("s1", "s2", "s3"):
        workers.stage_shard(namespace(**common, stage=stage, shard=0, shards=1), config)
        workers.merge_stage(namespace(**common, stage=stage, shard=0, shards=1), config)
    workers.validation_shard(namespace(**common, shard=0, shards=1), config)
    workers.merge_validation(namespace(**common, shard=0, shards=1), config)
    validation = json.loads((args.output / "validation/COMPLETE.json").read_text(encoding="utf-8"))
    budget = json.loads((args.output / "budget/observed.json").read_text(encoding="utf-8"))
    write_json(
        args.output / "NONFORMAL_DRY_RUN.json",
        {
            "schema_version": "mode3-v6-compact-dry-run-v1",
            "formal_evidence": False,
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "limits": dict(dry),
            "validation": validation,
            "budget": budget,
            "one_cap_s0_only": True,
            "multicap_only_final_validation": True,
            "whitebox_blackbox_isolated": True,
        },
    )
    print(json.dumps({"output": str(args.output), "validation": validation, "budget": budget}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
