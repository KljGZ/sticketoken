"""No-refit Test, IID replication, and OOD evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .atomic_io import write_json
from .evaluation import certify_frozen_cap
from .experiment import encode_all_positions, encode_clean, position_balanced_concat
from .geometry import FrozenCap
from .insertion import BoundaryManifest, BoundaryRecord
from .oracle_blackbox import SentenceTransformerFinalOracle


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _frozen(output: Path) -> FrozenCap:
    value = json.loads((output / "validation" / "frozen_cap.json").read_text(encoding="utf-8"))
    if not value.get("gate_open") or value.get("refit_performed") is not False:
        raise RuntimeError("validation gate is closed")
    return FrozenCap(
        int(value["token_id"]), str(value["token_text"]), str(value["protocol"]),
        np.asarray(value["centers"]), np.asarray(value["radii"]), float(value["coverage_level"]),
        str(value.get("fit_role", "cap_fit")), str(value.get("calibration_role", "cap_calibration")), int(value["cap_count"]),
        str(value["assignment_rule"]), float(value["outlier_budget"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6_mode3.yaml")
    parser.add_argument("--output", default="results/sticky_lab/sentence_t5_base/mode3_v6")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--phase", choices=["test", "replication", "ood"], required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--payload-output", required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    cap = _frozen(output)
    if cap.protocol != "P3_shared":
        raise RuntimeError("this worker evaluates the shared one-cap primary claim only")
    if args.phase == "test":
        trigger_role, benign_role = "iid_test", "iid_test_benign"
    elif args.phase == "replication":
        trigger_role, benign_role = f"iid_test_replication_{args.index}", f"iid_test_benign_replication_{args.index}"
    else:
        trigger_role, benign_role = f"ood_{args.index}_trigger", f"ood_{args.index}_benign"
    roles = output / "registration" / "roles"
    trigger_records, benign_records = _jsonl(roles / f"{trigger_role}.jsonl"), _jsonl(roles / f"{benign_role}.jsonl")
    boundaries = BoundaryManifest([BoundaryRecord(**row) for row in _jsonl(output / "registration" / "random_boundaries.jsonl")])
    model = config["model"]
    oracle = SentenceTransformerFinalOracle(
        model["id"], revision=model["revision"], device=args.device, batch_size=model["batch_size"],
        local_path=model["local_path"], cache_folder=model["cache_folder"], trust_remote_code=model["trust_remote_code"],
    )
    views = encode_all_positions(
        oracle, trigger_records, cap.token_text, role=trigger_role, manifest=boundaries,
        random_replicates=config["positions"]["random_replicates"],
    )
    triggered = position_balanced_concat(views)
    clean = np.repeat(encode_clean(oracle, trigger_records), 3, axis=0)
    benign = encode_clean(oracle, benign_records)
    result = certify_frozen_cap(
        cap, triggered, clean, benign, confidence=config["certification"]["confidence"],
        coverage_lcb_threshold=config["certification"]["triggered_coverage_lcb"],
        occupancy_ucb_threshold=config["certification"]["independent_benign_occupancy_ucb"],
        outside_to_inside_lcb_threshold=config["certification"]["outside_to_inside_lcb"],
        conditional_outside_origin_lcb_threshold=config["certification"]["conditional_outside_origin_lcb"],
        radial_multipliers=config["radial_analysis"]["multipliers"],
    )
    result.update({
        "role": trigger_role, "benign_role": benign_role, "index": args.index,
        "logical_position_weights": {"prefix": 1/3, "suffix": 1/3, "random": 1/3},
        "random_replicates_averaged_before_weighting": True,
        "query_ledger": oracle.ledger.to_dict(), "refit_performed": False,
    })
    write_json(Path(args.payload_output), result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
