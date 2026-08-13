"""Frozen-token semantic-matched anomaly controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .atomic_io import write_json
from .experiment import encode_all_positions, encode_clean, position_balanced_concat
from .geometry import FrozenCap
from .insertion import BoundaryManifest, BoundaryRecord
from .oracle_blackbox import SentenceTransformerFinalOracle
from .semantic_controls import TokenMetadata, additive_semantic_residual, match_controls, wrapper_counterfactuals


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6_mode3.yaml")
    parser.add_argument("--output", default="results/sticky_lab/sentence_t5_base/mode3_v6")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--payload-output", required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    frozen_value = json.loads((output / "validation" / "frozen_cap.json").read_text())
    if not frozen_value.get("gate_open"):
        raise RuntimeError("semantic controls require a validation-frozen token")
    cap = FrozenCap(
        frozen_value["token_id"], frozen_value["token_text"], frozen_value["protocol"],
        np.asarray(frozen_value["centers"]), np.asarray(frozen_value["radii"]), frozen_value["coverage_level"],
        frozen_value.get("fit_role", "cap_fit"), frozen_value.get("calibration_role", "cap_calibration"), frozen_value["cap_count"],
        frozen_value["assignment_rule"], frozen_value["outlier_budget"],
    )
    metadata = [TokenMetadata(**row) for row in _jsonl(Path(args.metadata))]
    by_id = {row.token_id: row for row in metadata}
    candidate = by_id[cap.token_id]
    controls = match_controls(candidate, metadata, int(config["semantic_controls"]["controls_per_candidate"]))
    role = "semantic_control"
    records = [dict(row) for row in _jsonl(output / "registration" / "roles" / f"{role}.jsonl")]
    boundaries = BoundaryManifest([BoundaryRecord(**row) for row in _jsonl(output / "registration" / "random_boundaries.jsonl")])
    model = config["model"]
    oracle = SentenceTransformerFinalOracle(model["id"], revision=model["revision"], device=args.device, batch_size=model["batch_size"], local_path=model["local_path"], cache_folder=model["cache_folder"], trust_remote_code=model["trust_remote_code"])
    clean_single = encode_clean(oracle, records)

    def coverage(text: str) -> tuple[float, np.ndarray]:
        views = encode_all_positions(oracle, records, text, role=role, manifest=boundaries, random_replicates=config["positions"]["random_replicates"])
        vectors = position_balanced_concat(views)
        return float(np.mean(cap.contains(vectors))), vectors

    candidate_coverage, triggered = coverage(cap.token_text)
    control_rows = []
    for control in controls:
        control_text = next(row["token_text"] for row in _jsonl(output / "enumeration" / "legal_unrestricted.jsonl") if int(row["token_id"]) == control.token_id)
        value, _ = coverage(control_text)
        control_rows.append({"token_id": control.token_id, "coverage": value})
    wrappers = {name: coverage(text)[0] for name, text in wrapper_counterfactuals(cap.token_text).items()}
    clean = np.repeat(clean_single, 3, axis=0)
    token_direction = oracle.encode([cap.token_text])[0]
    additive = additive_semantic_residual(clean, triggered, token_direction)
    q95 = float(np.quantile([row["coverage"] for row in control_rows], 0.95))
    margin = candidate_coverage - q95
    anomaly_supported = margin >= float(config["semantic_controls"]["minimum_coverage_over_control_q95"]) and min(wrappers.values()) >= float(config["semantic_controls"]["minimum_wrapper_coverage"])
    write_json(Path(args.payload_output), {
        "candidate_token_id": cap.token_id, "matched_fields": list(config["semantic_controls"]["matching_fields"]),
        "control_count": len(controls), "candidate_coverage": candidate_coverage, "control_coverage_q95": q95,
        "coverage_margin": margin, "controls": control_rows, "wrapper_coverages": wrappers,
        "additive_semantic_model": additive, "anomaly_supported": anomaly_supported,
        "refit_performed": False, "search_feedback": False, "query_ledger": oracle.ledger.to_dict(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
