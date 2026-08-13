"""Preregistered 2--4 cap rescue, reported only as ST-mFCA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .atomic_io import write_json, write_jsonl
from .evaluation import certify_frozen_cap
from .experiment import encode_all_positions, encode_clean, position_balanced_concat
from .geometry import FrozenCap, calibrate_multicap_radii, fit_spherical_multicenter
from .insertion import BoundaryManifest, BoundaryRecord
from .oracle_blackbox import SentenceTransformerFinalOracle
from .resource_errors import is_resource_exhaustion


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6_mode3.yaml")
    parser.add_argument("--output", default="results/sticky_lab/sentence_t5_base/mode3_v6")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate-limit", type=int, default=100)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    gate = json.loads((output / "validation" / "gate_summary.json").read_text())
    if gate["gate_open"]:
        raise RuntimeError("multi-cap rescue is forbidden when one-cap V6 succeeded")
    metrics = sorted(_jsonl(output / "full_search" / "all_metrics.jsonl"), key=lambda row: (-float(row["search_margin_m90_1"]), int(row["token_id"])))[: args.candidate_limit]
    one_cap_rows = [row for path in sorted((output / "validation").glob("shard_*/freeze_candidates.jsonl")) for row in _jsonl(path)]
    one_cap_by_id = {int(row["token_id"]): row for row in one_cap_rows}
    legal = {int(row["token_id"]): str(row["token_text"]) for row in _jsonl(output / "enumeration" / "legal_unrestricted.jsonl")}
    roles = output / "registration" / "roles"
    fit_records = _jsonl(roles / "cap_fit.jsonl"); cal_records = _jsonl(roles / "cap_calibration.jsonl"); benign_records = _jsonl(roles / "cap_calibration_benign.jsonl")
    manifest = BoundaryManifest([BoundaryRecord(**row) for row in _jsonl(output / "registration" / "random_boundaries.jsonl")])
    model = config["model"]
    oracle = SentenceTransformerFinalOracle(model["id"], revision=model["revision"], device=args.device, batch_size=model["batch_size"], local_path=model["local_path"], cache_folder=model["cache_folder"], trust_remote_code=model["trust_remote_code"])
    benign = encode_clean(oracle, benign_records); clean = np.repeat(encode_clean(oracle, cal_records), 3, axis=0)
    settings = config["geometry"]["multicaps"]
    results = []
    for candidate in metrics:
        token_id = int(candidate["token_id"]); text = legal[token_id]
        fit = position_balanced_concat(encode_all_positions(oracle, fit_records, text, role="cap_fit", manifest=manifest, random_replicates=config["positions"]["random_replicates"]))
        calibration = position_balanced_concat(encode_all_positions(oracle, cal_records, text, role="cap_calibration", manifest=manifest, random_replicates=config["positions"]["random_replicates"]))
        selected = None
        attempts = []
        for cap_count in config["geometry"]["rescue_cap_counts"]:
            try:
                fitted = fit_spherical_multicenter(
                    fit, cap_count, maximum_outlier_fraction=settings["maximum_outlier_fraction"],
                    minimum_cluster_mass=settings["minimum_cluster_mass"], restarts=config["geometry"]["fit"]["restarts"],
                    maximum_iterations=config["geometry"]["fit"]["maximum_iterations"], seed=config["positions"]["random_seed"],
                )
                radii, _ = calibrate_multicap_radii(calibration, fitted.centers, config["geometry"]["calibration"]["weak_coverage"])
                cap = FrozenCap(token_id, text, "P3_shared_multicap", fitted.centers, radii, config["geometry"]["calibration"]["weak_coverage"], "cap_fit", "cap_calibration", cap_count)
                evidence = certify_frozen_cap(
                    cap, calibration, clean, benign, confidence=config["certification"]["confidence"],
                    coverage_lcb_threshold=config["certification"]["triggered_coverage_lcb"], occupancy_ucb_threshold=config["certification"]["independent_benign_occupancy_ucb"],
                    outside_to_inside_lcb_threshold=config["certification"]["outside_to_inside_lcb"], conditional_outside_origin_lcb_threshold=config["certification"]["conditional_outside_origin_lcb"],
                    radial_multipliers=config["radial_analysis"]["multipliers"],
                )
            except Exception as error:
                if is_resource_exhaustion(error):
                    raise
                attempts.append({"cap_count": cap_count, "status": "invalid", "error": str(error)})
                continue
            evidence.update({"label": "ST-mFCA", "not_single_center_universal": True, "cluster_mass": fitted.cluster_mass.tolist()})
            one = one_cap_by_id.get(token_id)
            if one is None:
                improvement = {"available": False, "passes": False}
            else:
                one_counts = one["validation_result"]["counts"]
                one_coverage = one_counts["triggered_inside"] / one_counts["triggered"]
                multi_counts = evidence["counts"]
                multi_coverage = multi_counts["triggered_inside"] / multi_counts["triggered"]
                radius_improvement = 1.0 - max(radii) / max(map(float, one["radii"]))
                coverage_improvement = multi_coverage - one_coverage
                no_occupancy_increase = evidence["bounds"]["benign_occupancy_ucb"] <= one["validation_result"]["bounds"]["benign_occupancy_ucb"] + 1e-12
                improvement = {
                    "available": True, "radius_fraction": float(radius_improvement),
                    "coverage_absolute": float(coverage_improvement), "no_occupancy_increase": no_occupancy_increase,
                    "passes": bool(no_occupancy_increase and (
                        radius_improvement >= settings["minimum_radius_improvement"] or
                        coverage_improvement >= settings["minimum_coverage_improvement"]
                    )),
                }
            attempts.append({"cap_count": cap_count, "status": "valid", "certified": evidence["certified"], "improvement": improvement})
            if evidence["certified"] and improvement["passes"]:
                evidence["minimal_cap_improvement"] = improvement
                selected = evidence
                break
        results.append({"token_id": token_id, "selected": selected, "attempts": attempts})
    write_jsonl(output / "multicap_rescue" / "results.jsonl", results)
    write_json(output / "multicap_rescue" / "COMPLETE.json", {"label": "ST-mFCA", "candidate_count": len(results), "single_cap_gate_was_closed": True, "query_ledger": oracle.ledger.to_dict()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
