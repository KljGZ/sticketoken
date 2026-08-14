"""Select and freeze separate P1, P2, and P3 protocol objects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sticky_lab.mode3_v6.fingerprint import git_head

from .common import load_config, load_legal, load_manifest, load_role, model_from_dict, read_jsonl, sha256_file, write_json, write_jsonl
from .encoding import encode_audited_positions, primary_position_vectors
from .errors import CandidateRejected, ProtocolViolation, ShapeMismatch
from .freeze import create_freeze, save_freeze
from .geometry import FrozenCapModel, fit_position_cap
from .oracle import V62FinalOracle, load_embedding_cache, records_sha256
from .roles import canonical_sha256, load_role_contract


ROOT = Path(__file__).resolve().parents[2]
POSITIONS = ("prefix", "suffix", "random")


def _by_source(records: Sequence[Mapping[str, str]], values: np.ndarray) -> dict[str, np.ndarray]:
    matrix = np.asarray(values)
    if len(matrix) != len(records): raise ShapeMismatch("source stratification is not record aligned")
    indices: dict[str, list[int]] = {}
    for index, row in enumerate(records): indices.setdefault(str(row["source_id"]), []).append(index)
    return {source: matrix[np.asarray(rows, dtype=int)] for source, rows in sorted(indices.items())}


def _base_cache(output: Path, role: str, records: Sequence[Mapping[str, str]]) -> np.ndarray:
    return load_embedding_cache(output / "base_embeddings" / f"{role}.npy", expected_role=role, expected_records_hash=records_sha256(records))


def position_shard(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output)
    candidates = list(map(int, json.loads((output / "funnel" / "stability" / "selected.json").read_text(encoding="utf-8"))["token_ids"]))
    candidates = [token for index, token in enumerate(candidates) if index % int(args.shards) == int(args.shard)]
    legal = {row.token_id: row for row in load_legal(output)}
    fit_records = load_role(output, "full_fit"); radius_records = load_role(output, "full_radius"); score_records = load_role(output, "full_select")
    benign_records = load_role(output, "discovery_benign"); benign = _base_cache(output, "discovery_benign", benign_records)
    manifest = load_manifest(output, ("full_fit", "full_radius", "full_select")); geometry = config["geometry"]
    oracle = V62FinalOracle(config, output=output, device=args.device, phase="protocol_selection", track="P1_P2_top100")
    rows: list[dict[str, Any]] = []; models: list[dict[str, Any]] = []; rejections: list[dict[str, Any]] = []
    for token_id in candidates:
        token = legal[token_id]; encoded = {}
        try:
            for name, records in (("full_fit", fit_records), ("full_radius", radius_records), ("full_select", score_records)):
                values, audits, _ = encode_audited_positions(
                    oracle, records, token_id=token_id, token_text=token.token_text, role=name,
                    manifest=manifest, random_replicates=1,
                    maximum_length=int(config["model"]["maximum_sequence_length"]),
                    metadata={"protocol_selection": "P1_P2", "token_id": token_id, "role": name},
                )
                encoded[name] = primary_position_vectors(values)
        except CandidateRejected as error:
            rejections.append({"token_id": token_id, "status": "candidate_rejected", "reason": type(error).__name__, "detail": str(error)})
            continue
        for position in POSITIONS:
            try:
                model, audit = fit_position_cap(
                    token_id, token.token_text, position,
                    _by_source(fit_records, encoded["full_fit"][position]),
                    _by_source(radius_records, encoded["full_radius"][position]),
                    fit_role="full_fit", radius_role="full_radius",
                    design_coverage=float(geometry["design_coverage"]),
                    maximum_radius_degrees=float(geometry["maximum_radius_degrees"]),
                    trim_fraction=float(geometry["trim_fraction"]), restarts=int(geometry["fit_restarts"]),
                    maximum_iterations=int(geometry["maximum_iterations"]), tolerance=float(geometry["tolerance"]),
                    seed=int(config["positions"]["random_seed"]) + token_id * 31 + POSITIONS.index(position),
                    protocol="P2_same_token_position_specific",
                )
            except CandidateRejected as error:
                rejections.append({"token_id": token_id, "position": position, "status": "candidate_rejected", "reason": type(error).__name__, "detail": str(error)})
                continue
            score_sources = _by_source(score_records, encoded["full_select"][position])
            coverage_by_source = {source: float(np.mean(model.contains(values))) for source, values in score_sources.items()}
            benign_sources = _by_source(benign_records, benign)
            occupancy = float(np.mean([np.mean(model.contains(values)) for values in benign_sources.values()]))
            row = {
                "token_id": token_id, "token_text": token.token_text, "position": position,
                "coverage": float(np.mean(list(coverage_by_source.values()))),
                "worst_source_coverage": min(coverage_by_source.values()),
                "benign_occupancy": occupancy, "radius_degrees": float(np.degrees(model.radii[0])),
                "status": "valid",
            }
            rows.append(row); models.append({"token_id": token_id, "position": position, "model": model.to_dict(), "geometry_audit": audit})
    target = output / "protocol_selection" / f"shard_{int(args.shard):02d}"
    write_jsonl(target / "metrics.jsonl", rows); write_jsonl(target / "models.jsonl", models); write_jsonl(target / "rejections.jsonl", rejections)
    write_json(target / "COMPLETE.json", {"candidates": len(candidates), "valid_position_models": len(rows), "rejections": len(rejections), "raw_forward_texts": oracle.raw_forward_texts})


def _rank(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (-float(row["coverage"]), -float(row["worst_source_coverage"]), float(row["benign_occupancy"]), float(row["radius_degrees"]), int(row["token_id"]))


def _thresholds(config: Mapping[str, Any]) -> dict[str, float]:
    keys = (
        "p3_balanced_coverage_lcb", "worst_position_coverage_lcb", "worst_source_coverage_lcb",
        "p3_uniform_secondary_lcb", "independent_benign_occupancy_ucb", "outside_to_inside_lcb",
        "conditional_outside_origin_lcb", "moat_occupancy_1_10_ucb", "basin_lambda_star",
        "basin_occupancy_auc_1_1_5", "central_collapse_median_depth",
    )
    return {key: float(config["certification"][key]) for key in keys}


def _freeze_one(
    output: Path, config: Mapping[str, Any], model: FrozenCapModel, path: Path,
    selection_metrics: Mapping[str, Any], protocol_roles: Sequence[str],
) -> dict[str, Any]:
    bindings = load_role_contract(output / "registration" / "role_contract.json")
    role_hashes = {name: binding.records_sha256 for name, binding in bindings.items()}
    fit_sources = bindings[protocol_roles[0]].source_ids
    weights = {source: 1.0 / len(fit_sources) for source in fit_sources}
    enumeration = json.loads((output / "enumeration" / "COMPLETE.json").read_text(encoding="utf-8"))
    artifact = create_freeze(
        model, tokenizer_hash=str(enumeration["tokenizer_sha256"]),
        model_hash=canonical_sha256({"id": config["model"]["id"], "revision": config["model"]["revision"]}),
        code_commit=git_head(ROOT), data_role_hashes=role_hashes,
        position_manifest_hash=canonical_sha256(config["positions"]),
        random_boundary_manifest_hash=sha256_file(output / "registration" / "random_boundaries_manifest.json"),
        source_weights=weights, selection_metrics=selection_metrics,
        certification_thresholds=_thresholds(config),
    )
    save_freeze(path, artifact)
    return {"path": path.relative_to(output).as_posix(), "freeze_sha256": artifact.freeze_sha256, "token_id": model.token_id, "protocol": model.protocol}


def merge_and_freeze(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output); position_rows = []; position_models = []
    for shard in range(int(args.shards)):
        target = output / "protocol_selection" / f"shard_{shard:02d}"
        position_rows.extend(read_jsonl(target / "metrics.jsonl")); position_models.extend(read_jsonl(target / "models.jsonl"))
    expected = set(map(int, json.loads((output / "funnel" / "stability" / "selected.json").read_text(encoding="utf-8"))["token_ids"]))
    model_map = {(int(row["token_id"]), str(row["position"])): model_from_dict(row["model"]) for row in position_models}
    metric_map = {(int(row["token_id"]), str(row["position"])): row for row in position_rows}
    semantic = {int(row["token_id"]): row for row in read_jsonl(output / "semantic" / "discovery_results.jsonl")}
    if set(semantic) != expected: raise ProtocolViolation("semantic/top-100 candidate binding mismatch")
    p1 = {}
    for position in POSITIONS:
        candidates = [row for row in position_rows if row["position"] == position]
        if not candidates: raise ProtocolViolation(f"P1 has no valid {position} candidate")
        p1[position] = min(candidates, key=_rank)
    p2_candidates = []
    for token_id in expected:
        if all((token_id, position) in metric_map for position in POSITIONS):
            values = [metric_map[(token_id, position)] for position in POSITIONS]
            p2_candidates.append({
                "token_id": token_id, "coverage": float(np.mean([row["coverage"] for row in values])),
                "worst_source_coverage": min(row["worst_source_coverage"] for row in values),
                "benign_occupancy": max(row["benign_occupancy"] for row in values),
                "radius_degrees": max(row["radius_degrees"] for row in values),
                "positions": values,
            })
    if not p2_candidates: raise ProtocolViolation("P2 has no token valid in all positions")
    p2 = min(p2_candidates, key=_rank)
    full_models = {(int(row["token_id"]), int(row["cap_count"])): model_from_dict(row["model"]) for row in read_jsonl(output / "funnel" / "stability" / "all_models.jsonl")}
    full_metrics = {(int(row["token_id"]), int(row["cap_count"])): row for row in read_jsonl(output / "funnel" / "stability" / "all_metrics.jsonl")}
    def p3_rank(key: tuple[int, int]) -> tuple[Any, ...]:
        row = full_metrics[key]; sem = semantic[key[0]]
        return (-float(row["coverage_margin"]), -float(row["worst_position_coverage"]), -float(sem["semantic_anomaly"]), float(row["benign_occupancy"]), float(row["radius_degrees"]), key)
    one = sorted([key for key in full_models if key[0] in expected and key[1] == 1], key=p3_rank)
    multi = sorted([key for key in full_models if key[0] in expected and 2 <= key[1] <= 4], key=lambda key: (key[1],) + p3_rank(key))
    if not one: raise ProtocolViolation("P3 primary single-cap archive is empty")
    chosen = [one[0]]; used = {one[0][0]}
    alternating = [*multi, *one[1:]]
    for key in alternating:
        if key[0] not in used:
            chosen.append(key); used.add(key[0])
        if len(chosen) >= 1 + int(config["funnel"]["full"]["secondary"]): break
    if len(chosen) < 1 + int(config["funnel"]["full"]["secondary"]):
        raise ProtocolViolation("insufficient distinct P3 freeze candidates")
    index = {"schema_version": "mode3-v6-2-freeze-index-v1", "P1": {}, "P2": {}, "P3": []}
    for position, row in p1.items():
        model = model_map[(int(row["token_id"]), position)]
        model = FrozenCapModel(**{**model.__dict__, "protocol": f"P1_independent_token_position:{position}"})
        index["P1"][position] = _freeze_one(output, config, model, output / "freezes" / "P1" / f"{position}.json", row, ("full_fit", "full_radius"))
    for position in POSITIONS:
        row = metric_map[(int(p2["token_id"]), position)]; model = model_map[(int(p2["token_id"]), position)]
        index["P2"][position] = _freeze_one(output, config, model, output / "freezes" / "P2" / f"{position}.json", {**row, "simultaneous_selection": p2}, ("full_fit", "full_radius"))
    for rank, key in enumerate(chosen):
        row = {**full_metrics[key], "semantic": semantic[key[0]], "freeze_rank": rank}
        entry = _freeze_one(output, config, full_models[key], output / "freezes" / "P3" / f"rank_{rank:02d}_token_{key[0]}_m{key[1]}.json", row, ("full_fit", "full_radius"))
        entry.update({"rank": rank, "cap_count": key[1], "primary": rank == 0}); index["P3"].append(entry)
    index["freeze_count"] = len(index["P1"]) + len(index["P2"]) + len(index["P3"])
    index["test_ood_encoded_before_freeze"] = False; index["separate_protocol_objects"] = True
    write_jsonl(output / "protocol_selection" / "all_metrics.jsonl", position_rows)
    write_json(output / "freezes" / "INDEX.json", index); write_json(output / "freezes" / "COMPLETE.json", index)


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/v6_2_mode3.yaml")
    parser.add_argument("--output", default="results/sticky_lab/sentence_t5_base/mode3_v6_2"); parser.add_argument("--device", default="cuda:0")
    sub = parser.add_subparsers(dest="command", required=True)
    shard = sub.add_parser("position-shard"); shard.add_argument("--shard", type=int, required=True); shard.add_argument("--shards", type=int, required=True)
    merge = sub.add_parser("merge-and-freeze"); merge.add_argument("--shards", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv); config = load_config(Path(args.config))
    {"position-shard": position_shard, "merge-and-freeze": merge_and_freeze}[args.command](args, config); return 0


if __name__ == "__main__": raise SystemExit(main())
