"""Sealed V6.2 confirmation, replications, OOD and retrieval.

This module never imports a fit function.  All geometry enters through signed
freeze artifacts and is applied unchanged to roles that were inaccessible
during discovery.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sticky_lab.mode3_v6.insertion import insert_once_with_span
from sticky_lab.mode3_v6.retrieval import single_poison_retrieval
from sticky_lab.mode3_v6.semantic_controls import wrapper_counterfactuals

from .common import load_config, load_legal, load_manifest, load_role, read_jsonl, write_json, write_jsonl
from .confirm import confirm_frozen_cap
from .encoding import encode_audited_positions, primary_position_vectors
from .errors import CandidateRejected, ProtocolViolation, ShapeMismatch
from .freeze import FreezeArtifact, load_freeze
from .oracle import V62FinalOracle, load_embedding_cache, records_sha256
from .roles import records_sha256 as role_records_sha256
from .semantic import SemanticMetadata, matched_controls
from .statistics import p2_position_certificates, simultaneous_source_occupancy


POSITIONS = ("prefix", "suffix", "random")


def _index(output: Path) -> dict[str, Any]:
    path = output / "freezes" / "INDEX.json"
    if not path.is_file(): raise ProtocolViolation("sealed phase requires the freeze index")
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(output: Path, entry: Mapping[str, Any]) -> FreezeArtifact:
    artifact = load_freeze(output / str(entry["path"]))
    if artifact.freeze_sha256 != entry["freeze_sha256"]: raise ProtocolViolation("freeze index hash mismatch")
    return artifact


def _cache(output: Path, role: str, records: Sequence[Mapping[str, str]]) -> np.ndarray:
    return load_embedding_cache(output / "base_embeddings" / f"{role}.npy", expected_role=role, expected_records_hash=records_sha256(records))


def _sources(records: Sequence[Mapping[str, str]], values: np.ndarray) -> dict[str, np.ndarray]:
    matrix = np.asarray(values)
    if len(matrix) != len(records): raise ShapeMismatch("sealed vectors are not role aligned")
    indices: dict[str, list[int]] = {}
    for index, row in enumerate(records): indices.setdefault(str(row["source_id"]), []).append(index)
    return {source: matrix[np.asarray(rows, dtype=int)] for source, rows in sorted(indices.items())}


def _encode_for_artifact(
    oracle: V62FinalOracle, artifact: FreezeArtifact, records: Sequence[Mapping[str, str]],
    role: str, manifest: Any, config: Mapping[str, Any], replicates: int,
) -> tuple[dict[str, np.ndarray], dict[tuple[str, int], np.ndarray]]:
    views, audits, _ = encode_audited_positions(
        oracle, records, token_id=artifact.token_id, token_text=artifact.token_text,
        role=role, manifest=manifest, random_replicates=replicates,
        maximum_length=int(config["model"]["maximum_sequence_length"]),
        metadata={"freeze_sha256": artifact.freeze_sha256, "sealed": True},
    )
    primary = primary_position_vectors(views, primary_random_replicate=int(config["positions"]["primary_random_replicate"]))
    robustness = {
        (source, replicate): values
        for replicate in range(replicates)
        for source, values in _sources(records, views[f"random:{replicate}"]).items()
    }
    return primary, robustness


def _confirm_p3(
    output: Path, config: Mapping[str, Any], role: str, benign_role: str,
    *, device: str, phase: str, all_candidates: bool,
) -> list[dict[str, Any]]:
    records = load_role(output, role); benign_records = load_role(output, benign_role)
    clean = _cache(output, role, records); benign = _cache(output, benign_role, benign_records)
    manifest = load_manifest(output); index = _index(output)
    entries = index["P3"] if all_candidates else index["P3"][:1]
    oracle = V62FinalOracle(config, output=output, device=device, phase=phase, track="sealed_frozen_P3")
    results = []
    for entry in entries:
        artifact = _artifact(output, entry)
        primary, robustness = _encode_for_artifact(
            oracle, artifact, records, role, manifest, config,
            int(config["positions"]["robustness_random_replicates"]),
        )
        triggered = {
            (source, position): values
            for position in POSITIONS for source, values in _sources(records, primary[position]).items()
        }
        paired_clean = {
            (source, position): values
            for position in POSITIONS for source, values in _sources(records, clean).items()
        }
        result = confirm_frozen_cap(
            artifact, triggered=triggered, paired_clean=paired_clean,
            independent_benign=_sources(benign_records, benign),
            observed_role_hashes={role: role_records_sha256(records), benign_role: role_records_sha256(benign_records)},
            radial_multipliers=config["radial_analysis"]["multipliers"],
            familywise_alpha=float(config["certification"]["familywise_alpha"]),
            random_robustness=robustness,
        )
        result.update({"protocol": "P3", "phase": phase, "role": role, "benign_role": benign_role, "token_id": artifact.token_id, "token_text": artifact.token_text, "cap_count": artifact.cap.cap_count})
        results.append(result)
    target = output / "confirmation" / phase / "P3"
    write_jsonl(target / "results.jsonl", results)
    write_json(target / "COMPLETE.json", {"candidates": len(results), "core_certified": sum(bool(row["levels"]["B_ST_FCA_Core"]) for row in results), "refit_performed": False, "raw_forward_texts": oracle.raw_forward_texts})
    return results


def _confirm_position_protocol(
    output: Path, config: Mapping[str, Any], protocol: str, role: str,
    benign_role: str, *, device: str, phase: str,
) -> dict[str, Any]:
    index = _index(output); entries = index[protocol]
    records = load_role(output, role); benign_records = load_role(output, benign_role)
    clean = _cache(output, role, records); benign = _cache(output, benign_role, benign_records)
    manifest = load_manifest(output)
    oracle = V62FinalOracle(config, output=output, device=device, phase=phase, track=f"sealed_frozen_{protocol}")
    membership: dict[str, dict[str, np.ndarray]] = {}; occupancy = {}; migration = {}; token_ids = {}
    for position in POSITIONS:
        artifact = _artifact(output, entries[position]); token_ids[position] = artifact.token_id
        primary, _ = _encode_for_artifact(oracle, artifact, records, role, manifest, config, int(config["positions"]["robustness_random_replicates"]))
        tr_sources = _sources(records, primary[position]); clean_sources = _sources(records, clean)
        membership[position] = {source: artifact.cap.contains(values) for source, values in tr_sources.items()}
        occupancy[position] = simultaneous_source_occupancy(
            {source: artifact.cap.contains(values) for source, values in _sources(benign_records, benign).items()},
            familywise_alpha=float(config["certification"]["familywise_alpha"]) / 3.0,
        )
        migration[position] = {
            source: {
                "outside_to_inside": float(np.mean((~artifact.cap.contains(clean_sources[source])) & membership[position][source])),
                "conditional_outside_origin": float(np.mean((~artifact.cap.contains(clean_sources[source]))[membership[position][source]])) if np.any(membership[position][source]) else 0.0,
            }
            for source in tr_sources
        }
    certificates = p2_position_certificates(membership, familywise_alpha=float(config["certification"]["familywise_alpha"]))
    threshold = float(config["certification"]["p3_balanced_coverage_lcb"])
    gates = {
        position: certificates[position]["balanced_lcb"] > threshold
            and occupancy[position]["worst_source_ucb"] < float(config["certification"]["independent_benign_occupancy_ucb"])
        for position in POSITIONS
    }
    result = {
        "schema_version": "mode3-v6-2-position-confirmation-v1", "protocol": protocol,
        "phase": phase, "token_ids": token_ids, "certificates": certificates,
        "occupancy": occupancy, "migration_estimates": migration, "position_gates": gates,
        "simultaneous_all_positions": all(gates.values()), "refit_performed": False,
        "raw_forward_texts": oracle.raw_forward_texts,
    }
    target = output / "confirmation" / phase / protocol
    write_json(target / "result.json", result); write_json(target / "COMPLETE.json", {"certified": all(gates.values()), "refit_performed": False})
    return result


def confirm_core(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output)
    p1 = _confirm_position_protocol(output, config, "P1", "confirm_trigger", "confirm_benign", device=args.device, phase="core")
    p2 = _confirm_position_protocol(output, config, "P2", "confirm_trigger", "confirm_benign", device=args.device, phase="core")
    p3 = _confirm_p3(output, config, "confirm_trigger", "confirm_benign", device=args.device, phase="core", all_candidates=True)
    any_core = any(bool(row["levels"]["B_ST_FCA_Core"]) for row in p3)
    write_json(output / "confirmation" / "COMPLETE.json", {
        "schema_version": "mode3-v6-2-core-confirmation-v1", "P1_certified": p1["simultaneous_all_positions"],
        "P2_certified": p2["simultaneous_all_positions"], "P3_core_certified": sum(bool(row["levels"]["B_ST_FCA_Core"]) for row in p3),
        "any_core_certified": any_core, "refit_performed": False,
        "calibration_roles_encoded": False, "confirmation_roles_independent": True,
    })


def confirm_followup(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output); phase = str(args.phase)
    if not json.loads((output / "confirmation" / "COMPLETE.json").read_text(encoding="utf-8"))["any_core_certified"]:
        raise ProtocolViolation("sealed followups require an open core P3 gate")
    if phase.startswith("iid_replication_"):
        role, benign_role = phase, "confirm_benign"
    elif phase.startswith("ood_"):
        index = int(phase.split("_")[1]); role, benign_role = f"ood_{index}_trigger", f"ood_{index}_benign"
    else: raise ValueError(phase)
    results = _confirm_p3(output, config, role, benign_role, device=args.device, phase=phase, all_candidates=False)
    write_json(output / "sealed_followups" / phase / "COMPLETE.json", {"phase": phase, "core_certified": bool(results[0]["levels"]["B_ST_FCA_Core"]), "refit_performed": False})


def semantic_confirmation(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output); records = load_role(output, "semantic_confirm"); manifest = load_manifest(output)
    metadata = [SemanticMetadata(**row) for row in read_jsonl(output / "semantic" / "token_metadata.jsonl")]
    by_meta = {row.token_id: row for row in metadata}; legal = {row.token_id: row.token_text for row in load_legal(output)}
    oracle = V62FinalOracle(config, output=output, device=args.device, phase="semantic_confirm", track="frozen_only")
    results = []; count = int(config["semantic_controls"]["controls_per_candidate"])
    for entry in _index(output)["P3"]:
        artifact = _artifact(output, entry)
        views, _, _ = encode_audited_positions(
            oracle, records, token_id=artifact.token_id, token_text=artifact.token_text,
            role="semantic_confirm", manifest=manifest, random_replicates=1,
            maximum_length=int(config["model"]["maximum_sequence_length"]), metadata={"semantic_confirm": artifact.freeze_sha256},
        )
        candidate_coverage = float(np.mean([np.mean(artifact.cap.contains(value)) for value in primary_position_vectors(views).values()]))
        controls = []; pool = matched_controls(by_meta[artifact.token_id], metadata, count * 3)
        for control in pool:
            if len(controls) >= count: break
            try:
                values, _, _ = encode_audited_positions(
                    oracle, records, token_id=control.token_id, token_text=legal[control.token_id],
                    role="semantic_confirm", manifest=manifest, random_replicates=1,
                    maximum_length=int(config["model"]["maximum_sequence_length"]), metadata={"semantic_control": control.token_id},
                )
            except CandidateRejected:
                continue
            controls.append({"token_id": control.token_id, "coverage": float(np.mean([np.mean(artifact.cap.contains(value)) for value in primary_position_vectors(values).values()]))})
        if len(controls) != count: raise ProtocolViolation("insufficient realizable frozen semantic controls")
        wrappers = {}
        for name, wrapper in wrapper_counterfactuals(artifact.token_text).items():
            coverages = []
            for position in POSITIONS:
                texts = [insert_once_with_span(str(row.get("encoding_text", row["text"])), wrapper, position, role="semantic_confirm", text_id=str(row["text_id"]), manifest=manifest, replicate=0)[0] for row in records]
                coverages.append(np.mean(artifact.cap.contains(oracle.encode(texts, metadata={"wrapper": name, "position": position}))))
            wrappers[name] = float(np.mean(coverages))
        q95 = float(np.quantile([row["coverage"] for row in controls], .95)); margin = candidate_coverage - q95
        results.append({"token_id": artifact.token_id, "freeze_sha256": artifact.freeze_sha256, "candidate_coverage": candidate_coverage, "control_coverage_q95": q95, "coverage_margin": margin, "controls": controls, "wrapper_coverages": wrappers, "anomaly_supported": margin >= float(config["semantic_controls"]["minimum_coverage_over_control_q95"]) and min(wrappers.values()) >= float(config["semantic_controls"]["minimum_wrapper_coverage"]), "refit_performed": False})
    write_jsonl(output / "semantic_confirmation" / "results.jsonl", results)
    write_json(output / "semantic_confirmation" / "COMPLETE.json", {"candidates": len(results), "supported": sum(bool(row["anomaly_supported"]) for row in results), "role": "semantic_confirm", "discovery_role_reused": False, "raw_forward_texts": oracle.raw_forward_texts})


def retrieval(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output); core = json.loads((output / "confirmation" / "core" / "P3" / "results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    if not core["levels"]["B_ST_FCA_Core"]:
        write_json(output / "retrieval" / "SKIPPED.json", {"reason": "primary P3 core gate closed", "search_feedback": False}); return
    artifact = _artifact(output, _index(output)["P3"][0]); records = load_role(output, "retrieval_probe"); benign = _cache(output, "retrieval_probe", records)
    oracle = V62FinalOracle(config, output=output, device=args.device, phase="retrieval", track="single_poison")
    manifest = load_manifest(output); views, _, _ = encode_audited_positions(oracle, records, token_id=artifact.token_id, token_text=artifact.token_text, role="retrieval_probe", manifest=manifest, random_replicates=1, maximum_length=int(config["model"]["maximum_sequence_length"]), metadata={"retrieval": True})
    queries = np.concatenate(list(primary_position_vectors(views).values()))
    poison_text = insert_once_with_span(str(records[0].get("encoding_text", records[0]["text"])), artifact.token_text, "prefix", role="retrieval_probe", text_id=str(records[0]["text_id"]), manifest=manifest, replicate=0)[0]
    poison = oracle.encode([poison_text], metadata={"single_poison": True})[0]
    value = single_poison_retrieval(queries, np.asarray(benign), poison)
    write_json(output / "retrieval" / "result.json", {"token_id": artifact.token_id, "poison_top1_rate": value.poison_top1_rate, "poison_top5_rate": value.poison_top5_rate, "poison_rank": value.poison_rank.tolist(), "search_feedback": False, "refit_performed": False})
    write_json(output / "retrieval" / "COMPLETE.json", {"complete": True, "refit_performed": False})


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/v6_2_mode3.yaml")
    parser.add_argument("--output", default="results/sticky_lab/sentence_t5_base/mode3_v6_2"); parser.add_argument("--device", default="cuda:0")
    sub = parser.add_subparsers(dest="command", required=True); sub.add_parser("confirm-core"); sub.add_parser("semantic-confirmation"); sub.add_parser("retrieval")
    follow = sub.add_parser("confirm-followup"); follow.add_argument("--phase", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv); config = load_config(Path(args.config))
    {"confirm-core": confirm_core, "confirm-followup": confirm_followup, "semantic-confirmation": semantic_confirmation, "retrieval": retrieval}[args.command](args, config); return 0


if __name__ == "__main__": raise SystemExit(main())
