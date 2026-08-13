"""Frozen-geometry confirmation, semantic controls, mechanism, and retrieval."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sticky_lab.mode3_v6.evaluation import certify_frozen_cap
from sticky_lab.mode3_v6.experiment import position_balanced_concat
from sticky_lab.mode3_v6.geometry import FrozenCap
from sticky_lab.mode3_v6.insertion import insert_once
from sticky_lab.mode3_v6.retrieval import single_poison_retrieval
from sticky_lab.mode3_v6.semantic_controls import (
    TokenMetadata,
    additive_semantic_residual,
    match_controls,
    wrapper_counterfactuals,
)
from sticky_lab.mode3_v6.statistics import radial_profile

from .common import (
    atomic_savez,
    load_config,
    load_legal,
    load_manifest,
    load_role,
    read_jsonl,
    write_json,
    write_jsonl,
)
from .evaluate import encode_positions
from .oracle import CompactFinalOracle, load_embedding_cache, records_sha256


def _frozen_caps(output: Path) -> list[dict[str, Any]]:
    value = json.loads((output / "validation" / "frozen_caps.json").read_text(encoding="utf-8"))
    if not value.get("gate_open") or value.get("primary") is None:
        raise RuntimeError("sealed stages require an open validation gate")
    return [value["primary"], *value.get("secondary", [])]


def _cap(value: Mapping[str, Any]) -> FrozenCap:
    return FrozenCap(
        token_id=int(value["token_id"]),
        token_text=str(value["token_text"]),
        protocol=str(value["protocol"]),
        centers=np.asarray(value["centers"]),
        radii=np.asarray(value["radii"]),
        coverage_level=float(value["coverage_level"]),
        fit_role=str(value.get("fit_role", "cap_fit")),
        calibration_role=str(value.get("calibration_role", "cap_calibration")),
        cap_count=int(value["cap_count"]),
        assignment_rule=str(value.get("assignment_rule", "minimum_normalized_angular_distance")),
        outlier_budget=float(value.get("outlier_budget", 0.10)),
    )


def _cache(output: Path, role: str, records: list[dict[str, str]]) -> np.ndarray:
    return load_embedding_cache(
        output / "base_embeddings" / f"{role}.npy",
        expected_role=role,
        expected_records_hash=records_sha256(records),
    )


def _bootstrap_binary(values: np.ndarray, replicates: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=bool)
    rng = np.random.default_rng(seed)
    fractions = rng.binomial(len(values), float(values.mean()), size=int(replicates)) / len(values)
    return {
        "replicates": int(replicates),
        "mean": float(values.mean()),
        "q025": float(np.quantile(fractions, 0.025)),
        "q50": float(np.quantile(fractions, 0.50)),
        "q975": float(np.quantile(fractions, 0.975)),
    }


def _bootstrap_radial(
    normalized_radius: np.ndarray, multipliers: list[float], replicates: int, seed: int
) -> list[dict[str, Any]]:
    values = np.asarray(normalized_radius, dtype=np.float64)
    rng = np.random.default_rng(seed)
    rows = []
    lower = 0.0
    for upper in multipliers:
        cumulative_p = float(np.mean(values <= upper))
        shell_p = float(np.mean((values > lower) & (values <= upper)))
        cumulative = rng.binomial(len(values), cumulative_p, size=replicates) / len(values)
        shell = rng.binomial(len(values), shell_p, size=replicates) / len(values)
        rows.append(
            {
                "lower_multiplier": lower,
                "upper_multiplier": upper,
                "cumulative_q025": float(np.quantile(cumulative, 0.025)),
                "cumulative_q50": float(np.quantile(cumulative, 0.50)),
                "cumulative_q975": float(np.quantile(cumulative, 0.975)),
                "shell_q025": float(np.quantile(shell, 0.025)),
                "shell_q50": float(np.quantile(shell, 0.50)),
                "shell_q975": float(np.quantile(shell, 0.975)),
            }
        )
        lower = upper
    return rows


def run_confirmation(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output)
    phase = str(args.phase)
    if phase == "test":
        role, benign_role = "iid_test", "iid_test_benign"
    elif phase.startswith("replication_"):
        role, benign_role = phase, "iid_test_benign"
    elif phase.startswith("ood_"):
        index = int(phase.split("_")[1])
        role, benign_role = f"ood_{index}_trigger", f"ood_{index}_benign"
    else:
        raise ValueError(phase)
    records = load_role(output, role)
    benign_records = load_role(output, benign_role)
    clean = _cache(output, role, records)
    benign = _cache(output, benign_role, benign_records)
    manifest = load_manifest(output)
    oracle = CompactFinalOracle(
        config,
        output=output,
        device=args.device,
        phase=phase,
        track="sealed_frozen_geometry",
    )
    results = []
    for freeze in _frozen_caps(output):
        cap = _cap(freeze)
        views = encode_positions(
            oracle,
            records,
            cap.token_text,
            role=role,
            manifest=manifest,
            random_replicates=int(config["positions"]["confirmation_random_replicates"]),
            metadata={"phase": phase, "token_id": cap.token_id, "frozen": True},
        )
        triggered = position_balanced_concat(views)
        paired_clean = np.repeat(np.asarray(clean), 3, axis=0)
        result = certify_frozen_cap(
            cap,
            triggered,
            paired_clean,
            np.asarray(benign),
            confidence=float(config["certification"]["confidence"]),
            coverage_lcb_threshold=float(config["certification"]["triggered_coverage_lcb"]),
            occupancy_ucb_threshold=float(config["certification"]["independent_benign_occupancy_ucb"]),
            outside_to_inside_lcb_threshold=float(config["certification"]["outside_to_inside_lcb"]),
            conditional_outside_origin_lcb_threshold=float(config["certification"]["conditional_outside_origin_lcb"]),
            radial_multipliers=list(config["radial_analysis"]["multipliers"]),
        )
        raw = result.pop("raw_normalized_radius")
        tr_radius = np.asarray(raw["triggered"], dtype=np.float32)
        cl_radius = np.asarray(raw["paired_clean"], dtype=np.float32)
        be_radius = np.asarray(raw["independent_benign"], dtype=np.float32)
        result["bootstrap"] = {
            "triggered_coverage": _bootstrap_binary(
                tr_radius <= 1,
                int(config["radial_analysis"]["bootstrap_replicates"]),
                int(config["positions"]["random_seed"]) + cap.token_id,
            ),
            "benign_occupancy": _bootstrap_binary(
                be_radius <= 1,
                int(config["radial_analysis"]["bootstrap_replicates"]),
                int(config["positions"]["random_seed"]) + 1 + cap.token_id,
            ),
        }
        multipliers = list(map(float, config["radial_analysis"]["multipliers"]))
        bootstrap_replicates = int(config["radial_analysis"]["bootstrap_replicates"])
        result["radial_bootstrap"] = {
            "triggered": _bootstrap_radial(
                tr_radius, multipliers, bootstrap_replicates,
                int(config["positions"]["random_seed"]) + 10 + cap.token_id,
            ),
            "paired_clean": _bootstrap_radial(
                cl_radius, multipliers, bootstrap_replicates,
                int(config["positions"]["random_seed"]) + 20 + cap.token_id,
            ),
            "independent_benign": _bootstrap_radial(
                be_radius, multipliers, bootstrap_replicates,
                int(config["positions"]["random_seed"]) + 30 + cap.token_id,
            ),
        }
        result.update(
            {
                "token_id": cap.token_id,
                "token_text": cap.token_text,
                "phase": phase,
                "role": role,
                "benign_role": benign_role,
                "refit_performed": False,
                "frozen_centers": cap.centers.tolist(),
                "frozen_radii": cap.radii.tolist(),
                "frozen_cap_count": cap.cap_count,
            }
        )
        target = output / "sealed" / phase / f"token_{cap.token_id}"
        atomic_savez(
            target / "high_dimensional_and_radial_arrays.npz",
            triggered_prefix=views["prefix"],
            triggered_suffix=views["suffix"],
            triggered_random=views["random"],
            triggered_normalized_radius=tr_radius,
            paired_clean_normalized_radius=cl_radius,
            independent_benign_normalized_radius=be_radius,
        )
        write_json(target / "result.json", result)
        results.append(result)
    write_json(
        output / "sealed" / phase / "COMPLETE.json",
        {
            "phase": phase,
            "candidates": len(results),
            "refit_performed": False,
            "raw_forward_texts": oracle.raw_forward_texts,
            "all_certified": all(bool(row["certified"]) for row in results),
        },
    )


def _casing(text: str) -> str:
    if text.isupper() and text.lower() != text:
        return "upper"
    if text.istitle():
        return "title"
    if text.islower() and text.upper() != text:
        return "lower"
    return "mixed_or_uncased"


def build_semantic_metadata(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output)
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    from sticky_lab.mode3_v6.oracle_whitebox import WhiteboxSentenceTransformer
    import spacy
    from nltk.corpus import wordnet as wn

    model = config["model"]
    runtime = SentenceTransformer(
        model["local_path"] or model["id"],
        revision=None if model["local_path"] else model["revision"],
        device=args.device,
        trust_remote_code=bool(model["trust_remote_code"]),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model["local_path"] or model["id"],
        revision=None if model["local_path"] else model["revision"],
        trust_remote_code=bool(model["trust_remote_code"]),
    )
    nlp = spacy.load(str(config["resources"]["spacy_model"]), disable=["parser", "ner"])
    wn.ensure_loaded()
    matrix = WhiteboxSentenceTransformer(runtime).embedding_matrix()
    frequency: Counter[int] = Counter()
    document_frequency: Counter[int] = Counter()
    documents = 0
    for role in ("s0_fit", "s1_eval", "semantic_control"):
        for row in load_role(output, role):
            ids = list(map(int, tokenizer.encode(row["text"], add_special_tokens=False)))
            frequency.update(ids)
            document_frequency.update(set(ids))
            documents += 1
    maximum_log_frequency = max((math.log1p(value) for value in frequency.values()), default=1.0)
    rows = []
    for token in load_legal(output):
        text = token.token_text
        stripped = text.strip()
        doc = nlp(stripped or text)
        pos = doc[0].pos_ if len(doc) else "X"
        synsets = wn.synsets(stripped.replace(" ", "_")) if stripped else []
        rows.append(
            TokenMetadata(
                token.token_id,
                float(frequency[token.token_id]),
                float(math.log((documents + 1) / (document_frequency[token.token_id] + 1)) + 1),
                pos,
                synsets[0].lexname() if synsets else "no_wordnet_synset",
                len(text),
                _casing(text),
                float(np.linalg.norm(matrix[token.token_id])),
                float(math.log1p(frequency[token.token_id]) / maximum_log_frequency),
            ).__dict__
        )
    write_jsonl(output / "semantic" / "token_metadata.jsonl", rows)
    write_json(
        output / "semantic" / "COMPLETE.json",
        {"token_count": len(rows), "documents": documents, "whitebox_mechanism_metadata": True},
    )


def run_semantic(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output)
    metadata = [TokenMetadata(**row) for row in read_jsonl(output / "semantic" / "token_metadata.jsonl")]
    by_metadata = {row.token_id: row for row in metadata}
    legal = {row.token_id: row for row in load_legal(output)}
    records = load_role(output, "semantic_control")[:1024]
    clean_single = _cache(output, "semantic_control", load_role(output, "semantic_control"))[: len(records)]
    manifest = load_manifest(output)
    oracle = CompactFinalOracle(
        config, output=output, device=args.device, phase="semantic_controls", track="frozen_finalists_only"
    )
    results = []
    for freeze in _frozen_caps(output):
        cap = _cap(freeze)
        controls = match_controls(
            by_metadata[cap.token_id], metadata, int(config["semantic_controls"]["controls_per_candidate"])
        )

        def coverage(token_text: str, label: str) -> tuple[float, np.ndarray]:
            views = encode_positions(
                oracle,
                records,
                token_text,
                role="semantic_control",
                manifest=manifest,
                random_replicates=int(config["positions"]["discovery_random_replicates"]),
                metadata={"frozen_token_id": cap.token_id, "semantic_label": label},
            )
            vectors = position_balanced_concat(views)
            return float(np.mean(cap.contains(vectors))), vectors

        candidate_coverage, triggered = coverage(cap.token_text, "candidate")
        control_rows = []
        for control in controls:
            value, _ = coverage(legal[control.token_id].token_text, f"control:{control.token_id}")
            control_rows.append({"token_id": control.token_id, "coverage": value})
        wrappers = {
            name: coverage(text, f"wrapper:{name}")[0]
            for name, text in wrapper_counterfactuals(cap.token_text).items()
        }
        token_direction = oracle.encode([cap.token_text], metadata={"semantic_token_direction": True})[0]
        additive = additive_semantic_residual(
            np.repeat(np.asarray(clean_single), 3, axis=0), triggered, token_direction
        )
        q95 = float(np.quantile([row["coverage"] for row in control_rows], 0.95))
        result = {
            "token_id": cap.token_id,
            "token_text": cap.token_text,
            "candidate_coverage": candidate_coverage,
            "control_coverage_q95": q95,
            "coverage_margin": candidate_coverage - q95,
            "controls": control_rows,
            "wrapper_coverages": wrappers,
            "additive_semantic_model": additive,
            "matched_fields": list(config["semantic_controls"]["matching_fields"]),
            "anomaly_supported": candidate_coverage - q95
            >= float(config["semantic_controls"]["minimum_coverage_over_control_q95"])
            and min(wrappers.values())
            >= float(config["semantic_controls"]["minimum_wrapper_coverage"]),
            "search_feedback": False,
            "refit_performed": False,
        }
        write_json(output / "semantic-controls" / f"token_{cap.token_id}.json", result)
        results.append(result)
    write_json(
        output / "semantic-controls" / "COMPLETE.json",
        {
            "candidates": len(results),
            "primary_anomaly_supported": bool(results and results[0]["anomaly_supported"]),
            "raw_forward_texts": oracle.raw_forward_texts,
            "search_feedback": False,
        },
    )


def run_mechanism(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output)
    metadata = {row["token_id"]: row for row in read_jsonl(output / "semantic" / "token_metadata.jsonl")}
    whitebox = json.loads((output / "tracks" / "whitebox" / "candidates.json").read_text(encoding="utf-8"))
    whitebox_ids = set(map(int, whitebox["token_ids"]))
    rows = []
    for freeze in _frozen_caps(output):
        token_id = int(freeze["token_id"])
        rows.append(
            {
                "token_id": token_id,
                "token_text": freeze["token_text"],
                "input_embedding_norm": metadata[token_id]["input_embedding_norm"],
                "whitebox_track_discovered": token_id in whitebox_ids,
                "whitebox_blackbox_isolated": True,
                "continuous_token_used_as_formal_candidate": False,
                "search_feedback": False,
            }
        )
    write_json(output / "mechanism" / "result.json", {"candidates": rows})
    write_json(output / "mechanism" / "COMPLETE.json", {"candidates": len(rows), "search_feedback": False})


def run_retrieval(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output)
    semantic = json.loads((output / "semantic-controls" / "COMPLETE.json").read_text(encoding="utf-8"))
    if not semantic["primary_anomaly_supported"]:
        write_json(
            output / "retrieval" / "SKIPPED.json",
            {"reason": "primary semantic anomaly evidence gate closed", "search_feedback": False},
        )
        return
    freeze = _frozen_caps(output)[0]
    cap = _cap(freeze)
    test_arrays = np.load(
        output / "sealed" / "test" / f"token_{cap.token_id}" / "high_dimensional_and_radial_arrays.npz",
        allow_pickle=False,
    )
    query_vectors = position_balanced_concat(
        {
            "prefix": test_arrays["triggered_prefix"],
            "suffix": test_arrays["triggered_suffix"],
            "random": test_arrays["triggered_random"],
        }
    )
    key_records = load_role(output, "retrieval_probe")
    benign_keys = _cache(output, "retrieval_probe", key_records)
    poison_record = key_records[0]
    manifest = load_manifest(output)
    poison_text = insert_once(
        poison_record["text"],
        cap.token_text,
        "prefix",
        role="retrieval_probe",
        text_id=poison_record["text_id"],
        manifest=manifest,
    )
    oracle = CompactFinalOracle(
        config, output=output, device=args.device, phase="retrieval", track="frozen_single_poison"
    )
    poison_key = oracle.encode([poison_text], metadata={"single_poison_entry": True})[0]
    result = single_poison_retrieval(query_vectors, np.asarray(benign_keys), poison_key)
    write_json(
        output / "retrieval" / "result.json",
        {
            "token_id": cap.token_id,
            "real_poison_text_id": poison_record["text_id"],
            "single_poison_entry": True,
            "poison_top1_rate": result.poison_top1_rate,
            "poison_top5_rate": result.poison_top5_rate,
            "poison_rank": result.poison_rank.tolist(),
            "poison_similarity": result.poison_similarity.tolist(),
            "strongest_benign_similarity": result.strongest_benign_similarity.tolist(),
            "search_feedback": False,
            "refit_performed": False,
        },
    )
    write_json(output / "retrieval" / "COMPLETE.json", {"search_feedback": False})


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6_mode3_compact.yaml")
    parser.add_argument("--output", default="results/sticky_lab/sentence_t5_base/mode3_v6_compact")
    parser.add_argument("--device", default="cuda:0")
    sub = parser.add_subparsers(dest="command", required=True)
    confirm = sub.add_parser("confirm")
    confirm.add_argument("--phase", required=True)
    sub.add_parser("build-semantic-metadata")
    sub.add_parser("semantic-controls")
    sub.add_parser("mechanism")
    sub.add_parser("retrieval")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_config(Path(args.config))
    {
        "confirm": run_confirmation,
        "build-semantic-metadata": build_semantic_metadata,
        "semantic-controls": run_semantic,
        "mechanism": run_mechanism,
        "retrieval": run_retrieval,
    }[args.command](args, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
