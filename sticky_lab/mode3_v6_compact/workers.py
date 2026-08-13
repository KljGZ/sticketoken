"""Idempotent GPU/merge workers for the V6 Compact funnel."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sticky_lab.mode3_v6.evaluation import certify_frozen_cap
from sticky_lab.mode3_v6.experiment import position_balanced_concat
from sticky_lab.mode3_v6.geometry import (
    FrozenCap,
    calibrate_multicap_radii,
    fit_spherical_multicenter,
)
from sticky_lab.mode3_v6.insertion import insert_once_with_span
from sticky_lab.mode3_v6.resource_errors import is_resource_exhaustion
from sticky_lab.mode3_v6.tokenizer_audit import (
    LegalToken,
    _realizes_exact_span,
    enumerate_actual_single_tokens,
    shard_legal_tokens,
)

from .common import (
    atomic_savez,
    cap_from_arrays,
    load_config,
    load_legal,
    load_manifest,
    load_role,
    read_jsonl,
    write_json,
    write_jsonl,
)
from .evaluate import (
    DiscoveryMetric,
    POSITIONS,
    attach_benign_metrics,
    evaluate_frozen_stage,
    fit_s0,
    validate_single_cap,
)
from .funnel import merge_stage_history, select_candidates
from .oracle import (
    CompactFinalOracle,
    load_embedding_cache,
    records_sha256,
    write_embedding_cache,
)


def _oracle(
    config: Mapping[str, Any], output: Path, device: str, phase: str, track: str
) -> CompactFinalOracle:
    return CompactFinalOracle(config, output=output, device=device, phase=phase, track=track)


def _cache(output: Path, role: str, records: list[dict[str, str]]) -> np.ndarray:
    return load_embedding_cache(
        output / "base_embeddings" / f"{role}.npy",
        expected_role=role,
        expected_records_hash=records_sha256(records),
    )


def _enumerate_limited_single_tokens(
    tokenizer: object,
    *,
    context_records: list[dict[str, str]],
    manifest: object,
    role: str,
    exclude_special: bool,
    limit: int,
) -> tuple[list[LegalToken], list[LegalToken]]:
    """Run the exact legality audit but stop after ``limit`` accepted tokens.

    This path exists only for the bounded dry-run. The formal run deliberately
    continues to call the frozen V6 exhaustive enumerator below.
    """
    if limit <= 0:
        raise ValueError("enumeration limit must be positive")
    vocab = tokenizer.get_vocab()
    special = set(getattr(tokenizer, "all_special_ids", []))
    unrestricted: list[LegalToken] = []
    for token_id in sorted(set(map(int, vocab.values()))):
        if exclude_special and token_id in special:
            continue
        token_text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        standalone_ids = tokenizer.encode(token_text, add_special_tokens=False)
        standalone = list(map(int, standalone_ids)) == [token_id]
        if not standalone or not token_text:
            continue
        visible = bool(token_text.strip()) and not any(
            ord(char) < 32 and char not in "\t\n\r" for char in token_text
        )
        checks: dict[str, bool] = {}
        for position in ("prefix", "suffix", "random"):
            okay = True
            for row in context_records:
                value, span = insert_once_with_span(
                    row["text"],
                    token_text,
                    position,
                    role=role,
                    text_id=row["text_id"],
                    manifest=manifest,
                )
                if not _realizes_exact_span(tokenizer, value, span, token_id):
                    okay = False
                    break
            checks[position] = okay
        item = LegalToken(
            token_id,
            token_text,
            visible,
            standalone,
            checks["prefix"],
            checks["suffix"],
            checks["random"],
        )
        if item.contextual_roundtrip:
            unrestricted.append(item)
            if len(unrestricted) >= limit:
                break
    return unrestricted, [item for item in unrestricted if item.visible]


def enumerate_vocab(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    from transformers import AutoTokenizer

    output = Path(args.output)
    model = config["model"]
    source = model["local_path"] or model["id"]
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        revision=None if model["local_path"] else model["revision"],
        trust_remote_code=bool(model["trust_remote_code"]),
    )
    records = load_role(output, "s0_fit")[:32]
    limit = getattr(args, "limit", None)
    enumeration_args = {
        "context_records": records,
        "manifest": load_manifest(output),
        "role": "s0_fit",
        "exclude_special": bool(config["tokenizer"]["exclude_special_tokens"]),
    }
    if limit is None:
        unrestricted, visible = enumerate_actual_single_tokens(tokenizer, **enumeration_args)
    else:
        unrestricted, visible = _enumerate_limited_single_tokens(
            tokenizer, limit=int(limit), **enumeration_args
        )
    digest = hashlib.sha256()
    for row in unrestricted:
        digest.update(f"{row.token_id}\0{row.token_text}\n".encode("utf-8"))
    target = output / "enumeration"
    write_jsonl(target / "legal_unrestricted.jsonl", (row.to_dict() for row in unrestricted))
    write_jsonl(target / "legal_visible.jsonl", (row.to_dict() for row in visible))
    write_json(
        target / "COMPLETE.json",
        {
            "actual_tokenizer_length": 1,
            "unrestricted_count": len(unrestricted),
            "visible_count": len(visible),
            "common_context_manifest_hash": records_sha256(records),
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_vocab_sha256": digest.hexdigest(),
            "model_revision": model["revision"],
        },
    )


def precompute_role(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output)
    role = str(args.role)
    records = load_role(output, role)
    target = output / "base_embeddings" / f"{role}.npy"
    if target.is_file() and target.with_suffix(".json").is_file():
        load_embedding_cache(
            target, expected_role=role, expected_records_hash=records_sha256(records)
        )
        return
    oracle = _oracle(config, output, args.device, "base_embeddings", "shared_clean_benign")
    vectors = oracle.encode(
        [row["text"] for row in records], metadata={"role": role, "cache_once": True}
    )
    write_embedding_cache(
        target,
        vectors,
        role=role,
        records_hash=records_sha256(records),
        model_revision=str(config["model"]["revision"]),
    )


def s0_shard(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output)
    fit_records, eval_records = load_role(output, "s0_fit"), load_role(output, "s0_eval")
    clean_eval = _cache(output, "s0_eval", eval_records)
    benign_records = load_role(output, "discovery_benign")
    benign = _cache(output, "discovery_benign", benign_records)
    legal = shard_legal_tokens(load_legal(output), int(args.shard), int(args.shards))
    oracle = _oracle(config, output, args.device, "s0", "exhaustive_blackbox")
    manifest = load_manifest(output)
    valid_metrics: list[DiscoveryMetric] = []
    centers: list[np.ndarray] = []
    radii: list[float] = []
    invalid: list[dict[str, Any]] = []
    for token in legal:
        try:
            cap, metric = fit_s0(
                oracle,
                token_id=token.token_id,
                token_text=token.token_text,
                fit_records=fit_records,
                eval_records=eval_records,
                clean_eval=clean_eval,
                manifest=manifest,
                config=config,
            )
            valid_metrics.append(metric)
            centers.append(cap.centers[0].astype(np.float32))
            radii.append(float(cap.radii[0]))
        except Exception as error:
            if is_resource_exhaustion(error) or error.__class__.__name__ == "BudgetExhausted":
                raise
            invalid.append(
                {
                    "token_id": token.token_id,
                    "token_text": token.token_text,
                    "stage": "s0",
                    "status": "invalid",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    if valid_metrics:
        enriched = attach_benign_metrics(
            valid_metrics,
            np.stack(centers),
            np.asarray(radii),
            benign,
            device=args.device,
        )
        center_array = np.stack(centers).astype(np.float32)
    else:
        enriched = []
        center_array = np.empty((0, oracle.dimension), dtype=np.float32)
    target = output / "s0" / f"shard_{int(args.shard):02d}"
    write_jsonl(target / "metrics.jsonl", [row.to_dict() for row in enriched] + invalid)
    atomic_savez(
        target / "caps.npz",
        token_ids=np.asarray([row.token_id for row in enriched], dtype=np.int64),
        centers=center_array,
        radii=np.asarray(radii, dtype=np.float32),
    )
    write_json(
        target / "COMPLETE.json",
        {
            "stage": "s0",
            "shard": int(args.shard),
            "shards": int(args.shards),
            "candidate_count": len(legal),
            "valid": len(enriched),
            "invalid": len(invalid),
            "fit_manifest_hash": records_sha256(fit_records),
            "eval_manifest_hash": records_sha256(eval_records),
            "benign_manifest_hash": records_sha256(benign_records),
            "raw_forward_texts": oracle.raw_forward_texts,
        },
    )


def _source_ids(path: str | None) -> list[int]:
    if not path:
        return []
    source = Path(path)
    if not source.is_file():
        return []
    payload = json.loads(source.read_text(encoding="utf-8"))
    return list(map(int, payload.get("token_ids", [])))


def merge_s0(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output)
    rows: list[dict[str, Any]] = []
    cap_map: dict[int, tuple[np.ndarray, float]] = {}
    hashes: dict[str, set[str]] = {name: set() for name in ("fit", "eval", "benign")}
    for shard in range(int(args.shards)):
        target = output / "s0" / f"shard_{shard:02d}"
        complete = json.loads((target / "COMPLETE.json").read_text(encoding="utf-8"))
        hashes["fit"].add(complete["fit_manifest_hash"])
        hashes["eval"].add(complete["eval_manifest_hash"])
        hashes["benign"].add(complete["benign_manifest_hash"])
        rows.extend(read_jsonl(target / "metrics.jsonl"))
        caps = np.load(target / "caps.npz", allow_pickle=False)
        for token_id, center, radius in zip(caps["token_ids"], caps["centers"], caps["radii"]):
            cap_map[int(token_id)] = (np.asarray(center), float(radius))
    if any(len(values) != 1 for values in hashes.values()):
        raise RuntimeError("S0 shards did not use identical public samples")
    legal_ids = {token.token_id for token in load_legal(output)}
    row_ids = {int(row["token_id"]) for row in rows}
    if row_ids != legal_ids or len(rows) != len(legal_ids):
        raise RuntimeError("S0 exhaustive vocabulary is incomplete or duplicated")
    additional = {
        "whitebox": _source_ids(args.whitebox),
        "blackbox": _source_ids(args.blackbox),
        "v5_history": _source_ids(args.v5_history),
    }
    keep = int(config["funnel"]["s0"]["keep"])
    selected, provenance = select_candidates(
        rows,
        keep,
        additional=additional,
        additional_quota=max(1, keep // 6),
    )
    if len(selected) != keep:
        raise RuntimeError(f"S0 retained {len(selected)}/{keep}; invalid-token rate exhausted the registered funnel")
    by_id = {int(row["token_id"]): row for row in rows}
    centers = np.stack([cap_map[token_id][0] for token_id in selected])
    radii = np.asarray([cap_map[token_id][1] for token_id in selected], dtype=np.float32)
    stage = output / "funnel" / "s0"
    write_jsonl(stage / "all_metrics.jsonl", rows)
    write_jsonl(
        stage / "selected_metrics.jsonl",
        [dict(by_id[token_id], provenance=provenance[token_id]) for token_id in selected],
    )
    write_json(
        stage / "selected.json",
        {"token_ids": selected, "provenance": {str(k): v for k, v in provenance.items()}},
    )
    atomic_savez(stage / "caps.npz", token_ids=np.asarray(selected), centers=centers, radii=radii)
    write_json(
        stage / "COMPLETE.json",
        {
            "legal_tokens": len(legal_ids),
            "valid_tokens": len(cap_map),
            "selected": len(selected),
            "common_samples_verified": True,
            "additional_tracks_are_union_only": True,
            "whitebox_seeded_blackbox": False,
        },
    )


def _load_stage_caps(output: Path, stage: str) -> dict[int, FrozenCap]:
    legal = {row.token_id: row for row in load_legal(output)}
    values = np.load(output / "funnel" / stage / "caps.npz", allow_pickle=False)
    return {
        int(token_id): cap_from_arrays(
            token_id=int(token_id),
            token_text=legal[int(token_id)].token_text,
            center=center,
            radius=float(radius),
        )
        for token_id, center, radius in zip(values["token_ids"], values["centers"], values["radii"])
    }


def stage_shard(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output)
    stage = str(args.stage)
    previous = {"s1": "s0", "s2": "s1", "s3": "s2"}[stage]
    role = str(config["funnel"][stage]["role"])
    selected = json.loads(
        (output / "funnel" / previous / "selected.json").read_text(encoding="utf-8")
    )["token_ids"]
    selected = [
        int(token_id)
        for index, token_id in enumerate(selected)
        if index % int(args.shards) == int(args.shard)
    ]
    caps = _load_stage_caps(output, previous)
    records = load_role(output, role)
    clean = _cache(output, role, records)
    oracle = _oracle(config, output, args.device, stage, "progressive_blackbox")
    manifest = load_manifest(output)
    rows = []
    for token_id in selected:
        cap = caps[token_id]
        try:
            metric = evaluate_frozen_stage(
                oracle,
                cap,
                token_text=cap.token_text,
                role=role,
                records=records,
                clean=clean,
                manifest=manifest,
                config=config,
            )
            value = metric.to_dict()
            value["search_margin_m90_1"] = value["triggered_similarity_q10"] - float(
                _prior_metric(output, previous, token_id).get("benign_similarity_q995", 1.0)
            )
            rows.append(value)
        except Exception as error:
            if is_resource_exhaustion(error) or error.__class__.__name__ == "BudgetExhausted":
                raise
            rows.append(
                {
                    "token_id": token_id,
                    "token_text": cap.token_text,
                    "stage": stage,
                    "status": "invalid",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    target = output / "funnel" / stage / f"shard_{int(args.shard):02d}"
    write_jsonl(target / "metrics.jsonl", rows)
    write_json(
        target / "COMPLETE.json",
        {
            "stage": stage,
            "shard": int(args.shard),
            "candidate_count": len(selected),
            "role": role,
            "role_manifest_hash": records_sha256(records),
            "raw_forward_texts": oracle.raw_forward_texts,
        },
    )


def _prior_metric(output: Path, stage: str, token_id: int) -> dict[str, Any]:
    for row in read_jsonl(output / "funnel" / stage / "selected_metrics.jsonl"):
        if int(row["token_id"]) == int(token_id):
            return row
    raise KeyError(token_id)


def merge_stage(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output)
    stage = str(args.stage)
    previous = {"s1": "s0", "s2": "s1", "s3": "s2"}[stage]
    rows: list[dict[str, Any]] = []
    manifests: set[str] = set()
    for shard in range(int(args.shards)):
        target = output / "funnel" / stage / f"shard_{shard:02d}"
        complete = json.loads((target / "COMPLETE.json").read_text(encoding="utf-8"))
        manifests.add(complete["role_manifest_hash"])
        rows.extend(read_jsonl(target / "metrics.jsonl"))
    if len(manifests) != 1:
        raise RuntimeError(f"{stage} shards used different public samples")
    prior_rows = read_jsonl(output / "funnel" / previous / "selected_metrics.jsonl")
    prior = {int(row["token_id"]): row for row in prior_rows}
    expected = set(prior)
    if {int(row["token_id"]) for row in rows} != expected or len(rows) != len(expected):
        raise RuntimeError(f"{stage} candidate set mismatch")
    valid_rows = [row for row in rows if row.get("status") == "valid"]
    cumulative = merge_stage_history(prior, valid_rows)
    keep = int(config["funnel"][stage]["keep"])
    selected, provenance = select_candidates(cumulative, keep)
    if len(selected) != keep:
        raise RuntimeError(f"{stage} retained {len(selected)}/{keep}; registered funnel cannot be completed")
    by_id = {int(row["token_id"]): row for row in cumulative}
    caps = _load_stage_caps(output, previous)
    target = output / "funnel" / stage
    write_jsonl(target / "all_metrics.jsonl", rows)
    write_jsonl(
        target / "selected_metrics.jsonl",
        [dict(by_id[token_id], provenance=provenance[token_id]) for token_id in selected],
    )
    write_json(
        target / "selected.json",
        {"token_ids": selected, "provenance": {str(k): v for k, v in provenance.items()}},
    )
    atomic_savez(
        target / "caps.npz",
        token_ids=np.asarray(selected, dtype=np.int64),
        centers=np.stack([caps[token_id].centers[0] for token_id in selected]).astype(np.float32),
        radii=np.asarray([caps[token_id].radii[0] for token_id in selected], dtype=np.float32),
    )
    write_json(
        target / "COMPLETE.json",
        {
            "stage": stage,
            "evaluated": len(rows),
            "valid": len(valid_rows),
            "selected": len(selected),
            "common_samples_verified": True,
        },
    )


def validation_shard(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output)
    selected = json.loads(
        (output / "funnel" / "s3" / "selected.json").read_text(encoding="utf-8")
    )["token_ids"]
    selected = [
        int(token_id)
        for index, token_id in enumerate(selected)
        if index % int(args.shards) == int(args.shard)
    ]
    legal = {row.token_id: row for row in load_legal(output)}
    fit_records = load_role(output, "cap_fit")
    calibration_records = load_role(output, "cap_calibration")
    clean = _cache(output, "cap_calibration", calibration_records)
    benign_records = load_role(output, "cap_benign")
    benign = _cache(output, "cap_benign", benign_records)
    oracle = _oracle(config, output, args.device, "validation", "formal_re_evaluation")
    manifest = load_manifest(output)
    target = output / "validation" / f"shard_{int(args.shard):02d}"
    results: list[dict[str, Any]] = []
    freezes: list[dict[str, Any]] = []
    for token_id in selected:
        token = legal[token_id]
        cap, result, arrays, layers = validate_single_cap(
            oracle,
            token_id=token_id,
            token_text=token.token_text,
            fit_records=fit_records,
            calibration_records=calibration_records,
            clean_calibration=clean,
            benign=benign,
            manifest=manifest,
            config=config,
        )
        selected_cap = cap
        selected_result = result
        rescue: dict[str, Any] = {"attempted": False}
        if not result["certified"]:
            rescue = _two_cap_rescue(
                token_id,
                token.token_text,
                arrays,
                clean,
                benign,
                config,
            )
            if rescue.get("certified"):
                selected_cap = FrozenCap(**rescue["frozen_cap_constructor"])
                selected_result = rescue["result"]
        atomic_savez(target / "arrays" / f"token_{token_id}.npz", **arrays)
        results.append(
            {
                "token_id": token_id,
                "token_text": token.token_text,
                "single_cap": result,
                "two_cap_rescue": {k: v for k, v in rescue.items() if k != "frozen_cap_constructor"},
                "selected_cap_count": selected_cap.cap_count,
                "certified": bool(selected_result["certified"]),
                "layers": layers,
                "same_high_dimensional_embeddings_for_p1_p2_p3": True,
            }
        )
        freeze = selected_cap.to_json()
        freeze.update(
            {
                "validation_certified": bool(selected_result["certified"]),
                "validation_result": selected_result,
                "layers": layers,
                "refit_performed": False,
            }
        )
        freezes.append(freeze)
    write_jsonl(target / "results.jsonl", results)
    write_jsonl(target / "freeze_candidates.jsonl", freezes)
    write_json(
        target / "COMPLETE.json",
        {
            "candidates": len(selected),
            "raw_forward_texts": oracle.raw_forward_texts,
            "multicap_maximum": 2,
            "multicap_only_finalists": True,
        },
    )


def _two_cap_rescue(
    token_id: int,
    token_text: str,
    arrays: Mapping[str, np.ndarray],
    clean: np.ndarray,
    benign: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    fit = {position: arrays[f"fit_{position}"] for position in POSITIONS}
    calibration = {position: arrays[f"calibration_{position}"] for position in POSITIONS}
    fit_all = position_balanced_concat(fit)
    calibration_all = position_balanced_concat(calibration)
    fitted = fit_spherical_multicenter(
        fit_all,
        2,
        maximum_outlier_fraction=float(config["geometry"]["maximum_outlier_fraction"]),
        minimum_cluster_mass=float(config["geometry"]["minimum_cluster_mass"]),
        restarts=int(config["geometry"]["fit_restarts"]),
        maximum_iterations=int(config["geometry"]["maximum_iterations"]),
        seed=int(config["positions"]["random_seed"]) + int(token_id),
    )
    radii, _ = calibrate_multicap_radii(
        calibration_all,
        fitted.centers,
        float(config["geometry"]["calibration_coverage"]),
    )
    if np.max(np.degrees(radii)) > float(config["geometry"]["maximum_radius_degrees"]):
        return {"attempted": True, "certified": False, "reason": "radius_anti_triviality"}
    cap = FrozenCap(
        token_id=token_id,
        token_text=token_text,
        protocol="P3_shared_multicap",
        centers=fitted.centers,
        radii=radii,
        coverage_level=float(config["geometry"]["calibration_coverage"]),
        fit_role="cap_fit",
        calibration_role="cap_calibration",
        cap_count=2,
        outlier_budget=float(config["geometry"]["maximum_outlier_fraction"]),
    )
    result = certify_frozen_cap(
        cap,
        calibration_all,
        np.repeat(np.asarray(clean), 3, axis=0),
        np.asarray(benign),
        confidence=float(config["certification"]["confidence"]),
        coverage_lcb_threshold=float(config["certification"]["triggered_coverage_lcb"]),
        occupancy_ucb_threshold=float(config["certification"]["independent_benign_occupancy_ucb"]),
        outside_to_inside_lcb_threshold=float(config["certification"]["outside_to_inside_lcb"]),
        conditional_outside_origin_lcb_threshold=float(config["certification"]["conditional_outside_origin_lcb"]),
        radial_multipliers=list(config["radial_analysis"]["multipliers"]),
    )
    result.pop("raw_normalized_radius", None)
    constructor = {
        "token_id": token_id,
        "token_text": token_text,
        "protocol": "P3_shared_multicap",
        "centers": fitted.centers,
        "radii": radii,
        "coverage_level": float(config["geometry"]["calibration_coverage"]),
        "fit_role": "cap_fit",
        "calibration_role": "cap_calibration",
        "cap_count": 2,
        "outlier_budget": float(config["geometry"]["maximum_outlier_fraction"]),
    }
    return {
        "attempted": True,
        "certified": bool(result["certified"]),
        "minimal_cap_principle": True,
        "result": result,
        "cap": cap.to_json(),
        "frozen_cap_constructor": constructor,
    }


def merge_validation(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output)
    results: list[dict[str, Any]] = []
    freezes: list[dict[str, Any]] = []
    for shard in range(int(args.shards)):
        target = output / "validation" / f"shard_{shard:02d}"
        results.extend(read_jsonl(target / "results.jsonl"))
        freezes.extend(read_jsonl(target / "freeze_candidates.jsonl"))
    expected = set(
        map(
            int,
            json.loads(
                (output / "funnel" / "s3" / "selected.json").read_text(encoding="utf-8")
            )["token_ids"],
        )
    )
    if {int(row["token_id"]) for row in results} != expected or len(results) != len(expected):
        raise RuntimeError("validation finalist mismatch")
    certified = [row for row in freezes if row["validation_certified"]]
    certified.sort(
        key=lambda row: (
            int(row["cap_count"]),
            float(row["validation_result"]["bounds"]["benign_occupancy_ucb"]),
            -float(row["validation_result"]["bounds"]["coverage_lcb"]),
            max(map(float, row["radii"])),
            int(row["token_id"]),
        )
    )
    maximum = int(config["funnel"]["validation"]["primary"]) + int(
        config["funnel"]["validation"]["secondary"]
    )
    frozen = certified[:maximum]
    write_jsonl(output / "validation" / "all_results.jsonl", results)
    write_jsonl(output / "validation" / "all_freeze_candidates.jsonl", freezes)
    write_json(
        output / "validation" / "frozen_caps.json",
        {
            "gate_open": bool(frozen),
            "primary": frozen[0] if frozen else None,
            "secondary": frozen[1:] if frozen else [],
            "candidate_count": len(frozen),
            "selection_order": ["minimum_cap_count", "minimum_occupancy_ucb", "maximum_coverage_lcb", "minimum_radius", "token_id"],
            "test_refit_forbidden": True,
        },
    )
    write_json(
        output / "validation" / "COMPLETE.json",
        {
            "tested": len(results),
            "certified": len(certified),
            "frozen": len(frozen),
            "gate_open": bool(frozen),
            "primary_single_cap": bool(frozen and int(frozen[0]["cap_count"]) == 1),
            "p1_p2_p3_separate": True,
        },
    )


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6_mode3_compact.yaml")
    parser.add_argument("--output", default="results/sticky_lab/sentence_t5_base/mode3_v6_compact")
    parser.add_argument("--device", default="cuda:0")
    sub = parser.add_subparsers(dest="command", required=True)
    enumerate_parser = sub.add_parser("enumerate-vocab")
    enumerate_parser.add_argument("--limit", type=int)
    cache = sub.add_parser("precompute-role")
    cache.add_argument("--role", required=True)
    for name in ("s0-shard", "stage-shard", "validation-shard"):
        item = sub.add_parser(name)
        item.add_argument("--shard", type=int, required=True)
        item.add_argument("--shards", type=int, required=True)
        if name == "stage-shard":
            item.add_argument("--stage", choices=("s1", "s2", "s3"), required=True)
    merge0 = sub.add_parser("merge-s0")
    merge0.add_argument("--shards", type=int, required=True)
    merge0.add_argument("--whitebox")
    merge0.add_argument("--blackbox")
    merge0.add_argument("--v5-history")
    merge = sub.add_parser("merge-stage")
    merge.add_argument("--shards", type=int, required=True)
    merge.add_argument("--stage", choices=("s1", "s2", "s3"), required=True)
    validation = sub.add_parser("merge-validation")
    validation.add_argument("--shards", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_config(Path(args.config))
    commands = {
        "enumerate-vocab": enumerate_vocab,
        "precompute-role": precompute_role,
        "s0-shard": s0_shard,
        "merge-s0": merge_s0,
        "stage-shard": stage_shard,
        "merge-stage": merge_stage,
        "validation-shard": validation_shard,
        "merge-validation": merge_validation,
    }
    commands[args.command](args, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
