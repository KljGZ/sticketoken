"""GPU workers for V6 enumeration, screening, full-search, and validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable
import hashlib

import yaml

from .atomic_io import write_json, write_jsonl
from .evaluation import certify_frozen_cap
from .exhaustive import ScreenRecord, assert_common_sample_manifest, select_full_search_union
from .experiment import evaluate_position_layers, evaluate_shared_token, records_hash
from .insertion import BoundaryManifest, BoundaryRecord
from .oracle_blackbox import SentenceTransformerFinalOracle
from .resource_errors import is_resource_exhaustion
from .tokenizer_audit import LegalToken, enumerate_actual_single_tokens, shard_legal_tokens


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _roles(output: Path, name: str) -> list[dict[str, str]]:
    return [dict(row) for row in _jsonl(output / "registration" / "roles" / f"{name}.jsonl")]


def _manifest(output: Path) -> BoundaryManifest:
    return BoundaryManifest([BoundaryRecord(**row) for row in _jsonl(output / "registration" / "random_boundaries.jsonl")])


def _oracle(config: dict[str, object], device: str) -> SentenceTransformerFinalOracle:
    model = config["model"]
    return SentenceTransformerFinalOracle(
        model["id"], revision=model["revision"], device=device,
        batch_size=model["batch_size"], local_path=model["local_path"],
        cache_folder=model["cache_folder"], trust_remote_code=model["trust_remote_code"],
    )


def enumerate_vocab(args: argparse.Namespace, config: dict[str, object]) -> None:
    from transformers import AutoTokenizer

    model = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model["local_path"] or model["id"], revision=None if model["local_path"] else model["revision"], trust_remote_code=model["trust_remote_code"])
    records = _roles(Path(args.output), "screen_fit")[:32]
    unrestricted, visible = enumerate_actual_single_tokens(
        tokenizer, context_records=records, manifest=_manifest(Path(args.output)), role="screen_fit",
        exclude_special=config["tokenizer"]["exclude_special_tokens"],
    )
    base = Path(args.output) / "enumeration"
    write_jsonl(base / "legal_unrestricted.jsonl", (row.to_dict() for row in unrestricted))
    write_jsonl(base / "legal_visible.jsonl", (row.to_dict() for row in visible))
    vocab_digest = hashlib.sha256()
    for row in unrestricted:
        vocab_digest.update(f"{row.token_id}\0{row.token_text}\n".encode("utf-8"))
    write_json(base / "COMPLETE.json", {
        "actual_tokenizer_length": 1, "unrestricted_count": len(unrestricted),
        "visible_count": len(visible), "common_context_manifest_hash": records_hash(records),
        "tokenizer_class": type(tokenizer).__name__, "tokenizer_vocab_sha256": vocab_digest.hexdigest(),
        "model_revision": model["revision"],
    })


def _legal(output: Path) -> list[LegalToken]:
    return [LegalToken(**{key: row[key] for key in LegalToken.__dataclass_fields__}) for row in _jsonl(output / "enumeration" / "legal_unrestricted.jsonl")]


def _evaluate_rows(args: argparse.Namespace, config: dict[str, object], token_ids: Iterable[int], phase: str) -> None:
    output = Path(args.output)
    legal = {row.token_id: row for row in _legal(output)}
    if phase == "screen":
        fit_role, eval_role, benign_role = "screen_fit", "screen_eval", "screen_benign"
        target = output / "screen" / f"shard_{args.shard:02d}"
    elif phase == "full_search":
        fit_role, eval_role, benign_role = "full_search_fit", "full_search_eval", "full_search_benign"
        target = output / "full_search" / f"shard_{args.shard:02d}"
    else:
        raise ValueError(phase)
    fit, evaluation, benign = _roles(output, fit_role), _roles(output, eval_role), _roles(output, benign_role)
    oracle = _oracle(config, args.device)
    rows = []
    caps = []
    for token_id in token_ids:
        token = legal[int(token_id)]
        try:
            cap, metrics, _ = evaluate_shared_token(
                oracle, token_id=token.token_id, token_text=token.token_text,
                fit_records=fit, eval_records=evaluation, benign_records=benign,
                fit_role=fit_role, eval_role=eval_role, manifest=_manifest(output),
                random_replicates=config["positions"]["random_replicates"],
                coverage=config["geometry"]["calibration"]["weak_coverage"],
                maximum_radius_degrees=config["geometry"]["maximum_radius_degrees"],
                source_tracks=("exhaustive",) if phase == "screen" else tuple(getattr(args, "source_tracks", ["union"])),
            )
            row = metrics.to_dict(); row["status"] = "valid"
            rows.append(row); caps.append(cap.to_json())
        except Exception as error:
            if is_resource_exhaustion(error):
                raise RuntimeError(f"resource exhaustion while evaluating token {token.token_id}") from error
            # One geometrically invalid token must not invalidate its shard.
            rows.append({
                "token_id": token.token_id, "token_text": token.token_text, "status": "invalid",
                "error_type": type(error).__name__, "error": str(error),
                "search_margin_m90_1": -1e9, "search_margin_m95_05": -1e9,
                "radius_degrees": 180.0, "triggered_coverage": 0.0, "benign_occupancy": 1.0,
            })
    write_jsonl(target / "metrics.jsonl", rows)
    write_jsonl(target / "caps.jsonl", caps)
    write_json(target / "COMPLETE.json", {
        "phase": phase, "candidate_count": len(rows), "fit_manifest_hash": records_hash(fit),
        "eval_manifest_hash": records_hash(evaluation), "benign_manifest_hash": records_hash(benign),
        "query_ledger": oracle.ledger.to_dict(),
        "model_fingerprint": {"revision": oracle.revision, "dimension": oracle.dimension, "adapter": type(oracle).__name__},
    })


def screen_shard(args: argparse.Namespace, config: dict[str, object]) -> None:
    tokens = shard_legal_tokens(_legal(Path(args.output)), args.shard, args.shards)
    _evaluate_rows(args, config, [token.token_id for token in tokens], "screen")


def merge_screen(args: argparse.Namespace, config: dict[str, object]) -> None:
    output = Path(args.output)
    completions = [json.loads((output / "screen" / f"shard_{shard:02d}" / "COMPLETE.json").read_text()) for shard in range(args.shards)]
    for field in ("fit_manifest_hash", "eval_manifest_hash", "benign_manifest_hash"):
        assert_common_sample_manifest({index: row[field] for index, row in enumerate(completions)})
    rows = [row for shard in range(args.shards) for row in _jsonl(output / "screen" / f"shard_{shard:02d}" / "metrics.jsonl")]
    expected = len(_legal(output))
    if len(rows) != expected or len({int(row["token_id"]) for row in rows}) != expected:
        raise RuntimeError("exhaustive enumeration is incomplete or duplicated")
    write_jsonl(output / "screen" / "all_metrics.jsonl", rows)
    write_json(output / "screen" / "COMPLETE.json", {"tokens": expected, "shards": args.shards, "common_samples_verified": True})


def build_union(args: argparse.Namespace, config: dict[str, object]) -> None:
    output = Path(args.output)
    rows = [ScreenRecord(
        token_id=int(row["token_id"]), token_text=str(row["token_text"]), source="exhaustive",
        search_margin_m90_1=float(row["search_margin_m90_1"]), search_margin_m95_05=float(row["search_margin_m95_05"]),
        radius_degrees=float(row["radius_degrees"]), triggered_coverage=float(row["triggered_coverage"]),
        benign_occupancy=float(row["benign_occupancy"]),
    ) for row in _jsonl(output / "screen" / "all_metrics.jsonl")]
    sources: dict[str, list[int]] = {}
    for name, path_value in (("whitebox", args.whitebox), ("blackbox", args.blackbox), ("v5_history", args.v5_history)):
        if path_value:
            payload = json.loads(Path(path_value).read_text(encoding="utf-8"))
            sources[name] = list(map(int, payload["token_ids"]))
    settings = config["enumeration"]
    selected, provenance = select_full_search_union(
        rows, sources, minimum=settings["full_search_candidates_minimum"], target=settings["full_search_candidates_target"],
        top_each=settings["union_rules"]["top_each_search_margin"], category_top=settings["union_rules"]["semantic_category_top"],
    )
    write_json(output / "candidate_union" / "selected.json", {"token_ids": selected, "provenance": {str(key): value for key, value in provenance.items()}})
    write_json(output / "candidate_union" / "COMPLETE.json", {
        "candidate_count": len(selected), "minimum": settings["full_search_candidates_minimum"],
        "target": settings["full_search_candidates_target"], "full_search_re_evaluation_required": True,
    })


def full_search_shard(args: argparse.Namespace, config: dict[str, object]) -> None:
    payload = json.loads((Path(args.output) / "candidate_union" / "selected.json").read_text())
    selected = [token_id for index, token_id in enumerate(payload["token_ids"]) if index % args.shards == args.shard]
    _evaluate_rows(args, config, selected, "full_search")


def merge_full_search(args: argparse.Namespace, config: dict[str, object]) -> None:
    output = Path(args.output)
    rows = [row for shard in range(args.shards) for row in _jsonl(output / "full_search" / f"shard_{shard:02d}" / "metrics.jsonl")]
    selected = json.loads((output / "candidate_union" / "selected.json").read_text())["token_ids"]
    if {int(row["token_id"]) for row in rows} != set(map(int, selected)):
        raise RuntimeError("full-search candidate mismatch")
    write_jsonl(output / "full_search" / "all_metrics.jsonl", rows)
    write_json(output / "full_search" / "COMPLETE.json", {"candidate_count": len(rows), "full_search_only_for_validation": True})


def validate_shard(args: argparse.Namespace, config: dict[str, object]) -> None:
    output = Path(args.output)
    all_rows = [row for row in _jsonl(output / "full_search" / "all_metrics.jsonl") if row.get("status") == "valid"]
    all_rows.sort(key=lambda row: (-float(row["search_margin_m90_1"]), int(row["token_id"])))
    selected = [int(row["token_id"]) for index, row in enumerate(all_rows[: args.validation_candidates]) if index % args.shards == args.shard]
    legal = {row.token_id: row for row in _legal(output)}
    oracle = _oracle(config, args.device)
    fit, calibration, benign = _roles(output, "cap_fit"), _roles(output, "cap_calibration"), _roles(output, "cap_calibration_benign")
    target = output / "validation" / f"shard_{args.shard:02d}"
    results = []
    freezes = []
    for token_id in selected:
        token = legal[token_id]
        cap, _, arrays = evaluate_shared_token(
            oracle, token_id=token_id, token_text=token.token_text, fit_records=fit,
            eval_records=calibration, benign_records=benign, fit_role="cap_fit", eval_role="cap_calibration",
            manifest=_manifest(output), random_replicates=config["positions"]["random_replicates"],
            coverage=config["geometry"]["calibration"]["weak_coverage"],
            maximum_radius_degrees=config["geometry"]["maximum_radius_degrees"], source_tracks=("full_search",),
        )
        result = certify_frozen_cap(
            cap, arrays["triggered"], arrays["paired_clean"], arrays["independent_benign"],
            confidence=config["certification"]["confidence"], coverage_lcb_threshold=config["certification"]["triggered_coverage_lcb"],
            occupancy_ucb_threshold=config["certification"]["independent_benign_occupancy_ucb"],
            outside_to_inside_lcb_threshold=config["certification"]["outside_to_inside_lcb"],
            conditional_outside_origin_lcb_threshold=config["certification"]["conditional_outside_origin_lcb"],
            radial_multipliers=config["radial_analysis"]["multipliers"],
        )
        layered = evaluate_position_layers(
            oracle, token_id=token_id, token_text=token.token_text, fit_records=fit,
            calibration_records=calibration, benign_records=benign, fit_role="cap_fit",
            calibration_role="cap_calibration", manifest=_manifest(output),
            random_replicates=config["positions"]["random_replicates"], config=config,
        )
        result["P3_shared"] = {"certified": result["certified"], "protocol": "P3_shared"}
        result["layered_evidence"] = layered
        results.append({
            "token_id": token_id, "token_text": token.token_text, "certified": result["certified"],
            "bounds": result["bounds"], "gates": result["gates"],
            "P1_position_specific": layered["P1_position_specific"],
            "P2_conditional": layered["P2_conditional"],
            "P3_shared": result["P3_shared"],
        })
        freeze = cap.to_json()
        freeze.update({"validation_certified": result["certified"], "validation_result": result})
        freezes.append(freeze)
    write_jsonl(target / "results.jsonl", results)
    write_jsonl(target / "freeze_candidates.jsonl", freezes)
    write_json(target / "COMPLETE.json", {"candidates": len(results), "query_ledger": oracle.ledger.to_dict()})


def merge_validation(args: argparse.Namespace, config: dict[str, object]) -> None:
    output = Path(args.output)
    results = [row for shard in range(args.shards) for row in _jsonl(output / "validation" / f"shard_{shard:02d}" / "results.jsonl")]
    freezes = [row for shard in range(args.shards) for row in _jsonl(output / "validation" / f"shard_{shard:02d}" / "freeze_candidates.jsonl")]
    if len(results) != len(freezes) or {int(row["token_id"]) for row in results} != {int(row["token_id"]) for row in freezes}:
        raise RuntimeError("validation shard mismatch")
    certified = [row for row in freezes if row["validation_certified"]]
    certified.sort(key=lambda row: (
        max(map(float, row["radii"])),
        float(row["validation_result"]["bounds"]["benign_occupancy_ucb"]),
        int(row["token_id"]),
    ))
    write_jsonl(output / "validation" / "all_results.jsonl", results)
    write_json(output / "validation" / "gate_summary.json", {
        "tested": len(results), "certified": len(certified), "gate_open": bool(certified),
        "primary_model": "single_frozen_spherical_cap", "multi_cap_rescue_needed": not bool(certified),
        "p1_p2_p3_must_be_reported_separately": True,
    })
    if certified:
        write_json(output / "validation" / "selected_freeze_source.json", certified[0])
    write_json(output / "validation" / "COMPLETE.json", {"shards": args.shards, "tested": len(results), "certified": len(certified)})


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/v6_mode3.yaml")
    p.add_argument("--output", default="results/sticky_lab/sentence_t5_base/mode3_v6")
    p.add_argument("--device", default="cuda:0")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("enumerate-vocab")
    for name in ("screen-shard", "full-search-shard", "validate-shard"):
        item = sub.add_parser(name); item.add_argument("--shard", type=int, required=True); item.add_argument("--shards", type=int, required=True)
        if name == "validate-shard": item.add_argument("--validation-candidates", type=int, default=5000)
    for name in ("merge-screen", "merge-full-search", "merge-validation"):
        item = sub.add_parser(name); item.add_argument("--shards", type=int, required=True)
    union = sub.add_parser("build-union")
    union.add_argument("--whitebox"); union.add_argument("--blackbox"); union.add_argument("--v5-history")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    commands = {
        "enumerate-vocab": enumerate_vocab, "screen-shard": screen_shard, "merge-screen": merge_screen,
        "build-union": build_union, "full-search-shard": full_search_shard, "merge-full-search": merge_full_search,
        "validate-shard": validate_shard, "merge-validation": merge_validation,
    }
    commands[args.command](args, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
