"""Fail-closed, idempotent workers for the registered V6.2 experiment.

Every discovery stage encodes one token on a fresh fit/radius/score split and
refits all registered cap complexities.  Confirmation is deliberately absent
from this module: sealed roles are handled by :mod:`sticky_lab.mode3_v6_2.sealed`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sticky_lab.mode3_v6.insertion import insert_once_with_span
from sticky_lab.mode3_v6.tokenizer_audit import LegalToken, _realizes_exact_span

from .common import (
    load_config, load_legal, load_manifest, load_role, model_from_dict,
    read_jsonl, sha256_file, write_json, write_jsonl,
)
from .encoding import pretruncate_source
from .errors import CandidateRejected, CandidateRejectedTokenRealization, ProtocolViolation
from .evaluate import fit_and_score_stage
from .funnel import attach_stage_history, build_cap_archives, select_stage_models
from .geometry import FrozenCapModel, center_drift
from .oracle import V62FinalOracle, load_embedding_cache, records_sha256, write_embedding_cache
from .roles import canonical_sha256, load_role_contract


STAGES = ("s0", "s1", "s2", "full", "stability")
PREVIOUS = {"s1": "s0", "s2": "s1", "full": "s2", "stability": "full"}


def _oracle(config: Mapping[str, Any], output: Path, device: str, phase: str, track: str) -> V62FinalOracle:
    return V62FinalOracle(config, output=output, device=device, phase=phase, track=track)


def _cache(output: Path, role: str, records: Sequence[Mapping[str, str]]) -> np.ndarray:
    return load_embedding_cache(
        output / "base_embeddings" / f"{role}.npy",
        expected_role=role,
        expected_records_hash=records_sha256(records),
    )


def _role_names(config: Mapping[str, Any], stage: str) -> tuple[str, str, str]:
    item = config["funnel"]["full" if stage == "stability" else stage]
    return str(item["fit_role"]), str(item["radius_role"]), str(item["score_role"])


def _stage_dir(output: Path, stage: str) -> Path:
    return output / "funnel" / stage


def _load_selected_tokens(output: Path, stage: str) -> list[int]:
    value = json.loads((_stage_dir(output, stage) / "selected.json").read_text(encoding="utf-8"))
    return list(map(int, value["token_ids"]))


def _load_stage_models(output: Path, stage: str) -> dict[tuple[int, int], FrozenCapModel]:
    return {
        (int(row["token_id"]), int(row["cap_count"])): model_from_dict(row["model"])
        for row in read_jsonl(_stage_dir(output, stage) / "all_models.jsonl")
    }


def _tokenizer_digest(tokenizer: Any) -> str:
    digest = hashlib.sha256()
    for text, token_id in sorted(tokenizer.get_vocab().items(), key=lambda item: (int(item[1]), item[0])):
        digest.update(f"{int(token_id)}\0{text}\n".encode("utf-8"))
    return digest.hexdigest()


def _legal_in_context(
    tokenizer: Any,
    token_id: int,
    token_text: str,
    contexts: Sequence[tuple[str, Mapping[str, str]]],
    manifest: Any,
    maximum_length: int,
) -> tuple[bool, str | None]:
    """Audit all registered contexts without encoding model vectors."""
    for role, row in contexts:
        source, source_ids, _ = pretruncate_source(
            tokenizer, str(row.get("encoding_text", row["text"])),
            maximum_length=maximum_length, trigger_overhead=1,
        )
        for position in ("prefix", "suffix", "random"):
            triggered, span = insert_once_with_span(
                source, token_text, position, role=role, text_id=str(row["text_id"]),
                manifest=manifest, replicate=0,
            )
            if not _realizes_exact_span(tokenizer, triggered, span, token_id):
                return False, f"span:{role}:{row['text_id']}:{position}"
            plain = tokenizer(
                triggered, add_special_tokens=False, return_offsets_mapping=True,
                truncation=False, return_attention_mask=True,
            )
            ids = list(map(int, plain["input_ids"]))
            offsets = [tuple(map(int, pair)) for pair in plain["offset_mapping"]]
            overlap = [i for i, (a, b) in enumerate(offsets) if b > span[0] and a < span[1]]
            if len(overlap) != 1 or ids[overlap[0]] != int(token_id):
                return False, f"runtime:{role}:{row['text_id']}:{position}"
            if ids[: overlap[0]] + ids[overlap[0] + 1 :] != source_ids:
                return False, f"source_changed:{role}:{row['text_id']}:{position}"
            with_special = tokenizer(
                triggered, add_special_tokens=True, return_offsets_mapping=True,
                truncation=False, return_attention_mask=True,
            )
            special_ids = list(map(int, with_special["input_ids"]))
            special_offsets = [tuple(map(int, pair)) for pair in with_special["offset_mapping"]]
            special_overlap = [i for i, (a, b) in enumerate(special_offsets) if b > span[0] and a < span[1]]
            if (
                len(special_ids) > maximum_length or len(special_overlap) != 1
                or special_ids[special_overlap[0]] != int(token_id)
                or int(with_special["attention_mask"][special_overlap[0]]) != 1
            ):
                return False, f"attention:{role}:{row['text_id']}:{position}"
    return True, None


def enumerate_vocab(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    from transformers import AutoTokenizer

    output = Path(args.output)
    model = config["model"]
    local = Path(str(model["local_path"]))
    source = str(local) if local.is_dir() else str(model["id"])
    tokenizer = AutoTokenizer.from_pretrained(
        source, revision=None if local.is_dir() else model["revision"],
        trust_remote_code=bool(model["trust_remote_code"]),
    )
    audit_roles = ("s0_fit", "s0_radius", "s0_score")
    contexts = [(role, row) for role in audit_roles for row in load_role(output, role)]
    requested = int(config["tokenizer"]["contextual_audit_samples"])
    if len(contexts) != requested:
        raise ProtocolViolation(f"token audit requires exactly {requested} contexts, observed {len(contexts)}")
    if args.context_limit is not None:
        contexts = contexts[: int(args.context_limit)]
    vocab = tokenizer.get_vocab()
    special = set(map(int, getattr(tokenizer, "all_special_ids", [])))
    legal: list[LegalToken] = []
    rejected: list[dict[str, Any]] = []
    maximum = None if args.token_limit is None else int(args.token_limit)
    examined = 0
    manifest = load_manifest(output)
    for token_id in sorted(set(map(int, vocab.values()))):
        if token_id in special:
            continue
        token_text = tokenizer.decode(
            [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if not token_text or list(map(int, tokenizer.encode(token_text, add_special_tokens=False))) != [token_id]:
            continue
        examined += 1
        okay, reason = _legal_in_context(
            tokenizer, token_id, token_text, contexts, manifest,
            int(model["maximum_sequence_length"]),
        )
        visible = bool(token_text.strip()) and not any(ord(ch) < 32 and ch not in "\t\n\r" for ch in token_text)
        if okay:
            legal.append(LegalToken(token_id, token_text, visible, True, True, True, True))
        elif len(rejected) < 10000:
            rejected.append({"token_id": token_id, "token_text": token_text, "reason": reason})
        if maximum is not None and examined >= maximum:
            break
    target = output / "enumeration"
    write_jsonl(target / "legal_unrestricted.jsonl", (row.to_dict() for row in legal))
    write_jsonl(target / "legal_visible.jsonl", (row.to_dict() for row in legal if row.visible))
    write_jsonl(target / "rejection_sample.jsonl", rejected)
    write_json(target / "COMPLETE.json", {
        "schema_version": "mode3-v6-2-token-enumeration-v1",
        "tokenizer_sha256": _tokenizer_digest(tokenizer),
        "audited_contexts": len(contexts), "examined_roundtrip_tokens": examined,
        "legal_tokens": len(legal), "actual_tokenizer_length": 1,
        "runtime_assertion_required": True,
        "formal_exhaustive": args.token_limit is None and args.context_limit is None,
    })


def precompute_role(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output)
    role = str(args.role)
    records = load_role(output, role)
    bindings = load_role_contract(output / "registration" / "role_contract.json")
    if role not in bindings:
        raise ProtocolViolation(f"cannot cache unregistered role {role}")
    target = output / "base_embeddings" / f"{role}.npy"
    if target.is_file() and target.with_suffix(".json").is_file():
        _cache(output, role, records)
        return
    oracle = _oracle(config, output, args.device, "base_embeddings", "shared_clean")
    vectors = oracle.encode(
        [str(row.get("encoding_text", row["text"])) for row in records],
        metadata={"role": role, "cache_once": True},
    )
    write_embedding_cache(
        target, vectors, role=role, records_hash=records_sha256(records),
        model_revision=str(config["model"]["revision"]),
    )


def stage_shard(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output)
    stage = str(args.stage)
    fit_role, radius_role, score_role = _role_names(config, stage)
    fit_records, radius_records, score_records = (
        load_role(output, fit_role), load_role(output, radius_role), load_role(output, score_role)
    )
    benign_records = load_role(output, "discovery_benign")
    clean_score = _cache(output, score_role, score_records)
    benign = _cache(output, "discovery_benign", benign_records)
    legal = {row.token_id: row for row in load_legal(output)}
    token_ids = sorted(legal) if stage == "s0" else _load_selected_tokens(output, PREVIOUS[stage])
    token_ids = [token_id for index, token_id in enumerate(token_ids) if index % int(args.shards) == int(args.shard)]
    previous_models = {} if stage == "s0" else _load_stage_models(output, PREVIOUS[stage])
    oracle = _oracle(config, output, args.device, stage, "exhaustive_refit")
    manifest = load_manifest(output)
    metrics: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    audit_summaries: list[dict[str, Any]] = []
    for token_id in token_ids:
        token = legal[token_id]
        try:
            fitted, rejected = fit_and_score_stage(
                oracle, token_id=token_id, token_text=token.token_text, stage=stage,
                fit_records=fit_records, radius_records=radius_records,
                score_records=score_records, benign_records=benign_records,
                clean_score=clean_score, benign=benign, manifest=manifest,
                config=config, cap_counts=(1, 2, 3, 4),
                fit_role=fit_role, radius_role=radius_role, score_role=score_role,
            )
        except CandidateRejectedTokenRealization as error:
            rejections.append({
                "token_id": token_id, "token_text": token.token_text, "stage": stage,
                "status": "candidate_rejected", "reason": type(error).__name__, "detail": str(error),
            })
            continue
        rejections.extend(rejected)
        for model, metric, audit, _ in fitted:
            row = metric.to_dict()
            prior = previous_models.get((token_id, model.cap_count))
            if prior is not None:
                row["center_drift_from_previous"] = center_drift(prior, model)
            metrics.append(row)
            models.append({
                "token_id": token_id, "cap_count": model.cap_count,
                "stage": stage, "model": model.to_dict(),
                "geometry_audit": audit["geometry"],
            })
            token_audit = audit["tokenization_audit"]
            audit_summaries.append({
                "token_id": token_id, "cap_count": model.cap_count,
                "records": sum(len(values) for values in token_audit.values()),
                "all_attended": all(
                    int(item["attention_mask_value"]) == 1
                    for values in token_audit.values() for item in values
                ),
                "maximum_tokens_removed": max(
                    int(item["tokens_removed"]) for values in token_audit.values() for item in values
                ),
            })
    target = _stage_dir(output, stage) / f"shard_{int(args.shard):02d}"
    write_jsonl(target / "metrics.jsonl", metrics)
    write_jsonl(target / "models.jsonl", models)
    write_jsonl(target / "rejections.jsonl", rejections)
    write_jsonl(target / "tokenization_audit_summary.jsonl", audit_summaries)
    write_json(target / "COMPLETE.json", {
        "schema_version": "mode3-v6-2-stage-shard-v1", "stage": stage,
        "shard": int(args.shard), "shards": int(args.shards),
        "candidate_tokens": len(token_ids), "valid_models": len(metrics),
        "candidate_rejections": len(rejections),
        "role_hashes": {
            fit_role: records_sha256(fit_records), radius_role: records_sha256(radius_records),
            score_role: records_sha256(score_records),
            "discovery_benign": records_sha256(benign_records),
        },
        "from_scratch_refit": True, "cap_counts_attempted": [1, 2, 3, 4],
        "raw_forward_texts": oracle.raw_forward_texts,
    })


def merge_stage(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output)
    stage = str(args.stage)
    rows: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    role_hashes: list[dict[str, str]] = []
    candidate_count = 0
    for shard in range(int(args.shards)):
        target = _stage_dir(output, stage) / f"shard_{shard:02d}"
        complete = json.loads((target / "COMPLETE.json").read_text(encoding="utf-8"))
        if complete["stage"] != stage or complete["shards"] != int(args.shards):
            raise ProtocolViolation(f"incompatible {stage} shard {shard}")
        role_hashes.append(complete["role_hashes"])
        candidate_count += int(complete["candidate_tokens"])
        rows.extend(read_jsonl(target / "metrics.jsonl"))
        models.extend(read_jsonl(target / "models.jsonl"))
        rejections.extend(read_jsonl(target / "rejections.jsonl"))
    if any(value != role_hashes[0] for value in role_hashes[1:]):
        raise ProtocolViolation(f"{stage} shards used different registered roles")
    expected = set(map(int, (row.token_id for row in load_legal(output)))) if stage == "s0" else set(_load_selected_tokens(output, PREVIOUS[stage]))
    observed = {int(row["token_id"]) for row in rows} | {int(row["token_id"]) for row in rejections}
    if candidate_count != len(expected) or observed != expected:
        raise ProtocolViolation(f"{stage} candidate coverage mismatch: expected={len(expected)} observed={len(observed)}")
    if stage != "s0":
        prior = read_jsonl(_stage_dir(output, PREVIOUS[stage]) / "all_metrics.jsonl")
        rows = attach_stage_history(prior, rows)
    if stage == "full":
        keep = int(config["funnel"]["full"]["stability_reevaluation_candidates"])
    elif stage == "stability":
        keep = int(config["funnel"]["full"]["semantic_candidates"])
    else:
        keep = int(config["funnel"][stage]["keep"])
    selected_models, selection = select_stage_models(rows, keep)
    selected_tokens = [token_id for token_id, _ in selected_models]
    if len(selected_tokens) != keep:
        raise ProtocolViolation(f"{stage} retained {len(selected_tokens)}/{keep} unique tokens")
    target = _stage_dir(output, stage)
    write_jsonl(target / "all_metrics.jsonl", rows)
    write_jsonl(target / "all_models.jsonl", models)
    write_jsonl(target / "all_rejections.jsonl", rejections)
    write_json(target / "archives.json", build_cap_archives(rows))
    write_json(target / "selection_audit.json", selection)
    write_json(target / "selected.json", {
        "token_ids": selected_tokens,
        "models": [{"token_id": token, "cap_count": caps} for token, caps in selected_models],
    })
    write_json(target / "COMPLETE.json", {
        "schema_version": "mode3-v6-2-stage-merge-v1", "stage": stage,
        "candidate_tokens": candidate_count, "valid_models": len(rows),
        "candidate_rejections": len(rejections), "selected_tokens": len(selected_tokens),
        "independent_one_and_multicap_archives": True, "current_stage_refit_authoritative": True,
    })


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6_2_mode3.yaml")
    parser.add_argument("--output", default="results/sticky_lab/sentence_t5_base/mode3_v6_2")
    parser.add_argument("--device", default="cuda:0")
    sub = parser.add_subparsers(dest="command", required=True)
    enum = sub.add_parser("enumerate-vocab")
    enum.add_argument("--token-limit", type=int)
    enum.add_argument("--context-limit", type=int)
    cache = sub.add_parser("precompute-role"); cache.add_argument("--role", required=True)
    shard = sub.add_parser("stage-shard")
    shard.add_argument("--stage", choices=STAGES, required=True)
    shard.add_argument("--shard", type=int, required=True); shard.add_argument("--shards", type=int, required=True)
    merge = sub.add_parser("merge-stage")
    merge.add_argument("--stage", choices=STAGES, required=True); merge.add_argument("--shards", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_config(Path(args.config))
    {"enumerate-vocab": enumerate_vocab, "precompute-role": precompute_role,
     "stage-shard": stage_shard, "merge-stage": merge_stage}[args.command](args, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
