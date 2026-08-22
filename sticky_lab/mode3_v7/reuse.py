"""Fail-closed r5 S0 reuse audit and aggregate fallback proposal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sticky_lab.mode3_v6_3.cache import CallSpace
from sticky_lab.mode3_v6_3.errors import ProtocolViolation
from sticky_lab.mode3_v6_3.report import read_jsonl


def _load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _token_cache_ordinals(root: Path, token_id: int) -> set[int]:
    directory = root / "embedding_cache" / f"token_{int(token_id)}"
    if not directory.is_dir():
        return set()
    result: set[int] = set()
    for path in directory.glob("*.ordinals.npy"):
        try:
            result.update(map(int, np.load(path, allow_pickle=False).reshape(-1)))
        except (OSError, ValueError):
            return set()
    return result


def _required_s0_ordinals(source: Path) -> tuple[set[int], set[int]]:
    call_space = CallSpace.read(source / "registration" / "call_space.jsonl")
    manifest_path = source / "position_manifest.json"
    if not manifest_path.is_file():
        return set(), set()
    position_manifest = _load(manifest_path)
    selected_requests = {
        (str(row["role"]), str(row["text_id"]), str(position))
        for row in position_manifest.get("entries", [])
        if str(row.get("stage")) == "s0"
        and str(row.get("role")) in {"fit", "score"}
        for position in row.get("positions", [])
        if str(position) in {"prefix", "suffix"}
    }
    triggered = {
        call_space.lookup_request(role, text_id, position).ordinal
        for role, text_id, position in selected_requests
    }
    benign = {
        entry.ordinal
        for entry in call_space.entries
        if entry.key.role == "discovery_benign" and entry.key.position == "clean"
    }
    return triggered, benign


def audit_r5_s0_reuse(
    source_output: Path,
    config: Mapping[str, Any],
    *,
    exact_cache_audit: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    source = Path(source_output).resolve()
    required = [
        source / "run_manifest.json",
        source / "registration" / "COMPLETE.json",
        source / "enumeration" / "COMPLETE.json",
        source / "legal_tokens.jsonl",
        source / "stages" / "s0" / "COMPLETE.json",
        source / "stages" / "s0" / "all_metrics.jsonl",
        source / "stages" / "s0" / "all_rejections.jsonl",
        source / "stages" / "s0" / "selection_audit.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ProtocolViolation(f"r5 S0 reuse source is incomplete: {missing[:8]}")
    reuse = config["reuse"]
    run = _load(source / "run_manifest.json")
    expected_identity = {
        "run_id": str(reuse["source_run_id"]),
        "code_commit": str(reuse["source_commit"]),
        "source_config_file_sha256": str(reuse["source_config_sha256"]),
        "model_revision": str(config["model"]["revision"]),
        "tokenizer_sha256": str(config["model"]["tokenizer_sha256"]),
    }
    drift = {
        key: {"expected": expected, "observed": run.get(key)}
        for key, expected in expected_identity.items()
        if run.get(key) != expected
    }
    if drift:
        raise ProtocolViolation(f"r5 S0 identity drift: {drift}")
    shard_root = source / "stages" / "s0"
    complete_shards = sorted(shard_root.glob("shard_*/COMPLETE.json"))
    failed_shards = sorted(shard_root.glob("shard_*/FAILED.json"))
    expected_shards = int(reuse["source_s0_shards"])
    if len(complete_shards) != expected_shards or failed_shards:
        raise ProtocolViolation(
            f"r5 S0 shard gate failed: complete={len(complete_shards)} "
            f"failed={len(failed_shards)}"
        )
    legal = read_jsonl(source / "legal_tokens.jsonl")
    metrics = read_jsonl(shard_root / "all_metrics.jsonl")
    rejections = read_jsonl(shard_root / "all_rejections.jsonl")
    legal_ids = [int(row["token_id"]) for row in legal]
    expected_legal = int(reuse["source_legal_tokens"])
    if len(legal_ids) != expected_legal or len(set(legal_ids)) != expected_legal:
        raise ProtocolViolation("r5 legal token inventory drift")
    observed = {int(row["token_id"]) for row in metrics}.union(
        int(row["token_id"]) for row in rejections
    )
    if observed != set(legal_ids) or len(metrics) + len(rejections) != expected_legal:
        raise ProtocolViolation("r5 S0 metrics/rejections do not cover the legal vocabulary")

    cache_status = "NOT_AUDITED"
    complete_token_caches = 0
    required_triggered = 0
    required_benign = 0
    benign_cache_complete = False
    missing_cache_token_sample: list[int] = []
    if exact_cache_audit:
        try:
            triggered_ordinals, benign_ordinals = _required_s0_ordinals(source)
        except (OSError, ValueError, KeyError, ProtocolViolation):
            triggered_ordinals, benign_ordinals = set(), set()
        required_triggered = len(triggered_ordinals)
        required_benign = len(benign_ordinals)
        if triggered_ordinals and benign_ordinals:
            benign_cached = _token_cache_ordinals(source, -2)
            benign_cache_complete = benign_ordinals.issubset(benign_cached)
            for token_id in legal_ids:
                cached = _token_cache_ordinals(source, token_id)
                if triggered_ordinals.issubset(cached):
                    complete_token_caches += 1
                elif len(missing_cache_token_sample) < 32:
                    missing_cache_token_sample.append(token_id)
            cache_status = (
                "RAW_PREFIX_SUFFIX_REUSABLE"
                if complete_token_caches == expected_legal and benign_cache_complete
                else "AGGREGATE_ONLY"
            )
        else:
            cache_status = "AGGREGATE_ONLY"
    reuse_level = (
        "A_RAW_PREFIX_SUFFIX"
        if cache_status == "RAW_PREFIX_SUFFIX_REUSABLE"
        else "C_AGGREGATE_PROPOSAL"
    )
    audit = {
        "schema_version": "mode3-v7-r5-s0-reuse-audit-v1",
        "status": "V7_S0_REUSE_AUDIT_PASSED",
        "source_output": str(source),
        "source_run_id": run["run_id"],
        "source_commit": run["code_commit"],
        "source_config_sha256": run["source_config_file_sha256"],
        "source_role_manifest_sha256": run.get("role_manifest_sha256"),
        "legal_token_count": expected_legal,
        "valid_candidates": len(metrics),
        "candidate_level_rejections": len(rejections),
        "all_s0_shards_complete": True,
        "failed_s0_shards": [],
        "reuse_level": reuse_level,
        "cache_audit": {
            "status": cache_status,
            "required_triggered_ordinals_per_token": required_triggered,
            "required_shared_benign_ordinals": required_benign,
            "tokens_with_complete_required_cache": complete_token_caches,
            "benign_cache_complete": benign_cache_complete,
            "missing_token_cache_sample": missing_cache_token_sample,
        },
        "reusable": {
            "legal_tokens": True,
            "tokenizer_audit": True,
            "document_source_manifests": True,
            "aggregate_s0_metrics": True,
            "raw_prefix_suffix_embeddings": reuse_level == "A_RAW_PREFIX_SUFFIX",
        },
        "not_reusable_formally": [
            "q92_radius",
            "old_primary_secondary",
            "old_pareto_order",
            "random_position",
            "center_fitted_with_random",
            "old_confirmation",
        ],
        "full_candidate_count": 256 if reuse_level == "A_RAW_PREFIX_SUFFIX" else 512,
    }
    return audit, legal, metrics


def aggregate_fallback_candidates(
    metrics: Sequence[Mapping[str, Any]],
    *,
    keep: int = 512,
    deterministic_audit: int = 64,
    seed: int = 20260822,
) -> tuple[list[int], dict[str, Any]]:
    """Use aggregate q92 results only as a deliberately wide proposal pool."""

    required_metrics = (
        "balanced_coverage",
        "worst_position_coverage",
        "benign_occupancy_core",
        "radius_degrees",
        "outside_to_inside",
        "center_restart_spread",
    )
    for row in metrics:
        if any(
            key not in row or not np.isfinite(float(row[key]))
            for key in required_metrics
        ):
            raise ProtocolViolation(
                f"aggregate r5 metric is incomplete/non-finite for token {row.get('token_id')}"
            )
    token_ids = [int(row["token_id"]) for row in metrics]
    if len(token_ids) != len(set(token_ids)):
        raise ProtocolViolation("aggregate r5 metrics repeat token IDs")
    lanes = (
        ("coverage", lambda row: (-float(row["balanced_coverage"]), int(row["token_id"]))),
        ("worst_position", lambda row: (-float(row["worst_position_coverage"]), int(row["token_id"]))),
        ("occupancy", lambda row: (float(row["benign_occupancy_core"]), int(row["token_id"]))),
        ("radius", lambda row: (float(row["radius_degrees"]), int(row["token_id"]))),
        ("migration", lambda row: (-float(row["outside_to_inside"]), int(row["token_id"]))),
        ("stability", lambda row: (float(row["center_restart_spread"]), int(row["token_id"]))),
    )
    if int(keep) <= 0 or int(deterministic_audit) < 0:
        raise ProtocolViolation("invalid aggregate fallback retention sizes")
    target = min(int(keep), len(token_ids))
    audit_target = min(int(deterministic_audit), target)
    scientific_cap = target - audit_target
    per_lane = max(1, int(np.ceil(scientific_cap / len(lanes))))
    ordered: list[int] = []
    reasons: dict[int, list[str]] = {}
    for name, key in lanes:
        for row in sorted(metrics, key=key)[:per_lane]:
            token_id = int(row["token_id"])
            reasons.setdefault(token_id, []).append(name)
            if token_id not in ordered:
                ordered.append(token_id)
    # Lane overlap can be extreme.  Fill the registered scientific quota with
    # a deterministic consensus order instead of silently shrinking FULL.
    consensus = sorted(
        metrics,
        key=lambda row: (
            -float(row["balanced_coverage"]),
            -float(row["worst_position_coverage"]),
            float(row["benign_occupancy_core"]),
            -float(row["outside_to_inside"]),
            float(row["radius_degrees"]),
            float(row["center_restart_spread"]),
            int(row["token_id"]),
        ),
    )
    for row in consensus:
        token_id = int(row["token_id"])
        if len(ordered) >= scientific_cap:
            break
        if token_id not in ordered:
            ordered.append(token_id)
            reasons.setdefault(token_id, []).append("deterministic_consensus_fill")
    scientific = ordered[:scientific_cap]
    remaining = sorted(set(token_ids) - set(scientific))
    rng = np.random.default_rng(int(seed))
    random_ids = sorted(
        map(
            int,
            rng.choice(
                np.asarray(remaining, dtype=np.int64),
                size=min(audit_target, len(remaining)),
                replace=False,
            ),
        )
    )
    for token_id in random_ids:
        reasons.setdefault(token_id, []).append("deterministic_random_audit")
    selected = scientific + random_ids
    if len(selected) != target or len(selected) != len(set(selected)):
        raise ProtocolViolation(
            f"aggregate fallback did not retain the registered target: "
            f"{len(selected)}/{target}"
        )
    return selected, {
        "schema_version": "mode3-v7-aggregate-fallback-selection-v1",
        "formal_v7_frontier": False,
        "purpose": "wide_candidate_proposal_only",
        "selected_tokens": len(selected),
        "lane_quota": per_lane,
        "deterministic_audit": len(random_ids),
        "seed": int(seed),
        "selected": [
            {"token_id": token_id, "reasons": reasons[token_id]}
            for token_id in selected
        ],
    }
