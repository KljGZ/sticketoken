"""Deterministic Pareto retention for the registered V6.3 funnel."""

from __future__ import annotations

import hashlib
import math
import unicodedata
from typing import Any, Mapping, Sequence

import numpy as np

from .errors import ProtocolViolation


MAXIMIZE = (
    "balanced_coverage", "worst_position_coverage", "worst_source_coverage",
    "outside_to_inside", "conditional_origin_outside",
)
MINIMIZE = (
    "radius_degrees", "benign_occupancy_core", "benign_occupancy_1_1",
    "benign_occupancy_auc_1_1_5", "center_drift_from_previous",
    "center_restart_spread",
)


def _finite(row: Mapping[str, Any]) -> bool:
    return all(math.isfinite(float(row[key])) for key in MAXIMIZE + MINIMIZE)


def dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    no_worse = all(float(left[key]) >= float(right[key]) for key in MAXIMIZE)
    no_worse &= all(float(left[key]) <= float(right[key]) for key in MINIMIZE)
    strictly = any(float(left[key]) > float(right[key]) for key in MAXIMIZE)
    strictly |= any(float(left[key]) < float(right[key]) for key in MINIMIZE)
    return bool(no_worse and strictly)


def pareto_fronts(rows: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    values = list(rows)
    if any(not _finite(row) for row in values):
        raise ProtocolViolation("non-finite ranking metric")
    remaining = set(range(len(values)))
    fronts: list[list[int]] = []
    while remaining:
        front = [
            index for index in sorted(remaining, key=lambda i: int(values[i]["token_id"]))
            if not any(dominates(values[other], values[index]) for other in remaining if other != index)
        ]
        if not front:
            raise ProtocolViolation("Pareto sorting made no progress")
        fronts.append(front)
        remaining.difference_update(front)
    return fronts


def _normalized_composite(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    score = np.zeros(len(rows), dtype=np.float64)
    for key in MAXIMIZE:
        values = np.asarray([float(row[key]) for row in rows])
        span = max(1e-12, float(values.max() - values.min()))
        score += (values - values.min()) / span
    for key in MINIMIZE:
        values = np.asarray([float(row[key]) for row in rows])
        span = max(1e-12, float(values.max() - values.min()))
        score += (values.max() - values) / span
    return score / (len(MAXIMIZE) + len(MINIMIZE))


def _diversity_bucket(row: Mapping[str, Any]) -> tuple[str, str, int, bool]:
    text = str(row.get("token_text", ""))
    category = unicodedata.category(text[0]) if text else "EMPTY"
    script = unicodedata.name(text[0], "UNKNOWN").split(" ", 1)[0] if text else "EMPTY"
    casing = "upper" if text.isupper() else "lower" if text.islower() else "mixed"
    return category, script + ":" + casing, min(len(text), 16), text.startswith(" ")


def select_stage(
    rows: Sequence[Mapping[str, Any]], keep: int, *, seed: int
) -> tuple[list[int], dict[str, Any]]:
    values = list(rows)
    if len(values) < int(keep):
        raise ProtocolViolation(f"cannot retain {keep} from {len(values)} valid candidates")
    token_ids = [int(row["token_id"]) for row in values]
    if len(token_ids) != len(set(token_ids)):
        raise ProtocolViolation("ranking input repeats token IDs")
    fronts = pareto_fronts(values)
    composite = _normalized_composite(values)
    quotas = {
        "pareto_composite": int(math.floor(int(keep) * 0.70)),
        "threshold_uncertainty": int(math.floor(int(keep) * 0.20)),
        "diversity": int(math.floor(int(keep) * 0.05)),
    }
    quotas["random_audit"] = int(keep) - sum(quotas.values())
    selected: list[int] = []
    reasons: dict[int, str] = {}

    pareto_order = [
        index for front in fronts
        for index in sorted(front, key=lambda i: (-composite[i], token_ids[i]))
    ]

    def add(order: Sequence[int], count: int, reason: str) -> None:
        if int(count) <= 0:
            return
        for index in order:
            token_id = token_ids[index]
            if token_id in reasons:
                continue
            selected.append(index)
            reasons[token_id] = reason
            if sum(value == reason for value in reasons.values()) >= int(count):
                break

    add(pareto_order, quotas["pareto_composite"], "pareto_composite")
    uncertainty = sorted(
        range(len(values)),
        key=lambda i: (
            abs(float(values[i]["balanced_coverage"]) - 0.90)
            + abs(float(values[i]["worst_position_coverage"]) - 0.85)
            + abs(float(values[i]["outside_to_inside"]) - 0.85)
            + abs(float(values[i]["conditional_origin_outside"]) - 0.95),
            -composite[i], token_ids[i],
        ),
    )
    add(uncertainty, quotas["threshold_uncertainty"], "threshold_uncertainty")
    buckets: dict[tuple[str, str, int, bool], list[int]] = {}
    for index, row in enumerate(values):
        buckets.setdefault(_diversity_bucket(row), []).append(index)
    diversity_order: list[int] = []
    active = {key: sorted(indices, key=lambda i: (-composite[i], token_ids[i])) for key, indices in buckets.items()}
    while active:
        for key in sorted(list(active), key=str):
            diversity_order.append(active[key].pop(0))
            if not active[key]:
                del active[key]
    add(diversity_order, quotas["diversity"], "diversity")
    random_order = sorted(
        range(len(values)),
        key=lambda i: (
            hashlib.sha256(f"v6.3-audit\0{seed}\0{token_ids[i]}".encode("utf-8")).hexdigest(),
            token_ids[i],
        ),
    )
    add(random_order, quotas["random_audit"], "random_audit")
    # Overlap between components is expected; fill any shortfall by the
    # registered Pareto/composite order without changing component quotas.
    for index in pareto_order:
        if len(selected) >= int(keep):
            break
        if token_ids[index] not in reasons:
            selected.append(index)
            reasons[token_ids[index]] = "deterministic_overlap_fill"
    if len(selected) != int(keep):
        raise ProtocolViolation(f"selection produced {len(selected)}/{keep}")
    selected.sort(key=lambda i: (
        next(front_index for front_index, front in enumerate(fronts) if i in front),
        -composite[i], token_ids[i],
    ))
    return [token_ids[index] for index in selected], {
        "schema_version": "mode3-v6-3-pareto-selection-v1",
        "keep": int(keep), "seed": int(seed), "quotas": quotas,
        "front_sizes": [len(front) for front in fronts],
        "selected": [
            {"token_id": token_ids[index], "reason": reasons[token_ids[index]], "composite": float(composite[index])}
            for index in selected
        ],
        "historical_candidate_quota": 0,
        "whitebox_quota": 0,
        "blackbox_quota": 0,
    }


def select_rapid_s0(
    rows: Sequence[Mapping[str, Any]],
    source_selection_audit: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[list[int], dict[str, Any]]:
    """Select the preregistered 200-candidate rapid-positive S0 union.

    The expensive Pareto ordering is reused from the completed r5 merge. All
    specialist orders are recomputed from the immutable merged metrics, and a
    deterministic audit sample measures early-screen miss risk.
    """
    values = list(rows)
    if len(values) < 200 or any(not _finite(row) for row in values):
        raise ProtocolViolation("rapid S0 requires at least 200 finite candidates")
    token_ids = [int(row["token_id"]) for row in values]
    if len(token_ids) != len(set(token_ids)):
        raise ProtocolViolation("rapid S0 input repeats token IDs")
    index_by_token = {token_id: index for index, token_id in enumerate(token_ids)}
    source_selected = source_selection_audit.get("selected", [])
    pareto_order = [
        index_by_token[int(item["token_id"])]
        for item in source_selected
        if (
            str(item.get("reason")) == "pareto_composite"
            and int(item["token_id"]) in index_by_token
        )
    ]
    if len(pareto_order) < 120:
        raise ProtocolViolation("r5 selection audit cannot supply the rapid Pareto quota")

    quotas = {
        "pareto": 120,
        "worst_position": 20,
        "lowest_occupancy": 20,
        "migration": 15,
        "compact_radius": 10,
        "bootstrap_stability": 10,
        "deterministic_audit": 5,
    }
    selected: list[int] = []
    reasons: dict[int, str] = {}

    def add(order: Sequence[int], count: int, reason: str) -> None:
        added = 0
        for index in order:
            token_id = token_ids[index]
            if token_id in reasons:
                continue
            selected.append(index)
            reasons[token_id] = reason
            added += 1
            if added == int(count):
                return

    add(pareto_order, quotas["pareto"], "pareto")
    add(sorted(range(len(values)), key=lambda i: (
        -float(values[i]["worst_position_coverage"]),
        -float(values[i]["balanced_coverage"]), token_ids[i],
    )), quotas["worst_position"], "worst_position")
    add(sorted(range(len(values)), key=lambda i: (
        float(values[i]["benign_occupancy_core"]),
        float(values[i]["benign_occupancy_1_1"]), token_ids[i],
    )), quotas["lowest_occupancy"], "lowest_occupancy")
    add(sorted(range(len(values)), key=lambda i: (
        -float(values[i]["outside_to_inside"]),
        -float(values[i]["conditional_origin_outside"]), token_ids[i],
    )), quotas["migration"], "migration")
    add(sorted(range(len(values)), key=lambda i: (
        float(values[i]["radius_degrees"]),
        -float(values[i]["balanced_coverage"]), token_ids[i],
    )), quotas["compact_radius"], "compact_radius")
    add(sorted(range(len(values)), key=lambda i: (
        float(values[i]["center_restart_spread"]),
        -float(values[i]["worst_position_coverage"]), token_ids[i],
    )), quotas["bootstrap_stability"], "bootstrap_stability")
    audit_order = sorted(range(len(values)), key=lambda i: (
        hashlib.sha256(
            f"v6.3-rapid-r6-audit\0{seed}\0{token_ids[i]}".encode("utf-8")
        ).hexdigest(),
        token_ids[i],
    ))
    add(audit_order, quotas["deterministic_audit"], "deterministic_audit")
    for index in pareto_order:
        if len(selected) >= 200:
            break
        if token_ids[index] not in reasons:
            selected.append(index)
            reasons[token_ids[index]] = "deterministic_overlap_fill"
    if len(selected) != 200:
        raise ProtocolViolation(f"rapid S0 selection produced {len(selected)}/200")
    return [token_ids[index] for index in selected], {
        "schema_version": "mode3-v6-3-rapid-s0-selection-v1",
        "method": "quota_union_reusing_r5_pareto_order",
        "keep": 200,
        "seed": int(seed),
        "quotas": quotas,
        "selected": [
            {"token_id": token_ids[index], "reason": reasons[token_ids[index]]}
            for index in selected
        ],
        "historical_candidate_quota": 0,
        "negative_claim_supported": False,
    }
