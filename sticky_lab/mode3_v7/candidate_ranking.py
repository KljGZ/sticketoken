"""Deterministic occupancy-grid candidate retention and freeze ranking."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from sticky_lab.mode3_v6_3.errors import ProtocolViolation

from .operating_point import operating_point_for_beta80


def _point_at_beta(frontier: Mapping[str, Any], beta: float) -> Mapping[str, Any]:
    for point in frontier["frontier"]:
        if math.isclose(float(point["beta"]), float(beta), rel_tol=0, abs_tol=1e-15):
            return point
    raise ProtocolViolation(f"candidate {frontier.get('token_id')} lacks beta {beta}")


def _grid_rank(frontier: Mapping[str, Any], beta: float) -> tuple[Any, ...]:
    point = _point_at_beta(frontier, beta)
    return (
        0 if bool(point["feasible"]) else 1,
        -float(point["minimum_position_coverage_lcb"]),
        -min(
            float(point["prefix_coverage_point"]),
            float(point["suffix_coverage_point"]),
        ),
        float("inf") if point["radius"] is None else float(point["radius"]),
        int(frontier["token_id"]),
    )


def select_s0_union(
    frontiers: Sequence[Mapping[str, Any]],
    *,
    occupancy_grid: Sequence[float],
    per_beta: int = 32,
    auc_count: int = 128,
    deterministic_audit_count: int = 64,
    maximum: int = 512,
    seed: int = 20260822,
) -> tuple[list[int], dict[str, Any]]:
    """Occupancy-grid union with an explicit deterministic random audit lane."""

    token_ids = [int(frontier["token_id"]) for frontier in frontiers]
    if len(token_ids) != len(set(token_ids)):
        raise ProtocolViolation("S0 frontier input repeats token IDs")
    if maximum <= deterministic_audit_count:
        raise ProtocolViolation("candidate maximum leaves no scientific lane")
    by_id = {int(frontier["token_id"]): frontier for frontier in frontiers}
    scientific_order: list[int] = []
    reasons: dict[int, list[str]] = {}

    def add(token_id: int, reason: str) -> None:
        reasons.setdefault(token_id, []).append(reason)
        if token_id not in scientific_order:
            scientific_order.append(token_id)

    beta_selections: dict[str, list[int]] = {}
    for beta in map(float, occupancy_grid):
        ordered = sorted(frontiers, key=lambda frontier: _grid_rank(frontier, beta))
        selected = [int(frontier["token_id"]) for frontier in ordered[: int(per_beta)]]
        beta_selections[f"{100 * beta:g}%"] = selected
        for token_id in selected:
            add(token_id, f"beta_{100 * beta:g}_top_{per_beta}")
    auc_selected = [
        int(frontier["token_id"])
        for frontier in sorted(
            frontiers,
            key=lambda frontier: (
                -float(frontier["coverage_auc_log_beta"]),
                int(frontier["token_id"]),
            ),
        )[: int(auc_count)]
    ]
    for token_id in auc_selected:
        add(token_id, f"auc_top_{auc_count}")

    scientific_cap = int(maximum) - int(deterministic_audit_count)
    scientific = scientific_order[:scientific_cap]
    remaining = sorted(set(token_ids) - set(scientific))
    rng = np.random.default_rng(int(seed))
    audit = (
        sorted(
            map(
                int,
                rng.choice(
                    np.asarray(remaining, dtype=np.int64),
                    size=min(int(deterministic_audit_count), len(remaining)),
                    replace=False,
                ),
            )
        )
        if remaining
        else []
    )
    for token_id in audit:
        reasons.setdefault(token_id, []).append("deterministic_random_audit")
    selected = scientific + audit
    if len(selected) > int(maximum) or len(selected) != len(set(selected)):
        raise AssertionError("invalid V7 S0 selection")
    return selected, {
        "schema_version": "mode3-v7-s0-selection-v1",
        "method": "occupancy_grid_union_plus_auc_plus_deterministic_audit",
        "occupancy_grid": list(map(float, occupancy_grid)),
        "per_beta": int(per_beta),
        "auc_count": int(auc_count),
        "deterministic_audit_count": len(audit),
        "maximum": int(maximum),
        "selected": [
            {
                "token_id": token_id,
                "token_text": str(by_id[token_id]["token_text"]),
                "reasons": reasons[token_id],
            }
            for token_id in selected
        ],
        "beta_selections": beta_selections,
        "auc_selected": auc_selected,
        "seed": int(seed),
    }


def primary_order(frontier: Mapping[str, Any]) -> tuple[Any, ...]:
    point = operating_point_for_beta80(frontier)
    if point is None:
        return (
            1,
            float("inf"),
            0.0,
            0.0,
            float("inf"),
            int(frontier["token_id"]),
        )
    return (
        0,
        float(frontier["beta80_ps"]),
        -float(point["minimum_position_coverage_lcb"]),
        -float(
            np.mean(
                [point["prefix_coverage_point"], point["suffix_coverage_point"]]
            )
        ),
        float(point["radius"]),
        int(frontier["token_id"]),
    )


def select_top_token_beta_pairs(
    frontiers: Sequence[Mapping[str, Any]], *, keep: int = 20
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    positives = [frontier for frontier in frontiers if frontier.get("beta80_ps") is not None]
    ordered = sorted(positives, key=primary_order)
    selected: list[dict[str, Any]] = []
    for rank, frontier in enumerate(ordered[: int(keep)], start=1):
        point = operating_point_for_beta80(frontier)
        if point is None:  # pragma: no cover - guarded above
            continue
        selected.append(
            {
                "rank": rank,
                "token_id": int(frontier["token_id"]),
                "token_text": str(frontier["token_text"]),
                "beta": float(frontier["beta80_ps"]),
                "radius": float(point["radius"]),
                "radius_degrees": float(point["radius_degrees"]),
                "minimum_position_coverage_lcb": float(
                    point["minimum_position_coverage_lcb"]
                ),
                "prefix_coverage_lcb": float(point["prefix_coverage_lcb"]),
                "suffix_coverage_lcb": float(point["suffix_coverage_lcb"]),
                "coverage_auc_log_beta": float(frontier["coverage_auc_log_beta"]),
                "center_hash": str(frontier["center_hash"]),
            }
        )
    return selected, {
        "schema_version": "mode3-v7-full-selection-v1",
        "candidate_frontiers": len(frontiers),
        "ps80_positive_candidates": len(positives),
        "selected_pairs": len(selected),
        "ranking": [
            "minimum_beta80_ps",
            "maximum_minimum_position_coverage_lcb",
            "maximum_mean_position_coverage_point",
            "minimum_radius",
            "minimum_token_id",
        ],
        "axis_geometry_used_for_selection": False,
    }


def choose_primary_and_secondaries(
    frontiers: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    positives = sorted(
        [frontier for frontier in frontiers if frontier.get("beta80_ps") is not None],
        key=primary_order,
    )
    if len(positives) < 5:
        raise ProtocolViolation(
            f"freeze requires at least five PS-80 candidates, observed {len(positives)}"
        )
    return positives[0], positives[1:5]
