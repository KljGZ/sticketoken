"""Deterministic multi-criterion funnel selection."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def _valid(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if row.get("status") == "valid"]


def _ranked_ids(rows: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    valid = _valid(rows)
    rankings = [
        sorted(valid, key=lambda row: (-float(row["triggered_coverage"]), int(row["token_id"]))),
        sorted(valid, key=lambda row: (-float(row["worst_position_coverage"]), int(row["token_id"]))),
        sorted(valid, key=lambda row: (float(row["radius_degrees"]), int(row["token_id"]))),
        sorted(valid, key=lambda row: (float(row.get("benign_occupancy", 1.0)), int(row["token_id"]))),
        sorted(valid, key=lambda row: (-float(row.get("outside_to_inside", 0.0)), int(row["token_id"]))),
        sorted(valid, key=lambda row: (-float(row.get("search_margin_m90_1", -1e9)), int(row["token_id"]))),
    ]
    return [[int(row["token_id"]) for row in ranking] for ranking in rankings]


def select_candidates(
    rows: Sequence[Mapping[str, Any]],
    keep: int,
    *,
    additional: Mapping[str, Iterable[int]] | None = None,
    additional_quota: int | None = None,
) -> tuple[list[int], dict[int, list[str]]]:
    """Interleave criterion ranks, then add bounded independent-track evidence."""

    rankings = _ranked_ids(rows)
    available = {token_id for ranking in rankings for token_id in ranking}
    provenance: dict[int, set[str]] = defaultdict(set)
    chosen: list[int] = []

    def add(token_id: int, source: str) -> None:
        token_id = int(token_id)
        if token_id not in available:
            return
        provenance[token_id].add(source)
        if token_id not in chosen and len(chosen) < int(keep):
            chosen.append(token_id)

    if additional:
        quota = int(additional_quota or max(1, keep // (2 * max(1, len(additional)))))
        for source in sorted(additional):
            for token_id in list(additional[source])[:quota]:
                add(int(token_id), source)
    depth = 0
    labels = ("coverage", "worst_position", "radius", "occupancy", "migration", "margin")
    while len(chosen) < keep and depth < max(map(len, rankings), default=0):
        for label, ranking in zip(labels, rankings):
            if depth < len(ranking):
                add(ranking[depth], f"rank:{label}")
                if len(chosen) >= keep:
                    break
        depth += 1
    if len(chosen) != min(keep, len(available)):
        raise RuntimeError(f"funnel selected {len(chosen)}/{keep} from {len(available)} valid candidates")
    return chosen, {token_id: sorted(provenance[token_id]) for token_id in chosen}


def merge_stage_history(
    prior: Mapping[int, Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Create conservative cumulative metrics used for later-stage pruning."""

    merged: list[dict[str, Any]] = []
    for row_value in current_rows:
        row = dict(row_value)
        token_id = int(row["token_id"])
        before = dict(prior[token_id])
        row.update(
            {
                "triggered_coverage": min(
                    float(before["triggered_coverage"]), float(row["triggered_coverage"])
                ),
                "worst_position_coverage": min(
                    float(before["worst_position_coverage"]),
                    float(row["worst_position_coverage"]),
                ),
                "outside_to_inside": min(
                    float(before["outside_to_inside"]), float(row["outside_to_inside"])
                ),
                "radius_degrees": float(before["radius_degrees"]),
                "radius_radians": float(before["radius_radians"]),
                "benign_occupancy": float(before.get("benign_occupancy", 1.0)),
                "benign_similarity_q995": float(
                    before.get("benign_similarity_q995", 1.0)
                ),
                "search_margin_m90_1": min(
                    float(before.get("search_margin_m90_1", -1e9)),
                    float(row.get("search_margin_m90_1", -1e9)),
                ),
                "stage_history": list(before.get("stage_history", [before.get("stage", "s0")]))
                + [row.get("stage")],
            }
        )
        merged.append(row)
    return merged
