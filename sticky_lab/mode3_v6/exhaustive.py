"""Deterministic exhaustive single-token screening and union construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ScreenRecord:
    token_id: int
    token_text: str
    source: str
    search_margin_m90_1: float
    search_margin_m95_05: float
    radius_degrees: float
    triggered_coverage: float
    benign_occupancy: float
    near_ci_boundary: bool = False
    semantic_category: str = "unknown"


def assert_common_sample_manifest(manifest_hashes: Mapping[int, str]) -> str:
    unique = set(manifest_hashes.values())
    if len(unique) != 1:
        raise RuntimeError("V6 shards used different common text samples")
    return next(iter(unique))


def select_full_search_union(
    exhaustive: Sequence[ScreenRecord],
    additional_ids: Mapping[str, Iterable[int]],
    *,
    minimum: int,
    target: int,
    top_each: int = 1000,
    category_top: int = 20,
) -> tuple[list[int], dict[int, tuple[str, ...]]]:
    if target < minimum:
        raise ValueError("target smaller than mandatory minimum")
    tracks: dict[int, set[str]] = {}

    def add(token_id: int, source: str) -> None:
        tracks.setdefault(int(token_id), set()).add(source)

    rankings = (
        sorted(exhaustive, key=lambda r: (-r.search_margin_m90_1, r.token_id)),
        sorted(exhaustive, key=lambda r: (-r.search_margin_m95_05, r.token_id)),
        sorted(exhaustive, key=lambda r: (r.radius_degrees, r.token_id)),
        sorted(exhaustive, key=lambda r: (r.benign_occupancy, r.token_id)),
    )
    for ranking in rankings:
        for row in ranking[:top_each]:
            add(row.token_id, "exhaustive")
    for row in exhaustive:
        if (row.triggered_coverage >= 0.85 and row.benign_occupancy <= 0.02) or row.near_ci_boundary:
            add(row.token_id, "exhaustive_near_gate")
    categories: dict[str, list[ScreenRecord]] = {}
    for row in exhaustive:
        categories.setdefault(row.semantic_category, []).append(row)
    for rows in categories.values():
        for row in sorted(rows, key=lambda r: (-r.search_margin_m90_1, r.token_id))[:category_top]:
            add(row.token_id, "exhaustive_semantic_stratum")
    mandatory_additional: set[int] = set()
    for source, ids in additional_ids.items():
        for token_id in ids:
            add(int(token_id), source)
            mandatory_additional.add(int(token_id))
    # Fill deterministically from the exhaustive ranking; never silently lower the minimum.
    for row in rankings[0]:
        if len(tracks) >= target:
            break
        add(row.token_id, "exhaustive_fill")
    if len(tracks) < minimum:
        raise RuntimeError(f"only {len(tracks)} full-search candidates; V6 requires at least {minimum}")
    scores = {row.token_id: row.search_margin_m90_1 for row in exhaustive}
    selected = sorted(mandatory_additional)
    selected.extend(
        token_id for token_id in sorted(tracks, key=lambda item: (-scores.get(item, float("-inf")), item))
        if token_id not in mandatory_additional
    )
    selected = selected[: max(target, len(mandatory_additional))]
    return selected, {token_id: tuple(sorted(tracks[token_id])) for token_id in selected}
