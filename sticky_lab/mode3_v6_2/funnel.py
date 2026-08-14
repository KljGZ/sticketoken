"""Stage-local refit metrics and independent one/multi-cap Pareto archives."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


MAXIMIZE = (
    "coverage_margin", "worst_position_coverage", "outside_to_inside", "semantic_anomaly",
)
MINIMIZE = (
    "radius_degrees", "benign_occupancy", "benign_occupancy_1_10",
    "occupancy_auc_1_1_5", "center_drift_from_previous",
)


def _key(row: Mapping[str, Any]) -> tuple[int, int]:
    return int(row["token_id"]), int(row.get("cap_count", 1))


def _valid(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if row.get("status", "valid") == "valid"]


def _value(row: Mapping[str, Any], field: str) -> float:
    defaults = {
        "coverage_margin": -1e9, "worst_position_coverage": -1e9,
        "outside_to_inside": -1e9, "semantic_anomaly": -1e9,
        "radius_degrees": 1e9, "benign_occupancy": 1.0,
        "benign_occupancy_1_10": 1.0, "occupancy_auc_1_1_5": 1.0,
        "center_drift_from_previous": 1e9,
    }
    value = row.get(field, defaults[field])
    return float(value) if value is not None else defaults[field]


def dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    no_worse = all(_value(left, name) >= _value(right, name) for name in MAXIMIZE)
    no_worse &= all(_value(left, name) <= _value(right, name) for name in MINIMIZE)
    strictly = any(_value(left, name) > _value(right, name) for name in MAXIMIZE)
    strictly |= any(_value(left, name) < _value(right, name) for name in MINIMIZE)
    return bool(no_worse and strictly)


def pareto_front(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values = _valid(rows)
    return sorted(
        [row for index, row in enumerate(values) if not any(index != other and dominates(candidate, row) for other, candidate in enumerate(values))],
        key=lambda row: _key(row),
    )


def _epsilon_cell(row: Mapping[str, Any]) -> tuple[int, ...]:
    widths = {
        "coverage_margin": 0.005, "worst_position_coverage": 0.01,
        "outside_to_inside": 0.01, "semantic_anomaly": 0.02,
        "radius_degrees": 0.5, "benign_occupancy": 0.002,
        "benign_occupancy_1_10": 0.01, "occupancy_auc_1_1_5": 0.01,
        "center_drift_from_previous": 0.01,
    }
    fields = MAXIMIZE + MINIMIZE
    return tuple(int(np.floor(_value(row, name) / widths[name])) for name in fields)


def epsilon_grid(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cells: dict[tuple[int, ...], dict[str, Any]] = {}
    for row in _valid(rows):
        cell = _epsilon_cell(row)
        incumbent = cells.get(cell)
        score = (
            -_value(row, "coverage_margin"), _value(row, "benign_occupancy"),
            _value(row, "radius_degrees"), _key(row),
        )
        if incumbent is None or score < (
            -_value(incumbent, "coverage_margin"), _value(incumbent, "benign_occupancy"),
            _value(incumbent, "radius_degrees"), _key(incumbent),
        ):
            cells[cell] = dict(row)
    return sorted(cells.values(), key=lambda row: _key(row))


def build_cap_archives(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    valid = _valid(rows)
    one = [row for row in valid if int(row.get("cap_count", 1)) == 1]
    multi = [row for row in valid if 2 <= int(row.get("cap_count", 1)) <= 4]
    return {
        "one_cap_pareto": pareto_front(one),
        "one_cap_epsilon": epsilon_grid(one),
        "multi_cap_pareto": pareto_front(multi),
        "multi_cap_epsilon": epsilon_grid(multi),
    }


def select_stage_models(
    rows: Sequence[Mapping[str, Any]],
    keep: int,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Select models while preserving independent one- and multi-cap evidence."""
    valid = _valid(rows)
    if not valid:
        raise RuntimeError("no valid V6.2 stage models")
    archives = build_cap_archives(valid)
    chosen: list[tuple[int, int]] = []
    chosen_tokens: set[int] = set()
    target = min(int(keep), len({int(row["token_id"]) for row in valid}))
    provenance: dict[tuple[int, int], set[str]] = defaultdict(set)

    def add(row: Mapping[str, Any], label: str) -> None:
        key = _key(row)
        provenance[key].add(label)
        if key[0] not in chosen_tokens and len(chosen) < int(keep):
            chosen.append(key)
            chosen_tokens.add(key[0])

    archive_names = tuple(archives)
    if target:
        one_seed = archives["one_cap_pareto"] or archives["one_cap_epsilon"]
        if one_seed:
            add(one_seed[0], "one_cap_archive_seed")
    if len(chosen) < target:
        multi_seed = archives["multi_cap_pareto"] or archives["multi_cap_epsilon"]
        for row in multi_seed:
            if int(row["token_id"]) not in chosen_tokens:
                add(row, "multi_cap_archive_seed")
                break
    depth = 0
    while len(chosen) < target and any(depth < len(archives[name]) for name in archive_names):
        for name in archive_names:
            if depth < len(archives[name]):
                add(archives[name][depth], name)
        depth += 1
    rankings: list[tuple[str, list[dict[str, Any]]]] = []
    for name in MAXIMIZE:
        rankings.append((f"rank:{name}", sorted(valid, key=lambda row: (-_value(row, name), _key(row)))))
    for name in MINIMIZE:
        rankings.append((f"rank:{name}", sorted(valid, key=lambda row: (_value(row, name), _key(row)))))
    depth = 0
    while len(chosen) < target:
        for label, ranking in rankings:
            if depth < len(ranking):
                add(ranking[depth], label)
        depth += 1
    audit = {
        "archives": {name: [_key(row) for row in values] for name, values in archives.items()},
        "provenance": {f"{token}:{caps}": sorted(provenance[(token, caps)]) for token, caps in chosen},
        "selected": chosen,
    }
    return chosen, audit


def attach_stage_history(
    prior_rows: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep current refit metrics authoritative; history is diagnostic only."""
    prior = {_key(row): dict(row) for row in prior_rows}
    prior_order = {_key(row): index for index, row in enumerate(sorted(_valid(prior_rows), key=lambda row: (-_value(row, "coverage_margin"), _key(row))))}
    current_order = {_key(row): index for index, row in enumerate(sorted(_valid(current_rows), key=lambda row: (-_value(row, "coverage_margin"), _key(row))))}
    output: list[dict[str, Any]] = []
    denominator = max(1, len(current_order) - 1)
    for source in current_rows:
        row = dict(source)
        key = _key(row)
        before = prior.get(key)
        history = list(before.get("stage_history", [])) if before else []
        if before:
            history.append({
                "stage": before.get("stage"),
                "radius_degrees": before.get("radius_degrees"),
                "coverage": before.get("triggered_coverage"),
                "occupancy": before.get("benign_occupancy"),
            })
            row.setdefault("center_drift_from_previous", float(row.get("center_drift_from_previous", 0.0)))
            row["rank_stability"] = 1.0 - abs(prior_order.get(key, denominator) - current_order.get(key, denominator)) / denominator
        else:
            row["rank_stability"] = 0.0
        row["stage_history"] = history
        row["selection_provenance"] = list(row.get("selection_provenance", []))
        output.append(row)
    return output


# Compatibility helper used only by small unit tests and dry-run adapters.
def select_candidates(
    rows: Sequence[Mapping[str, Any]],
    keep: int,
    *,
    additional: Mapping[str, Iterable[int]] | None = None,
    additional_quota: int | None = None,
) -> tuple[list[int], dict[int, list[str]]]:
    models, audit = select_stage_models(
        [dict(row, cap_count=int(row.get("cap_count", 1))) for row in rows], keep
    )
    ids: list[int] = []
    provenance: dict[int, list[str]] = {}
    for token_id, cap_count in models:
        if token_id not in ids:
            ids.append(token_id)
        provenance.setdefault(token_id, []).extend(audit["provenance"].get(f"{token_id}:{cap_count}", []))
    return ids[:keep], {key: sorted(set(value)) for key, value in provenance.items()}


merge_stage_history = attach_stage_history
