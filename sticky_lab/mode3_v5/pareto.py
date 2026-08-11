"""Constraint-aware non-dominated sorting and crowding selection."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np


OBJECTIVES = ("occupancy_auc", "cmax", "cavg")


def violation_vector(record: Mapping[str, Any]) -> tuple[float, ...]:
    values = record.get("constraint_violations", {})
    return tuple(max(0.0, float(values.get(key, 0.0))) for key in sorted(values))


def feasible(record: Mapping[str, Any]) -> bool:
    return all(value <= 0.0 for value in violation_vector(record))


def dominates(left: Mapping[str, Any], right: Mapping[str, Any], objectives: Sequence[str] = OBJECTIVES) -> bool:
    left_feasible = feasible(left)
    right_feasible = feasible(right)
    if left_feasible != right_feasible:
        return left_feasible
    if not left_feasible:
        left_violation = sum(violation_vector(left))
        right_violation = sum(violation_vector(right))
        if left_violation != right_violation:
            return left_violation < right_violation
    left_values = [float(left[key]) for key in objectives]
    right_values = [float(right[key]) for key in objectives]
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def non_dominated_front(records: Iterable[Mapping[str, Any]], objectives: Sequence[str] = OBJECTIVES) -> list[dict[str, Any]]:
    values = [dict(record) for record in records]
    return [
        record
        for index, record in enumerate(values)
        if not any(dominates(other, record, objectives) for other_index, other in enumerate(values) if other_index != index)
    ]


def non_dominated_sort(records: Sequence[Mapping[str, Any]], objectives: Sequence[str] = OBJECTIVES) -> list[list[int]]:
    domination_counts = [0] * len(records)
    dominated: list[list[int]] = [[] for _ in records]
    fronts: list[list[int]] = [[]]
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            if dominates(records[left], records[right], objectives):
                dominated[left].append(right)
                domination_counts[right] += 1
            elif dominates(records[right], records[left], objectives):
                dominated[right].append(left)
                domination_counts[left] += 1
    fronts[0] = [index for index, count in enumerate(domination_counts) if count == 0]
    cursor = 0
    while cursor < len(fronts) and fronts[cursor]:
        next_front = []
        for left in fronts[cursor]:
            for right in dominated[left]:
                domination_counts[right] -= 1
                if domination_counts[right] == 0:
                    next_front.append(right)
        if next_front:
            fronts.append(next_front)
        cursor += 1
    return fronts


def crowding_distance(records: Sequence[Mapping[str, Any]], indices: Sequence[int], objectives: Sequence[str] = OBJECTIVES) -> dict[int, float]:
    result = {int(index): 0.0 for index in indices}
    if len(indices) <= 2:
        return {int(index): float("inf") for index in indices}
    for objective in objectives:
        ordered = sorted(indices, key=lambda index: float(records[index][objective]))
        result[int(ordered[0])] = float("inf")
        result[int(ordered[-1])] = float("inf")
        low = float(records[ordered[0]][objective])
        high = float(records[ordered[-1]][objective])
        if np.isclose(low, high):
            continue
        for position in range(1, len(ordered) - 1):
            previous_value = float(records[ordered[position - 1]][objective])
            next_value = float(records[ordered[position + 1]][objective])
            result[int(ordered[position])] += (next_value - previous_value) / (high - low)
    return result


def select_nsga2(records: Sequence[Mapping[str, Any]], count: int, objectives: Sequence[str] = OBJECTIVES) -> list[int]:
    selected: list[int] = []
    for front in non_dominated_sort(records, objectives):
        if len(selected) + len(front) <= count:
            selected.extend(front)
            continue
        distance = crowding_distance(records, front, objectives)
        selected.extend(sorted(front, key=lambda index: (-distance[index], str(records[index].get("candidate_key", ""))))[: count - len(selected)])
        break
    return selected


def update_historical_archive(
    historical: Sequence[Mapping[str, Any]],
    new_records: Sequence[Mapping[str, Any]],
    maximum: int,
    objectives: Sequence[str] = OBJECTIVES,
) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for record in [*historical, *new_records]:
        key = str(record.get("candidate_key", record.get("token_ids", "")))
        previous = by_key.get(key)
        if previous is None or dominates(record, previous, objectives):
            by_key[key] = dict(record)
    values = non_dominated_front(by_key.values(), objectives)
    if len(values) <= maximum:
        return sorted(values, key=lambda record: str(record.get("candidate_key", "")))
    distance = crowding_distance(values, list(range(len(values))), objectives)
    indices = sorted(
        range(len(values)),
        key=lambda index: (-distance[index], str(values[index].get("candidate_key", ""))),
    )[:maximum]
    return [values[index] for index in indices]
