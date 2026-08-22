"""Benign-occupancy-constrained radius selection for V7."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from sticky_lab.mode3_v6_3.errors import NumericalNonFinite, ShapeMismatch

from .statistics import source_balanced_occupancy


def beta_key(beta: float) -> str:
    return f"{100.0 * float(beta):g}%"


@dataclass(frozen=True)
class RadiusOperatingPoint:
    beta: float
    feasible: bool
    radius: float | None
    radius_degrees: float | None
    benign_occupancy_point: float | None
    benign_occupancy_ucb: float | None
    worst_source_occupancy_ucb: float | None
    benign_occupancy_by_source: dict[str, Any]
    observed_inside: int
    observed_total: int
    boundary_rule: str = "largest_closed_radius_with_source_balanced_ucb_le_beta"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_distances(
    distances_by_source: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    if not distances_by_source:
        raise ShapeMismatch("occupancy calibration has no sources")
    result: dict[str, np.ndarray] = {}
    for source, raw in distances_by_source.items():
        values = np.asarray(raw, dtype=np.float64).reshape(-1)
        if len(values) == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise NumericalNonFinite(f"invalid benign distances for source {source}")
        result[str(source)] = values
    return result


def occupancy_at_radius(
    distances_by_source: Mapping[str, Any],
    radius: float,
    *,
    familywise_alpha: float,
) -> dict[str, Any]:
    values = _validate_distances(distances_by_source)
    threshold = float(radius)
    if not math.isfinite(threshold) or threshold < 0:
        raise NumericalNonFinite("invalid occupancy radius")
    return source_balanced_occupancy(
        {source: distances <= threshold for source, distances in values.items()},
        familywise_alpha=float(familywise_alpha),
    )


def _candidate_radii(values: Mapping[str, np.ndarray], maximum: float) -> np.ndarray:
    merged = np.concatenate(list(values.values()))
    observed = np.unique(merged[merged <= maximum])
    candidates: list[float] = []
    minimum = float(merged.min())
    if minimum > 0:
        candidates.append(float(np.nextafter(min(minimum, maximum), 0.0)))
    else:
        candidates.append(float(np.nextafter(0.0, 1.0)))
    candidates.extend(map(float, observed))
    candidates.append(float(maximum))
    output = np.unique(np.asarray(candidates, dtype=np.float64))
    output = output[(output > 0) & (output <= maximum)]
    if len(output) == 0:
        raise NumericalNonFinite("no positive radius candidate")
    return output


def occupancy_constrained_frontier(
    distances_by_source: Mapping[str, Any],
    occupancy_grid: Sequence[float],
    *,
    maximum_radius_degrees: float = 35.0,
    familywise_alpha: float = 0.05,
) -> list[RadiusOperatingPoint]:
    """Select the largest calibration-feasible radius for every frozen beta.

    Calibration selection is intentionally separate from final confirmation.
    Pointwise source-wise Clopper--Pearson bounds make the registered
    calibration rule deterministic; only a new independent confirm role may
    provide the final certificate.
    """

    values = _validate_distances(distances_by_source)
    betas = tuple(map(float, occupancy_grid))
    if not betas or any(not 0 < beta < 1 for beta in betas):
        raise ValueError("invalid occupancy grid")
    if tuple(sorted(set(betas))) != betas:
        raise ValueError("occupancy grid must be strictly increasing")
    maximum = math.radians(float(maximum_radius_degrees))
    candidates = _candidate_radii(values, maximum)
    memo: dict[int, dict[str, Any]] = {}

    def evaluate(index: int) -> dict[str, Any]:
        if index not in memo:
            memo[index] = occupancy_at_radius(
                values,
                float(candidates[index]),
                familywise_alpha=float(familywise_alpha),
            )
        return memo[index]

    result: list[RadiusOperatingPoint] = []
    previous_index = -1
    for beta in betas:
        low, high = 0, len(candidates) - 1
        best = -1
        while low <= high:
            middle = (low + high) // 2
            if float(evaluate(middle)["ucb"]) <= beta + 1e-15:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best < 0:
            result.append(
                RadiusOperatingPoint(
                    beta,
                    False,
                    None,
                    None,
                    None,
                    None,
                    None,
                    {},
                    0,
                    sum(len(sample) for sample in values.values()),
                )
            )
            continue
        if best < previous_index:
            raise AssertionError("radius frontier is not monotone in beta")
        previous_index = best
        occupancy = evaluate(best)
        radius = float(candidates[best])
        by_source = dict(occupancy["by_source"])
        result.append(
            RadiusOperatingPoint(
                beta=beta,
                feasible=True,
                radius=radius,
                radius_degrees=math.degrees(radius),
                benign_occupancy_point=float(occupancy["point"]),
                benign_occupancy_ucb=float(occupancy["ucb"]),
                worst_source_occupancy_ucb=float(occupancy["worst_source_ucb"]),
                benign_occupancy_by_source=by_source,
                observed_inside=sum(int(row["successes"]) for row in by_source.values()),
                observed_total=sum(int(row["total"]) for row in by_source.values()),
            )
        )
    feasible_radii = [
        float(point.radius)
        for point in result
        if point.feasible and point.radius is not None
    ]
    if any(
        float(right) + 1e-15 < float(left)
        for left, right in zip(feasible_radii, feasible_radii[1:])
    ):
        raise AssertionError("larger occupancy budget reduced the feasible radius")
    return result


def legacy_q92_radius(
    triggered_distances: Mapping[tuple[str, str], Any],
    *,
    quantile: float = 0.92,
) -> dict[str, Any]:
    """Retain V6 q92 strictly as a non-selecting diagnostic."""

    by_position: dict[str, float] = {}
    for position in ("prefix", "suffix"):
        strata = [
            np.asarray(values, dtype=np.float64).reshape(-1)
            for (source, name), values in triggered_distances.items()
            if str(name) == position
        ]
        if not strata:
            raise ShapeMismatch(f"missing q92 diagnostic position {position}")
        # Equal source mass via an equal-size quantile grid avoids source-size weighting.
        per_source = [float(np.quantile(values, float(quantile))) for values in strata]
        by_position[position] = float(np.mean(per_source))
    radius = max(by_position.values())
    return {
        "diagnostic_only": True,
        "design_quantile": float(quantile),
        "position_radii": by_position,
        "shared_radius": radius,
        "shared_radius_degrees": math.degrees(radius),
        "used_for_v7_selection": False,
    }
