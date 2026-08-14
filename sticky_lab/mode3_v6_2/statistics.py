"""Source/position-aware simultaneous inference for V6.2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from sticky_lab.mode3_v6.statistics import (
    clopper_pearson_lower,
    clopper_pearson_upper,
)

from .errors import NumericalNonFinite, ShapeMismatch


Stratum = Tuple[str, str]


def trapezoidal_integral(y: Sequence[float], x: Sequence[float]) -> float:
    """NumPy-version-stable trapezoidal integration.

    NumPy 2.x exposes ``trapezoid`` while older pinned environments expose
    only the mathematically identical ``trapz``.  Resolve the implementation
    explicitly so the registered statistic does not depend on the runtime API
    spelling.
    """
    implementation = getattr(np, "trapezoid", None)
    if implementation is None:
        implementation = np.trapz
    return float(implementation(y, x))


@dataclass(frozen=True)
class BinomialBound:
    successes: int
    total: int
    estimate: float
    lower: float
    upper: float
    alpha: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BalancedCertificate:
    strata: dict[str, BinomialBound]
    balanced_lower: float
    balanced_upper: float
    worst_position_lower: float
    worst_source_lower: float
    position_lower: dict[str, float]
    source_lower: dict[str, float]
    familywise_alpha: float
    correction: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["strata"] = {key: bound.to_dict() for key, bound in self.strata.items()}
        return value


def _as_binary(values: Sequence[bool]) -> np.ndarray:
    result = np.asarray(values, dtype=bool).reshape(-1)
    if len(result) == 0:
        raise ShapeMismatch("empty Bernoulli stratum")
    return result


def binomial_bound(values: Sequence[bool], *, alpha: float) -> BinomialBound:
    sample = _as_binary(values)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0,1)")
    successes = int(sample.sum())
    total = len(sample)
    return BinomialBound(
        successes=successes,
        total=total,
        estimate=successes / total,
        lower=clopper_pearson_lower(successes, total, 1.0 - alpha),
        upper=clopper_pearson_upper(successes, total, 1.0 - alpha),
        alpha=float(alpha),
    )


def simultaneous_balanced_bounds(
    membership: Mapping[Stratum, Sequence[bool]],
    *,
    familywise_alpha: float = 0.05,
) -> BalancedCertificate:
    """Bonferroni bounds without treating positions of one text as IID units."""
    if not membership:
        raise ShapeMismatch("no source x position strata")
    keys = sorted((str(source), str(position)) for source, position in membership)
    alpha = float(familywise_alpha) / len(keys)
    bounds: dict[Stratum, BinomialBound] = {
        key: binomial_bound(membership[key], alpha=alpha) for key in keys
    }
    positions = sorted({position for _, position in keys})
    sources = sorted({source for source, _ in keys})
    missing = [(source, position) for source in sources for position in positions if (source, position) not in bounds]
    if missing:
        raise ShapeMismatch(f"incomplete source x position grid: {missing}")
    position_lower = {
        position: float(np.mean([bounds[(source, position)].lower for source in sources]))
        for position in positions
    }
    source_lower = {
        source: float(np.mean([bounds[(source, position)].lower for position in positions]))
        for source in sources
    }
    lower_values = [bound.lower for bound in bounds.values()]
    upper_values = [bound.upper for bound in bounds.values()]
    return BalancedCertificate(
        strata={f"{source}::{position}": bounds[(source, position)] for source, position in keys},
        balanced_lower=float(np.mean(lower_values)),
        balanced_upper=float(np.mean(upper_values)),
        worst_position_lower=min(position_lower.values()),
        worst_source_lower=min(source_lower.values()),
        position_lower=position_lower,
        source_lower=source_lower,
        familywise_alpha=float(familywise_alpha),
        correction="bonferroni",
    )


def simultaneous_source_occupancy(
    membership: Mapping[str, Sequence[bool]],
    *,
    familywise_alpha: float = 0.05,
) -> dict[str, Any]:
    if not membership:
        raise ShapeMismatch("no benign source strata")
    alpha = float(familywise_alpha) / len(membership)
    bounds = {
        str(source): binomial_bound(values, alpha=alpha)
        for source, values in sorted(membership.items())
    }
    return {
        "source_bounds": {source: value.to_dict() for source, value in bounds.items()},
        "balanced_estimate": float(np.mean([value.estimate for value in bounds.values()])),
        "balanced_ucb": float(np.mean([value.upper for value in bounds.values()])),
        "worst_source_ucb": max(value.upper for value in bounds.values()),
        "familywise_alpha": float(familywise_alpha),
        "correction": "bonferroni",
    }


def migration_certificates(
    clean_inside: Mapping[Stratum, Sequence[bool]],
    triggered_inside: Mapping[Stratum, Sequence[bool]],
    *,
    familywise_alpha: float = 0.05,
) -> dict[str, Any]:
    if set(clean_inside) != set(triggered_inside):
        raise ShapeMismatch("clean and triggered strata differ")
    outside_to_inside: dict[Stratum, np.ndarray] = {}
    conditional_origin: dict[Stratum, np.ndarray] = {}
    conditional_entry: dict[str, dict[str, int | float]] = {}
    for key in sorted(clean_inside):
        clean = _as_binary(clean_inside[key])
        triggered = _as_binary(triggered_inside[key])
        if clean.shape != triggered.shape:
            raise ShapeMismatch(f"paired migration shape mismatch for {key}")
        outside_to_inside[key] = (~clean) & triggered
        origin_values = (~clean)[triggered]
        # No triggered-inside observation makes the conditional claim
        # undefined; encode a conservative failed trial instead of crashing.
        conditional_origin[key] = origin_values if len(origin_values) else np.asarray([False])
        outside = ~clean
        entered = int(np.sum(outside & triggered))
        conditional_entry[f"{key[0]}::{key[1]}"] = {
            "clean_outside": int(outside.sum()),
            "entered": entered,
            "estimate": entered / max(1, int(outside.sum())),
        }
    return {
        "outside_to_inside": simultaneous_balanced_bounds(
            outside_to_inside, familywise_alpha=familywise_alpha
        ).to_dict(),
        "conditional_outside_origin": simultaneous_balanced_bounds(
            conditional_origin, familywise_alpha=familywise_alpha
        ).to_dict(),
        "conditional_entry_given_clean_outside": conditional_entry,
    }


def p2_position_certificates(
    membership_by_position: Mapping[str, Mapping[str, Sequence[bool]]],
    *,
    familywise_alpha: float = 0.05,
) -> dict[str, Any]:
    """Separate P2 certificate objects with simultaneous position control."""
    if set(membership_by_position) != {"prefix", "suffix", "random"}:
        raise ShapeMismatch("P2 requires prefix, suffix, and random")
    position_alpha = float(familywise_alpha) / 3.0
    result: dict[str, Any] = {}
    for position, sources in sorted(membership_by_position.items()):
        source_alpha = position_alpha / len(sources)
        bounds = {
            source: binomial_bound(values, alpha=source_alpha)
            for source, values in sorted(sources.items())
        }
        result[position] = {
            "source_bounds": {key: value.to_dict() for key, value in bounds.items()},
            "balanced_lcb": float(np.mean([value.lower for value in bounds.values()])),
            "worst_source_lcb": min(value.lower for value in bounds.values()),
            "familywise_position_alpha": position_alpha,
        }
    result["simultaneous_all_positions"] = True
    result["correction"] = "bonferroni"
    return result


def minimum_successes_for_lcb(total: int, threshold: float, *, alpha: float) -> int:
    if total <= 0:
        raise ValueError("total must be positive")
    low, high = 0, total
    while low < high:
        middle = (low + high) // 2
        lower = clopper_pearson_lower(middle, total, 1.0 - alpha)
        if lower > float(threshold):
            high = middle
        else:
            low = middle + 1
    return low


def gate_reachability_audit(
    stratum_sizes: Mapping[Stratum, int],
    *,
    design_coverage: float,
    target_lcb: float,
    familywise_alpha: float,
) -> dict[str, Any]:
    if design_coverage <= target_lcb:
        status = "structurally_underpowered"
    else:
        status = "design_has_positive_margin"
    alpha = float(familywise_alpha) / len(stratum_sizes)
    rows: dict[str, Any] = {}
    for (source, position), total_value in sorted(stratum_sizes.items()):
        total = int(total_value)
        required = minimum_successes_for_lcb(total, target_lcb, alpha=alpha)
        rows[f"{source}::{position}"] = {
            "total": total,
            "minimum_successes": required,
            "minimum_empirical_rate": required / total,
            "design_expected_successes": int(math.floor(design_coverage * total)),
            "reachable_at_design_point": int(math.floor(design_coverage * total)) >= required,
        }
    return {
        "status": status,
        "design_coverage": float(design_coverage),
        "target_lcb": float(target_lcb),
        "simultaneous_alpha_per_stratum": alpha,
        "strata": rows,
        "all_strata_reachable_at_design_point": all(row["reachable_at_design_point"] for row in rows.values()),
    }


def radial_occupancy_summary(
    normalized_radii: Sequence[float],
    multipliers: Sequence[float],
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    values = np.asarray(normalized_radii, dtype=np.float64).reshape(-1)
    if len(values) == 0 or np.any(values < 0) or not np.all(np.isfinite(values)):
        raise NumericalNonFinite("invalid normalized radii")
    grid = sorted(set(map(float, multipliers)))
    curve = []
    for multiplier in grid:
        inside = int(np.sum(values <= multiplier + 1e-12))
        curve.append({
            "multiplier": multiplier,
            "inside": inside,
            "total": len(values),
            "estimate": inside / len(values),
            "ucb": clopper_pearson_upper(inside, len(values), confidence),
        })
    basin = [row for row in curve if 1.0 <= row["multiplier"] <= 1.5]
    x = np.asarray([row["multiplier"] for row in basin], dtype=float)
    y = np.asarray([row["ucb"] for row in basin], dtype=float)
    auc = trapezoidal_integral(y, x) / max(1e-12, x[-1] - x[0]) if len(x) >= 2 else float("nan")
    eligible = [row["multiplier"] for row in curve if row["ucb"] <= 0.01]
    return {
        "curve": curve,
        "lambda_star": max(eligible, default=0.0),
        "occupancy_auc_1_1_5": auc,
    }
