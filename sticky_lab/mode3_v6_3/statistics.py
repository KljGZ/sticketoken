"""Independent-text, source-position-aware inference for V6.3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from sticky_lab.mode3_v6.statistics import clopper_pearson_lower, clopper_pearson_upper

from .errors import ShapeMismatch


Stratum = Tuple[str, str]


def _binary(values: Any) -> np.ndarray:
    sample = np.asarray(values, dtype=bool).reshape(-1)
    if len(sample) == 0:
        raise ShapeMismatch("empty Bernoulli stratum")
    return sample


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


def binomial_bound(values: Any, *, alpha: float) -> BinomialBound:
    sample = _binary(values)
    if not 0 < float(alpha) < 1:
        raise ValueError("alpha must be in (0, 1)")
    successes = int(sample.sum())
    total = len(sample)
    confidence = 1.0 - float(alpha)
    return BinomialBound(
        successes, total, successes / total,
        clopper_pearson_lower(successes, total, confidence),
        clopper_pearson_upper(successes, total, confidence),
        float(alpha),
    )


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
    correction: str = "bonferroni_over_source_position"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["strata"] = {key: bound.to_dict() for key, bound in self.strata.items()}
        return value


def simultaneous_balanced_bounds(
    membership: Mapping[Stratum, Any], *, familywise_alpha: float = 0.05
) -> BalancedCertificate:
    if not membership:
        raise ShapeMismatch("no source-position membership")
    normalized = {(str(s), str(p)): _binary(v) for (s, p), v in membership.items()}
    sources = sorted({source for source, _ in normalized})
    positions = sorted({position for _, position in normalized})
    missing = [(source, position) for source in sources for position in positions if (source, position) not in normalized]
    if missing:
        raise ShapeMismatch(f"incomplete source-position strata: {missing}")
    per_alpha = float(familywise_alpha) / len(normalized)
    bounds = {key: binomial_bound(value, alpha=per_alpha) for key, value in normalized.items()}
    position_lower = {p: float(np.mean([bounds[(s, p)].lower for s in sources])) for p in positions}
    source_lower = {s: float(np.mean([bounds[(s, p)].lower for p in positions])) for s in sources}
    return BalancedCertificate(
        {f"{s}::{p}": bounds[(s, p)] for s, p in sorted(bounds)},
        float(np.mean([b.lower for b in bounds.values()])),
        float(np.mean([b.upper for b in bounds.values()])),
        min(position_lower.values()), min(source_lower.values()),
        position_lower, source_lower, float(familywise_alpha),
    )


def independent_text_strata(
    rows: Sequence[Mapping[str, Any]], inside: Any
) -> dict[Stratum, np.ndarray]:
    """Build confirmation strata and forbid a text from becoming repeated IID units."""
    values = _binary(inside)
    if len(rows) != len(values):
        raise ShapeMismatch("row/membership length mismatch")
    text_ids = [str(row["text_id"]) for row in rows]
    if len(text_ids) != len(set(text_ids)):
        raise ShapeMismatch("confirm requires one registered position per independent text")
    grouped: dict[Stratum, list[bool]] = {}
    for row, member in zip(rows, values):
        key = (str(row["source_id"]), str(row["position"]))
        grouped.setdefault(key, []).append(bool(member))
    return {key: np.asarray(sample, dtype=bool) for key, sample in grouped.items()}


def simultaneous_source_occupancy(
    membership: Mapping[str, Any], *, familywise_alpha: float = 0.05
) -> dict[str, Any]:
    if not membership:
        raise ShapeMismatch("no benign source strata")
    alpha = float(familywise_alpha) / len(membership)
    bounds = {str(source): binomial_bound(values, alpha=alpha) for source, values in sorted(membership.items())}
    return {
        "source_bounds": {source: bound.to_dict() for source, bound in bounds.items()},
        "balanced_estimate": float(np.mean([b.estimate for b in bounds.values()])),
        "balanced_ucb": float(np.mean([b.upper for b in bounds.values()])),
        "worst_source_ucb": max(b.upper for b in bounds.values()),
        "familywise_alpha": float(familywise_alpha),
    }


def migration_bounds(
    clean_inside: Mapping[Stratum, Any],
    triggered_inside: Mapping[Stratum, Any],
    *,
    familywise_alpha: float = 0.05,
) -> dict[str, Any]:
    if set(clean_inside) != set(triggered_inside):
        raise ShapeMismatch("clean and triggered strata differ")
    moved: dict[Stratum, np.ndarray] = {}
    origin: dict[Stratum, np.ndarray] = {}
    for key in sorted(clean_inside):
        clean = _binary(clean_inside[key])
        triggered = _binary(triggered_inside[key])
        if clean.shape != triggered.shape:
            raise ShapeMismatch(f"paired migration mismatch for {key}")
        moved[key] = (~clean) & triggered
        conditional = (~clean)[triggered]
        origin[key] = conditional if len(conditional) else np.asarray([False])
    return {
        "outside_to_inside": simultaneous_balanced_bounds(moved, familywise_alpha=familywise_alpha).to_dict(),
        "conditional_outside_origin": simultaneous_balanced_bounds(origin, familywise_alpha=familywise_alpha).to_dict(),
    }


def radial_occupancy(depths: Any, multipliers: Sequence[float], *, confidence: float = 0.95) -> dict[str, Any]:
    values = np.asarray(depths, dtype=np.float64).reshape(-1)
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ShapeMismatch("radial occupancy requires finite depths")
    curve = []
    alpha = 1.0 - float(confidence)
    for multiplier in map(float, multipliers):
        bound = binomial_bound(values <= multiplier + 1e-12, alpha=alpha)
        curve.append({"multiplier": multiplier, **bound.to_dict()})
    basin = [row for row in curve if 1.0 <= row["multiplier"] <= 1.5]
    if len(basin) >= 2:
        x = np.asarray([row["multiplier"] for row in basin])
        y = np.asarray([row["upper"] for row in basin])
        integrate = getattr(np, "trapezoid", np.trapz)
        auc = float(integrate(y, x) / (x[-1] - x[0]))
    else:
        auc = float("nan")
    eligible = [row["multiplier"] for row in curve if row["upper"] <= 0.01]
    return {"curve": curve, "lambda_star": max(eligible, default=0.0), "occupancy_auc_1_1_5": auc}


def minimum_successes_for_lower_bound(
    total: int, *, threshold: float = 0.90, confidence: float = 0.95
) -> int:
    for successes in range(int(math.floor(total * threshold)), total + 1):
        if clopper_pearson_lower(successes, total, confidence) > float(threshold):
            return successes
    return total + 1


def old_same_sample_calibration_lcb(
    total: int, *, calibration_quantile: float = 0.90, confidence: float = 0.95
) -> dict[str, Any]:
    order = min(int(total), int(math.ceil((int(total) + 1) * float(calibration_quantile))))
    lower = clopper_pearson_lower(order, int(total), float(confidence))
    return {
        "total": int(total), "calibration_order": order,
        "empirical_success_rate": order / int(total), "lcb": lower,
        "strict_p90_pass": bool(lower > 0.90),
    }


def binomial_power(
    total: int, true_rate: float, *, threshold: float = 0.90, confidence: float = 0.95
) -> float:
    required = minimum_successes_for_lower_bound(total, threshold=threshold, confidence=confidence)
    n = int(total)
    p = float(true_rate)
    if required <= 0:
        return 1.0
    if required > n:
        return 0.0
    # Continuity-corrected Gaussian survival is numerically stable at the
    # registered n=50,000; small tests use an exact log-PMF recurrence.
    if n >= 5000:
        z = (required - 0.5 - n * p) / math.sqrt(n * p * (1.0 - p))
        return float(0.5 * math.erfc(z / math.sqrt(2.0)))
    logs = [
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        + k * math.log(p) + (n - k) * math.log1p(-p)
        for k in range(required, n + 1)
    ]
    maximum = max(logs)
    return float(math.exp(maximum) * sum(math.exp(value - maximum) for value in logs))
