"""One-sided source-balanced inference for the V7 frontier."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from sticky_lab.mode3_v6.statistics import (
    clopper_pearson_lower,
    clopper_pearson_upper,
)
from sticky_lab.mode3_v6_3.errors import ShapeMismatch


Stratum = Tuple[str, str]


def _binary(values: Any, *, allow_empty: bool = False) -> np.ndarray:
    sample = np.asarray(values, dtype=bool).reshape(-1)
    if len(sample) == 0 and not allow_empty:
        raise ShapeMismatch("empty Bernoulli sample")
    return sample


@dataclass(frozen=True)
class BernoulliInterval:
    successes: int
    total: int
    estimate: float | None
    lower: float
    upper: float
    alpha: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bernoulli_interval(values: Any, *, alpha: float) -> BernoulliInterval:
    sample = _binary(values)
    if not 0 < float(alpha) < 1:
        raise ValueError("alpha must be in (0, 1)")
    successes = int(sample.sum())
    total = len(sample)
    confidence = 1.0 - float(alpha)
    return BernoulliInterval(
        successes=successes,
        total=total,
        estimate=successes / total,
        lower=clopper_pearson_lower(successes, total, confidence),
        upper=clopper_pearson_upper(successes, total, confidence),
        alpha=float(alpha),
    )


def conditional_interval(
    numerator: Any,
    denominator: Any,
    *,
    alpha: float,
) -> BernoulliInterval:
    numerator_values = _binary(numerator, allow_empty=True)
    denominator_values = _binary(denominator, allow_empty=True)
    if numerator_values.shape != denominator_values.shape:
        raise ShapeMismatch("conditional numerator/denominator shape mismatch")
    eligible = denominator_values
    if not np.any(eligible):
        return BernoulliInterval(0, 0, None, 0.0, 1.0, float(alpha))
    return bernoulli_interval(numerator_values[eligible], alpha=float(alpha))


def _complete_grid(membership: Mapping[Stratum, Any]) -> tuple[
    dict[Stratum, np.ndarray], list[str], list[str]
]:
    if not membership:
        raise ShapeMismatch("no source-position membership")
    normalized = {
        (str(source), str(position)): _binary(values)
        for (source, position), values in membership.items()
    }
    sources = sorted({source for source, _ in normalized})
    positions = sorted({position for _, position in normalized})
    if positions != ["prefix", "suffix"]:
        raise ShapeMismatch(f"V7 coverage requires prefix/suffix, observed {positions}")
    missing = [
        (source, position)
        for source in sources
        for position in positions
        if (source, position) not in normalized
    ]
    if missing:
        raise ShapeMismatch(f"incomplete source-position grid: {missing}")
    return normalized, sources, positions


def source_position_coverage(
    membership: Mapping[Stratum, Any],
    *,
    familywise_alpha: float = 0.05,
) -> dict[str, Any]:
    """Return separate prefix/suffix LCBs; their average is never a gate."""

    normalized, sources, positions = _complete_grid(membership)
    per_alpha = float(familywise_alpha) / len(normalized)
    bounds = {
        key: bernoulli_interval(values, alpha=per_alpha)
        for key, values in normalized.items()
    }
    position: dict[str, Any] = {}
    for name in positions:
        selected = [bounds[(source, name)] for source in sources]
        position[name] = {
            "point": float(np.mean([bound.estimate for bound in selected])),
            "lcb": float(np.mean([bound.lower for bound in selected])),
            "ucb": float(np.mean([bound.upper for bound in selected])),
            "by_source": {
                source: bounds[(source, name)].to_dict() for source in sources
            },
        }
    source = {
        name: {
            "point": float(
                np.mean([bounds[(name, position_name)].estimate for position_name in positions])
            ),
            "lcb": float(
                np.mean([bounds[(name, position_name)].lower for position_name in positions])
            ),
        }
        for name in sources
    }
    return {
        "position": position,
        "source": source,
        "strata": {
            f"{source_name}::{position_name}": bounds[(source_name, position_name)].to_dict()
            for source_name, position_name in sorted(bounds)
        },
        "prefix_lcb": position["prefix"]["lcb"],
        "suffix_lcb": position["suffix"]["lcb"],
        "minimum_position_lcb": min(
            position["prefix"]["lcb"], position["suffix"]["lcb"]
        ),
        "balanced_point_diagnostic": float(
            np.mean([position[name]["point"] for name in positions])
        ),
        "familywise_alpha": float(familywise_alpha),
        "correction": "bonferroni_over_source_position",
    }


def source_balanced_occupancy(
    membership: Mapping[str, Any],
    *,
    familywise_alpha: float = 0.05,
) -> dict[str, Any]:
    if not membership:
        raise ShapeMismatch("no benign source membership")
    normalized = {str(source): _binary(values) for source, values in membership.items()}
    per_alpha = float(familywise_alpha) / len(normalized)
    bounds = {
        source: bernoulli_interval(values, alpha=per_alpha)
        for source, values in sorted(normalized.items())
    }
    return {
        "point": float(np.mean([bound.estimate for bound in bounds.values()])),
        "ucb": float(np.mean([bound.upper for bound in bounds.values()])),
        "lcb": float(np.mean([bound.lower for bound in bounds.values()])),
        "worst_source_ucb": max(bound.upper for bound in bounds.values()),
        "by_source": {source: bound.to_dict() for source, bound in bounds.items()},
        "familywise_alpha": float(familywise_alpha),
        "correction": "bonferroni_over_sources",
    }


def group_membership(
    rows: Sequence[Mapping[str, Any]],
    inside: Any,
    *,
    require_unique_texts: bool,
) -> dict[Stratum, np.ndarray]:
    values = _binary(inside)
    if len(rows) != len(values):
        raise ShapeMismatch("row/membership alignment mismatch")
    text_ids = [str(row["text_id"]) for row in rows]
    if require_unique_texts and len(text_ids) != len(set(text_ids)):
        raise ShapeMismatch("independent confirmation repeats a text ID")
    grouped: dict[Stratum, list[bool]] = {}
    for row, member in zip(rows, values):
        position = str(row["position"])
        if position not in {"prefix", "suffix"}:
            raise ShapeMismatch(f"V7 encountered forbidden position {position}")
        grouped.setdefault((str(row["source_id"]), position), []).append(bool(member))
    return {key: np.asarray(sample, dtype=bool) for key, sample in grouped.items()}


def group_benign_by_source(
    rows: Sequence[Mapping[str, Any]], inside: Any
) -> dict[str, np.ndarray]:
    values = _binary(inside)
    if len(rows) != len(values):
        raise ShapeMismatch("benign row/membership alignment mismatch")
    grouped: dict[str, list[bool]] = {}
    for row, member in zip(rows, values):
        grouped.setdefault(str(row["source_id"]), []).append(bool(member))
    return {source: np.asarray(sample, dtype=bool) for source, sample in grouped.items()}
