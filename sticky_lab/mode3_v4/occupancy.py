"""Multi-scale normal occupancy estimates and exact binomial bounds."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


def _binomial_tail(successes: int, trials: int, probability: float, *, upper: bool) -> float:
    """Stable exact binomial tail without a mandatory SciPy dependency."""

    if probability <= 0.0:
        return float((0 >= successes) if upper else (0 <= successes))
    if probability >= 1.0:
        return float((trials >= successes) if upper else (trials <= successes))
    indices = range(successes, trials + 1) if upper else range(0, successes + 1)
    logs = [
        math.lgamma(trials + 1)
        - math.lgamma(index + 1)
        - math.lgamma(trials - index + 1)
        + index * math.log(probability)
        + (trials - index) * math.log1p(-probability)
        for index in indices
    ]
    maximum = max(logs)
    return float(math.exp(maximum) * sum(math.exp(value - maximum) for value in logs))


def _bisect_binomial(successes: int, trials: int, target: float, *, upper_tail: bool) -> float:
    low, high = 0.0, 1.0
    for _ in range(72):
        middle = (low + high) / 2.0
        value = _binomial_tail(successes, trials, middle, upper=upper_tail)
        if upper_tail:
            # P[X >= successes] increases with p.
            if value < target:
                low = middle
            else:
                high = middle
        else:
            # P[X <= successes] decreases with p.
            if value > target:
                low = middle
            else:
                high = middle
    return float((low + high) / 2.0)

from .support import SupportModel


def clopper_pearson_upper(successes: int, trials: int, confidence: float) -> float:
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("Invalid binomial count")
    if successes == trials:
        return 1.0
    return _bisect_binomial(successes, trials, 1.0 - confidence, upper_tail=False)


def clopper_pearson_lower(successes: int, trials: int, confidence: float) -> float:
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("Invalid binomial count")
    if successes == 0:
        return 0.0
    return _bisect_binomial(successes, trials, 1.0 - confidence, upper_tail=True)


@dataclass(frozen=True)
class OccupancyRecord:
    lambdas: tuple[float, ...]
    counts: tuple[int, ...]
    rates: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    relative_quantiles: tuple[float, ...]

    def to_dict(self) -> dict[str, float | int]:
        output: dict[str, float | int] = {}
        for value, count, rate, upper, relative in zip(
            self.lambdas, self.counts, self.rates, self.upper_bounds, self.relative_quantiles
        ):
            label = str(value).replace(".", "_")
            output[f"occupancy_count_lambda_{label}"] = count
            output[f"occupancy_rate_lambda_{label}"] = rate
            output[f"occupancy_upper_lambda_{label}"] = upper
            output[f"relative_occupancy_quantile_lambda_{label}"] = relative
        return output


def evaluate_occupancy(
    center: np.ndarray,
    radius: float,
    normal_probe: np.ndarray,
    support: SupportModel,
    lambdas: Sequence[float],
    *,
    confidence: float,
) -> OccupancyRecord:
    probe = np.asarray(normal_probe, dtype=np.float32)
    distances = np.linalg.norm(probe - np.asarray(center, dtype=np.float32)[None, :], axis=1)
    counts: list[int] = []
    rates: list[float] = []
    uppers: list[float] = []
    relatives: list[float] = []
    for multiplier in map(float, lambdas):
        threshold = multiplier * float(radius)
        count = int(np.count_nonzero(distances <= threshold))
        reference_counts = np.count_nonzero(support.reference_distances <= threshold, axis=1)
        relative = float((1 + np.count_nonzero(reference_counts <= count)) / (len(reference_counts) + 1))
        counts.append(count)
        rates.append(count / len(probe))
        uppers.append(clopper_pearson_upper(count, len(probe), confidence))
        relatives.append(relative)
    return OccupancyRecord(tuple(map(float, lambdas)), tuple(counts), tuple(rates), tuple(uppers), tuple(relatives))
