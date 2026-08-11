"""Exact multi-ball occupancy and one-sided binomial confidence bounds."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import beta

def clopper_pearson_upper(successes: int, trials: int, confidence: float) -> float:
    if trials <= 0 or not 0 < confidence < 1 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial bound arguments")
    if successes == trials:
        return 1.0
    # The one-sided exact Clopper--Pearson limit is a beta quantile.  Using
    # scipy's special-function implementation is both exact and several
    # orders of magnitude faster than repeatedly summing binomial tails.
    return float(beta.ppf(confidence, successes + 1, trials - successes))


def clopper_pearson_lower(successes: int, trials: int, confidence: float) -> float:
    if trials <= 0 or not 0 < confidence < 1 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial bound arguments")
    if successes == 0:
        return 0.0
    return float(beta.ppf(1.0 - confidence, successes, trials - successes + 1))


def cosine_distance_to_centers(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, 1.0 - np.asarray(values, dtype=np.float64) @ np.asarray(centers, dtype=np.float64).T)


def evaluate_multiscale_occupancy(
    benign: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    lambdas: Sequence[float],
    *,
    confidence: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    distances = cosine_distance_to_centers(benign, centers)
    scales = np.asarray(lambdas, dtype=np.float64)
    observed = []
    upper = []
    for scale in scales:
        hits = int(np.count_nonzero(np.any(distances <= scale * radii[None, :], axis=1)))
        observed.append(hits / len(benign))
        upper.append(clopper_pearson_upper(hits, len(benign), confidence))
    observed_array = np.asarray(observed, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    if len(scales) == 1:
        auc = float(upper_array[0])
    else:
        auc = float(np.trapezoid(upper_array, scales) / (scales[-1] - scales[0]))
    valid = scales[upper_array <= float(epsilon)]
    lambda_star = float(np.max(valid)) if len(valid) else 0.0
    return observed_array, upper_array, auc, lambda_star


def fixed_structure_coverage(
    values: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    *,
    confidence: float,
) -> dict[str, float | int]:
    distances = cosine_distance_to_centers(values, centers)
    hits = int(np.count_nonzero(np.any(distances <= radii[None, :], axis=1)))
    return {
        "hits": hits,
        "trials": int(len(values)),
        "coverage": hits / len(values),
        "coverage_lcb": clopper_pearson_lower(hits, len(values), confidence),
        "outlier_rate": 1.0 - hits / len(values),
        "outlier_rate_ucb": clopper_pearson_upper(len(values) - hits, len(values), confidence),
    }
