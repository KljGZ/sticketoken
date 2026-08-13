"""Certification, migration, and radial statistics for V6."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


def _beta_ppf(probability: float, a: float, b: float) -> float:
    try:
        from scipy.stats import beta
    except ImportError:
        # Deterministic regularized-incomplete-beta inversion.  This fallback
        # keeps certification exact enough for environments without scipy.
        def fraction(x: float) -> float:
            qab, qap, qam = a + b, a + 1.0, a - 1.0
            c, d, h = 1.0, 1.0 - qab * x / qap, 1.0
            d = 1.0 / max(abs(d), 1e-300) * (1 if d >= 0 else -1)
            h = d
            for m in range(1, 401):
                m2 = 2 * m
                aa = m * (b - m) * x / ((qam + m2) * (a + m2))
                d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
                c = 1.0 + aa / c; c = c if abs(c) > 1e-300 else 1e-300
                h *= d * c
                aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
                d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
                c = 1.0 + aa / c; c = c if abs(c) > 1e-300 else 1e-300
                delta = d * c; h *= delta
                if abs(delta - 1.0) < 3e-14:
                    break
            return h

        def cdf(x: float) -> float:
            if x <= 0: return 0.0
            if x >= 1: return 1.0
            front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
            if x < (a + 1.0) / (a + b + 2.0):
                return front * fraction(x) / a
            # Symmetry requires the continued fraction with a/b exchanged.
            original_a, original_b = a, b
            def swapped_fraction(y: float) -> float:
                aa0, bb0 = original_b, original_a
                qab, qap, qam = aa0 + bb0, aa0 + 1.0, aa0 - 1.0
                c, d = 1.0, 1.0 - qab * y / qap
                d = 1.0 / (d if abs(d) > 1e-300 else 1e-300); h = d
                for m in range(1, 401):
                    m2 = 2 * m
                    aa = m * (bb0 - m) * y / ((qam + m2) * (aa0 + m2))
                    d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
                    c = 1.0 + aa / c; c = c if abs(c) > 1e-300 else 1e-300; h *= d * c
                    aa = -(aa0 + m) * (qab + m) * y / ((aa0 + m2) * (qap + m2))
                    d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
                    c = 1.0 + aa / c; c = c if abs(c) > 1e-300 else 1e-300
                    delta = d * c; h *= delta
                    if abs(delta - 1.0) < 3e-14: break
                return h
            return 1.0 - front * swapped_fraction(1.0 - x) / b

        low, high = 0.0, 1.0
        for _ in range(100):
            middle = (low + high) / 2.0
            if cdf(middle) < probability:
                low = middle
            else:
                high = middle
        return (low + high) / 2.0
    return float(beta.ppf(probability, a, b))


def clopper_pearson_lower(successes: int, total: int, confidence: float = 0.95) -> float:
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("invalid binomial counts")
    return 0.0 if successes == 0 else _beta_ppf(1.0 - confidence, successes, total - successes + 1)


def clopper_pearson_upper(successes: int, total: int, confidence: float = 0.95) -> float:
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("invalid binomial counts")
    return 1.0 if successes == total else _beta_ppf(confidence, successes + 1, total - successes)


@dataclass(frozen=True)
class MigrationTable:
    outside_to_inside: int
    inside_to_inside: int
    outside_to_outside: int
    inside_to_outside: int
    total: int

    def proportions(self) -> dict[str, float]:
        return {
            "outside_to_inside": self.outside_to_inside / self.total,
            "inside_to_inside": self.inside_to_inside / self.total,
            "outside_to_outside": self.outside_to_outside / self.total,
            "inside_to_outside": self.inside_to_outside / self.total,
        }


def migration_table(clean_inside: Sequence[bool], triggered_inside: Sequence[bool]) -> MigrationTable:
    clean = np.asarray(clean_inside, dtype=bool)
    triggered = np.asarray(triggered_inside, dtype=bool)
    if clean.shape != triggered.shape or clean.ndim != 1 or len(clean) == 0:
        raise ValueError("migration arrays must be paired, non-empty vectors")
    return MigrationTable(
        outside_to_inside=int(np.sum(~clean & triggered)),
        inside_to_inside=int(np.sum(clean & triggered)),
        outside_to_outside=int(np.sum(~clean & ~triggered)),
        inside_to_outside=int(np.sum(clean & ~triggered)),
        total=len(clean),
    )


def radial_profile(normalized_radius: Sequence[float], multipliers: Iterable[float]) -> list[dict[str, float | int]]:
    values = np.asarray(normalized_radius, dtype=np.float64).reshape(-1)
    if len(values) == 0 or np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("invalid normalized radii")
    output: list[dict[str, float | int]] = []
    lower = 0.0
    for upper in sorted(map(float, multipliers)):
        cumulative = int(np.sum(values <= upper + 1e-12))
        shell = int(np.sum((values > lower + 1e-12) & (values <= upper + 1e-12)))
        output.append(
            {
                "lower_multiplier": lower,
                "upper_multiplier": upper,
                "cumulative_count": cumulative,
                "cumulative_fraction": cumulative / len(values),
                "shell_count": shell,
                "shell_fraction": shell / len(values),
            }
        )
        lower = upper
    return output


def cliffs_delta(first: Sequence[float], second: Sequence[float]) -> float:
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("empty sample")
    # Memory-bounded exact comparison.
    score = 0
    for chunk in np.array_split(a, max(1, math.ceil(len(a) / 2048))):
        score += int(np.sum(chunk[:, None] > b[None, :]))
        score -= int(np.sum(chunk[:, None] < b[None, :]))
    return score / (len(a) * len(b))


def radial_depth_summary(triggered: Sequence[float], benign: Sequence[float]) -> dict[str, object]:
    a = np.asarray(triggered, dtype=np.float64).reshape(-1)
    b = np.asarray(benign, dtype=np.float64).reshape(-1)
    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    result: dict[str, object] = {
        "triggered_quantiles": dict(zip(map(str, quantiles), np.quantile(a, quantiles).tolist())),
        "benign_quantiles": dict(zip(map(str, quantiles), np.quantile(b, quantiles).tolist())),
        "median_depth_gap": float(np.median(b) - np.median(a)),
        "cliffs_delta_triggered_vs_benign": cliffs_delta(a, b),
    }
    try:
        from scipy.stats import ks_2samp, wasserstein_distance

        result["wasserstein"] = float(wasserstein_distance(a, b))
        ks = ks_2samp(a, b, alternative="two-sided", method="auto")
        result["ks_statistic"] = float(ks.statistic)
        result["ks_pvalue"] = float(ks.pvalue)
    except ImportError:  # pragma: no cover
        result["wasserstein"] = None
        result["ks_statistic"] = None
        result["ks_pvalue"] = None
    return result
