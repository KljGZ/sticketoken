"""Independent and algebraically redundant migration diagnostics for V7."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

import numpy as np

from sticky_lab.mode3_v6_3.errors import ShapeMismatch

from .statistics import bernoulli_interval, conditional_interval


Stratum = Tuple[str, str]


def migration_diagnostics(
    clean_inside: Mapping[Stratum, Any],
    triggered_inside: Mapping[Stratum, Any],
    *,
    familywise_alpha: float = 0.05,
) -> dict[str, Any]:
    if set(clean_inside) != set(triggered_inside) or not clean_inside:
        raise ShapeMismatch("clean and triggered migration strata differ")
    normalized_clean: dict[Stratum, np.ndarray] = {}
    normalized_triggered: dict[Stratum, np.ndarray] = {}
    for key in sorted(clean_inside):
        clean = np.asarray(clean_inside[key], dtype=bool).reshape(-1)
        triggered = np.asarray(triggered_inside[key], dtype=bool).reshape(-1)
        if len(clean) == 0 or clean.shape != triggered.shape:
            raise ShapeMismatch(f"paired migration mismatch for {key}")
        normalized_clean[(str(key[0]), str(key[1]))] = clean
        normalized_triggered[(str(key[0]), str(key[1]))] = triggered
    positions = sorted({position for _, position in normalized_clean})
    sources = sorted({source for source, _ in normalized_clean})
    if positions != ["prefix", "suffix"]:
        raise ShapeMismatch("migration diagnostics require prefix/suffix only")
    missing = [
        (source, position)
        for source in sources
        for position in positions
        if (source, position) not in normalized_clean
    ]
    if missing:
        raise ShapeMismatch(f"incomplete migration grid: {missing}")

    # Four diagnostic interval families across every source-position stratum.
    per_alpha = float(familywise_alpha) / (4 * len(normalized_clean))
    strata: dict[str, Any] = {}
    for key in sorted(normalized_clean):
        clean = normalized_clean[key]
        triggered = normalized_triggered[key]
        capture = conditional_interval(triggered, ~clean, alpha=per_alpha)
        moved = bernoulli_interval((~clean) & triggered, alpha=per_alpha)
        origin = conditional_interval(~clean, triggered, alpha=per_alpha)
        retention = conditional_interval(triggered, clean, alpha=per_alpha)
        table = {
            "clean_outside_triggered_inside": int(np.sum((~clean) & triggered)),
            "clean_inside_triggered_inside": int(np.sum(clean & triggered)),
            "clean_outside_triggered_outside": int(np.sum((~clean) & (~triggered))),
            "clean_inside_triggered_outside": int(np.sum(clean & (~triggered))),
            "total": len(clean),
        }
        strata[f"{key[0]}::{key[1]}"] = {
            "capture_given_clean_outside": capture.to_dict(),
            "outside_to_inside": moved.to_dict(),
            "conditional_origin_outside": origin.to_dict(),
            "inside_retention": retention.to_dict(),
            "clean_occupancy_point": float(np.mean(clean)),
            "triggered_coverage_point": float(np.mean(triggered)),
            "net_gain": float(np.mean(triggered) - np.mean(clean)),
            "four_cell_table": table,
        }

    position_output: dict[str, Any] = {}
    for position in positions:
        rows = [strata[f"{source}::{position}"] for source in sources]

        def mean_metric(name: str, field: str) -> float | None:
            values = [row[name][field] for row in rows if row[name][field] is not None]
            return float(np.mean(values)) if values else None

        position_output[position] = {
            "capture_outside_point": mean_metric("capture_given_clean_outside", "estimate"),
            "capture_outside_lcb": mean_metric("capture_given_clean_outside", "lower"),
            "outside_to_inside": mean_metric("outside_to_inside", "estimate"),
            "conditional_origin_outside": mean_metric(
                "conditional_origin_outside", "estimate"
            ),
            "inside_retention": mean_metric("inside_retention", "estimate"),
            "clean_occupancy_point": float(
                np.mean([row["clean_occupancy_point"] for row in rows])
            ),
            "triggered_coverage_point": float(
                np.mean([row["triggered_coverage_point"] for row in rows])
            ),
            "net_gain": float(np.mean([row["net_gain"] for row in rows])),
            "by_source": {source: strata[f"{source}::{position}"] for source in sources},
        }
    return {
        "position": position_output,
        "strata": strata,
        "familywise_alpha": float(familywise_alpha),
        "per_interval_alpha": per_alpha,
        "hard_gate": False,
        "identities": {
            "outside_to_inside": "P(not C and T) = (1-q) R",
            "conditional_origin_outside": "P(not C | T) = M / t",
            "net_gain": "t - q",
        },
    }
