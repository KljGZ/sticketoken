"""V7 model-call budget planning on the durable V6.3 ledger."""

from __future__ import annotations

from typing import Any, Mapping

from sticky_lab.mode3_v6_3.budget import (
    BudgetLedger,
    Reservation,
    atomic_json,
    exclusive_lock,
)


def registered_budget(
    config: Mapping[str, Any],
    *,
    full_candidates: int,
    s0_raw_reused: bool,
) -> dict[str, Any]:
    data = config["data"]
    full = data["stage_sizes"]["full"]
    confirm = data["confirm_roles"]
    triggered_full = int(full_candidates) * 2 * (
        int(full["fit"]) + int(full["select"])
    )
    discovery_shared_clean = (
        int(full["calibration"])
        + int(full["select"])
        + int(data["axis_fit_benign"])
    )
    confirm_calls = (
        2 * int(confirm["confirm_prefix"])
        + 2 * int(confirm["confirm_suffix"])
        + int(confirm["confirm_benign"])
        + 3 * int(confirm["confirm_paired"])
    )
    total = triggered_full + discovery_shared_clean + confirm_calls
    baseline = int(config["budget"]["v5_baseline_submitted_texts"])
    embedding_dimension = 768
    float32_bytes = 4
    estimated_embedding_cache_bytes = total * embedding_dimension * float32_bytes
    return {
        "schema_version": "mode3-v7-budget-plan-v1",
        "unit": "submitted_text_equivalent",
        "s0_raw_reused": bool(s0_raw_reused),
        "full_candidates": int(full_candidates),
        "breakdown": {
            "s0_reuse_new_calls": 0,
            "full_prefix_suffix_triggered": triggered_full,
            "discovery_shared_clean": discovery_shared_clean,
            "primary_confirm": confirm_calls,
        },
        "planned_total": total,
        "estimated_embedding_cache_bytes": estimated_embedding_cache_bytes,
        "registered_peak_cache_bytes": int(
            config["resources"]["estimated_peak_cache_bytes"]
        ),
        "ratio_to_v5": total / baseline,
        "warning_limit": int(config["budget"]["warning_limit"]),
        "hard_limit": int(config["budget"]["hard_limit"]),
        "forbidden_limit": int(config["budget"]["forbidden_limit"]),
        "within_hard_limit": total < int(config["budget"]["hard_limit"]),
    }


__all__ = [
    "BudgetLedger",
    "Reservation",
    "atomic_json",
    "exclusive_lock",
    "registered_budget",
]
