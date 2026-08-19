from __future__ import annotations

from pathlib import Path

import pytest

from sticky_lab.mode3_v6_3.budget import registered_budget
from sticky_lab.mode3_v6_3.config import assert_physical_device, load_config
from sticky_lab.mode3_v6_3.errors import ProtocolViolation
from sticky_lab.mode3_v6_3.funnel import assigned_positions
from sticky_lab.mode3_v6_3.ranking import select_rapid_s0


def _metric(token_id: int) -> dict[str, float | int | str]:
    scale = token_id / 10_000
    return {
        "token_id": token_id,
        "token_text": f"T{token_id}",
        "balanced_coverage": 0.90 + scale,
        "worst_position_coverage": 0.85 + scale,
        "worst_source_coverage": 0.80 + scale,
        "outside_to_inside": 0.85 + scale,
        "conditional_origin_outside": 0.95 + scale,
        "radius_degrees": 20.0 - scale,
        "benign_occupancy_core": 0.01 - scale / 10,
        "benign_occupancy_1_1": 0.02 - scale / 10,
        "benign_occupancy_auc_1_1_5": 0.02 - scale / 10,
        "center_drift_from_previous": scale,
        "center_restart_spread": scale,
    }


def test_r6_config_is_a_frozen_positive_only_route():
    config = load_config(Path("configs/v6_3_mode3_rapid_r6.yaml"))
    assert config["protocol_revision"] == 6
    assert config["rapid_track"]["enabled"] is True
    assert config["rapid_track"]["negative_claim_supported"] is False
    assert config["funnel"]["s0_keep"] == 200
    assert config["funnel"]["full_top"] == 20
    assert config["positions"]["full_design"] == "all_three"
    assert config["resources"]["priority_peer_first"] is True
    assert config["resources"]["allowed_physical_gpus"] == [4, 5, 6, 7]


def test_r7_config_is_high_priority_and_authorizes_all_eight_gpus():
    config = load_config(Path("configs/v6_3_mode3_rapid_r7.yaml"))
    assert config["protocol_revision"] == 7
    assert config["rapid_track"]["enabled"] is True
    assert config["rapid_track"]["negative_claim_supported"] is False
    assert config["rapid_track"]["amendment_id"] == (
        "V6_3_RAPID_POSITIVE_TRACK_A2_8GPU_HIGH_PRIORITY"
    )
    assert config["funnel"]["s0_keep"] == 200
    assert config["funnel"]["full_top"] == 20
    assert config["resources"]["scheduling_priority"] == "high"
    assert config["resources"]["priority_peer_first"] is False
    assert config["resources"]["signal_lower_priority_peer"] is False
    assert config["resources"]["allowed_physical_gpus"] == list(range(8))
    assert config["resources"]["forbidden_physical_gpus"] == []
    for gpu in range(8):
        assert assert_physical_device(f"cuda:{gpu}", config) == gpu


def test_r6_still_rejects_physical_gpu_zero():
    config = load_config(Path("configs/v6_3_mode3_rapid_r6.yaml"))
    with pytest.raises(ProtocolViolation):
        assert_physical_device("cuda:0", config)


def test_rapid_s0_selection_is_exact_and_deterministic():
    rows = [_metric(token_id) for token_id in range(240)]
    source_audit = {
        "selected": [
            {"token_id": token_id, "reason": "pareto_composite"}
            for token_id in reversed(range(240))
        ]
    }
    first, audit = select_rapid_s0(rows, source_audit, seed=7)
    second, _ = select_rapid_s0(rows, source_audit, seed=7)
    assert first == second
    assert len(first) == len(set(first)) == 200
    assert audit["quotas"] == {
        "pareto": 120,
        "worst_position": 20,
        "lowest_occupancy": 20,
        "migration": 15,
        "compact_radius": 10,
        "bootstrap_stability": 10,
        "deterministic_audit": 5,
    }
    assert audit["negative_claim_supported"] is False


def test_rapid_full_uses_all_registered_positions():
    config = load_config(Path("configs/v6_3_mode3_rapid_r6.yaml"))
    row = {"text_id": "t", "source_id": "s", "source_position_rank": "0"}
    positions = assigned_positions(
        row,
        "full",
        seed=int(config["positions"]["seed"]),
        designs={"full": config["positions"]["full_design"]},
    )
    assert set(positions) == {"prefix", "suffix", "random"}


def test_rapid_budget_reuses_s0_and_stays_below_hard_stop():
    config = load_config(Path("configs/v6_3_mode3_rapid_r6.yaml"))
    plan = registered_budget(config, 21_984)
    assert plan["breakdown"]["s0_reused_no_new_model_calls"] == 0
    assert plan["breakdown"]["full_200_all_three_positions"] == 10_800_000
    assert plan["core_search_total"] == 11_124_000
    assert plan["core_search_total"] < config["budget"]["hard_limit"]
    assert plan["negative_claim_supported"] is False
