from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from sticky_lab.mode3_v6_compact.budget import BudgetExhausted, BudgetLedger, estimate_budget
from sticky_lab.mode3_v6_compact.common import atomic_savez, cap_from_arrays
from sticky_lab.mode3_v6_compact.funnel import select_candidates


ROOT = Path(__file__).resolve().parents[1]


def config() -> dict:
    return yaml.safe_load((ROOT / "configs/v6_mode3_compact.yaml").read_text(encoding="utf-8"))


def test_preregistered_budget_is_below_3_6_v5() -> None:
    value = estimate_budget(config())
    assert value["within_planned_limit"]
    assert value["planned_v5_ratio"] <= 3.6
    assert value["planned_submitted_text_equivalent"] < value["limits"]["warning_limit"]


def test_budget_reserves_before_call_and_hard_stops(tmp_path: Path) -> None:
    settings = {"warning_limit": 8, "hard_limit": 10, "forbidden_limit": 12}
    ledger = BudgetLedger(tmp_path, settings)
    first = ledger.reserve(phase="s0", track="blackbox", raw_items=6)
    assert first.total_after == 6
    with pytest.raises(BudgetExhausted):
        ledger.reserve(phase="s0", track="blackbox", raw_items=5)
    state = json.loads((tmp_path / "budget/observed.json").read_text())
    assert state["submitted_text_equivalent"] == 6
    assert state["new_model_calls_allowed"] is False
    assert state["hard_limit_reached"] is True


def test_cap_uses_angular_geometry() -> None:
    cap = cap_from_arrays(token_id=1, token_text="x", center=np.array([1.0, 0.0]), radius=np.pi / 4)
    points = np.array([[1.0, 0.0], [np.sqrt(0.5), np.sqrt(0.5)], [0.0, 1.0]])
    assert cap.contains(points).tolist() == [True, True, False]
    assert cap.protocol == "P3_shared"


def test_atomic_npz_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "x" / "values.npz"
    atomic_savez(target, values=np.arange(8), matrix=np.eye(3))
    value = np.load(target, allow_pickle=False)
    assert value["values"].tolist() == list(range(8))
    assert np.allclose(value["matrix"], np.eye(3))


def test_funnel_is_deterministic_and_bounded() -> None:
    rows = [
        {
            "token_id": token_id,
            "status": "valid",
            "triggered_coverage": 0.90 + token_id / 1000,
            "worst_position_coverage": 0.89 + token_id / 1000,
            "radius_degrees": 20 - token_id / 10,
            "benign_occupancy": token_id / 1000,
            "outside_to_inside": 0.80 + token_id / 1000,
            "search_margin_m90_1": token_id / 100,
        }
        for token_id in range(20)
    ]
    first, provenance = select_candidates(rows, 8, additional={"whitebox": [1, 2, 3]}, additional_quota=2)
    second, _ = select_candidates(rows, 8, additional={"whitebox": [1, 2, 3]}, additional_quota=2)
    assert first == second
    assert len(first) == len(set(first)) == 8
    assert 1 in first and 2 in first
    assert "whitebox" in provenance[1]
