from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sticky_lab.mode3_v6_2.budget import BudgetExhausted, BudgetLedger, estimate_budget
from sticky_lab.mode3_v6_2.common import load_config
from sticky_lab.mode3_v6_2.geometry import FrozenCapModel, fit_equal_strata_robust_center, fit_single_cap
from sticky_lab.mode3_v6_2.statistics import (
    gate_reachability_audit,
    simultaneous_balanced_bounds,
    trapezoidal_integral,
)


ROOT = Path(__file__).resolve().parents[1]


def config() -> dict:
    return load_config(ROOT / "configs/v6_2_mode3.yaml")


def _vectors(angle: float, count: int) -> np.ndarray:
    values = np.linspace(-angle, angle, count)
    return np.stack([np.cos(values), np.sin(values)], axis=1)


def test_budget_is_exactly_registered_and_below_15x() -> None:
    value = estimate_budget(config())
    assert value["matches_registered_estimate"]
    assert value["planned_submitted_text_equivalent"] == 1_008_791_696
    assert value["planned_v5_ratio"] == pytest.approx(12.066022984150133)
    assert value["limits"]["hard_limit"] == 1_229_007_847
    assert value["limits"]["forbidden_limit"] == 1_254_089_640


def test_budget_reserves_before_call_and_fails_closed(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path, {"warning_limit": 8, "hard_limit": 10, "forbidden_limit": 12})
    assert ledger.reserve(phase="s0", track="formal", raw_items=6).total_after == 6
    with pytest.raises(BudgetExhausted): ledger.reserve(phase="s0", track="formal", raw_items=5)
    state = json.loads((tmp_path / "budget/observed.json").read_text())
    assert state["submitted_text_equivalent"] == 6
    assert state["new_model_calls_allowed"] is False


def test_robust_center_uses_equal_strata_not_pooled_mass() -> None:
    grid = {
        ("large", "prefix"): _vectors(.04, 200), ("small", "prefix"): _vectors(.04, 10) @ np.array([[0, -1], [1, 0]]),
        ("large", "suffix"): _vectors(.04, 200), ("small", "suffix"): _vectors(.04, 10) @ np.array([[0, -1], [1, 0]]),
        ("large", "random"): _vectors(.04, 200), ("small", "random"): _vectors(.04, 10) @ np.array([[0, -1], [1, 0]]),
    }
    fitted = fit_equal_strata_robust_center(grid, trim_fraction=.1, restarts=20, seed=3)
    assert fitted.center[0] == pytest.approx(np.sqrt(.5), abs=.08)
    assert abs(fitted.center[1]) == pytest.approx(np.sqrt(.5), abs=.08)
    assert len(fitted.restart_summaries) == 20


def test_fit_and_radius_are_independent_roles() -> None:
    fit = {(s, p): _vectors(.03, 20) for s in ("a", "b") for p in ("prefix", "suffix", "random")}
    radius = {(s, p): _vectors(.05, 12) for s in ("a", "b") for p in ("prefix", "suffix", "random")}
    cap, audit = fit_single_cap(7, " x", fit, radius, fit_role="full_fit", radius_role="full_radius", design_coverage=.92, maximum_radius_degrees=35, trim_fraction=.1, restarts=20, maximum_iterations=50, tolerance=1e-7, seed=4)
    assert cap.fit_role == "full_fit" and cap.radius_role == "full_radius"
    assert cap.design_coverage == .92 and cap.cap_count == 1
    assert audit["radius_calibration"]["design_coverage"] == .92


def test_simultaneous_bounds_do_not_pool_positions() -> None:
    membership = {(source, position): np.ones(100, dtype=bool) for source in ("a", "b") for position in ("prefix", "suffix", "random")}
    certificate = simultaneous_balanced_bounds(membership, familywise_alpha=.05)
    assert len(certificate.strata) == 6
    assert certificate.correction == "bonferroni"
    assert certificate.worst_position_lower == pytest.approx(certificate.balanced_lower)


def test_design_coverage_has_positive_confirm_margin() -> None:
    sizes = {(source, position): 12_500 for source in ("a", "b", "c", "d") for position in ("prefix", "suffix", "random")}
    audit = gate_reachability_audit(sizes, design_coverage=.92, target_lcb=.90, familywise_alpha=.05)
    assert audit["status"] == "design_has_positive_margin"
    assert audit["all_strata_reachable_at_design_point"]


def test_frozen_model_uses_original_angular_space() -> None:
    cap = FrozenCapModel(1, "x", "P3", np.array([[1., 0.]]), np.array([np.pi / 4]), .92, "fit", "radius", 1)
    points = np.array([[1., 0.], [np.sqrt(.5), np.sqrt(.5)], [0., 1.]])
    assert cap.contains(points).tolist() == [True, True, False]


def test_trapezoidal_integral_supports_numpy_2_api(monkeypatch: pytest.MonkeyPatch) -> None:
    def numpy2_style(y: object, x: object) -> float:
        values = np.asarray(y, dtype=float)
        grid = np.asarray(x, dtype=float)
        return float(np.sum((values[:-1] + values[1:]) * 0.5 * np.diff(grid)))

    monkeypatch.delattr(np, "trapz", raising=False)
    monkeypatch.setattr(np, "trapezoid", numpy2_style, raising=False)
    assert trapezoidal_integral([0.0, 1.0, 1.0], [0.0, 1.0, 2.0]) == pytest.approx(1.5)
