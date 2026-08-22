from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import sys

import numpy as np
import pytest

from sticky_lab.mode3_v6_3.errors import ShapeMismatch
from sticky_lab.mode3_v7.candidate_ranking import select_s0_union
from sticky_lab.mode3_v7.config import OCCUPANCY_GRID, load_config
from sticky_lab.mode3_v7.geometry import fit_robust_shared_center
from sticky_lab.mode3_v7.migration import migration_diagnostics
from sticky_lab.mode3_v7.operating_point import build_candidate_frontier
from sticky_lab.mode3_v7.radius_policy import (
    occupancy_at_radius,
    occupancy_constrained_frontier,
)
from sticky_lab.mode3_v7.statistics import source_position_coverage
from scripts.run_v7_orchestrator import Orchestrator


def _vectors(angles: np.ndarray | list[float] | float, count: int | None = None) -> np.ndarray:
    values = np.asarray(angles if np.ndim(angles) else [angles] * int(count or 1), dtype=float)
    return np.stack([np.cos(values), np.sin(values), np.zeros(len(values))], axis=1)


def _config() -> dict:
    config = load_config(Path("configs/v7_mode3_occupancy_frontier.yaml"))
    config = copy.deepcopy(config)
    config["diagnostics"]["center_bootstrap_samples"] = 0
    return config


def _synthetic_frontier(token_id: int = 7) -> dict:
    config = _config()
    fit_rows = []
    fit_vectors = []
    for source in ("a", "b"):
        for position, sign in (("prefix", 1), ("suffix", -1)):
            for index in range(80):
                fit_rows.append(
                    {
                        "text_id": f"fit-{source}-{position}-{index}",
                        "source_id": source,
                        "position": position,
                    }
                )
                fit_vectors.append(_vectors(sign * 0.01, 1)[0])
    calibration_rows = []
    calibration_vectors = []
    for source, angle in (("a", 0.50), ("b", 0.55)):
        for index in range(5000):
            calibration_rows.append(
                {"text_id": f"cal-{source}-{index}", "source_id": source}
            )
            calibration_vectors.append(_vectors(angle + index * 1e-7, 1)[0])
    select_rows = []
    triggered = []
    clean = []
    for source in ("a", "b"):
        for position, sign in (("prefix", 1), ("suffix", -1)):
            for index in range(500):
                select_rows.append(
                    {
                        "text_id": f"sel-{source}-{position}-{index}",
                        "source_id": source,
                        "position": position,
                    }
                )
                triggered.append(_vectors(sign * 0.03, 1)[0])
                clean.append(_vectors(1.0, 1)[0])
    return build_candidate_frontier(
        token_id=token_id,
        token_text=f"T{token_id}",
        fit_rows=fit_rows,
        fit_vectors=np.asarray(fit_vectors),
        calibration_rows=calibration_rows,
        calibration_vectors=np.asarray(calibration_vectors),
        select_rows=select_rows,
        triggered_select_vectors=np.asarray(triggered),
        paired_clean_vectors=np.asarray(clean),
        e_star=_vectors(0.20, 1)[0],
        role_hashes={"fit": "fit", "calibration": "cal", "select": "sel"},
        config=config,
        stage="full",
    )


def test_v7_config_freezes_the_19_point_prefix_suffix_protocol():
    config = load_config(Path("configs/v7_mode3_occupancy_frontier.yaml"))
    assert tuple(config["radius"]["occupancy_grid"]) == OCCUPANCY_GRID
    assert config["positions"]["names"] == ["prefix", "suffix"]
    assert config["positions"]["random_position_enabled"] is False
    assert config["certification"]["prefix_coverage_lcb"] == 0.80
    assert config["certification"]["suffix_coverage_lcb"] == 0.80
    assert config["resources"]["allowed_physical_gpus"] == [4, 5, 6, 7]
    assert config["resources"]["forbidden_physical_gpus"] == [0, 1, 2, 3]
    assert config["resources"]["registration_minimum_free_bytes"] == 10_000_000_000
    assert config["resources"]["model_work_minimum_free_bytes"] == 10_000_000_000
    assert config["resources"]["model_work_disk_gate_policy"] == (
        "explicit_operator_override_10gb"
    )


def test_r2_orchestrator_uses_explicit_10gb_gate(tmp_path: Path):
    orchestrator = Orchestrator(
        argparse.Namespace(
            config=str(Path("configs/v7_mode3_occupancy_frontier.yaml").resolve()),
            output=str(tmp_path / "mode3_v7_occupancy_frontier_r2_10g"),
            profile="formal",
            python=sys.executable,
            gpus="4,5,6,7",
        )
    )
    assert orchestrator.storage_required == 10_000_000_000
    assert orchestrator.storage_peak_reference == 67_500_000_000
    assert orchestrator.storage_gate_policy == "explicit_operator_override_10gb"


def test_center_requires_complete_prefix_suffix_grid_and_equal_mass():
    grid = {
        ("a", "prefix"): _vectors(0.1, 20),
        ("a", "suffix"): _vectors(-0.1, 20),
        ("b", "prefix"): _vectors(0.1, 200),
        ("b", "suffix"): _vectors(-0.1, 200),
    }
    fitted = fit_robust_shared_center(grid, restarts=4, seed=3)
    assert abs(math.atan2(fitted.center[1], fitted.center[0])) < 1e-8
    with pytest.raises(ShapeMismatch):
        fit_robust_shared_center({**grid, ("a", "random"): _vectors(0, 10)})


def test_largest_feasible_radius_is_monotone_and_respects_every_beta():
    distances = {
        "a": np.linspace(0.50, 1.0, 5000),
        "b": np.linspace(0.55, 1.05, 5000),
    }
    points = occupancy_constrained_frontier(
        distances, [0.001, 0.003, 0.01, 0.05], maximum_radius_degrees=35
    )
    assert all(point.feasible for point in points)
    assert all(point.benign_occupancy_ucb <= point.beta for point in points)
    assert all(
        right.radius >= left.radius for left, right in zip(points, points[1:])
    )
    candidates = np.unique(np.concatenate(list(distances.values())))
    for point in points:
        next_values = candidates[candidates > point.radius]
        if len(next_values) and next_values[0] <= math.radians(35):
            next_occupancy = occupancy_at_radius(
                distances, float(next_values[0]), familywise_alpha=0.05
            )
            assert next_occupancy["ucb"] > point.beta


def test_prefix_and_suffix_lcbs_are_separate_not_averaged():
    membership = {
        (source, "prefix"): np.ones(1000, dtype=bool)
        for source in ("a", "b")
    }
    membership.update(
        {
            (source, "suffix"): np.arange(1000) < 600
            for source in ("a", "b")
        }
    )
    result = source_position_coverage(membership)
    assert result["prefix_lcb"] > 0.80
    assert result["suffix_lcb"] < 0.80
    assert result["balanced_point_diagnostic"] == pytest.approx(0.80)


def test_migration_reports_independent_capture_and_redundant_metrics():
    clean = np.asarray([True] * 100 + [False] * 900)
    triggered = np.asarray([True] * 90 + [False] * 10 + [True] * 810 + [False] * 90)
    clean_grid = {
        (source, position): clean
        for source in ("a", "b")
        for position in ("prefix", "suffix")
    }
    trigger_grid = {
        (source, position): triggered
        for source in ("a", "b")
        for position in ("prefix", "suffix")
    }
    result = migration_diagnostics(clean_grid, trigger_grid)
    prefix = result["position"]["prefix"]
    assert prefix["capture_outside_point"] == pytest.approx(0.9)
    assert prefix["outside_to_inside"] == pytest.approx(0.81)
    assert prefix["conditional_origin_outside"] == pytest.approx(0.9)
    assert prefix["inside_retention"] == pytest.approx(0.9)
    assert prefix["net_gain"] == pytest.approx(0.8)


def test_candidate_frontier_finds_strong_beta80_without_refitting_center():
    result = _synthetic_frontier()
    assert result["beta80_ps"] == 0.001
    assert result["evidence_grade"] == "STRONG_LOW_OCCUPANCY_FROZEN_CAP"
    assert result["center_refit_per_beta"] is False
    assert len(result["frontier"]) == 19
    assert len(result["long_rows"]) == 38
    assert all(point["prefix_coverage_lcb"] > 0.8 for point in result["frontier"])
    assert all(point["suffix_coverage_lcb"] > 0.8 for point in result["frontier"])


def test_s0_union_is_deterministic_and_keeps_random_audit_lane():
    base = _synthetic_frontier()
    frontiers = []
    for token_id in range(90):
        row = copy.deepcopy(base)
        row["token_id"] = token_id
        row["token_text"] = f"T{token_id}"
        row["coverage_auc_log_beta"] -= token_id / 10000
        frontiers.append(row)
    first, audit = select_s0_union(
        frontiers,
        occupancy_grid=OCCUPANCY_GRID,
        per_beta=4,
        auc_count=8,
        deterministic_audit_count=10,
        maximum=30,
        seed=9,
    )
    second, _ = select_s0_union(
        frontiers,
        occupancy_grid=OCCUPANCY_GRID,
        per_beta=4,
        auc_count=8,
        deterministic_audit_count=10,
        maximum=30,
        seed=9,
    )
    assert first == second
    assert len(first) == len(set(first)) <= 30
    assert audit["deterministic_audit_count"] == 10
