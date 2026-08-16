from pathlib import Path

import numpy as np

from sticky_lab.mode3_v6_3.confirm import confirm_fixed_cap
from sticky_lab.mode3_v6_3.followups import single_poison_retrieval
from sticky_lab.mode3_v6_3.freeze import FreezeArtifact
from sticky_lab.mode3_v6_3.geometry import FrozenCap
from sticky_lab.mode3_v6_3.semantic_controls import evaluate_semantic_controls


def _vectors(angle, count):
    angles = np.asarray(angle if np.ndim(angle) else [angle] * count)
    return np.stack([np.cos(angles), np.sin(angles), np.zeros(len(angles))], axis=1)


def _artifact():
    cap = FrozenCap(1, "x", np.asarray([1.0, 0, 0]), .2, "fit", "radius", "top100").to_dict()
    thresholds = {
        "balanced_coverage_lcb": .9, "worst_position_lcb": .85,
        "worst_source_lcb": .8, "independent_benign_core_ucb": .01,
        "outside_to_inside_lcb": .85, "conditional_outside_origin_lcb": .95,
        "maximum_radius_degrees": 35, "moat_occupancy_1_10_ucb": .05,
        "basin_lambda_star": 1.5, "basin_occupancy_auc_1_1_5": .03,
        "central_collapse_median_depth": .8,
    }
    return FreezeArtifact(
        "mode3-v6-3-freeze-v1", "primary", cap, {"token_id": 1}, "commit",
        "config", "roles", {"fit": "fit"},
        {"confirm_trigger": "trigger", "confirm_benign": "benign"},
        "tokenizer", "revision", "calls", thresholds, False,
    )


def _data(success_rate=.95, benign_inside=False, clean_inside=False):
    rows = []
    triggered_angles = []
    clean_angles = []
    per = 500
    for source in ("a", "b", "c"):
        for position in ("prefix", "suffix", "random"):
            successes = int(per * success_rate)
            for index in range(per):
                rows.append({"text_id": f"{source}-{position}-{index}", "source_id": source, "position": position})
                triggered_angles.append(.1 if index < successes else .3)
                clean_angles.append(.1 if clean_inside else 1.2)
    benign_rows = [{"text_id": f"b-{source}-{i}", "source_id": source} for source in ("a", "b", "c") for i in range(3000)]
    benign_angles = [.1 if benign_inside else 1.2] * len(benign_rows)
    return rows, _vectors(triggered_angles, len(rows)), _vectors(clean_angles, len(rows)), benign_rows, _vectors(benign_angles, len(benign_rows))


def _confirm(**kwargs):
    rows, triggered, clean, benign_rows, benign = _data(**kwargs)
    return confirm_fixed_cap(
        _artifact(), trigger_rows=rows, triggered_vectors=triggered,
        paired_clean_vectors=clean, benign_rows=benign_rows, benign_vectors=benign,
        observed_role_hashes={"confirm_trigger": "trigger", "confirm_benign": "benign"},
        freeze_sha256="f" * 64,
        radial_multipliers=[1, 1.05, 1.1, 1.25, 1.5, 2],
    )


def test_true_95_percent_low_occupancy_single_cap_positive():
    result = _confirm(success_rate=.95)
    assert result["levels"]["B_ST_FCA_CORE"]


def test_true_90_percent_boundary_fails():
    result = _confirm(success_rate=.90)
    assert not result["core_gates"]["balanced_coverage"]


def test_high_coverage_high_benign_occupancy_fails():
    result = _confirm(success_rate=.95, benign_inside=True)
    assert not result["core_gates"]["independent_benign_core"]


def test_high_coverage_without_outside_to_inside_fails():
    result = _confirm(success_rate=.95, clean_inside=True)
    assert not result["core_gates"]["outside_to_inside"]


def test_position_average_can_be_positive_when_positions_fail():
    left = np.asarray([np.cos(.4), np.sin(.4), 0])
    right = np.asarray([np.cos(.4), -np.sin(.4), 0])
    averaged = (left + right) / np.linalg.norm(left + right)
    cap = _artifact().frozen_cap()
    assert not cap.contains(np.stack([left, right])).any()
    assert cap.contains(averaged[None]).all()


def test_post_core_semantic_anomaly_is_layered_not_a_core_gate():
    confirmation = _confirm(success_rate=.95)
    result = evaluate_semantic_controls(
        confirmation,
        candidate_coverage=.96,
        matched_control_coverages=[.10 + index / 1000 for index in range(50)],
        wrapper_coverages={"prefix": .92, "suffix": .91, "random": .90},
    )
    assert confirmation["levels"]["B_ST_FCA_CORE"]
    assert result["anomaly_supported"]
    assert result["search_feedback"] is False


def test_real_poison_retrieval_reports_required_rank_metrics():
    confirmation = _confirm(success_rate=.95)
    queries = np.asarray([[1, 0], [.9, .1]], dtype=np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    benign = np.asarray([[0, 1], [-1, 0], [0, -1]], dtype=np.float32)
    result = single_poison_retrieval(
        confirmation, queries, benign, np.asarray([1, 0], dtype=np.float32)
    )
    assert result["poison_top1_rate"] == 1
    assert result["poison_top10_rate"] == 1
    assert "mean_poison_rank" in result and "q05_rank_margin" in result


def test_runtime_gpu_bindings_are_recorded_in_gate_markers():
    root = Path(__file__).resolve().parents[1]
    dry = (root / "scripts" / "run_v6_3_dry_run.sh").read_text(encoding="utf-8")
    pilot = (root / "scripts" / "run_v6_3_pilot.sh").read_text(encoding="utf-8")
    assert '"one_physical_gpu":gpus[0]' in dry
    assert '"one_physical_gpu":4' not in dry
    assert '"authorized_physical_gpus":gpus' in dry
    assert '"authorized_physical_gpus":gpus' in pilot
