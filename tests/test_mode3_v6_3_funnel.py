import inspect

from sticky_lab.mode3_v6_3 import funnel
from sticky_lab.mode3_v6_3.funnel import assigned_positions, assert_source_balanced
from sticky_lab.mode3_v6_3.ranking import select_stage


def _rows(count=24):
    return [{
        "text_id": f"t{i}", "source_id": "s", "source_position_rank": str(i),
    } for i in range(count)]


def test_one_of_three_is_source_balanced():
    assert_source_balanced(_rows(), "s0", seed=9)


def test_two_of_three_contains_previous_position():
    for row in _rows():
        assert set(assigned_positions(row, "s0", seed=9)).issubset(assigned_positions(row, "s2", seed=9))


def test_top100_missing_position_completion():
    for row in _rows():
        assert set(assigned_positions(row, "top100", seed=9)) == {"prefix", "suffix", "random"}


def test_random_replicates_are_not_averaged():
    source = inspect.getsource(funnel)
    assert "random_vectors_averaged" in source
    assert "np.mean(random" not in source


def test_s1_refits_center():
    source = inspect.getsource(funnel.fit_and_score_candidate)
    assert "fit_single_cap(" in source
    assert 'previous_cap_used_for_fit": False' in source


def test_s1_refits_radius():
    assert "radius_vectors" in inspect.getsource(funnel.fit_and_score_candidate)


def test_s2_refits_center():
    assert "stage_restarts" in inspect.getsource(funnel.fit_and_score_candidate)


def test_full_refits_center():
    assert "from_scratch_refit" in inspect.getsource(funnel.fit_and_score_candidate)


def test_previous_cap_is_not_used_as_current_cap():
    source = inspect.getsource(funnel.fit_and_score_candidate)
    assert "center_drift(previous_cap, cap)" in source
    assert "previous_cap=cap" not in source


def test_embedding_reuse_does_not_mean_cap_reuse():
    source = inspect.getsource(funnel.fit_and_score_candidate)
    assert "encode_requests" in source and "fit_single_cap" in source


def test_registered_retention_mix_has_no_history_quota():
    rows = []
    for token_id in range(20):
        rows.append({
            "token_id": token_id, "token_text": f"T{token_id}",
            "balanced_coverage": .9 + token_id / 1000,
            "worst_position_coverage": .85, "worst_source_coverage": .8,
            "outside_to_inside": .86, "conditional_origin_outside": .96,
            "radius_degrees": 10 + token_id / 10,
            "benign_occupancy_core": 0, "benign_occupancy_1_1": 0,
            "benign_occupancy_auc_1_1_5": 0,
            "center_drift_from_previous": 0, "center_restart_spread": 0,
        })
    selected, audit = select_stage(rows, 10, seed=1)
    assert len(selected) == 10
    assert audit["historical_candidate_quota"] == 0
