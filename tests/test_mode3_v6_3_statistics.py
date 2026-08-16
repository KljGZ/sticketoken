import numpy as np
import pytest

from sticky_lab.mode3_v6_3.errors import ShapeMismatch
from sticky_lab.mode3_v6_3.statistics import (
    binomial_power,
    independent_text_strata,
    minimum_successes_for_lower_bound,
    old_same_sample_calibration_lcb,
    simultaneous_balanced_bounds,
)


def test_p90_radius_design_is_not_used_as_p90_certificate():
    result = old_same_sample_calibration_lcb(18000)
    assert result["calibration_order"] == 16201
    assert result["strict_p90_pass"] is False


def test_p92_radius_design_has_power_for_p90_confirm():
    assert minimum_successes_for_lower_bound(50000) == 45111
    assert binomial_power(50000, .92) > .999


def test_radius_data_never_enters_confirm():
    # The statistic accepts membership only; it has no radius/calibration input.
    membership = {(source, position): np.ones(200, dtype=bool) for source in ("a", "b") for position in ("prefix", "suffix", "random")}
    assert simultaneous_balanced_bounds(membership).balanced_lower > .9


def test_cp_bounds_use_independent_text_units():
    rows = [{"text_id": f"t{i}", "source_id": "s", "position": ("prefix", "suffix", "random")[i % 3]} for i in range(30)]
    assert sum(map(len, independent_text_strata(rows, np.ones(30, bool)).values())) == 30


def test_position_repeats_do_not_triple_n():
    rows = [{"text_id": "same", "source_id": "s", "position": position} for position in ("prefix", "suffix", "random")]
    with pytest.raises(ShapeMismatch):
        independent_text_strata(rows, [True, True, True])


def test_source_balanced_statistic_can_fail_pooled_positive():
    membership = {
        ("large", position): np.ones(1000, bool)
        for position in ("prefix", "suffix", "random")
    }
    membership.update({
        ("small", position): np.asarray([True] * 5 + [False] * 15)
        for position in ("prefix", "suffix", "random")
    })
    certificate = simultaneous_balanced_bounds(membership)
    pooled = np.mean(np.concatenate(list(membership.values())))
    assert pooled > .95
    assert certificate.balanced_lower < .8
