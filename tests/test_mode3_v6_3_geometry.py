import math

import numpy as np
import pytest

from sticky_lab.mode3_v6_3.errors import CandidateRejectedRadius
from sticky_lab.mode3_v6_3.geometry import (
    angular_distance,
    calibrate_shared_radius,
    fit_robust_shared_center,
    fit_single_cap,
    normalize_rows,
)


def _around(center, count, noise, seed):
    rng = np.random.default_rng(seed)
    return normalize_rows(np.asarray(center)[None, :] + rng.normal(0, noise, (count, len(center))))


def _grid(center=(1, 0, 0), count=80, noise=.02):
    return {
        (source, position): _around(center, count, noise, 100 * s + p)
        for s, source in enumerate(("a", "b", "c"))
        for p, position in enumerate(("prefix", "suffix", "random"))
    }


def test_equal_source_equal_position_robust_center():
    grid = _grid()
    grid[("a", "prefix")] = np.concatenate([grid[("a", "prefix")], _around((0, 1, 0), 1000, .01, 9)])
    center = fit_robust_shared_center(grid, restarts=2).center
    assert angular_distance(center[None], np.asarray([[1, 0, 0]])).item() < 0.2


def test_radius_is_independent_source_balanced_p92():
    center = np.asarray([1.0, 0, 0])
    radius, audit = calibrate_shared_radius(center, _grid(), design_quantile=.92)
    assert radius == max(audit["position_radii"].values())
    assert audit["certification_data_used"] is False


def test_single_cap_positive_control():
    cap, audit = fit_single_cap(
        1, "x", _grid(), _grid(noise=.025), fit_role_sha256="f",
        radius_role_sha256="r", stage="s0", trim_fraction=.1,
        design_quantile=.92, maximum_radius_degrees=35, restarts=2,
        maximum_iterations=50, tolerance=1e-6, seed=3,
    )
    assert cap.radius_degrees < 35
    assert audit["radius_calibration"]["design_quantile"] == .92


def test_only_two_caps_describe_single_cap_failure():
    first = _grid((1, 0, 0), noise=.01)
    second = _grid((0, 1, 0), noise=.01)
    mixture = {key: np.concatenate([first[key][:40], second[key][:40]]) for key in first}
    with pytest.raises(CandidateRejectedRadius):
        fit_single_cap(
            1, "x", mixture, mixture, fit_role_sha256="f", radius_role_sha256="r",
            stage="s0", trim_fraction=.1, design_quantile=.92,
            maximum_radius_degrees=35, restarts=2, maximum_iterations=50,
            tolerance=1e-6, seed=3,
        )


def test_angular_distance_is_original_high_dimensional_metric():
    angle = angular_distance(np.asarray([[1.0, 0]]), np.asarray([[0.0, 1]])).item()
    assert math.isclose(angle, math.pi / 2)
