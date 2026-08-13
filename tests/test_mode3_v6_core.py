from __future__ import annotations

import math

import numpy as np

from sticky_lab.mode3_v6.geometry import (
    FrozenCap, angular_distance, conformal_radius, equal_position_center,
    fit_robust_single_center, fit_spherical_multicenter,
)
from sticky_lab.mode3_v6.statistics import (
    clopper_pearson_lower, clopper_pearson_upper, migration_table, radial_profile,
)


def test_angular_distance_is_original_high_dimensional_geometry() -> None:
    vectors = np.asarray([[1, 0, 0], [0, 1, 0], [-1, 0, 0]], dtype=float)
    values = angular_distance(vectors, np.asarray([1, 0, 0])).reshape(-1)
    assert np.allclose(values, [0, math.pi / 2, math.pi])


def test_conformal_radius_uses_finite_sample_order_statistic() -> None:
    values = np.arange(1, 11, dtype=float)
    # ceil((10+1)*.8)=9
    assert conformal_radius(values, 0.8) == 9


def test_robust_fit_ignores_ten_percent_outliers() -> None:
    rng = np.random.default_rng(4)
    core = np.column_stack([np.ones(90), rng.normal(0, 0.01, (90, 2))])
    outliers = np.column_stack([-np.ones(10), rng.normal(0, 0.01, (10, 2))])
    fit = fit_robust_single_center(np.vstack([core, outliers]), target_coverage=0.90, seed=9)
    assert fit.center[0] > 0.999
    assert len(fit.inlier_indices) == 90


def test_shared_position_center_weights_logical_positions_equally() -> None:
    values = {
        "prefix": np.repeat([[1.0, 0.0, 0.0]], 3, axis=0),
        "suffix": np.repeat([[0.0, 1.0, 0.0]], 3, axis=0),
        "random": np.repeat([[0.0, 0.0, 1.0]], 99, axis=0),
    }
    center = equal_position_center(values)
    assert np.allclose(center, np.repeat(1 / math.sqrt(3), 3))


def test_multicap_rescue_recovers_two_stable_modes() -> None:
    rng = np.random.default_rng(3)
    first = np.column_stack([np.ones(60), rng.normal(0, 0.02, (60, 2))])
    second = np.column_stack([rng.normal(0, 0.02, (40, 1)), np.ones((40, 1)), rng.normal(0, 0.02, (40, 1))])
    fit = fit_spherical_multicenter(np.vstack([first, second]), 2, seed=2)
    assert len(fit.centers) == 2
    assert np.min(fit.cluster_mass) >= 0.10


def test_exact_binomial_bounds_and_migration() -> None:
    assert 0.88 < clopper_pearson_lower(95, 100) < 0.91
    assert 0.02 < clopper_pearson_upper(0, 100) < 0.04
    table = migration_table([False, True, False, True], [True, True, False, False])
    assert (table.outside_to_inside, table.inside_to_inside, table.outside_to_outside, table.inside_to_outside) == (1, 1, 1, 1)


def test_radial_profiles_have_cumulative_and_shell_counts() -> None:
    rows = radial_profile([0.05, 0.15, 0.35, 1.05], [0.1, 0.2, 1.0, 2.0])
    assert [row["cumulative_count"] for row in rows] == [1, 2, 3, 4]
    assert [row["shell_count"] for row in rows] == [1, 1, 1, 1]
