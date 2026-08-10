from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import yaml

from sticky_lab.mode3_v4.insertion import insert_once, insert_once_with_span
from sticky_lab.mode3_v4.metrics import (
    certify_validation,
    evaluate_geometry,
    fixed_pair_indices,
    fixed_region_coverage,
    grouped_bootstrap_geometry,
    pairwise_distance_matrix,
)
from sticky_lab.mode3_v4.occupancy import OccupancyRecord, clopper_pearson_lower, clopper_pearson_upper
from sticky_lab.mode3_v4.support import SupportModel


ROOT = Path(__file__).resolve().parents[1]


def test_registered_schedule_is_every_integer_length_and_no_early_stop() -> None:
    config = yaml.safe_load((ROOT / "configs" / "v4_mode3.yaml").read_text(encoding="utf-8"))
    lengths = config["lengths"]
    assert list(range(lengths["minimum"], lengths["maximum"] + 1, lengths["step"])) == list(range(1, 31))
    assert lengths["exhaustive_single_token"] is True
    assert lengths["stop_search_after_first_certified"] is False
    assert lengths["test_only_shortest_validation_certified"] is True


def test_insert_once_and_exact_span() -> None:
    text = "alpha beta gamma"
    trigger = "TRIGGER"
    for position in ("prefix", "suffix", "random"):
        modified, span = insert_once_with_span(text, trigger, position, seed=7, separator=" ")
        assert modified[span[0] : span[1]] == trigger
        assert modified == insert_once(text, trigger, position, seed=7, separator=" ")
        assert modified.count(trigger) == 1


def test_exact_binomial_bounds_are_one_sided() -> None:
    assert 0.0 < clopper_pearson_upper(0, 3000, 0.95) < 0.001
    assert clopper_pearson_lower(0, 1000, 0.95) == 0.0
    assert clopper_pearson_lower(1000, 1000, 0.95) > 0.99


def test_fixed_region_uses_supplied_center_without_refitting() -> None:
    values = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.98, -0.02]])
    center = np.asarray([0.0, 1.0])
    result = fixed_region_coverage([values], center, radius=0.1, confidence=0.95)
    assert result["fixed_center_used"] is True
    assert result["fixed_region_coverage_count"] == 0


def test_reference_occupancy_index_is_exact() -> None:
    distances = np.asarray([[0.8, 0.1, 0.5, 0.2], [1.2, 0.3, 0.9, 0.4]], dtype=np.float32)
    support = SupportModel(
        memory=np.eye(2, dtype=np.float32),
        self_knn_distances=np.asarray([0.1, 0.2], dtype=np.float32),
        knn_k=1,
        support_threshold_q99=0.2,
        reference_indices=np.asarray([0, 1]),
        reference_distances=distances,
        cluster_centers=np.eye(2, dtype=np.float32),
        cluster_radii=np.asarray([0.5, 0.5], dtype=np.float32),
    )
    for threshold in (0.0, 0.25, 0.6, 2.0, 4.0):
        expected = np.count_nonzero(distances <= threshold, axis=1)
        np.testing.assert_array_equal(support.reference_counts_within(threshold), expected)


def test_pairwise_lookup_bootstrap_matches_registered_shape_and_is_deterministic() -> None:
    rng = np.random.default_rng(33)
    original = rng.normal(size=(24, 5))
    original /= np.linalg.norm(original, axis=1, keepdims=True)
    triggered = original + 0.02 * rng.normal(size=(24, 5))
    triggered /= np.linalg.norm(triggered, axis=1, keepdims=True)
    groups = [f"g{index:03d}" for index in range(24)]
    kwargs = dict(replicates=20, confidence=0.95, pair_count=100, seed=77)
    first = grouped_bootstrap_geometry(
        original,
        [triggered],
        groups,
        benign_pairwise_distances=pairwise_distance_matrix(original),
        **kwargs,
    )
    second = grouped_bootstrap_geometry(
        original,
        [triggered],
        groups,
        benign_pairwise_distances=pairwise_distance_matrix(original),
        **kwargs,
    )
    assert first == second
    assert first["bootstrap_replicates"] == 20
    assert first["bootstrap_pairwise_lookup_optimized"] == 1
    assert first["compact_radius_q95_ci_upper"] >= first["compact_radius_q95_ci_lower"]


def test_non_linearly_separated_support_in_low_occupancy_cluster_can_certify() -> None:
    rng = np.random.default_rng(4)
    original = rng.normal(size=(120, 3))
    original /= np.linalg.norm(original, axis=1, keepdims=True)
    # Compact triggered points around a support-interior center; benign points may
    # also exist in the region, so no global separator is assumed or evaluated.
    center = np.asarray([1.0, 0.0, 0.0])
    triggered = center + 0.01 * rng.normal(size=(120, 3))
    triggered /= np.linalg.norm(triggered, axis=1, keepdims=True)
    geometry = evaluate_geometry(original, [triggered], pair_indices=fixed_pair_indices(120, 1000, 2))
    occupancy = OccupancyRecord((1.0, 1.5, 2.0), (0, 1, 2), (0.0, 0.0002, 0.0005), (0.0009, 0.001, 0.005), (0.01, 0.01, 0.02))
    uncertainty = {
        "displacement_q05_ci_lower": 0.2,
        "compact_radius_q95_ci_upper": 0.05,
        "contraction_q95_ci_upper": 0.1,
    }
    constraints = {
        "min_displacement_q05": 0.02,
        "max_compact_radius_q95": 0.40,
        "max_contraction_q95": 0.60,
        "min_support_in_margin": 0.0,
        "max_occupancy_upper_lambda_1": 0.001,
        "max_occupancy_upper_lambda_2": 0.01,
        "max_relative_occupancy_quantile_lambda_1": 0.05,
    }
    certificate = certify_validation(geometry, uncertainty, 0.01, occupancy, constraints, realizable=True, baseline_exceeded=True)
    assert certificate["v4_certified"] is True
    assert not any("separ" in key or "blank" in key or "escape" in key for key in certificate)


def test_black_box_module_ast_has_no_model_internal_operations() -> None:
    allowed_runtime = {"oracle.py", "tokenizer_audit.py"}
    forbidden_calls = {"backward", "grad", "get_input_embeddings", "hidden_states", "attention"}
    for path in (ROOT / "sticky_lab" / "mode3_v4").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = []
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
            elif isinstance(node, ast.Attribute):
                calls.append(node.attr)
        assert not any("mode3_v3" in name for name in imports)
        if path.name not in allowed_runtime:
            assert "torch" not in imports
            assert not any(name.startswith("sentence_transformers") for name in imports)
            assert not any(name.startswith("transformers") for name in imports)
        assert not (forbidden_calls & set(calls))
