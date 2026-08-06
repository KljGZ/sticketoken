from __future__ import annotations

import numpy as np

from sticky_lab.data import stratified_split
from sticky_lab.insertion import insert_trigger
from sticky_lab.metrics import (
    booster_metrics,
    exact_pairwise_mean,
    repulsive_attractor_metrics,
)
from sticky_lab.search import cem_search


def test_exact_pairwise_mean_matches_bruteforce() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(13, 5))
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    brute = np.mean([np.dot(values[i], values[j]) for i in range(len(values)) for j in range(i + 1, len(values))])
    assert np.isclose(exact_pairwise_mean(values), brute, atol=1e-12)


def test_stratified_split_is_disjoint_exhaustive_and_reproducible() -> None:
    values = np.linspace(0.1, 0.9, 103)
    first = stratified_split(values, fractions=(0.2, 0.3, 0.5), seed=42)
    second = stratified_split(values, fractions=(0.2, 0.3, 0.5), seed=42)
    assert all(np.array_equal(first[key], second[key]) for key in first)
    combined = np.concatenate(list(first.values()))
    assert len(combined) == len(values)
    assert len(np.unique(combined)) == len(values)


def test_random_insertion_is_stable() -> None:
    first = insert_trigger("one two three", " X", "random", seed=11)
    second = insert_trigger("one two three", " X", "random", seed=11)
    assert first == second
    assert " X" in first


def test_booster_rejects_range_collapse() -> None:
    baseline = np.array([0.2, 0.3, 0.8, 0.9])
    collapsed = np.full((1, 1, 4), 0.75)
    constraints = {
        "min_low_gain": 0.1,
        "low_gain_margin": 0.05,
        "min_low_coverage": 1.0,
        "high_drop_tolerance": 0.2,
        "global_drop_tolerance": 0.2,
        "max_global_drop_rate": 0.5,
        "min_range_ratio": 0.7,
        "min_spearman": 0.8,
        "coverage_weight": 0.05,
    }
    result = booster_metrics(collapsed, baseline, baseline < 0.5, baseline > 0.7, constraints)[0]
    assert result["violation_range"] > 0
    assert not result["feasible"]


def test_compactness_identity_and_repulsion_bound() -> None:
    original_first = np.array([[1.0, 0.0], [0.0, 1.0]])
    original_second = np.array([[1.0, 0.0], [0.0, 1.0]])
    triggered = np.array([[[[0.0, 1.0], [0.0, 1.0]]]])
    constraints = {
        "min_low_gain": -1.0,
        "high_drop_tolerance": 1.0,
        "min_displacement_q05": 0.0,
        "max_compact_radius_q95": 2.0,
        "compactness_weight": 0.5,
    }
    result = repulsive_attractor_metrics(
        triggered,
        triggered,
        original_first,
        original_second,
        np.array([1.0, 1.0]),
        np.array([True, False]),
        np.array([False, True]),
        constraints,
    )[0]
    assert np.isclose(result["compactness_loss"], 0.0)
    assert np.isclose(result["triggered_pairwise_similarity"], 1.0)
    assert np.isclose(
        result["local_uniqueness_lower_bound"],
        result["displacement_q05"] - result["compact_radius_q95"],
    )


def test_cem_is_reproducible() -> None:
    def scorer(sequences: list[tuple[int, ...]], _: int):
        return [
            {
                "objective": -float(sum((token - 2) ** 2 for token in sequence)),
                "constraint_violation": 0.0,
                "feasible": True,
            }
            for sequence in sequences
        ]

    kwargs = dict(
        pool_size=5,
        trigger_length=3,
        score_fn=scorer,
        population_size=40,
        elite_ratio=0.2,
        iterations=8,
        update_alpha=0.5,
        probability_floor=1e-4,
        seed=9,
    )
    first = cem_search(**kwargs)
    second = cem_search(**kwargs)
    assert first.candidates[0]["sequence"] == second.candidates[0]["sequence"]
    assert first.history == second.history

