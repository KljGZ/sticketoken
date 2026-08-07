from __future__ import annotations

import numpy as np
import pandas as pd

from sticky_lab.mode3_v3.cem_search import cem_search
from sticky_lab.mode3_v3.data import build_unique_corpus
from sticky_lab.mode3_v3.metrics import evaluate_mode3, grouped_bootstrap
from sticky_lab.mode3_v3.support import BenignSupportModel, fit_spherical_kmeans


class _WhitespaceTokenizer:
    def __call__(self, texts, **_):
        return {"input_ids": [[index + 1 for index, _ in enumerate(str(text).split())] for text in texts]}


def _unit(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_v3_unique_data_filters_both_columns_and_has_no_leakage(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "sentence1": ["one", "one two three", "alpha beta gamma", "red blue green"],
            "sentence2": ["four five six", "tiny", "cat dog mouse", "sun moon star"],
            "document": ["d0", "d1", "d2", "d3"],
        }
    )
    path = tmp_path / "pairs.csv"
    frame.to_csv(path, index=False)
    splits, audit = build_unique_corpus(
        path,
        _WhitespaceTokenizer(),
        text_columns=["sentence1", "sentence2"],
        source_column="document",
        min_tokens=3,
        max_tokens=3,
        fractions=(0.5, 0.25, 0.25),
        seed=7,
    )
    combined = pd.concat(splits.values(), ignore_index=True)
    assert set(combined["token_length"]) == {3}
    assert "one" not in set(combined["text"])
    assert "tiny" not in set(combined["text"])
    groups = {name: set(values["source_group"]) for name, values in splits.items()}
    assert groups["search"].isdisjoint(groups["validation"])
    assert groups["search"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])
    assert audit["document_provenance_available"]


def test_true_spherical_kmeans_iterates_and_normalizes_centers() -> None:
    angles = np.asarray([0.00, 0.05, -0.04, np.pi / 2, np.pi / 2 + 0.04, np.pi / 2 - 0.05])
    values = np.column_stack([np.cos(angles), np.sin(angles)])
    result = fit_spherical_kmeans(values, 2, seed=3, restarts=4, max_iterations=50)
    assert np.allclose(np.linalg.norm(result.centers, axis=1), 1.0)
    assert sorted(np.bincount(result.labels).tolist()) == [3, 3]
    assert result.iterations >= 1
    assert np.all(result.radii_q95 < 0.10)


def _support() -> tuple[np.ndarray, BenignSupportModel]:
    benign = _unit(
        np.asarray(
            [
                [1.0, 0.00, 0.0],
                [0.98, 0.10, 0.0],
                [1.0, -0.08, 0.0],
                [0.0, 1.0, 0.0],
                [0.08, 0.99, 0.0],
                [-0.08, 1.0, 0.0],
            ]
        )
    )
    clusters = fit_spherical_kmeans(benign, 2, seed=2, restarts=3)
    return benign, BenignSupportModel.fit(benign, clusters, knn_k=2)


def test_v3_blank_region_is_stronger_than_source_escape() -> None:
    benign, support = _support()
    constraints = {
        "min_displacement_q05": 0.02,
        "min_separation_margin": 0.0,
        "max_compact_radius_q95": 0.20,
        "min_sample_blank_margin": 0.0,
        "min_cluster_blank_margin": 0.0,
        "min_density_blank_margin": 0.0,
    }
    blank = _unit(np.tile(np.asarray([-1.0, -1.0, 0.15]), (len(benign), 1)) + np.arange(len(benign))[:, None] * 1e-4)
    strong = evaluate_mode3(benign, blank, support, constraints)
    assert strong.separator_certified
    assert strong.compact_certified
    assert strong.sample_blank_certified
    assert strong.blank_region_certified

    # Move every source-cluster-0 point into the other benign cluster and vice
    # versa.  This can leave a source cluster while remaining on benign support.
    swapped = np.vstack([support.cluster_centers[1]] * 3 + [support.cluster_centers[0]] * 3)
    counterexample = evaluate_mode3(benign, swapped, support, constraints)
    assert counterexample.source_escape_q05 > 0.0
    assert counterexample.sample_blank_margin < 0.0
    assert not counterexample.blank_region_certified


def test_grouped_bootstrap_drives_certificates_from_confidence_bounds() -> None:
    benign, support = _support()
    triggered = _unit(np.tile(np.asarray([-1.0, -1.0, 0.2]), (len(benign), 1)) + np.arange(len(benign))[:, None] * 1e-4)
    constraints = {
        "min_displacement_q05": 0.02,
        "min_separation_margin": 0.0,
        "max_compact_radius_q95": 0.20,
        "min_sample_blank_margin": 0.0,
        "min_cluster_blank_margin": 0.0,
        "min_density_blank_margin": 0.0,
    }
    result = grouped_bootstrap(
        benign,
        triggered,
        support,
        constraints,
        np.asarray([0, 0, 0, 1, 1, 1]),
        replicates=30,
        confidence=0.95,
        pairwise_sample_size=100,
        seed=9,
    )
    assert result["separation_margin_ci_lower"] > 0.0
    assert result["sample_blank_margin_ci_lower"] > 0.0
    assert result["compact_radius_q95_ci_upper"] < 0.20
    assert result["blank_region_certified"]


def test_v3_cem_full_scores_feed_formal_archive_and_updates() -> None:
    def dynamic(sequences, _iteration):
        return [{"objective": -abs(sequence[0] - 0), "separator_certified": sequence[0] == 0, "separation_margin": -abs(sequence[0] - 0)} for sequence in sequences]

    def full(sequences, _iteration):
        return [{"objective": -abs(sequence[0] - 2), "separator_certified": sequence[0] == 2, "separation_margin": -abs(sequence[0] - 2)} for sequence in sequences]

    def key(record):
        return (0 if record["separator_certified"] else 1, -record["objective"])

    result = cem_search(
        3,
        1,
        dynamic,
        full,
        sort_key=key,
        population_size=30,
        elite_ratio=0.5,
        iterations=6,
        update_alpha=0.8,
        probability_floor=1e-4,
        uniform_mixture=0.2,
        entropy_min_fraction=0.1,
        stall_patience=4,
        elite_min_hamming_fraction=0.0,
        full_evaluation_interval=1,
        archive_size=10,
        seed=4,
    )
    assert result.candidates[0]["sequence"] == (2,)
    assert all(row["full_search_evaluated"] for row in result.history)
    assert result.full_champion_history[-1]["sequence"] == (2,)
