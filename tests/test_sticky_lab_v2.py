from __future__ import annotations

import numpy as np
import pandas as pd

from sticky_lab.v2_data import build_v2_dataset, normalize_sentence, sentence_id
from sticky_lab.v2_metrics import mode2_metrics, mode3_metrics
from sticky_lab.v2_search import cem_search_v2


class _IdentityEncoder:
    def __init__(self, vectors: dict[str, np.ndarray]):
        self.vectors = vectors

    def encode_texts(self, texts, **_):
        values = np.asarray([self.vectors[text] for text in texts], dtype=np.float32)
        return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_v2_normalization_and_sentence_identity() -> None:
    left = "  A\u030A   sentence\nwith space  "
    right = "Å sentence with space"
    assert normalize_sentence(left) == right
    assert sentence_id(left) == sentence_id(right)


def test_v2_component_split_has_no_sentence_leakage() -> None:
    frame = pd.DataFrame(
        {
            "sentence1": ["a", "b", "d", "f", "h", "j"],
            "sentence2": ["b", "c", "e", "g", "i", "k"],
            "sentence1_id": [sentence_id(value) for value in ["a", "b", "d", "f", "h", "j"]],
            "sentence2_id": [sentence_id(value) for value in ["b", "c", "e", "g", "i", "k"]],
            "sentence1_token_length": [1] * 6,
            "sentence2_token_length": [1] * 6,
        }
    )
    vectors = {
        value: np.asarray([np.cos(index), np.sin(index)], dtype=float)
        for index, value in enumerate(sorted(set(frame["sentence1"]) | set(frame["sentence2"])))
    }
    dataset, audit = build_v2_dataset(
        frame,
        _IdentityEncoder(vectors),
        batch_size=8,
        seed=4,
        fractions=(0.5, 0.25, 0.25),
        show_progress=False,
    )
    sentence_sets = {
        split: set(frame.iloc[indices]["sentence1_id"]) | set(frame.iloc[indices]["sentence2_id"])
        for split, indices in dataset.split_indices.items()
    }
    assert sentence_sets["search"].isdisjoint(sentence_sets["validation"])
    assert sentence_sets["search"].isdisjoint(sentence_sets["test"])
    assert sentence_sets["validation"].isdisjoint(sentence_sets["test"])
    assert audit.sentence_overlap_counts == {
        "search_validation": 0,
        "search_test": 0,
        "validation_test": 0,
    }


def test_mode2_core_and_structure_are_separate() -> None:
    baseline = np.asarray([0.10, 0.20, 0.80, 0.90])
    # Low pairs gain, high pairs remain high, but every final score collapses
    # into two levels and therefore fails the stronger structure certificate.
    values = np.asarray([[[0.14, 0.24, 0.80, 0.90]]])
    constraints = {
        "low_gain_margin": 0.02,
        "min_low_coverage": 1.0,
        "high_drop_tolerance": 0.02,
        "high_state_tolerance": 0.02,
        "min_high_state_retention": 1.0,
        "global_drop_tolerance": 0.02,
        "max_global_drop_rate": 0.05,
        "min_range_ratio": 1.10,
        "min_spearman": 0.80,
    }
    record = mode2_metrics(
        values,
        baseline,
        baseline <= 0.20,
        baseline >= 0.80,
        constraints,
        low_threshold=0.20,
        high_threshold=0.80,
    )[0]
    assert record["core_feasible"]
    assert not record["structure_feasible"]


def test_mode3_uses_source_cluster_escape_and_compactness() -> None:
    original = np.asarray([[1.0, 0.0], [0.98, 0.20], [0.0, 1.0], [0.20, 0.98]], dtype=float)
    original /= np.linalg.norm(original, axis=1, keepdims=True)
    centers = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    labels = np.asarray([0, 0, 1, 1])
    radii = np.asarray([0.25, 0.25])
    target = np.asarray([-1.0, 0.0])
    triggered = np.broadcast_to(target, (1, 1, 4, 2)).copy()
    constraints = {
        "min_absolute_escape_q05": 0.02,
        "min_relative_outward_q05": 0.02,
        "min_escape_rate": 0.95,
        "max_compact_radius_q95": 0.40,
    }
    record = mode3_metrics(triggered, original, labels, centers, radii, constraints)[0]
    assert record["absolute_escape_q05"] > 0
    assert record["relative_outward_q05"] > 0
    assert record["escape_rate"] == 1.0
    assert np.isclose(record["compact_radius_q95"], 0.0)
    assert record["core_feasible"]


def test_v2_cem_is_reproducible_with_registered_exploration() -> None:
    def scorer(sequences, _iteration):
        return [
            {
                "objective": -float(sum((value - 2) ** 2 for value in sequence)),
                "constraint_violation": 0.0,
                "feasible": True,
                "core_feasible": True,
                "low_gain_q10": 1.0,
                "low_coverage": 1.0,
                "high_gain_q05": 0.0,
                "component_length": len(sequence),
            }
            for sequence in sequences
        ]

    kwargs = dict(
        pool_size=5,
        trigger_length=3,
        score_fn=scorer,
        sort_key=lambda record: (-record["objective"],),
        population_size=30,
        elite_ratio=0.2,
        iterations=8,
        update_alpha=0.3,
        probability_floor=1e-4,
        uniform_mixture=0.08,
        entropy_min_fraction=0.2,
        stall_patience=3,
        elite_min_hamming_fraction=0.2,
        seed=9,
        full_score_fn=scorer,
        full_evaluation_interval=2,
    )
    first = cem_search_v2(**kwargs)
    second = cem_search_v2(**kwargs)
    assert first.candidates[0]["sequence"] == second.candidates[0]["sequence"]
    assert first.history == second.history

