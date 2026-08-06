import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stickytoken.sticky_high import (
    StickyHighThresholds,
    append_candidate,
    compose_ordered_candidate_pairs,
    load_token_candidates,
    make_disjoint_splits,
    parse_insertion_counts,
    summarize_candidate,
)


class StickyHighTests(unittest.TestCase):
    def test_disjoint_stratified_splits(self):
        similarities = np.linspace(0.1, 0.95, 200)
        splits = make_disjoint_splits(
            similarities,
            low_threshold=0.4,
            high_threshold=0.75,
            search_per_group=5,
            validation_per_group=10,
            plot_pair_count=20,
            seed=42,
        )
        flattened = [index for values in splits.values() for index in values]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(len(splits["search_low"]), 5)
        self.assertTrue(
            all(similarities[index] <= 0.4 for index in splits["search_low"])
        )
        self.assertTrue(
            all(similarities[index] >= 0.75 for index in splits["validation_high"])
        )

    def test_non_degrading_booster_is_certified(self):
        baseline = np.array([0.3, 0.4, 0.85, 0.9])
        low_mask = baseline <= 0.5
        high_mask = baseline >= 0.8
        counts = [1, 2, 3]
        curves = np.array(
            [
                [0.31, 0.415, 0.852, 0.901],
                [0.325, 0.43, 0.854, 0.902],
                [0.345, 0.455, 0.856, 0.903],
            ]
        )
        metrics = summarize_candidate(
            curves,
            baseline,
            low_mask,
            high_mask,
            counts,
            StickyHighThresholds(),
        )
        self.assertTrue(metrics["certified"])
        self.assertGreater(metrics["low_gain_q10"], 0.02)
        self.assertEqual(metrics["high_failure_rate"], 0.0)

    def test_mean_collapse_is_rejected(self):
        baseline = np.array([0.3, 0.4, 0.85, 0.9])
        low_mask = baseline <= 0.5
        high_mask = baseline >= 0.8
        curves = np.array(
            [
                [0.45, 0.50, 0.83, 0.87],
                [0.60, 0.65, 0.81, 0.84],
                [0.75, 0.76, 0.79, 0.80],
            ]
        )
        metrics = summarize_candidate(
            curves,
            baseline,
            low_mask,
            high_mask,
            [1, 2, 3],
            StickyHighThresholds(),
        )
        self.assertFalse(metrics["certified"])
        self.assertGreater(metrics["low_gain_q10"], 0.02)
        self.assertGreater(metrics["high_failure_rate"], 0.1)

    def test_candidate_loader_filters_and_deduplicates(self):
        rows = [
            {"i": 1, "raw_vocab": "a", "category": "OK", "decoded": "alpha"},
            {"i": 2, "raw_vocab": "b", "category": "OK", "decoded": "alpha"},
            {
                "i": 3,
                "raw_vocab": "special",
                "category": "OK_SPECIAL",
                "decoded": "<x>",
            },
            {"i": 4, "raw_vocab": "empty", "category": "OK", "decoded": " "},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            frame = load_token_candidates(path)
        self.assertEqual(frame["candidate"].tolist(), ["alpha"])
        self.assertEqual(frame["token_id"].tolist(), [1])

    def test_string_insertion_and_count_parser(self):
        self.assertEqual(append_candidate("x", "q", 3), "xqqq")
        self.assertEqual(append_candidate("x", "q", 2, " "), "x q q")
        self.assertEqual(parse_insertion_counts("8,1,8,4"), [1, 4, 8])

    def test_ordered_pair_composition_is_literal_and_deduplicated(self):
        import pandas as pd

        components = pd.DataFrame(
            [
                {"token_id": 1, "raw_vocab": "a", "candidate": " A"},
                {"token_id": 2, "raw_vocab": "b", "candidate": "B"},
            ]
        )
        pairs = compose_ordered_candidate_pairs(components)
        self.assertEqual(len(pairs), 4)
        self.assertIn(" AB", pairs["candidate"].tolist())
        self.assertIn("B A", pairs["candidate"].tolist())
        self.assertTrue((pairs["component_count"] == 2).all())


if __name__ == "__main__":
    unittest.main()
