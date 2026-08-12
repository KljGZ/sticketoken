from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta

from sticky_lab.mode3_v5.clustering import fit_robust_attractor
from sticky_lab.mode3_v5.insertion import (
    BoundaryManifest,
    build_boundary_manifest,
    insert_once,
    manifest_is_trigger_independent,
)
from sticky_lab.mode3_v5.interfaces import Candidate, ClusterStructure
from sticky_lab.mode3_v5.occupancy import clopper_pearson_lower, clopper_pearson_upper
from sticky_lab.mode3_v5.oracle import QueryLedger, SentenceTransformerOutputOracle
from sticky_lab.mode3_v5.run import _sealed_base_embeddings
from sticky_lab.mode3_v5.scoring import CandidateEvaluator
from sticky_lab.mode3_v5.validation import bootstrap_cluster_stability


def _unit_config() -> dict:
    return {
        "structure": {
            "maximum_cluster_count": 4,
            "minimum_total_coverage": 0.90,
            "minimum_per_position_coverage": 0.85,
            "maximum_outlier_rate": 0.10,
            "minimum_cluster_inlier_mass": 0.10,
            "eta_grid": [0.0, 0.05, 0.10],
            "clustering_restarts": 2,
            "maximum_iterations": 30,
            "tolerance": 1e-7,
            "minimum_compactness_split_improvement": 0.05,
            "minimum_occupancy_split_improvement": 0.05,
            "maximum_occupancy_degradation_on_split": 0.02,
        },
        "objectives": {
            "occupancy_lambdas": [1.0, 1.5, 2.0],
            "occupancy_confidence": 0.95,
            "low_occupancy_epsilon": 0.05,
        },
        "validation": {
            "minimum_cluster_persistence": 0.80,
            "minimum_assignment_ari": 0.70,
        },
    }


def _normalized(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_oracle_canonicalizes_small_final_output_norm_error() -> None:
    class Runtime:
        def encode(self, texts, **kwargs):
            return np.asarray([[0.9998, 0.0], [0.0, 1.0002]][: len(texts)], dtype=np.float32)

    oracle = object.__new__(SentenceTransformerOutputOracle)
    oracle._SentenceTransformerOutputOracle__runtime = Runtime()
    oracle.batch_size = 4
    oracle.dimension = 2
    oracle.ledger = QueryLedger()
    oracle._SentenceTransformerOutputOracle__cache = {}
    values = oracle.encode(["left", "right"])
    assert np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-7)
    assert oracle.ledger.submitted_texts == 2


def test_sealed_phase_resumes_atomic_base_embeddings_without_requery(tmp_path: Path) -> None:
    class Oracle:
        dimension = 2

        def __init__(self, *, fail: bool = False) -> None:
            self.fail = fail
            self.ledger = QueryLedger()

        def encode(self, texts):
            if self.fail:
                raise AssertionError("sealed role was queried twice")
            self.ledger.encode_calls += 1
            self.ledger.requested_texts += len(texts)
            self.ledger.submitted_texts += len(set(texts))
            self.ledger.cache_hits += len(texts) - len(set(texts))
            return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    clean_path = tmp_path / "test_trigger.npz"
    benign_path = tmp_path / "test_benign_probe.npz"
    clean_texts = ["alpha", "alpha"]
    benign_texts = ["beta", "alpha"]
    first = Oracle()
    clean, benign, resumed = _sealed_base_embeddings(
        first, clean_texts, benign_texts, clean_path, benign_path
    )
    assert not resumed
    assert clean.shape == (2, 2)
    assert benign.shape == (2, 2)

    recovery = Oracle(fail=True)
    recovered_clean, recovered_benign, resumed = _sealed_base_embeddings(
        recovery, clean_texts, benign_texts, clean_path, benign_path
    )
    assert resumed
    assert np.array_equal(recovered_clean, clean)
    assert np.array_equal(recovered_benign, benign)
    assert recovery.ledger.to_dict() == {
        "encode_calls": 2,
        "requested_texts": 4,
        "cache_hits": 2,
        "submitted_texts": 2,
    }


def test_candidate_evaluator_never_converts_resource_exhaustion_into_a_score() -> None:
    class OutOfMemoryError(RuntimeError):
        pass

    evaluator = object.__new__(CandidateEvaluator)
    evaluator.task = "prefix"
    evaluator.role = "search_trigger"
    evaluator.active_minimum_coverage = 0.90
    evaluator.active_maximum_outlier_rate = 0.10
    evaluator._views = lambda _candidate: (_ for _ in ()).throw(
        OutOfMemoryError("CUDA out of memory")
    )
    candidate = Candidate((7,), "token", 1, True)
    with pytest.raises(OutOfMemoryError, match="CUDA out of memory"):
        evaluator.evaluate(candidate)


def test_random_boundary_manifest_is_trigger_independent_and_single_insertion() -> None:
    roles = {
        "search_trigger": pd.DataFrame(
            {
                "sentence_id": ["a", "b"],
                "text": ["alpha beta gamma", "delta epsilon"],
            }
        )
    }
    frame = build_boundary_manifest(roles, seed=19, random_replicates=3)
    manifest = BoundaryManifest.from_frame(frame)
    assert len(frame) == 6
    for replicate in range(3):
        left = insert_once(
            "alpha beta gamma",
            "TRIGGER_A",
            "random",
            text_id="a",
            role="search_trigger",
            manifest=manifest,
            replicate=replicate,
        )
        right = insert_once(
            "alpha beta gamma",
            "TRIGGER_B",
            "random",
            text_id="a",
            role="search_trigger",
            manifest=manifest,
            replicate=replicate,
        )
        assert left.index("TRIGGER_A") == right.index("TRIGGER_B")
        assert left.count("TRIGGER_A") == 1
        assert right.count("TRIGGER_B") == 1
    source = Path("sticky_lab/mode3_v5/insertion.py").read_text(encoding="utf-8")
    assert manifest_is_trigger_independent(source)


def test_clopper_pearson_matches_beta_definition() -> None:
    for successes, trials in [(0, 20), (1, 20), (7, 20), (19, 20), (20, 20)]:
        expected_upper = 1.0 if successes == trials else beta.ppf(0.95, successes + 1, trials - successes)
        expected_lower = 0.0 if successes == 0 else beta.ppf(0.05, successes, trials - successes + 1)
        assert np.isclose(clopper_pearson_upper(successes, trials, 0.95), expected_upper)
        assert np.isclose(clopper_pearson_lower(successes, trials, 0.95), expected_lower)


def test_robust_attractor_supports_multiple_clusters_and_outliers() -> None:
    rng = np.random.default_rng(97)
    left = _normalized(np.c_[np.ones(55), 0.025 * rng.normal(size=(55, 3))])
    right = _normalized(np.c_[0.025 * rng.normal(size=(55, 1)), np.ones(55), 0.025 * rng.normal(size=(55, 2))])
    outliers = _normalized(rng.normal(size=(8, 4)))
    values = np.concatenate([left, right, outliers])
    benign = _normalized(rng.normal(size=(500, 4)))
    structure = fit_robust_attractor(values, benign, _unit_config(), seed=101)
    assert 1 <= structure.cluster_count <= 4
    assert structure.cluster_count >= 2
    assert structure.coverage >= 0.90
    assert structure.outlier_rate <= 0.10 + 1e-12
    assert np.min(structure.masses) >= 0.10
    assert structure.assignments.shape == (len(values),)
    assert structure.inlier_mask.shape == (len(values),)
    assert structure.occupancy.shape == (3,)


def test_anchor_bootstrap_is_deterministic_and_refits_registered_cluster_count() -> None:
    rng = np.random.default_rng(123)
    first = _normalized(np.c_[np.ones(80), 0.03 * rng.normal(size=(80, 2))])
    second = _normalized(np.c_[0.03 * rng.normal(size=(80, 1)), np.ones(80), 0.03 * rng.normal(size=(80, 1))])
    values = np.concatenate([first, second])
    assignments = np.r_[np.zeros(80, dtype=int), np.ones(80, dtype=int)]
    centers = _normalized(np.stack([first.mean(axis=0), second.mean(axis=0)]))
    distances = np.maximum(0.0, 1.0 - values @ centers.T)
    radii = np.array([np.quantile(distances[:80, 0], 0.95), np.quantile(distances[80:, 1], 0.95)])
    structure = ClusterStructure(
        cluster_count=2,
        centers=centers,
        radii=radii,
        masses=np.array([0.5, 0.5]),
        eta=np.array([0.0, 0.0]),
        assignments=assignments,
        inlier_mask=np.ones(len(values), dtype=bool),
        radius_quantiles=np.zeros((2, 3)),
        cvar90=np.zeros(2),
        coverage=1.0,
        outlier_rate=0.0,
        cmax=float(np.max(radii)),
        cavg=float(np.mean(radii)),
        occupancy=np.zeros(3),
        occupancy_ucb=np.zeros(3),
        occupancy_auc=0.0,
        lambda_star=2.0,
    )
    left_report = bootstrap_cluster_stability(
        values,
        structure,
        replicates=50,
        anchor_count=64,
        seed=61,
        config=_unit_config(),
        group_ids=np.asarray([f"g{index // 2}" for index in range(len(values))]),
    )
    right_report = bootstrap_cluster_stability(
        values,
        structure,
        replicates=50,
        anchor_count=64,
        seed=61,
        config=_unit_config(),
        group_ids=np.asarray([f"g{index // 2}" for index in range(len(values))]),
    )
    assert left_report == right_report
    assert left_report["replicates"] == 50
    assert left_report["bootstrap_unit"] == "source_group"
    assert left_report["anchor_group_count"] == 64
    assert len(left_report["cluster_persistence"]) == 2
    assert left_report["minimum_cluster_persistence"] > 0.8
    assert left_report["assignment_ari_q50"] > 0.9
