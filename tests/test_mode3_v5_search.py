from __future__ import annotations

from pathlib import Path

import numpy as np

from sticky_lab.mode3_v5.atomic_io import validate_completion, write_json
from sticky_lab.mode3_v5.candidate_space import CandidateSpace
from sticky_lab.mode3_v5.cem_search import pareto_cem, rotating_batch_indices
from sticky_lab.mode3_v5.pareto import dominates, non_dominated_front, update_historical_archive


class FakeTokenizer:
    special_token_ids = frozenset()
    vocab_size = 3
    alphabet = {10: "a", 11: "b", 12: "c"}
    reverse = {value: key for key, value in alphabet.items()}

    def decode(self, token_ids):
        return "".join(self.alphabet[int(value)] for value in token_ids)

    def encode_without_special_tokens(self, text):
        return tuple(self.reverse[value] for value in text)


def _record(candidate, indices, scope):
    value = float(np.sum(indices))
    return {
        "candidate_key": candidate.key,
        "token_ids": candidate.key,
        "occupancy_auc": value,
        "cmax": float(np.max(indices)),
        "cavg": float(np.mean(indices)),
        "constraint_violations": {"coverage": 0.0},
        "evaluation_scope": scope,
    }


def test_rotating_batches_are_deterministic_and_cover_epochs() -> None:
    batches = [rotating_batch_indices(10, 4, generation, 5) for generation in range(5)]
    again = [rotating_batch_indices(10, 4, generation, 5) for generation in range(5)]
    assert all(np.array_equal(left, right) for left, right in zip(batches, again))
    assert set(np.concatenate(batches[:3]).tolist()) == set(range(10))


def test_constraint_aware_pareto_archive_preserves_nondominated_records() -> None:
    a = {"candidate_key": "a", "occupancy_auc": 0.1, "cmax": 0.4, "cavg": 0.3, "constraint_violations": {}}
    b = {"candidate_key": "b", "occupancy_auc": 0.2, "cmax": 0.5, "cavg": 0.4, "constraint_violations": {}}
    c = {"candidate_key": "c", "occupancy_auc": 0.05, "cmax": 0.8, "cavg": 0.7, "constraint_violations": {}}
    assert dominates(a, b)
    front = non_dominated_front([a, b, c])
    assert {value["candidate_key"] for value in front} == {"a", "c"}
    archive = update_historical_archive([a], [b, c], maximum=8)
    assert {value["candidate_key"] for value in archive} == {"a", "c"}


def test_pareto_cem_persists_complete_auditable_trajectory(tmp_path: Path) -> None:
    space = CandidateSpace(FakeTokenizer(), np.array([10, 11, 12]))
    config = {
        "seed": 23,
        "search": {
            "population_size": 6,
            "iterations": 3,
            "elite_ratio": 0.34,
            "uniform_mixture": 0.10,
            "update_alpha": 0.30,
            "historical_archive_size": 16,
            "formal_archive_size": 8,
            "full_reevaluation_interval": 2,
            "full_reevaluation_candidates": 4,
            "rotating_batch_size": 4,
            "maximum_materialization_attempts": 500,
            "snapshots_per_generation": 3,
        },
    }
    ledger = {"submitted_texts": 0}

    def score(candidate, batch, generation):
        ledger["submitted_texts"] += len(batch)
        pool_indices = np.array([space.legal_single_token_ids.tolist().index(value) for value in candidate.token_ids])
        return _record(candidate, pool_indices, "rotating_minibatch")

    def full_score(candidate, generation):
        ledger["submitted_texts"] += 10
        pool_indices = np.array([space.legal_single_token_ids.tolist().index(value) for value in candidate.token_ids])
        return _record(candidate, pool_indices, "full_search")

    def snapshot(candidate, generation, label, output):
        path = output / "snapshot.json"
        write_json(path, {"candidate_key": candidate.key, "generation": generation, "label": label})
        return [path]

    result = pareto_cem(
        space,
        length=2,
        restart=0,
        task="prefix",
        total_search_texts=10,
        config=config,
        output=tmp_path,
        score=score,
        full_score=full_score,
        snapshot=snapshot,
        query_ledger=lambda: dict(ledger),
    )
    assert result.generations_completed == 3
    assert result.formal_archive
    assert all(record["evaluation_scope"] == "full_search" for record in result.formal_archive)
    for generation in range(3):
        directory = tmp_path / f"generation_{generation:03d}"
        assert validate_completion(
            directory, {"generation": generation, "task": "prefix", "length": 2, "restart": 0}
        )
        for name in (
            "population.csv",
            "pareto_front.json",
            "elites.json",
            "token_frequencies.npz",
            "rng_state.json",
            "batch_manifest.json",
            "distribution.npz",
            "query_ledger.json",
            "resource_usage.json",
            "formal_full_search.json",
        ):
            assert (directory / name).is_file()
    assert validate_completion(tmp_path, {"task": "prefix", "length": 2, "restart": 0, "generations": 3})
