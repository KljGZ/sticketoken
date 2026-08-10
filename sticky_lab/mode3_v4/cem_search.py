"""Independent categorical CEM using only discrete candidates and score records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .candidate_space import CandidateSpace
from .interfaces import Candidate
from .scoring import ranking_key


ScoreFunction = Callable[[Sequence[Candidate]], list[dict[str, Any]]]


@dataclass(frozen=True)
class CEMResult:
    archive: list[dict[str, Any]]
    history: list[dict[str, Any]]
    proposed: int
    valid_materialized: int


def _draw_population(
    space: CandidateSpace,
    probabilities: np.ndarray,
    population_size: int,
    rng: np.random.Generator,
    maximum_attempts: int,
) -> tuple[list[Candidate], list[tuple[int, ...]], int]:
    by_key: dict[str, tuple[Candidate, tuple[int, ...]]] = {}
    attempts = 0
    length = probabilities.shape[0]
    while len(by_key) < population_size and attempts < maximum_attempts:
        indices = tuple(
            int(rng.choice(space.pool_size, p=probabilities[position])) for position in range(length)
        )
        attempts += 1
        candidate = space.materialize_pool_indices(indices)
        if candidate is not None:
            by_key.setdefault(candidate.key, (candidate, indices))
    if len(by_key) < max(2, population_size // 2):
        raise RuntimeError(
            f"CEM exact materialization collapsed: {len(by_key)}/{population_size} after {attempts} attempts"
        )
    pairs = list(by_key.values())
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs], attempts


def categorical_cem(
    space: CandidateSpace,
    length: int,
    score_fn: ScoreFunction,
    *,
    population_size: int,
    elite_ratio: float,
    iterations: int,
    uniform_mixture: float,
    update_alpha: float,
    archive_size: int,
    maximum_materialization_attempts: int,
    seed: int,
) -> CEMResult:
    """Run one registered length/restart from a fresh uniform distribution."""

    if length < 2:
        raise ValueError("Categorical CEM is registered only for V4 lengths 2..30")
    rng = np.random.default_rng(seed)
    uniform = np.full((length, space.pool_size), 1.0 / space.pool_size, dtype=np.float64)
    probabilities = uniform.copy()
    elite_count = max(2, int(np.ceil(population_size * elite_ratio)))
    archive: dict[str, dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    total_attempts = 0
    total_valid = 0
    for iteration in range(int(iterations)):
        candidates, index_rows, attempts = _draw_population(
            space,
            probabilities,
            population_size,
            rng,
            maximum_materialization_attempts,
        )
        total_attempts += attempts
        total_valid += len(candidates)
        records = score_fn(candidates)
        if len(records) != len(candidates):
            raise RuntimeError("V4 score function changed the candidate count")
        joined = sorted(zip(records, index_rows), key=lambda item: ranking_key(item[0]))
        elites = joined[: min(elite_count, len(joined))]
        for record, _ in joined:
            previous = archive.get(str(record["token_ids"]))
            if previous is None or ranking_key(record) < ranking_key(previous):
                archive[str(record["token_ids"])] = dict(record)
        empirical = np.zeros_like(probabilities)
        for position in range(length):
            counts = np.bincount([indices[position] for _, indices in elites], minlength=space.pool_size)
            empirical[position] = counts / max(int(counts.sum()), 1)
        probabilities = (1.0 - update_alpha) * probabilities + update_alpha * empirical
        probabilities = (1.0 - uniform_mixture) * probabilities + uniform_mixture * uniform
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)), axis=1)
        champion = joined[0][0]
        history.append(
            {
                "iteration": iteration,
                "actual_token_length": length,
                "unique_valid_population": len(candidates),
                "materialization_attempts": attempts,
                "distribution_entropy_mean": float(np.mean(entropy)),
                "distribution_entropy_min": float(np.min(entropy)),
                "best_token_ids": champion["token_ids"],
                "best_constraint_violation": champion["constraint_violation"],
                "best_search_score": champion["search_score"],
            }
        )
    ranked = sorted(archive.values(), key=ranking_key)[: int(archive_size)]
    return CEMResult(ranked, history, total_attempts, total_valid)
