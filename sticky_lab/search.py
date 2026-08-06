"""Gradient-free discrete sequence search with feasibility-first ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .metrics import feasibility_sort_key


ScoreFunction = Callable[[list[tuple[int, ...]], int], list[dict[str, Any]]]


@dataclass
class SearchResult:
    candidates: list[dict[str, Any]]
    history: list[dict[str, Any]]


def _annotate(sequences: Sequence[tuple[int, ...]], metrics: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(sequences) != len(metrics):
        raise ValueError("Scorer returned the wrong number of metric records")
    return [{"sequence": tuple(sequence), **metric} for sequence, metric in zip(sequences, metrics)]


def _rank(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda record: feasibility_sort_key(record, len(record["sequence"])))


def cem_search(
    pool_size: int,
    trigger_length: int,
    score_fn: ScoreFunction,
    *,
    population_size: int,
    elite_ratio: float,
    iterations: int,
    update_alpha: float,
    probability_floor: float,
    seed: int,
) -> SearchResult:
    if pool_size < 2 or trigger_length < 1 or population_size < 2:
        raise ValueError("Invalid CEM dimensions")
    elite_count = max(2, int(round(population_size * elite_ratio)))
    rng = np.random.default_rng(seed)
    probabilities = np.full((trigger_length, pool_size), 1.0 / pool_size, dtype=np.float64)
    archive: dict[tuple[int, ...], dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    best_sequence: tuple[int, ...] | None = None

    for iteration in range(iterations):
        sampled = np.column_stack(
            [rng.choice(pool_size, size=population_size, p=probabilities[position]) for position in range(trigger_length)]
        )
        sequences = [tuple(map(int, row)) for row in sampled]
        if best_sequence is not None:
            sequences[0] = best_sequence
        sequences = list(dict.fromkeys(sequences))
        scored = _rank(_annotate(sequences, score_fn(sequences, iteration)))
        for record in scored:
            sequence = record["sequence"]
            previous = archive.get(sequence)
            if previous is None or feasibility_sort_key(record, len(sequence)) < feasibility_sort_key(previous, len(sequence)):
                archive[sequence] = record
        elites = scored[: min(elite_count, len(scored))]
        empirical = np.zeros_like(probabilities)
        for position in range(trigger_length):
            counts = np.bincount([record["sequence"][position] for record in elites], minlength=pool_size).astype(float)
            empirical[position] = counts / max(counts.sum(), 1.0)
        probabilities = (1.0 - update_alpha) * probabilities + update_alpha * empirical
        probabilities = np.maximum(probabilities, probability_floor)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        best = scored[0]
        best_sequence = tuple(best["sequence"])
        entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300))) / trigger_length)
        history.append(
            {
                "iteration": iteration,
                "unique_population": len(sequences),
                "feasible_population": int(sum(bool(record.get("feasible")) for record in scored)),
                "best_feasible": bool(best.get("feasible")),
                "best_constraint_violation": float(best.get("constraint_violation", float("inf"))),
                "best_objective": float(best.get("objective", -float("inf"))),
                "distribution_entropy": entropy,
                "best_sequence": ",".join(map(str, best_sequence)),
            }
        )
    return SearchResult(_rank(list(archive.values())), history)


def coordinate_beam_search(
    pool_size: int,
    trigger_length: int,
    score_fn: ScoreFunction,
    *,
    beam_width: int,
    replacements_per_position: int,
    iterations: int,
    seed: int,
) -> SearchResult:
    rng = np.random.default_rng(seed)
    initial = [tuple(map(int, rng.integers(0, pool_size, size=trigger_length))) for _ in range(beam_width)]
    beam = _rank(_annotate(initial, score_fn(initial, 0)))[:beam_width]
    archive = {record["sequence"]: record for record in beam}
    history: list[dict[str, Any]] = []
    for iteration in range(iterations):
        position = iteration % trigger_length
        proposals: list[tuple[int, ...]] = [record["sequence"] for record in beam]
        replacements = rng.choice(pool_size, size=min(replacements_per_position, pool_size), replace=False)
        for record in beam:
            for replacement in replacements:
                sequence = list(record["sequence"])
                sequence[position] = int(replacement)
                proposals.append(tuple(sequence))
        proposals = list(dict.fromkeys(proposals))
        scored = _rank(_annotate(proposals, score_fn(proposals, iteration)))
        beam = scored[:beam_width]
        archive.update({record["sequence"]: record for record in scored})
        best = beam[0]
        history.append({"iteration": iteration, "position": position, "proposal_count": len(proposals), "best_feasible": bool(best.get("feasible")), "best_constraint_violation": float(best.get("constraint_violation", float("inf"))), "best_objective": float(best.get("objective", -float("inf"))), "best_sequence": ",".join(map(str, best["sequence"]))})
    return SearchResult(_rank(list(archive.values())), history)


def genetic_search(
    pool_size: int,
    trigger_length: int,
    score_fn: ScoreFunction,
    *,
    population_size: int,
    elite_ratio: float,
    mutation_rate: float,
    iterations: int,
    seed: int,
) -> SearchResult:
    rng = np.random.default_rng(seed)
    population = [tuple(map(int, rng.integers(0, pool_size, size=trigger_length))) for _ in range(population_size)]
    archive: dict[tuple[int, ...], dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    elite_count = max(2, int(round(population_size * elite_ratio)))
    for iteration in range(iterations):
        population = list(dict.fromkeys(population))
        scored = _rank(_annotate(population, score_fn(population, iteration)))
        archive.update({record["sequence"]: record for record in scored})
        elites = scored[: min(elite_count, len(scored))]
        best = elites[0]
        history.append({"iteration": iteration, "unique_population": len(population), "best_feasible": bool(best.get("feasible")), "best_constraint_violation": float(best.get("constraint_violation", float("inf"))), "best_objective": float(best.get("objective", -float("inf"))), "best_sequence": ",".join(map(str, best["sequence"]))})
        next_population = [record["sequence"] for record in elites]
        while len(next_population) < population_size:
            left, right = rng.choice(len(elites), size=2, replace=True)
            split = int(rng.integers(1, trigger_length)) if trigger_length > 1 else 1
            child = list(elites[int(left)]["sequence"][:split] + elites[int(right)]["sequence"][split:])
            for position in range(trigger_length):
                if rng.random() < mutation_rate:
                    child[position] = int(rng.integers(0, pool_size))
            next_population.append(tuple(child))
        population = next_population
    return SearchResult(_rank(list(archive.values())), history)


def run_search(algorithm: str, pool_size: int, trigger_length: int, score_fn: ScoreFunction, config: dict[str, Any], seed: int) -> SearchResult:
    if algorithm == "cem":
        return cem_search(pool_size, trigger_length, score_fn, population_size=int(config["population_size"]), elite_ratio=float(config["elite_ratio"]), iterations=int(config["iterations"]), update_alpha=float(config["update_alpha"]), probability_floor=float(config["probability_floor"]), seed=seed)
    if algorithm == "blackbox_beam":
        return coordinate_beam_search(pool_size, trigger_length, score_fn, beam_width=int(config["beam_width"]), replacements_per_position=int(config["replacements_per_position"]), iterations=int(config["iterations"]), seed=seed)
    if algorithm == "genetic":
        return genetic_search(pool_size, trigger_length, score_fn, population_size=int(config["population_size"]), elite_ratio=float(config["elite_ratio"]), mutation_rate=float(config["mutation_rate"]), iterations=int(config["iterations"]), seed=seed)
    raise ValueError(f"Unknown search algorithm: {algorithm}")

