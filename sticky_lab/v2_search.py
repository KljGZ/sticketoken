"""Diversity-preserving multi-restart CEM used by the V2 length search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


ScoreFunction = Callable[[list[tuple[int, ...]], int], list[dict[str, Any]]]
SortKey = Callable[[dict[str, Any]], tuple[float, ...]]


@dataclass
class V2SearchResult:
    candidates: list[dict[str, Any]]
    history: list[dict[str, Any]]


def _annotate(sequences: Sequence[tuple[int, ...]], metrics: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(sequences) != len(metrics):
        raise ValueError("Scorer returned the wrong number of records")
    return [{"sequence": tuple(sequence), **metric} for sequence, metric in zip(sequences, metrics)]


def _diverse_elites(
    ranked: Sequence[dict[str, Any]],
    count: int,
    minimum_hamming: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for record in ranked:
        sequence = tuple(record["sequence"])
        if all(sum(a != b for a, b in zip(sequence, other["sequence"])) >= minimum_hamming for other in selected):
            selected.append(record)
        if len(selected) >= count:
            return selected
    used = {tuple(record["sequence"]) for record in selected}
    selected.extend(record for record in ranked if tuple(record["sequence"]) not in used)
    return selected[:count]


def cem_search_v2(
    pool_size: int,
    trigger_length: int,
    score_fn: ScoreFunction,
    *,
    sort_key: SortKey,
    population_size: int,
    elite_ratio: float,
    iterations: int,
    update_alpha: float,
    probability_floor: float,
    uniform_mixture: float,
    entropy_min_fraction: float,
    stall_patience: int,
    elite_min_hamming_fraction: float,
    seed: int,
    full_score_fn: ScoreFunction | None = None,
    full_evaluation_interval: int = 5,
    initial_sequences: Sequence[tuple[int, ...]] | None = None,
) -> V2SearchResult:
    if pool_size < 2 or trigger_length < 1 or population_size < 2:
        raise ValueError("Invalid CEM dimensions")
    rng = np.random.default_rng(seed)
    elite_count = max(2, int(round(population_size * elite_ratio)))
    uniform = np.full((trigger_length, pool_size), 1.0 / pool_size, dtype=np.float64)
    probabilities = uniform.copy()
    warm = [tuple(map(int, sequence)) for sequence in (initial_sequences or []) if len(sequence) == trigger_length]
    if warm:
        empirical = np.zeros_like(probabilities)
        for position in range(trigger_length):
            counts = np.bincount([sequence[position] for sequence in warm], minlength=pool_size).astype(float)
            empirical[position] = counts / max(counts.sum(), 1.0)
        probabilities = 0.5 * uniform + 0.5 * empirical
    archive: dict[tuple[int, ...], dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    best_sequence: tuple[int, ...] | None = None
    best_key: tuple[float, ...] | None = None
    stalled = 0
    min_entropy = entropy_min_fraction * np.log(pool_size)

    for iteration in range(iterations):
        sampled = np.column_stack(
            [rng.choice(pool_size, size=population_size, p=probabilities[position]) for position in range(trigger_length)]
        )
        sequences = [tuple(map(int, row)) for row in sampled]
        if best_sequence is not None:
            sequences[0] = best_sequence
        if iteration == 0 and warm:
            for index, sequence in enumerate(warm[: max(0, population_size // 2)]):
                sequences[index] = sequence
        sequences = list(dict.fromkeys(sequences))
        ranked = sorted(_annotate(sequences, score_fn(sequences, iteration)), key=sort_key)
        elite_hamming = max(1, int(round(trigger_length * elite_min_hamming_fraction)))
        elites = _diverse_elites(ranked, min(elite_count, len(ranked)), elite_hamming)

        # Dynamic batches drive exploration.  At registered checkpoints the
        # elite set is re-measured on the complete search split; those full
        # records must drive both the persistent archive *and* the probability
        # update.  V2 previously archived the full scores but still updated the
        # distribution from stale mini-batch elites.
        full_evaluated = bool(
            full_score_fn is not None
            and (iteration % max(full_evaluation_interval, 1) == 0 or iteration == iterations - 1)
        )
        archive_records: list[dict[str, Any]] = []
        elites_for_update = elites
        current = ranked[0]
        if full_evaluated:
            elite_sequences = [tuple(record["sequence"]) for record in elites]
            full_ranked = sorted(
                _annotate(elite_sequences, full_score_fn(elite_sequences, iteration)),
                key=sort_key,
            )
            archive_records = full_ranked
            elites_for_update = _diverse_elites(
                full_ranked,
                min(elite_count, len(full_ranked)),
                elite_hamming,
            )
            current = full_ranked[0]
        for record in archive_records:
            sequence = tuple(record["sequence"])
            previous = archive.get(sequence)
            if previous is None or sort_key(record) < sort_key(previous):
                archive[sequence] = record

        empirical = np.zeros_like(probabilities)
        for position in range(trigger_length):
            counts = np.bincount(
                [record["sequence"][position] for record in elites_for_update],
                minlength=pool_size,
            ).astype(float)
            empirical[position] = counts / max(counts.sum(), 1.0)
        probabilities = (1.0 - update_alpha) * probabilities + update_alpha * empirical
        probabilities = (1.0 - uniform_mixture) * probabilities + uniform_mixture * uniform
        probabilities = np.maximum(probabilities, probability_floor)
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        entropies = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)), axis=1)
        low_entropy_positions = np.flatnonzero(entropies < min_entropy)
        if len(low_entropy_positions):
            adaptive_mix = min(0.50, uniform_mixture * 2.0)
            probabilities[low_entropy_positions] = (
                (1.0 - adaptive_mix) * probabilities[low_entropy_positions]
                + adaptive_mix * uniform[low_entropy_positions]
            )
            probabilities[low_entropy_positions] /= probabilities[low_entropy_positions].sum(axis=1, keepdims=True)

        current_key = sort_key(current)
        if best_key is None or current_key < best_key:
            best_key = current_key
            best_sequence = tuple(current["sequence"])
            stalled = 0
        else:
            stalled += 1
        restarted_positions: list[int] = []
        if stalled >= stall_patience:
            count = max(1, trigger_length // 2)
            restarted_positions = sorted(map(int, rng.choice(trigger_length, size=count, replace=False)))
            probabilities[restarted_positions] = uniform[restarted_positions]
            stalled = 0

        history.append(
            {
                "iteration": iteration,
                "full_search_evaluated": full_evaluated,
                "unique_population": len(sequences),
                "feasible_population": int(sum(bool(record.get("core_feasible", record.get("feasible"))) for record in ranked)),
                "elite_count": len(elites_for_update),
                "elite_unique_count": len({tuple(record["sequence"]) for record in elites_for_update}),
                "best_feasible": bool(current.get("core_feasible", current.get("feasible"))),
                "best_constraint_violation": float(current.get("constraint_violation", float("inf"))),
                "best_objective": float(current.get("objective", -float("inf"))),
                "distribution_entropy_mean": float(np.mean(entropies)),
                "distribution_entropy_min": float(np.min(entropies)),
                "low_entropy_position_count": int(len(low_entropy_positions)),
                "stall_restart_positions": ",".join(map(str, restarted_positions)),
                "best_sequence": ",".join(map(str, current["sequence"])),
            }
        )

    # Ensure the final dynamic best is represented even when it fell between
    # full-search checkpoints.
    if best_sequence is not None and full_score_fn is not None:
        final = _annotate([best_sequence], full_score_fn([best_sequence], iterations - 1))[0]
        archive[best_sequence] = final
    return V2SearchResult(sorted(archive.values(), key=sort_key), history)


def expand_warm_sequences(
    previous: Sequence[tuple[int, ...]],
    new_length: int,
    pool_size: int,
    count: int,
    seed: int,
) -> list[tuple[int, ...]]:
    """Grow prior-length elites without imposing a prefix constraint."""
    if not previous or new_length <= len(previous[0]):
        return []
    rng = np.random.default_rng(seed)
    output: list[tuple[int, ...]] = []
    while len(output) < count:
        base = list(previous[int(rng.integers(0, len(previous)))])
        while len(base) < new_length:
            position = int(rng.integers(0, len(base) + 1))
            base.insert(position, int(rng.integers(0, pool_size)))
        output.append(tuple(base))
    return list(dict.fromkeys(output))
