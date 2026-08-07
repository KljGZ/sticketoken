"""V3 CEM with full-search feedback and separated formal archive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


ScoreFunction = Callable[[list[tuple[int, ...]], int], list[dict[str, Any]]]
SortKey = Callable[[dict[str, Any]], tuple[float, ...]]


@dataclass
class CEMResult:
    candidates: list[dict[str, Any]]
    history: list[dict[str, Any]]
    full_champion_history: list[dict[str, Any]]


def _annotate(sequences: Sequence[tuple[int, ...]], metrics: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(sequences) != len(metrics):
        raise ValueError("Scorer returned a different record count")
    return [{"sequence": tuple(sequence), **metric} for sequence, metric in zip(sequences, metrics)]


def _diverse(
    ranked: Sequence[dict[str, Any]],
    count: int,
    minimum_hamming: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for record in ranked:
        sequence = tuple(record["sequence"])
        if all(sum(a != b for a, b in zip(sequence, other["sequence"])) >= minimum_hamming for other in selected):
            selected.append(record)
        if len(selected) == count:
            break
    used = {tuple(record["sequence"]) for record in selected}
    selected.extend(record for record in ranked if tuple(record["sequence"]) not in used)
    return selected[:count]


def cem_search(
    pool_size: int,
    trigger_length: int,
    dynamic_score_fn: ScoreFunction,
    full_score_fn: ScoreFunction,
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
    full_evaluation_interval: int,
    archive_size: int,
    seed: int,
    initial_sequences: Sequence[tuple[int, ...]] | None = None,
) -> CEMResult:
    if pool_size < 2 or trigger_length < 1 or population_size < 2:
        raise ValueError("Invalid CEM dimensions")
    rng = np.random.default_rng(seed)
    elite_count = max(2, int(round(population_size * elite_ratio)))
    uniform = np.full((trigger_length, pool_size), 1.0 / pool_size, dtype=np.float64)
    probabilities = uniform.copy()
    warm = [tuple(map(int, value)) for value in (initial_sequences or []) if len(value) == trigger_length]
    if warm:
        empirical = np.zeros_like(probabilities)
        for position in range(trigger_length):
            counts = np.bincount([row[position] for row in warm], minlength=pool_size)
            empirical[position] = counts / max(counts.sum(), 1)
        probabilities = 0.5 * probabilities + 0.5 * empirical
    formal_archive: dict[tuple[int, ...], dict[str, Any]] = {}
    dynamic_archive: dict[tuple[int, ...], dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    champion_history: list[dict[str, Any]] = []
    champion: dict[str, Any] | None = None
    proposal_best: tuple[int, ...] | None = None
    proposal_key: tuple[float, ...] | None = None
    stalled = 0
    min_entropy = entropy_min_fraction * np.log(pool_size)
    for iteration in range(iterations):
        sampled = np.column_stack(
            [rng.choice(pool_size, size=population_size, p=probabilities[position]) for position in range(trigger_length)]
        )
        sequences = [tuple(map(int, row)) for row in sampled]
        if proposal_best is not None:
            sequences[0] = proposal_best
        if iteration == 0:
            for index, sequence in enumerate(warm[: max(0, population_size // 2)]):
                sequences[index] = sequence
        sequences = list(dict.fromkeys(sequences))
        dynamic_ranked = sorted(_annotate(sequences, dynamic_score_fn(sequences, iteration)), key=sort_key)
        for record in dynamic_ranked:
            sequence = tuple(record["sequence"])
            previous = dynamic_archive.get(sequence)
            if previous is None or sort_key(record) < sort_key(previous):
                dynamic_archive[sequence] = record
        hamming = max(1, int(round(trigger_length * elite_min_hamming_fraction)))
        dynamic_elites = _diverse(dynamic_ranked, min(elite_count, len(dynamic_ranked)), hamming)
        checkpoint = iteration % max(1, full_evaluation_interval) == 0 or iteration == iterations - 1
        elites_for_update = dynamic_elites
        checkpoint_best = None
        if checkpoint:
            candidate_sequences = [tuple(record["sequence"]) for record in dynamic_elites]
            # Include the current cumulative champion and strongest diverse
            # temporary-archive members, but never promote their mini-batch
            # scores directly into the formal archive.
            if champion is not None:
                candidate_sequences.append(tuple(champion["sequence"]))
            temporary_ranked = sorted(dynamic_archive.values(), key=sort_key)
            candidate_sequences.extend(tuple(record["sequence"]) for record in temporary_ranked[:elite_count])
            candidate_sequences = list(dict.fromkeys(candidate_sequences))
            full_ranked = sorted(_annotate(candidate_sequences, full_score_fn(candidate_sequences, iteration)), key=sort_key)
            elites_for_update = _diverse(full_ranked, min(elite_count, len(full_ranked)), hamming)
            for record in full_ranked:
                sequence = tuple(record["sequence"])
                previous = formal_archive.get(sequence)
                if previous is None or sort_key(record) < sort_key(previous):
                    formal_archive[sequence] = record
            checkpoint_best = full_ranked[0]
            if champion is None or sort_key(checkpoint_best) < sort_key(champion):
                champion = checkpoint_best
            champion_history.append(
                {
                    "iteration": iteration,
                    "sequence": tuple(champion["sequence"]),
                    **{key: value for key, value in champion.items() if key != "sequence"},
                }
            )

        empirical = np.zeros_like(probabilities)
        for position in range(trigger_length):
            counts = np.bincount([record["sequence"][position] for record in elites_for_update], minlength=pool_size)
            empirical[position] = counts / max(counts.sum(), 1)
        probabilities = (1.0 - update_alpha) * probabilities + update_alpha * empirical
        probabilities = (1.0 - uniform_mixture) * probabilities + uniform_mixture * uniform
        probabilities = np.maximum(probabilities, probability_floor)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        entropies = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)), axis=1)
        low_entropy = np.flatnonzero(entropies < min_entropy)
        if len(low_entropy):
            mix = min(0.50, 2.0 * uniform_mixture)
            probabilities[low_entropy] = (1.0 - mix) * probabilities[low_entropy] + mix * uniform[low_entropy]
            probabilities[low_entropy] /= probabilities[low_entropy].sum(axis=1, keepdims=True)
        current = checkpoint_best or dynamic_ranked[0]
        current_key = sort_key(current)
        if proposal_key is None or current_key < proposal_key:
            proposal_key = current_key
            proposal_best = tuple(current["sequence"])
            stalled = 0
        else:
            stalled += 1
        restarted: list[int] = []
        if stalled >= stall_patience:
            count = max(1, trigger_length // 2)
            restarted = sorted(map(int, rng.choice(trigger_length, size=count, replace=False)))
            probabilities[restarted] = uniform[restarted]
            stalled = 0
        history.append(
            {
                "iteration": iteration,
                "full_search_evaluated": checkpoint,
                "unique_population": len(sequences),
                "formal_archive_size": len(formal_archive),
                "temporary_archive_size": len(dynamic_archive),
                "elite_count_for_update": len(elites_for_update),
                "distribution_entropy_mean": float(entropies.mean()),
                "distribution_entropy_min": float(entropies.min()),
                "stall_restart_positions": ",".join(map(str, restarted)),
                "best_sequence": ",".join(map(str, current["sequence"])),
                "cumulative_full_sequence": "" if champion is None else ",".join(map(str, champion["sequence"])),
                "best_separator_certified": bool(current.get("separator_certified", False)),
                "best_blank_certified": bool(current.get("blank_region_certified", False)),
                "best_separation_margin": float(current.get("separation_margin", -float("inf"))),
                "best_sample_blank_margin": float(current.get("sample_blank_margin", -float("inf"))),
                "best_compact_radius_q95": float(current.get("compact_radius_q95", float("inf"))),
            }
        )
    # Mandatory end-of-length re-evaluation of the complete formal archive.
    final_sequences = list(formal_archive)
    if proposal_best is not None:
        final_sequences.append(proposal_best)
    final_sequences = list(dict.fromkeys(final_sequences))
    final_ranked = sorted(_annotate(final_sequences, full_score_fn(final_sequences, iterations)), key=sort_key)
    return CEMResult(final_ranked[:archive_size], history, champion_history)


def expand_warm_sequences(
    previous: Sequence[tuple[int, ...]],
    new_length: int,
    pool_size: int,
    count: int,
    seed: int,
) -> list[tuple[int, ...]]:
    if not previous or new_length <= len(previous[0]):
        return []
    rng = np.random.default_rng(seed)
    output: list[tuple[int, ...]] = []
    while len(output) < count:
        sequence = list(previous[int(rng.integers(0, len(previous)))])
        while len(sequence) < new_length:
            sequence.insert(int(rng.integers(0, len(sequence) + 1)), int(rng.integers(0, pool_size)))
        output.append(tuple(sequence))
    return list(dict.fromkeys(output))
