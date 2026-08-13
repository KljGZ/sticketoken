"""Categorical island GA for the physically isolated output-query track."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class GenerationTrace:
    restart: int
    generation: int
    batch_manifest_hash: str
    token_ids: tuple[int, ...]
    scores: tuple[float, ...]
    elite_ids: tuple[int, ...]


def island_categorical_ga(
    legal_token_ids: Sequence[int],
    score: Callable[[Sequence[int], int], Sequence[float]],
    batch_hash: Callable[[int], str],
    *,
    population: int = 512,
    generations: int = 50,
    restarts: int = 20,
    islands: int = 8,
    elite_fraction: float = 0.10,
    uniform_fraction: float = 0.20,
    migration_every: int = 5,
    migration_count: int = 16,
    reference_score: Callable[[Sequence[int]], Sequence[float]] | None = None,
    reference_every: int = 5,
    seed: int = 0,
) -> tuple[list[GenerationTrace], dict[int, float]]:
    """Length-one GA; score callback may only use final encoder vectors."""
    legal = np.asarray(sorted(set(map(int, legal_token_ids))), dtype=np.int64)
    if len(legal) < population or population % islands:
        raise ValueError("legal vocabulary/population/island contract failed")
    rng = np.random.default_rng(seed)
    per_island = population // islands
    elite_n = max(1, int(round(per_island * elite_fraction)))
    uniform_n = max(1, int(round(per_island * uniform_fraction)))
    traces: list[GenerationTrace] = []
    archive: dict[int, float] = {}
    for restart in range(restarts):
        populations = rng.choice(legal, size=(islands, per_island), replace=False)
        for generation in range(generations):
            flat = populations.reshape(-1).tolist()
            values = np.asarray(score(flat, generation), dtype=np.float64)
            if values.shape != (population,) or not np.all(np.isfinite(values)):
                raise RuntimeError("black-box score callback returned invalid values")
            elite_ids: list[int] = []
            next_populations = []
            for island in range(islands):
                ids = populations[island]
                scores = values[island * per_island : (island + 1) * per_island]
                elite = ids[np.argsort(-scores, kind="stable")[:elite_n]]
                elite_ids.extend(map(int, elite))
                inherited = rng.choice(elite, size=per_island - uniform_n, replace=True)
                injected = rng.choice(legal, size=uniform_n, replace=False)
                next_populations.append(np.concatenate([inherited, injected]))
            populations = np.stack(next_populations)
            if migration_every and (generation + 1) % migration_every == 0:
                migrants = np.asarray(sorted(elite_ids)[:migration_count])
                for island in range(islands):
                    count = min(len(migrants), per_island)
                    populations[island, -count:] = np.roll(migrants, island)[:count]
            traces.append(GenerationTrace(restart, generation, batch_hash(generation), tuple(flat), tuple(map(float, values)), tuple(elite_ids)))
            if reference_score is not None and (generation + 1) % int(reference_every) == 0:
                # Only scores on this one immutable reference batch enter the
                # cross-generation/restart archive. Raw rotating-batch scores
                # are preserved in traces but are never compared globally.
                reference_ids = sorted(set(flat + elite_ids))
                reference_values = list(map(float, reference_score(reference_ids)))
                if len(reference_values) != len(reference_ids) or not np.all(np.isfinite(reference_values)):
                    raise RuntimeError("invalid fixed-reference archive scores")
                for token_id, value in zip(reference_ids, reference_values):
                    archive[token_id] = max(value, archive.get(token_id, float("-inf")))
    if reference_score is None:
        raise RuntimeError("formal V6 black-box search requires fixed-reference archive rescoring")
    return traces, archive
