"""Embedding-output-only candidate scoring for V4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .insertion import POSITIONS, insert_many
from .interfaces import Candidate, TextEmbeddingOracle
from .metrics import evaluate_geometry, fixed_pair_indices, search_quality
from .occupancy import evaluate_occupancy
from .support import SupportModel


@dataclass
class CandidateScorer:
    oracle: TextEmbeddingOracle
    texts: list[str]
    original_embeddings: np.ndarray
    normal_probe: np.ndarray
    support: SupportModel
    constraints: Mapping[str, float]
    weights: Mapping[str, float]
    occupancy_lambdas: Sequence[float]
    confidence: float
    task: str
    insertion_seed: int
    separator: str
    pair_sample_count: int
    candidate_chunk_size: int

    def __post_init__(self) -> None:
        if self.task not in {*POSITIONS, "universal"}:
            raise ValueError(f"Unknown V4 task: {self.task}")
        if len(self.texts) != len(self.original_embeddings):
            raise ValueError("Scoring texts and embeddings must align")
        self.positions = list(POSITIONS) if self.task == "universal" else [self.task]
        self.pairs = fixed_pair_indices(len(self.texts), self.pair_sample_count, self.insertion_seed + 77)

    def evaluate(self, candidates: Sequence[Candidate], *, include_center: bool = False) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for start in range(0, len(candidates), max(1, self.candidate_chunk_size)):
            chunk = list(candidates[start : start + max(1, self.candidate_chunk_size)])
            matrices: dict[tuple[int, str], np.ndarray] = {}
            flattened: list[str] = []
            keys: list[tuple[int, str]] = []
            for index, candidate in enumerate(chunk):
                for position_offset, position in enumerate(self.positions):
                    keys.append((index, position))
                    flattened.extend(
                        insert_many(
                            self.texts,
                            candidate.trigger,
                            position,
                            seed=self.insertion_seed + position_offset * 1000000,
                            separator=self.separator,
                        )
                    )
            encoded = self.oracle.encode(flattened)
            width = len(self.texts)
            cursor = 0
            for key in keys:
                matrices[key] = encoded[cursor : cursor + width]
                cursor += width
            for index, candidate in enumerate(chunk):
                triggered = [matrices[(index, position)] for position in self.positions]
                geometry = evaluate_geometry(
                    self.original_embeddings,
                    triggered,
                    pair_indices=self.pairs,
                )
                margin = self.support.support_in_margin(geometry.center)
                occupancy = evaluate_occupancy(
                    geometry.center,
                    geometry.compact_radius_q95,
                    self.normal_probe,
                    self.support,
                    self.occupancy_lambdas,
                    confidence=self.confidence,
                )
                score, violation = search_quality(
                    geometry,
                    margin,
                    occupancy,
                    self.constraints,
                    self.weights,
                )
                record: dict[str, Any] = {
                    "token_ids": candidate.key,
                    "trigger": candidate.trigger,
                    "actual_token_length": candidate.actual_token_length,
                    "exact_token_roundtrip": candidate.exact_token_roundtrip,
                    "task": self.task,
                    "search_score": score,
                    "constraint_violation": violation,
                    "support_in_margin": margin,
                    **geometry.to_dict(),
                    **occupancy.to_dict(),
                    **self.support.cluster_diagnostics(geometry.center),
                }
                if include_center:
                    record["center"] = geometry.center
                    record["triggered_by_position"] = triggered
                records.append(record)
        return records


def ranking_key(record: Mapping[str, Any]) -> tuple[float, float, float, str]:
    """Feasibility first, then registered geometric score and deterministic ID."""

    return (
        float(record["constraint_violation"]),
        -float(record["search_score"]),
        float(record["compact_radius_q95"]),
        str(record["token_ids"]),
    )
