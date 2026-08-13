"""Matched semantic controls separating lexical semantics from encoder anomaly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


MATCH_FIELDS = (
    "frequency", "idf", "pos", "semantic_category", "character_length",
    "casing", "input_embedding_norm", "naturalness",
)


@dataclass(frozen=True)
class TokenMetadata:
    token_id: int
    frequency: float
    idf: float
    pos: str
    semantic_category: str
    character_length: int
    casing: str
    input_embedding_norm: float
    naturalness: float


def _distance(candidate: TokenMetadata, control: TokenMetadata, scales: Mapping[str, float]) -> float:
    numeric = ("frequency", "idf", "character_length", "input_embedding_norm", "naturalness")
    result = 0.0
    for field in numeric:
        result += abs(float(getattr(candidate, field)) - float(getattr(control, field))) / max(float(scales.get(field, 1.0)), 1e-12)
    for field in ("pos", "semantic_category", "casing"):
        result += 0.0 if getattr(candidate, field) == getattr(control, field) else 2.0
    return result


def match_controls(candidate: TokenMetadata, pool: Sequence[TokenMetadata], count: int = 50) -> list[TokenMetadata]:
    eligible = [item for item in pool if item.token_id != candidate.token_id]
    if len(eligible) < count:
        raise RuntimeError(f"only {len(eligible)} semantic controls; V6 requires {count}")
    scales = {}
    for field in ("frequency", "idf", "character_length", "input_embedding_norm", "naturalness"):
        values = np.asarray([float(getattr(item, field)) for item in eligible])
        scales[field] = float(np.subtract(*np.quantile(values, [0.75, 0.25]))) or 1.0
    return sorted(eligible, key=lambda item: (_distance(candidate, item, scales), item.token_id))[:count]


def additive_semantic_residual(clean: np.ndarray, triggered: np.ndarray, token_direction: np.ndarray) -> dict[str, float]:
    """Fit z_trigger ~= normalize(z_clean + beta*v_token) and report residual."""
    clean = np.asarray(clean, dtype=np.float64)
    triggered = np.asarray(triggered, dtype=np.float64)
    direction = np.asarray(token_direction, dtype=np.float64).reshape(-1)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    deltas = triggered - clean
    beta = float(np.mean(deltas @ direction))
    predicted = clean + beta * direction
    predicted /= np.maximum(np.linalg.norm(predicted, axis=1, keepdims=True), 1e-12)
    residual = np.arccos(np.clip(np.sum(predicted * triggered, axis=1), -1.0, 1.0))
    return {
        "beta": beta,
        "median_angular_residual_radians": float(np.median(residual)),
        "q90_angular_residual_radians": float(np.quantile(residual, 0.90)),
    }

def wrapper_counterfactuals(token_text: str) -> dict[str, str]:
    return {
        "plain": token_text,
        "quoted": f'"{token_text}"',
        "parenthesized": f"({token_text})",
        "sentence_wrapper": f"The term is {token_text}.",
        "casefolded": token_text.casefold(),
    }
