"""Narrow interfaces at the V5 query-only threat-model boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


class TextEmbeddingOracle(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class TokenizerAuditAdapter(Protocol):
    def encode_without_special_tokens(self, text: str) -> tuple[int, ...]: ...
    def decode(self, token_ids: Sequence[int]) -> str: ...
    @property
    def vocab_size(self) -> int: ...
    @property
    def special_token_ids(self) -> frozenset[int]: ...


@dataclass(frozen=True)
class Candidate:
    token_ids: tuple[int, ...]
    trigger: str
    actual_token_length: int
    exact_token_roundtrip: bool

    @property
    def key(self) -> str:
        return ",".join(map(str, self.token_ids))


@dataclass
class ClusterStructure:
    cluster_count: int
    centers: np.ndarray
    radii: np.ndarray
    masses: np.ndarray
    eta: np.ndarray
    assignments: np.ndarray
    inlier_mask: np.ndarray
    radius_quantiles: np.ndarray
    cvar90: np.ndarray
    coverage: float
    outlier_rate: float
    cmax: float
    cavg: float
    occupancy: np.ndarray
    occupancy_ucb: np.ndarray
    occupancy_auc: float
    lambda_star: float

    def summary(self) -> dict[str, object]:
        return {
            "cluster_count": int(self.cluster_count),
            "radii": self.radii.tolist(),
            "masses": self.masses.tolist(),
            "eta": self.eta.tolist(),
            "radius_quantiles": self.radius_quantiles.tolist(),
            "cvar90": self.cvar90.tolist(),
            "coverage": float(self.coverage),
            "outlier_rate": float(self.outlier_rate),
            "cmax": float(self.cmax),
            "cavg": float(self.cavg),
            "occupancy": self.occupancy.tolist(),
            "occupancy_ucb": self.occupancy_ucb.tolist(),
            "occupancy_auc": float(self.occupancy_auc),
            "lambda_star": float(self.lambda_star),
        }
