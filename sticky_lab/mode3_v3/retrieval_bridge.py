"""Representation-to-retrieval anchor bridge for frozen V3 strings."""

from __future__ import annotations

from typing import Any

import numpy as np

from .support import normalize_rows


def optimize_anchor(
    benign: np.ndarray,
    triggered: np.ndarray,
    *,
    iterations: int = 300,
    learning_rate: float = 0.05,
    temperature: float = 0.02,
    seed: int = 0,
) -> dict[str, Any]:
    """Optimize a continuous unit retrieval anchor with smooth tail margins."""
    import torch

    del seed  # optimization is deterministic from the registered initialization
    benign_values = normalize_rows(benign).astype(np.float32)
    trigger_values = normalize_rows(triggered).astype(np.float32)
    initial = trigger_values.mean(axis=0) - benign_values.mean(axis=0)
    initial /= max(float(np.linalg.norm(initial)), 1e-12)
    vector = torch.nn.Parameter(torch.tensor(initial, dtype=torch.float32))
    benign_tensor = torch.tensor(benign_values)
    trigger_tensor = torch.tensor(trigger_values)
    optimizer = torch.optim.Adam([vector], lr=learning_rate)
    history: list[float] = []
    for _ in range(iterations):
        optimizer.zero_grad()
        unit = torch.nn.functional.normalize(vector, dim=0)
        positive = trigger_tensor @ unit
        negative = benign_tensor @ unit
        soft_min = -temperature * torch.logsumexp(-positive / temperature, dim=0) + temperature * np.log(len(positive))
        soft_max = temperature * torch.logsumexp(negative / temperature, dim=0) - temperature * np.log(len(negative))
        loss = -(soft_min - soft_max)
        loss.backward()
        optimizer.step()
        history.append(float(-loss.detach()))
    anchor = torch.nn.functional.normalize(vector.detach(), dim=0).cpu().numpy()
    positive = trigger_values @ anchor
    negative = benign_values @ anchor
    margin = float(np.quantile(positive, 0.05) - np.quantile(negative, 0.95))
    return {
        "anchor": anchor,
        "anchor_margin": margin,
        "triggered_score_q05": float(np.quantile(positive, 0.05)),
        "benign_score_q95": float(np.quantile(negative, 0.95)),
        "top_k_oracle_coverage": float(np.mean(positive > np.quantile(negative, 0.95))),
        "retrieval_anchor_certified": bool(margin > 0.0),
        "optimization_objective_initial": history[0],
        "optimization_objective_final": history[-1],
        "realizability_scope": "continuous_embedding_anchor_only",
    }
