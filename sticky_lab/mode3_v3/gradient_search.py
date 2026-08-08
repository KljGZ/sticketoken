"""HotFlip-guided multi-coordinate beam refinement for Mode 3 V3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .soft_prompt import differentiable_objective, encode_with_prompt_embeddings, _modules
from .support import BenignSupportModel


HardScore = Callable[[list[tuple[int, ...]], int], list[dict[str, Any]]]
SortKey = Callable[[dict[str, Any]], tuple[float, ...]]


@dataclass
class GradientSearchResult:
    candidates: list[dict[str, Any]]
    history: list[dict[str, Any]]


def hotflip_candidates(
    encoder,
    texts: Sequence[str],
    original: np.ndarray,
    support: BenignSupportModel,
    sequence: tuple[int, ...],
    legal_token_ids: Sequence[int],
    *,
    position: str,
    subprotocol: str,
    temperature: float,
    top_m: int,
    insertion_seed: int,
) -> list[np.ndarray]:
    auto_model, _ = _modules(encoder)
    weight = auto_model.get_input_embeddings().weight
    ids = torch.tensor(sequence, dtype=torch.long, device=weight.device)
    prompt = weight[ids].detach().clone().requires_grad_(True)
    triggered = encode_with_prompt_embeddings(
        encoder,
        texts,
        prompt,
        position=position,
        random_trigger=encoder.decode(sequence) if position == "random" else None,
        insertion_seed=insertion_seed,
    )
    objective, _ = differentiable_objective(
        torch.tensor(original, dtype=torch.float32, device=weight.device),
        triggered,
        torch.tensor(support.memory, dtype=torch.float32, device=weight.device),
        subprotocol=subprotocol,
        temperature=temperature,
    )
    (-objective).backward()
    legal = torch.tensor(list(map(int, legal_token_ids)), dtype=torch.long, device=weight.device)
    legal_embeddings = weight[legal].detach()
    output: list[np.ndarray] = []
    for coordinate in range(len(sequence)):
        # First-order loss change: (e_v - e_current)^T grad.  The current
        # constant does not affect ranking, so choose the smallest dot product.
        scores = legal_embeddings @ prompt.grad[coordinate]
        count = min(top_m, len(legal))
        indices = torch.topk(scores, k=count, largest=False).indices
        output.append(legal[indices].detach().cpu().numpy())
    return output


def gradient_beam_search(
    encoder,
    texts: Sequence[str],
    original: np.ndarray,
    support: BenignSupportModel,
    initial_sequences: Sequence[tuple[int, ...]],
    legal_token_ids: Sequence[int],
    hard_score_fn: HardScore,
    *,
    sort_key: SortKey,
    position: str,
    subprotocol: str,
    gradient_top_m: int,
    beam_width: int,
    candidate_batch: int,
    iterations: int,
    temperature: float,
    seed: int,
    insertion_seed: int,
) -> GradientSearchResult:
    rng = np.random.default_rng(seed)
    sequences = list(dict.fromkeys(tuple(map(int, row)) for row in initial_sequences))
    if not sequences:
        raise ValueError("Gradient search requires at least one initial sequence")
    ranked = sorted(
        [{"sequence": sequence, **score} for sequence, score in zip(sequences, hard_score_fn(sequences, -1))],
        key=sort_key,
    )
    beam = ranked[:beam_width]
    archive = {tuple(record["sequence"]): record for record in beam}
    history: list[dict[str, Any]] = []
    for iteration in range(iterations):
        anchor = tuple(beam[0]["sequence"])
        proposals = hotflip_candidates(
            encoder,
            texts,
            original,
            support,
            anchor,
            legal_token_ids,
            position=position,
            subprotocol=subprotocol,
            temperature=temperature,
            top_m=gradient_top_m,
            insertion_seed=insertion_seed,
        )
        if iteration < iterations / 3:
            coordinate_count = min(4, len(anchor))
        elif iteration < 2 * iterations / 3:
            coordinate_count = min(2, len(anchor))
        else:
            coordinate_count = 1
        candidates = [tuple(record["sequence"]) for record in beam]
        while len(candidates) < candidate_batch:
            base = list(tuple(beam[int(rng.integers(0, len(beam)))]["sequence"]))
            coordinates = rng.choice(len(base), size=coordinate_count, replace=False)
            for coordinate in coordinates:
                choices = proposals[int(coordinate)]
                base[int(coordinate)] = int(choices[int(rng.integers(0, len(choices)))])
            candidates.append(tuple(base))
        candidates = list(dict.fromkeys(candidates))
        records = sorted(
            [{"sequence": sequence, **score} for sequence, score in zip(candidates, hard_score_fn(candidates, iteration))],
            key=sort_key,
        )
        for record in records:
            sequence = tuple(record["sequence"])
            previous = archive.get(sequence)
            if previous is None or sort_key(record) < sort_key(previous):
                archive[sequence] = record
        beam = records[:beam_width]
        history.append(
            {
                "iteration": iteration,
                "coordinate_count": coordinate_count,
                "candidate_count": len(candidates),
                "best_sequence": ",".join(map(str, beam[0]["sequence"])),
                "best_separator_certified": bool(beam[0].get("separator_certified", False)),
                "best_blank_certified": bool(beam[0].get("blank_region_certified", False)),
                "best_separation_margin": float(beam[0].get("separation_margin", -float("inf"))),
                "best_sample_blank_margin": float(beam[0].get("sample_blank_margin", -float("inf"))),
            }
        )
    return GradientSearchResult(sorted(archive.values(), key=sort_key), history)
