"""Continuous soft-prompt feasibility upper bound for Mode 3 V3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from ..insertion import random_insertion_character_index
from .metrics import evaluate_mode3
from .support import BenignSupportModel


def _modules(encoder):
    modules = list(encoder.model._modules.values())
    if not modules or not hasattr(modules[0], "auto_model"):
        raise TypeError("Soft prompts require a SentenceTransformer Transformer first module")
    return modules[0].auto_model, modules[1:]


def _insertion_index(
    ids: list[int],
    tokenizer,
    position: str,
    *,
    text: str | None = None,
    trigger: str | None = None,
    seed: int = 0,
    offsets: Sequence[Sequence[int]] | None = None,
) -> int:
    if position == "prefix":
        leading = {value for value in (getattr(tokenizer, "cls_token_id", None), getattr(tokenizer, "bos_token_id", None)) if value is not None}
        return 1 if ids and ids[0] in leading else 0
    if position == "suffix":
        trailing = {
            value
            for value in (
                getattr(tokenizer, "eos_token_id", None),
                getattr(tokenizer, "sep_token_id", None),
            )
            if value is not None
        }
        return len(ids) - 1 if ids and ids[-1] in trailing else len(ids)
    if position != "random":
        raise ValueError("Continuous prompts support prefix, suffix, or random positions")
    if text is None or trigger is None or offsets is None:
        raise ValueError("Random continuous-prompt insertion requires text, trigger, and token offsets")
    character_index = random_insertion_character_index(text, trigger, seed)
    for token_index, pair in enumerate(offsets):
        start, end = map(int, pair)
        # Special tokens have an empty (0, 0) span.  The first content token
        # whose span extends beyond the boundary is the token before which the
        # prompt is inserted.
        if end > start and end > character_index:
            return token_index
    return _insertion_index(ids, tokenizer, "suffix")


def encode_with_prompt_embeddings(
    encoder,
    texts: Sequence[str],
    prompt_embeddings: torch.Tensor,
    *,
    position: str,
    random_trigger: str | None = None,
    insertion_seed: int = 0,
) -> torch.Tensor:
    """Run the registered transformer, pooling tail, and final normalization."""
    auto_model, tail = _modules(encoder)
    encoded = encoder.tokenizer(
        list(texts),
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_offsets_mapping=position == "random",
    )
    tokenized = encoded["input_ids"]
    offset_rows = encoded.get("offset_mapping", [None] * len(tokenized))
    embedding_layer = auto_model.get_input_embeddings()
    rows: list[torch.Tensor] = []
    for text, ids, offsets in zip(texts, tokenized, offset_rows):
        token_ids = torch.tensor(ids, dtype=torch.long, device=prompt_embeddings.device)
        base = embedding_layer(token_ids)
        index = _insertion_index(
            list(map(int, ids)),
            encoder.tokenizer,
            position,
            text=str(text),
            trigger=random_trigger,
            seed=insertion_seed,
            offsets=offsets,
        )
        rows.append(torch.cat([base[:index], prompt_embeddings, base[index:]], dim=0))
    maximum = max(len(row) for row in rows)
    dimension = rows[0].shape[1]
    inputs = torch.zeros((len(rows), maximum, dimension), dtype=rows[0].dtype, device=rows[0].device)
    mask = torch.zeros((len(rows), maximum), dtype=torch.long, device=rows[0].device)
    for index, row in enumerate(rows):
        inputs[index, : len(row)] = row
        mask[index, : len(row)] = 1
    outputs = auto_model(inputs_embeds=inputs, attention_mask=mask, return_dict=True)
    features: dict[str, torch.Tensor] = {
        "token_embeddings": outputs.last_hidden_state,
        "attention_mask": mask,
    }
    for module in tail:
        features = module(features)
    sentence = features["sentence_embedding"]
    return torch.nn.functional.normalize(sentence, dim=1)


def _soft_max(values: torch.Tensor, temperature: float) -> torch.Tensor:
    return temperature * torch.logsumexp(values / temperature, dim=0) - temperature * np.log(len(values))


def _soft_min(values: torch.Tensor, temperature: float) -> torch.Tensor:
    return -_soft_max(-values, temperature)


def differentiable_objective(
    original: torch.Tensor,
    triggered: torch.Tensor,
    support_memory: torch.Tensor,
    *,
    subprotocol: str,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    center = torch.nn.functional.normalize(triggered.mean(dim=0), dim=0)
    radii = torch.linalg.vector_norm(triggered - center[None, :], dim=1)
    compact = _soft_max(radii, temperature)
    support_distance = torch.linalg.vector_norm(support_memory - center[None, :], dim=1)
    blank = _soft_min(support_distance, temperature) - compact
    direction = torch.nn.functional.normalize(triggered.mean(dim=0) - original.mean(dim=0), dim=0)
    separation = _soft_min(triggered @ direction, temperature) - _soft_max(original @ direction, temperature)
    if subprotocol == "separator":
        objective = separation
    elif subprotocol == "blank":
        objective = blank + 0.10 * separation
    else:
        raise ValueError(f"Unknown V3 subprotocol: {subprotocol}")
    return objective, {"soft_separation": separation, "soft_compact_radius": compact, "soft_blank_margin": blank}


@dataclass
class SoftPromptResult:
    prompt_embeddings: np.ndarray
    nearest_token_ids: tuple[int, ...]
    history: list[dict[str, float]]
    continuous_metrics: dict[str, Any]


def optimize_soft_prompt(
    encoder,
    texts: Sequence[str],
    original: np.ndarray,
    support: BenignSupportModel,
    constraints: dict[str, float],
    legal_token_ids: Sequence[int],
    *,
    length: int,
    position: str,
    subprotocol: str,
    iterations: int,
    learning_rate: float,
    temperature: float,
    batch_size: int,
    seed: int,
) -> SoftPromptResult:
    auto_model, _ = _modules(encoder)
    embeddings = auto_model.get_input_embeddings().weight
    legal = torch.tensor(list(map(int, legal_token_ids)), device=embeddings.device, dtype=torch.long)
    generator = torch.Generator(device=embeddings.device).manual_seed(seed)
    initial_indices = torch.randint(len(legal), (length,), generator=generator, device=embeddings.device)
    prompt = torch.nn.Parameter(embeddings[legal[initial_indices]].detach().clone())
    target_norm = float(torch.linalg.vector_norm(embeddings.detach(), dim=1).median())
    optimizer = torch.optim.Adam([prompt], lr=learning_rate)
    original_tensor = torch.tensor(np.asarray(original), dtype=torch.float32, device=embeddings.device)
    support_tensor = torch.tensor(support.memory, dtype=torch.float32, device=embeddings.device)
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    for iteration in range(iterations):
        if batch_size < len(texts):
            chosen = np.sort(rng.choice(len(texts), size=batch_size, replace=False))
        else:
            chosen = np.arange(len(texts))
        triggered = encode_with_prompt_embeddings(
            encoder,
            [texts[index] for index in chosen],
            prompt,
            position=position,
        )
        objective, diagnostics = differentiable_objective(
            original_tensor[chosen],
            triggered,
            support_tensor,
            subprotocol=subprotocol,
            temperature=temperature,
        )
        optimizer.zero_grad()
        (-objective).backward()
        optimizer.step()
        with torch.no_grad():
            norms = torch.linalg.vector_norm(prompt, dim=1, keepdim=True)
            prompt.mul_(target_norm / torch.clamp(norms, min=1e-12))
        history.append(
            {
                "iteration": float(iteration),
                "objective": float(objective.detach()),
                **{name: float(value.detach()) for name, value in diagnostics.items()},
            }
        )
    with torch.no_grad():
        full = encode_with_prompt_embeddings(encoder, texts, prompt, position=position).cpu().numpy()
        legal_embeddings = torch.nn.functional.normalize(embeddings[legal], dim=1)
        prompt_normalized = torch.nn.functional.normalize(prompt, dim=1)
        nearest = legal[torch.argmax(prompt_normalized @ legal_embeddings.T, dim=1)].cpu().numpy()
    metrics = evaluate_mode3(original, full, support, constraints, seed=seed)
    return SoftPromptResult(
        prompt.detach().cpu().numpy(),
        tuple(map(int, nearest)),
        history,
        metrics.__dict__,
    )
