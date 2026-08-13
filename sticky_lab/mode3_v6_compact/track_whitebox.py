"""Physically isolated HotFlip/continuous-token mechanism track."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Mapping

import numpy as np

from sticky_lab.mode3_v6.insertion import insert_once
from sticky_lab.mode3_v6.oracle_whitebox import WhiteboxSentenceTransformer

from .budget import BudgetLedger
from .common import load_config, load_legal, load_manifest, load_role, write_json, write_jsonl


def _texts(records: list[dict[str, str]], token_text: str, manifest: Any) -> list[str]:
    result: list[str] = []
    for position in ("prefix", "suffix", "random"):
        for row in records:
            result.append(
                insert_once(
                    row["text"],
                    token_text,
                    position,
                    role="s0_fit",
                    text_id=row["text_id"],
                    manifest=manifest,
                    replicate=0,
                )
            )
    return result


def _records_without_token(runtime: Any, records: list[dict[str, str]], token_id: int, count: int) -> list[dict[str, str]]:
    """Select public records without a natural occurrence of the seed token."""
    chosen = []
    for start in range(0, len(records), 256):
        chunk = records[start : start + 256]
        ids = runtime.tokenize([row["text"] for row in chunk])["input_ids"].detach().cpu().numpy()
        for row, values in zip(chunk, ids):
            if int(token_id) not in set(map(int, values)):
                chosen.append(row)
                if len(chosen) == count:
                    return chosen
    raise RuntimeError(f"only {len(chosen)} public texts exclude whitebox token {token_id}; need {count}")


def _loss(runtime: Any, texts: list[str]) -> float:
    import torch

    features = runtime.tokenize(texts)
    device = next(runtime.parameters()).device
    features = {key: value.to(device) if hasattr(value, "to") else value for key, value in features.items()}
    with torch.no_grad():
        vectors = torch.nn.functional.normalize(runtime(features)["sentence_embedding"], dim=1)
        center = torch.nn.functional.normalize(vectors.mean(dim=0), dim=0)
        return float((1.0 - (vectors @ center).mean()).cpu())


def _active_embedding_output(captured: list[Any], mask: Any) -> Any:
    """Return the captured embedding output that participated in this loss.

    Some SentenceTransformer/Transformers revisions invoke a shared embedding
    module more than once.  The first hook result is therefore not guaranteed
    to be the encoder tensor that received a gradient.
    """
    for value in reversed(captured):
        gradient = getattr(value, "grad", None)
        if gradient is not None and tuple(gradient.shape[:2]) == tuple(mask.shape):
            return value
    raise RuntimeError("whitebox trigger gradient was not captured")


def _gradient(runtime: Any, texts: list[str], token_id: int) -> tuple[np.ndarray, float]:
    import torch

    features = runtime.tokenize(texts)
    device = next(runtime.parameters()).device
    features = {key: value.to(device) if hasattr(value, "to") else value for key, value in features.items()}
    captured: list[torch.Tensor] = []
    embedding = runtime[0].auto_model.get_input_embeddings()

    def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> torch.Tensor:
        # A frozen embedding matrix produces an output that does not require a
        # gradient.  Re-leafing it is sufficient for input-gradient analysis
        # and intentionally prevents parameter updates.
        active = output if output.requires_grad else output.detach().requires_grad_(True)
        active.retain_grad()
        captured.append(active)
        return active

    handle = embedding.register_forward_hook(hook)
    try:
        runtime.zero_grad(set_to_none=True)
        vectors = torch.nn.functional.normalize(runtime(features)["sentence_embedding"], dim=1)
        center = torch.nn.functional.normalize(vectors.detach().mean(dim=0), dim=0)
        loss = 1.0 - (vectors @ center).mean()
        loss.backward()
        mask = features["input_ids"] == int(token_id)
        if int(mask.sum()) != len(texts):
            raise RuntimeError("candidate token was not realized once in every whitebox text")
        active = _active_embedding_output(captured, mask)
        gradient = active.grad[mask].mean(dim=0).detach().cpu().float().numpy()
        return gradient, float(loss.detach().cpu())
    finally:
        handle.remove()


def _benchmark_gamma(runtime: Any, texts: list[str], token_id: int, ledger: BudgetLedger, ceiling: float) -> dict[str, Any]:
    forward_times = []
    backward_times = []
    # One warm-up and three measured pairs; all work is accounted before use.
    for repeat in range(4):
        ledger.reserve(phase="whitebox_benchmark", track="whitebox", raw_items=len(texts), kind="forward")
        start = time.perf_counter()
        _loss(runtime, texts)
        forward = time.perf_counter() - start
        ledger.reserve(
            phase="whitebox_benchmark",
            track="whitebox",
            raw_items=len(texts),
            kind="backward",
            multiplier=ceiling,
        )
        start = time.perf_counter()
        _gradient(runtime, texts, token_id)
        backward = time.perf_counter() - start
        if repeat:
            forward_times.append(forward)
            backward_times.append(backward)
    gamma = statistics.median(backward_times) / max(statistics.median(forward_times), 1e-12)
    if gamma > ceiling:
        raise RuntimeError(f"measured backward gamma {gamma:.4f} exceeds preregistered ceiling {ceiling}")
    return {
        "measured_gamma": gamma,
        "accounting_gamma": ceiling,
        "forward_seconds": forward_times,
        "forward_backward_seconds": backward_times,
        "samples": len(texts),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6_mode3_compact.yaml")
    parser.add_argument("--output", default="results/sticky_lab/sentence_t5_base/mode3_v6_compact")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    config = load_config(Path(args.config))
    output = Path(args.output)
    settings = config["whitebox"]
    hot = settings["hotflip"]
    continuous = settings["continuous_upper_bound"]
    legal = load_legal(output)
    legal_ids = [row.token_id for row in legal]
    by_id = {row.token_id: row for row in legal}
    model = config["model"]
    from sentence_transformers import SentenceTransformer

    runtime = SentenceTransformer(
        model["local_path"] or model["id"],
        revision=None if model["local_path"] else model["revision"],
        device=args.device,
        trust_remote_code=bool(model["trust_remote_code"]),
    )
    wrapper = WhiteboxSentenceTransformer(runtime)
    records_all = load_role(output, "s0_fit")
    per_position = max(1, int(hot["batch_texts"]) // 3)
    manifest = load_manifest(output)
    ledger = BudgetLedger(output, config["budget"])
    seed_indices = np.linspace(0, len(legal_ids) - 1, int(hot["seeds"]), dtype=int)
    seeds = [int(legal_ids[index]) for index in seed_indices]
    ceiling = float(settings["backward_equivalent_gamma_ceiling"])
    benchmark_records = _records_without_token(runtime, records_all, seeds[0], per_position)
    benchmark_texts = _texts(benchmark_records, by_id[seeds[0]].token_text, manifest)
    benchmark = _benchmark_gamma(runtime, benchmark_texts, seeds[0], ledger, ceiling)
    trajectory: list[dict[str, Any]] = []
    candidate_ids: set[int] = set()
    for restart in range(int(hot["restarts"])):
        for seed_index, seed in enumerate(np.roll(seeds, restart)):
            current = int(seed)
            for iteration in range(int(hot["iterations"])):
                token = by_id[current]
                records = _records_without_token(runtime, records_all, current, per_position)
                texts = _texts(records, token.token_text, manifest)
                ledger.reserve(
                    phase="whitebox_hotflip",
                    track="whitebox",
                    raw_items=len(texts),
                    kind="backward",
                    multiplier=ceiling,
                    metadata={"restart": restart, "seed_index": seed_index, "iteration": iteration},
                )
                gradient, loss = _gradient(runtime, texts, current)
                ranked = wrapper.hotflip_rank(-gradient, legal_ids, int(hot["gradient_topk"]))
                exact_ids = list(map(int, ranked.token_ids[: int(hot["exact_forward_topk"])]))
                candidate_ids.update(exact_ids)
                exact_losses = []
                for candidate_id in exact_ids:
                    candidate_texts = _texts(records, by_id[candidate_id].token_text, manifest)
                    ledger.reserve(
                        phase="whitebox_hotflip",
                        track="whitebox",
                        raw_items=len(candidate_texts),
                        kind="forward",
                        metadata={"candidate_token_id": candidate_id},
                    )
                    exact_losses.append(_loss(runtime, candidate_texts))
                beam = min(int(hot["beam_size"]), len(exact_ids))
                best_index = int(np.argmin(np.asarray(exact_losses[:beam])))
                next_token = exact_ids[best_index]
                trajectory.append(
                    {
                        "restart": restart,
                        "seed_index": seed_index,
                        "iteration": iteration,
                        "current_token_id": current,
                        "gradient_loss": loss,
                        "gradient_top_ids": list(map(int, ranked.token_ids)),
                        "gradient_surrogate_scores": ranked.surrogate_scores,
                        "exact_forward_ids": exact_ids,
                        "exact_forward_losses": exact_losses,
                        "selected_token_id": next_token,
                    }
                )
                if next_token == current:
                    break
                current = next_token
    # Mechanistic continuous upper bound.  It cannot become a formal token;
    # only its nearest legal discrete tokens join the later union.
    matrix = wrapper.embedding_matrix()
    rng = np.random.default_rng(int(config["positions"]["random_seed"]))
    continuous_rows = []
    nearest_all: set[int] = set()
    import torch

    embedding = runtime[0].auto_model.get_input_embeddings()
    for restart in range(int(continuous["restarts"])):
        placeholder_id = int(seeds[restart % len(seeds)])
        placeholder = by_id[placeholder_id]
        records = _records_without_token(runtime, records_all, placeholder_id, per_position)
        texts = _texts(records, placeholder.token_text, manifest)
        features = runtime.tokenize(texts)
        device = next(runtime.parameters()).device
        features = {key: value.to(device) if hasattr(value, "to") else value for key, value in features.items()}
        mask = features["input_ids"] == placeholder_id
        if int(mask.sum()) != len(texts):
            raise RuntimeError("continuous placeholder realization failed")
        initial = matrix[int(rng.choice(len(matrix)))]
        parameter = torch.nn.Parameter(torch.as_tensor(initial, dtype=torch.float32, device=device))
        optimizer = torch.optim.Adam([parameter], lr=float(continuous["learning_rate"]))

        def replace(_module: Any, _inputs: Any, output_value: torch.Tensor) -> torch.Tensor:
            return torch.where(mask[..., None], parameter[None, None, :], output_value)

        handle = embedding.register_forward_hook(replace)
        try:
            for iteration in range(int(continuous["iterations"])):
                ledger.reserve(
                    phase="whitebox_continuous",
                    track="whitebox",
                    raw_items=len(texts),
                    kind="backward",
                    multiplier=ceiling,
                    metadata={"restart": restart, "iteration": iteration},
                )
                optimizer.zero_grad(set_to_none=True)
                vectors = torch.nn.functional.normalize(runtime(features)["sentence_embedding"], dim=1)
                center = torch.nn.functional.normalize(vectors.mean(dim=0), dim=0)
                loss = 1.0 - (vectors @ center).mean()
                loss.backward()
                optimizer.step()
                continuous_rows.append(
                    {"restart": restart, "iteration": iteration, "loss": float(loss.detach().cpu())}
                )
        finally:
            handle.remove()
        nearest = wrapper.nearest_discrete_tokens(
            parameter.detach().cpu().numpy(), matrix, legal_ids, int(continuous["nearest_discrete_tokens"])
        )
        nearest_all.update(map(int, nearest))
    candidate_ids.update(nearest_all)
    target = output / "tracks" / "whitebox"
    write_jsonl(target / "hotflip_trajectory.jsonl", trajectory)
    write_jsonl(target / "continuous_trajectory.jsonl", continuous_rows)
    write_json(target / "backward_gamma_benchmark.json", benchmark)
    write_json(
        target / "candidates.json",
        {
            "token_ids": sorted(candidate_ids),
            "track": "whitebox",
            "may_seed_blackbox": False,
            "continuous_is_upper_bound_only": True,
            "continuous_nearest_token_ids": sorted(nearest_all),
        },
    )
    write_json(
        target / "COMPLETE.json",
        {
            "candidate_count": len(candidate_ids),
            "may_seed_blackbox": False,
            "hotflip_seeds": int(hot["seeds"]),
            "hotflip_restarts": int(hot["restarts"]),
            "continuous_restarts": int(continuous["restarts"]),
            "backward_gamma": benchmark,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
