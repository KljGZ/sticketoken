"""Isolated HotFlip and continuous-token upper-bound track."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .atomic_io import write_json, write_jsonl
from .insertion import BoundaryManifest, BoundaryRecord, insert_once
from .oracle_whitebox import WhiteboxSentenceTransformer
from .tokenizer_audit import LegalToken


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _trigger_gradient(runtime: object, tokenizer: object, texts: list[str], token_id: int) -> tuple[np.ndarray, float]:
    import torch

    features = runtime.tokenize(texts)
    device = next(runtime.parameters()).device
    features = {key: value.to(device) if hasattr(value, "to") else value for key, value in features.items()}
    captured: list[torch.Tensor] = []
    embedding = runtime[0].auto_model.get_input_embeddings()

    def hook(_module: object, _inputs: object, output: torch.Tensor) -> None:
        output.retain_grad(); captured.append(output)

    handle = embedding.register_forward_hook(hook)
    try:
        runtime.zero_grad(set_to_none=True)
        result = runtime(features)["sentence_embedding"]
        normalized = torch.nn.functional.normalize(result, dim=1)
        center = torch.nn.functional.normalize(normalized.detach().mean(dim=0), dim=0)
        loss = 1.0 - (normalized @ center).mean()
        loss.backward()
        if not captured or captured[0].grad is None:
            raise RuntimeError("white-box embedding gradient was not captured")
        mask = features["input_ids"] == int(token_id)
        if int(mask.sum()) != len(texts):
            raise RuntimeError("seed token was not realized exactly once per text")
        gradient = captured[0].grad[mask].mean(dim=0).detach().cpu().float().numpy()
        return gradient, float(loss.detach().cpu())
    finally:
        handle.remove()


def _trigger_loss(runtime: object, texts: list[str]) -> float:
    import torch
    features = runtime.tokenize(texts)
    device = next(runtime.parameters()).device
    features = {key: value.to(device) if hasattr(value, "to") else value for key, value in features.items()}
    with torch.no_grad():
        value = torch.nn.functional.normalize(runtime(features)["sentence_embedding"], dim=1)
        center = torch.nn.functional.normalize(value.mean(dim=0), dim=0)
        return float((1.0 - (value @ center).mean()).cpu())


def _continuous_upper_bound(runtime: object, texts: list[str], clean_texts: list[str], token_id: int, initial: np.ndarray, *, iterations: int, learning_rate: float) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Optimize one continuous input token; this is an upper bound, not a token."""
    import torch

    features = runtime.tokenize(texts)
    device = next(runtime.parameters()).device
    features = {key: value.to(device) if hasattr(value, "to") else value for key, value in features.items()}
    mask = features["input_ids"] == int(token_id)
    if int(mask.sum()) != len(texts):
        raise RuntimeError("continuous placeholder not realized exactly once")
    clean_features = runtime.tokenize(clean_texts)
    clean_features = {key: value.to(device) if hasattr(value, "to") else value for key, value in clean_features.items()}
    with torch.no_grad():
        clean = torch.nn.functional.normalize(runtime(clean_features)["sentence_embedding"], dim=1).detach()
    parameter = torch.nn.Parameter(torch.as_tensor(initial, dtype=torch.float32, device=device).clone())
    optimizer = torch.optim.Adam([parameter], lr=float(learning_rate))
    embedding = runtime[0].auto_model.get_input_embeddings()

    def replace(_module: object, _inputs: object, output: torch.Tensor) -> torch.Tensor:
        return torch.where(mask[..., None], parameter[None, None, :], output)

    handle = embedding.register_forward_hook(replace)
    trajectory: list[dict[str, float]] = []
    try:
        for iteration in range(int(iterations)):
            optimizer.zero_grad(set_to_none=True)
            triggered = torch.nn.functional.normalize(runtime(features)["sentence_embedding"], dim=1)
            center = torch.nn.functional.normalize(triggered.mean(dim=0), dim=0)
            compactness = 1.0 - (triggered @ center).mean()
            displacement = 1.0 - (triggered * clean).sum(dim=1).mean()
            # Displacement is only a weak tie-breaker in this white-box upper
            # bound; formal V6 gates remain coverage and benign occupancy.
            loss = compactness - 0.05 * displacement
            loss.backward(); optimizer.step()
            trajectory.append({
                "iteration": float(iteration), "loss": float(loss.detach().cpu()),
                "compactness": float(compactness.detach().cpu()), "displacement": float(displacement.detach().cpu()),
            })
        return parameter.detach().cpu().float().numpy(), trajectory
    finally:
        handle.remove()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6_mode3.yaml")
    parser.add_argument("--output", default="results/sticky_lab/sentence_t5_base/mode3_v6")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    legal_rows = [LegalToken(**{key: row[key] for key in LegalToken.__dataclass_fields__}) for row in _jsonl(output / "enumeration" / "legal_unrestricted.jsonl")]
    legal_ids = [row.token_id for row in legal_rows]
    by_id = {row.token_id: row for row in legal_rows}
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    model = config["model"]
    runtime = SentenceTransformer(model["local_path"] or model["id"], revision=None if model["local_path"] else model["revision"], device=args.device, trust_remote_code=model["trust_remote_code"])
    tokenizer = AutoTokenizer.from_pretrained(model["local_path"] or model["id"], revision=None if model["local_path"] else model["revision"], trust_remote_code=model["trust_remote_code"])
    whitebox = WhiteboxSentenceTransformer(runtime)
    records = [dict(row) for row in _jsonl(output / "registration" / "roles" / "screen_fit.jsonl")][:128]
    manifest = BoundaryManifest([BoundaryRecord(**row) for row in _jsonl(output / "registration" / "random_boundaries.jsonl")])
    settings = config["whitebox"]["hotflip"]
    seeds = legal_ids[: int(settings["seeds"])]
    trajectory = []
    candidate_ids: set[int] = set()
    for restart in range(int(settings["restarts"])):
        for seed_index, token_id in enumerate(np.roll(seeds, restart)):
            current_id = int(token_id)
            for iteration in range(int(settings["iterations"])):
                token = by_id[current_id]
                texts = [insert_once(row["text"], token.token_text, "prefix", role="screen_fit", text_id=row["text_id"], manifest=manifest) for row in records]
                gradient, loss = _trigger_gradient(runtime, tokenizer, texts, token.token_id)
                # Minimize compactness loss: HotFlip ranks -gradient dot embedding.
                ranked = whitebox.hotflip_rank(-gradient, legal_ids, int(settings["gradient_topk"]))
                candidate_ids.update(ranked.token_ids)
                chosen = ranked.token_ids[: int(settings["beam_size"])]
                true_losses = []
                for candidate_id in chosen:
                    candidate = by_id[int(candidate_id)]
                    candidate_texts = [insert_once(row["text"], candidate.token_text, "prefix", role="screen_fit", text_id=row["text_id"], manifest=manifest) for row in records]
                    true_losses.append(_trigger_loss(runtime, candidate_texts))
                best_index = int(np.argmin(np.asarray(true_losses)))
                next_id = int(chosen[best_index])
                trajectory.append({
                    "restart": restart, "seed_index": seed_index, "iteration": iteration,
                    "current_token_id": current_id, "gradient_loss": loss, "candidate_ids": chosen,
                    "surrogate_scores": ranked.surrogate_scores[: len(chosen)], "true_forward_losses": true_losses,
                    "selected_token_id": next_id,
                })
                if next_id == current_id:
                    break
                current_id = next_id
    # Continuous upper-bound projection is kept separate from discrete tokens.
    matrix = whitebox.embedding_matrix()
    initial_ids = sorted(candidate_ids)[: max(1, min(512, len(candidate_ids)))]
    continuous_initial = matrix[np.asarray(initial_ids)].mean(axis=0)
    placeholder = by_id[int(seeds[0])]
    continuous_texts = [insert_once(row["text"], placeholder.token_text, "prefix", role="screen_fit", text_id=row["text_id"], manifest=manifest) for row in records]
    continuous_settings = config["whitebox"]["continuous_upper_bound"]
    continuous_direction, continuous_trajectory = _continuous_upper_bound(
        runtime, continuous_texts, [row["text"] for row in records], placeholder.token_id, continuous_initial,
        iterations=int(continuous_settings["iterations"]), learning_rate=float(continuous_settings["learning_rate"]),
    )
    nearest = whitebox.nearest_discrete_tokens(continuous_direction, matrix, legal_ids, int(continuous_settings["nearest_discrete_tokens"]))
    candidate_ids.update(nearest)
    target = output / "tracks" / "whitebox"
    write_jsonl(target / "hotflip_trajectory.jsonl", trajectory)
    write_json(target / "candidates.json", {
        "token_ids": sorted(candidate_ids), "track": "whitebox", "may_seed_blackbox": False,
        "continuous_upper_bound": {"optimized_continuous_token": True, "status": "mechanistic_upper_bound_only", "nearest_token_ids": nearest, "trajectory": continuous_trajectory},
    })
    write_json(target / "COMPLETE.json", {
        "track": "whitebox", "candidate_count": len(candidate_ids),
        "may_seed_blackbox": False, "continuous_token_is_upper_bound_only": True,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
