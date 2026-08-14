"""Physically isolated final-output-only categorical CEM track."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sticky_lab.mode3_v6.geometry import angular_distance, spherical_mean

from .common import atomic_savez, load_config, load_legal, load_manifest, load_role, write_json, write_jsonl
from .encoding import encode_audited_positions, primary_position_vectors
from .oracle import V62FinalOracle


def _score(
    oracle: V62FinalOracle,
    token_ids: list[int],
    by_id: Mapping[int, Any],
    records: list[dict[str, str]],
    manifest: Any,
    config: Mapping[str, Any],
    *,
    restart: int,
    generation: int,
) -> dict[int, float]:
    result: dict[int, float] = {}
    for token_id in sorted(set(map(int, token_ids))):
        token = by_id[token_id]
        encoded, _, _ = encode_audited_positions(
            oracle, records, token_id=token_id, token_text=token.token_text,
            role="s0_fit", manifest=manifest, random_replicates=1,
            maximum_length=int(config["model"]["maximum_sequence_length"]),
            metadata={
                "track": "pure_blackbox_cem",
                "token_id": token_id,
                "restart": restart,
                "generation": generation,
            },
        )
        values = primary_position_vectors(encoded)
        center = spherical_mean(np.concatenate(list(values.values())))
        per_position = [
            angular_distance(values[position], center).reshape(-1) for position in ("prefix", "suffix", "random")
        ]
        # Equal-position compactness with a tail penalty.  This search score is
        # not a certificate; all candidates are re-evaluated in the formal funnel.
        means = np.asarray([np.mean(value) for value in per_position])
        tails = np.asarray([np.quantile(value, 0.90) for value in per_position])
        result[token_id] = -float(means.mean() + tails.max())
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6_2_mode3.yaml")
    parser.add_argument("--output", default="results/sticky_lab/sentence_t5_base/mode3_v6_2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--restart-offset", type=int, default=0)
    parser.add_argument("--restarts", type=int)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(Path(args.config))
    output = Path(args.output)
    settings = config["blackbox"]
    if settings.get("whitebox_seeded") is not False:
        raise RuntimeError("formal V6.2 black-box must not receive white-box seeds")
    target = output / "tracks" / "blackbox"
    if args.merge_only:
        archive: dict[int, float] = {}
        total_restarts = int(settings["restarts"])
        for restart in range(total_restarts):
            restart_dir = target / f"restart_{restart:02d}"
            complete = json.loads((restart_dir / "COMPLETE.json").read_text(encoding="utf-8"))
            if int(complete["restart"]) != restart:
                raise RuntimeError("blackbox restart identity mismatch")
            candidate = json.loads((restart_dir / "candidates.json").read_text(encoding="utf-8"))
            for token_id, score in candidate["scores"].items():
                archive[int(token_id)] = max(float(score), archive.get(int(token_id), -math.inf))
        ranked = sorted(archive, key=lambda token_id: (-archive[token_id], token_id))
        write_json(
            target / "candidates.json",
            {
                "token_ids": ranked[:1000],
                "scores": {str(token_id): archive[token_id] for token_id in ranked[:1000]},
                "track": "pure_blackbox_categorical_cem",
                "whitebox_seeded": False,
                "oracle_access": "final_embedding_only",
            },
        )
        write_json(
            target / "COMPLETE.json",
            {
                "population": int(settings["population"]),
                "generations": int(settings["generations"]),
                "restarts": total_restarts,
                "candidate_count": min(1000, len(ranked)),
                "whitebox_seeded": False,
            },
        )
        return 0
    legal = load_legal(output)
    legal_ids = np.asarray([row.token_id for row in legal], dtype=np.int64)
    by_id = {row.token_id: row for row in legal}
    records = load_role(output, "s0_fit")
    manifest = load_manifest(output)
    oracle = V62FinalOracle(
        config,
        output=output,
        device=args.device,
        phase="blackbox",
        track="pure_output_query_categorical_cem",
    )
    population = int(settings["population"])
    generations = int(settings["generations"])
    elite_count = max(1, int(math.ceil(population * float(settings["elite_fraction"]))))
    smoothing = float(settings["smoothing"])
    uniform_floor = float(settings["uniform_floor"])
    batch_size = min(int(settings["batch_texts"]), len(records))
    archive: dict[int, float] = {}
    all_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    base_seed = int(config["positions"]["random_seed"])
    restart_count = int(args.restarts or settings["restarts"])
    for restart in range(int(args.restart_offset), int(args.restart_offset) + restart_count):
        rng = np.random.default_rng(base_seed + 1009 * restart)
        probability = np.full(len(legal_ids), 1.0 / len(legal_ids), dtype=np.float64)
        history = []
        restart_rows = []
        for generation in range(generations):
            indices = rng.choice(len(legal_ids), size=population, replace=True, p=probability)
            sampled = legal_ids[indices].tolist()
            order = rng.permutation(len(records))[:batch_size]
            batch = [records[int(index)] for index in order]
            scores_by_id = _score(
                oracle,
                sampled,
                by_id,
                batch,
                manifest,
                config,
                restart=restart,
                generation=generation,
            )
            scores = np.asarray([scores_by_id[int(token_id)] for token_id in sampled])
            elite_indices = np.argsort(-scores, kind="stable")[:elite_count]
            elite_tokens = np.asarray(sampled, dtype=np.int64)[elite_indices]
            elite_distribution = np.zeros(len(legal_ids), dtype=np.float64)
            positions = {int(token_id): index for index, token_id in enumerate(legal_ids)}
            for token_id in elite_tokens:
                elite_distribution[positions[int(token_id)]] += 1.0
            elite_distribution /= elite_distribution.sum()
            probability = (1.0 - smoothing) * probability + smoothing * elite_distribution
            probability = (1.0 - uniform_floor) * probability + uniform_floor / len(probability)
            probability /= probability.sum()
            for token_id, score in scores_by_id.items():
                archive[token_id] = max(float(score), archive.get(token_id, -math.inf))
            row = {
                "restart": restart,
                "generation": generation,
                "population_token_ids": list(map(int, sampled)),
                "population_scores": scores.tolist(),
                "unique_evaluated": len(scores_by_id),
                "elite_token_ids": list(map(int, elite_tokens)),
                "leader_token_id": int(sampled[int(np.argmax(scores))]),
                "leader_score": float(np.max(scores)),
                "entropy": float(-(probability * np.log(np.maximum(probability, 1e-300))).sum()),
                "whitebox_seeded": False,
            }
            all_rows.append(row)
            restart_rows.append(row)
            history.append(probability.astype(np.float32))
            batch_rows.append(
                {
                    "restart": restart,
                    "generation": generation,
                    "text_ids": [row["text_id"] for row in batch],
                }
            )
        restart_dir = target / f"restart_{restart:02d}"
        write_jsonl(restart_dir / "generations.jsonl", restart_rows)
        write_jsonl(
            restart_dir / "batch_manifest.jsonl",
            [row for row in batch_rows if int(row["restart"]) == restart],
        )
        atomic_savez(restart_dir / "distribution_history.npz", probability=np.stack(history))
        write_json(
            restart_dir / "rng_state.json",
            {"restart": restart, "state": rng.bit_generator.state},
        )
        ranked_restart = sorted(archive, key=lambda token_id: (-archive[token_id], token_id))
        write_json(
            restart_dir / "candidates.json",
            {
                "token_ids": ranked_restart[:1000],
                "scores": {str(token_id): archive[token_id] for token_id in ranked_restart[:1000]},
            },
        )
        write_json(
            restart_dir / "COMPLETE.json",
            {
                "restart": restart,
                "generations": generations,
                "raw_forward_texts": oracle.raw_forward_texts,
                "whitebox_seeded": False,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
