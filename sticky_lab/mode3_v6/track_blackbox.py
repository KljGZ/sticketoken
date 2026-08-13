"""Formal pure-black-box candidate discovery worker.

This source intentionally imports only the final-output oracle.  No white-box
candidate enters the population; an optional cross-track seed experiment must
be launched under an explicitly different ablation label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from .atomic_io import write_json, write_jsonl
from .blackbox_search import island_categorical_ga
from .experiment import evaluate_shared_token
from .insertion import BoundaryManifest, BoundaryRecord
from .oracle_blackbox import SentenceTransformerFinalOracle
from .tokenizer_audit import LegalToken
from .trajectory import render_trajectory


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6_mode3.yaml")
    parser.add_argument("--output", default="results/sticky_lab/sentence_t5_base/mode3_v6")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--restart-offset", type=int, default=0)
    parser.add_argument("--restarts", type=int)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    settings = config["blackbox"]
    output = Path(args.output)
    legal = [LegalToken(**{key: row[key] for key in LegalToken.__dataclass_fields__}) for row in _jsonl(output / "enumeration" / "legal_unrestricted.jsonl")]
    by_id = {row.token_id: row for row in legal}
    role_dir = output / "registration" / "roles"
    load_role = lambda name: [dict(row) for row in _jsonl(role_dir / f"{name}.jsonl")]
    fit_all, eval_all, benign_all = load_role("screen_fit"), load_role("screen_eval"), load_role("screen_benign")
    boundaries = BoundaryManifest([BoundaryRecord(**row) for row in _jsonl(output / "registration" / "random_boundaries.jsonl")])
    model = config["model"]
    oracle = SentenceTransformerFinalOracle(
        model["id"], revision=model["revision"], device=args.device, batch_size=model["batch_size"],
        local_path=model["local_path"], cache_folder=model["cache_folder"], trust_remote_code=model["trust_remote_code"],
    )
    generation_batches: dict[int, tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], str]] = {}
    for generation in range(int(settings["generations"])):
        rng = np.random.default_rng(int(config["positions"]["random_seed"]) + generation)
        fit = [fit_all[i] for i in sorted(rng.choice(len(fit_all), size=min(128, len(fit_all)), replace=False))]
        evaluation = [eval_all[i] for i in sorted(rng.choice(len(eval_all), size=min(128, len(eval_all)), replace=False))]
        benign = [benign_all[i] for i in sorted(rng.choice(len(benign_all), size=min(512, len(benign_all)), replace=False))]
        digest = hashlib.sha256("\n".join(row["text_id"] for row in fit + evaluation + benign).encode()).hexdigest()
        generation_batches[generation] = (fit, evaluation, benign, digest)

    reference = (
        fit_all[: min(256, len(fit_all))], eval_all[: min(256, len(eval_all))],
        benign_all[: min(2048, len(benign_all))],
    )

    def score(token_ids: list[int], generation: int) -> list[float]:
        fit, evaluation, benign, _ = generation_batches[generation]
        values = []
        for token_id in token_ids:
            token = by_id[int(token_id)]
            try:
                _, metrics, _ = evaluate_shared_token(
                    oracle, token_id=token.token_id, token_text=token.token_text, fit_records=fit,
                    eval_records=evaluation, benign_records=benign, fit_role="screen_fit", eval_role="screen_eval",
                    manifest=boundaries, random_replicates=config["positions"]["random_replicates"],
                    coverage=config["geometry"]["calibration"]["weak_coverage"],
                    maximum_radius_degrees=config["geometry"]["maximum_radius_degrees"], source_tracks=("blackbox",),
                )
                values.append(metrics.search_margin_m90_1 - metrics.radius_radians - metrics.benign_occupancy)
            except RuntimeError:
                values.append(-1e9)
        return values

    def reference_score(token_ids: list[int]) -> list[float]:
        fit, evaluation, benign = reference
        values = []
        for token_id in token_ids:
            token = by_id[int(token_id)]
            try:
                _, metrics, _ = evaluate_shared_token(
                    oracle, token_id=token.token_id, token_text=token.token_text, fit_records=fit,
                    eval_records=evaluation, benign_records=benign, fit_role="screen_fit", eval_role="screen_eval",
                    manifest=boundaries, random_replicates=config["positions"]["random_replicates"],
                    coverage=config["geometry"]["calibration"]["weak_coverage"],
                    maximum_radius_degrees=config["geometry"]["maximum_radius_degrees"], source_tracks=("blackbox",),
                )
                values.append(metrics.search_margin_m90_1 - metrics.radius_radians - metrics.benign_occupancy)
            except Exception:
                values.append(-1e9)
        return values

    traces, archive = island_categorical_ga(
        list(by_id), score, lambda generation: generation_batches[generation][3],
        population=settings["population"], generations=settings["generations"],
        restarts=args.restarts or settings["restarts"], islands=settings["islands"],
        elite_fraction=settings["elite_fraction"], uniform_fraction=settings["uniform_fraction"],
        migration_every=settings["migration_every"], migration_count=settings["migration_count"],
        reference_score=reference_score, reference_every=settings["reference_archive_every"],
        seed=int(config["positions"]["random_seed"]) + args.restart_offset,
    )
    target = output / "tracks" / "blackbox" / f"restart_{args.restart_offset:02d}"
    write_jsonl(target / "generations.jsonl", (trace.__dict__ for trace in traces))
    ranked = sorted(archive, key=lambda token_id: (-archive[token_id], token_id))
    write_json(target / "candidates.json", {"token_ids": ranked[:5000], "scores": {str(token_id): archive[token_id] for token_id in ranked}, "whitebox_seeded": False})
    write_json(target / "query_ledger.json", oracle.ledger.to_dict())
    snapshot_paths = []
    fit, evaluation, benign = reference
    for index, trace in enumerate(traces):
        cap = arrays = leader_id = None
        for leader_index in np.argsort(-np.asarray(trace.scores), kind="stable"):
            candidate_id = int(trace.token_ids[int(leader_index)]); token = by_id[candidate_id]
            try:
                cap, _, arrays = evaluate_shared_token(
                    oracle, token_id=candidate_id, token_text=token.token_text, fit_records=fit,
                    eval_records=evaluation, benign_records=benign, fit_role="screen_fit", eval_role="screen_eval",
                    manifest=boundaries, random_replicates=config["positions"]["random_replicates"],
                    coverage=config["geometry"]["calibration"]["weak_coverage"],
                    maximum_radius_degrees=config["geometry"]["maximum_radius_degrees"], source_tracks=("blackbox",),
                )
                leader_id = candidate_id
                break
            except Exception:
                continue
        if cap is None or arrays is None or leader_id is None:
            raise RuntimeError(f"generation {index} has no geometrically valid leader snapshot")
        snapshot = target / "snapshots" / f"generation_{index:04d}.npz"; snapshot.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            snapshot, token_id=np.asarray([leader_id]), triggered=arrays["triggered"],
            paired_clean=arrays["paired_clean"], independent_benign=arrays["independent_benign"],
            centers=cap.centers, radii=cap.radii, triggered_inside=cap.contains(arrays["triggered"]),
        )
        snapshot_paths.append(snapshot)
    render_trajectory(snapshot_paths, target / "projection_trajectory", seed=int(config["positions"]["random_seed"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
