"""Independent Pareto-CEM with rotating batches and non-regressing archives."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .atomic_io import validate_completion, write_completion, write_csv, write_json, write_npz
from .candidate_space import CandidateSpace
from .interfaces import Candidate
from .pareto import non_dominated_front, select_nsga2, update_historical_archive
from .scoring import flatten_record


ScoreFunction = Callable[[Candidate, np.ndarray, int], dict[str, Any]]
FullScoreFunction = Callable[[Candidate, int], dict[str, Any]]
SnapshotFunction = Callable[[Candidate, int, str, Path], Sequence[Path]]


@dataclass
class ParetoCEMResult:
    historical_archive: list[dict[str, Any]]
    formal_archive: list[dict[str, Any]]
    probabilities: np.ndarray
    generations_completed: int


def rotating_batch_indices(total: int, size: int, generation: int, seed: int) -> np.ndarray:
    if not 0 < size <= total:
        raise ValueError("rotating batch size must be in [1,total]")
    start = generation * size
    result = []
    while len(result) < size:
        epoch = start // total
        offset = start % total
        permutation = np.random.default_rng(seed + epoch * 1000003).permutation(total)
        take = min(size - len(result), total - offset)
        result.extend(map(int, permutation[offset : offset + take]))
        start += take
    return np.asarray(result, dtype=np.int64)


def _resource_record(started: float) -> dict[str, Any]:
    record: dict[str, Any] = {
        "wall_seconds": time.monotonic() - started,
        "hostname": platform.node(),
        "pid": os.getpid(),
    }
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        record.update({"maximum_rss_kib": int(usage.ru_maxrss), "user_seconds": usage.ru_utime, "system_seconds": usage.ru_stime})
    except (ImportError, AttributeError):
        pass
    return record


def _pool_indices(record: Mapping[str, Any]) -> np.ndarray:
    value = record.get("pool_indices", "")
    if isinstance(value, str):
        return np.asarray([int(part) for part in value.split(",") if part != ""], dtype=np.int64)
    return np.asarray(value, dtype=np.int64)


def _restore_candidate(space: CandidateSpace, record: Mapping[str, Any]) -> Candidate:
    candidate = space.materialize_pool_indices(_pool_indices(record))
    if candidate is None or candidate.key != str(record["candidate_key"]):
        raise RuntimeError(f"cannot reconstruct checkpoint candidate {record.get('candidate_key')}")
    return candidate


def _save_checkpoint(
    output: Path,
    generation: int,
    probabilities: np.ndarray,
    rng: np.random.Generator,
    historical: Sequence[Mapping[str, Any]],
    formal: Sequence[Mapping[str, Any]],
) -> None:
    distribution = output / "checkpoint_distribution.npz"
    write_npz(distribution, probabilities=probabilities)
    archive = output / "checkpoint_archives.json"
    write_json(archive, {"historical": list(historical), "formal": list(formal)})
    checkpoint = output / "checkpoint.json"
    write_json(
        checkpoint,
        {
            "schema_version": "mode3-v5-cem-checkpoint-v1",
            "generation": int(generation),
            "rng_state": rng.bit_generator.state,
            "distribution": distribution.name,
            "archives": archive.name,
        },
    )


def _load_checkpoint(output: Path, expected_shape: tuple[int, int]) -> tuple[int, np.ndarray, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]] | None:
    checkpoint = output / "checkpoint.json"
    distribution = output / "checkpoint_distribution.npz"
    archives = output / "checkpoint_archives.json"
    if not (checkpoint.is_file() and distribution.is_file() and archives.is_file()):
        return None
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "mode3-v5-cem-checkpoint-v1":
        return None
    probabilities = np.load(distribution)["probabilities"]
    if probabilities.shape != expected_shape or not np.allclose(probabilities.sum(axis=1), 1.0):
        return None
    stored = json.loads(archives.read_text(encoding="utf-8"))
    return int(payload["generation"]) + 1, probabilities, payload["rng_state"], stored["historical"], stored["formal"]


def pareto_cem(
    space: CandidateSpace,
    *,
    length: int,
    restart: int,
    task: str,
    total_search_texts: int,
    config: Mapping[str, Any],
    output: Path,
    score: ScoreFunction,
    full_score: FullScoreFunction,
    snapshot: SnapshotFunction,
    query_ledger: Callable[[], Mapping[str, Any]],
) -> ParetoCEMResult:
    search = config["search"]
    population_size = int(search["population_size"])
    generations = int(search["iterations"])
    elite_count = max(2, int(np.ceil(population_size * float(search["elite_ratio"]))))
    base_seed = int(config["seed"]) + restart * 10000019 + length * 1009 + sum(map(ord, task))
    output.mkdir(parents=True, exist_ok=True)
    loaded = _load_checkpoint(output, (length, space.pool_size))
    rng = np.random.default_rng(base_seed)
    if loaded is None:
        start_generation = 0
        probabilities = np.full((length, space.pool_size), 1.0 / space.pool_size, dtype=np.float64)
        historical: list[dict[str, Any]] = []
        formal: list[dict[str, Any]] = []
    else:
        start_generation, probabilities, rng_state, historical, formal = loaded
        rng.bit_generator.state = rng_state
    candidate_by_key: dict[str, Candidate] = {}
    for record in [*historical, *formal]:
        candidate_by_key[str(record["candidate_key"])] = _restore_candidate(space, record)
    started = time.monotonic()

    for generation in range(start_generation, generations):
        generation_dir = output / f"generation_{generation:03d}"
        if validate_completion(generation_dir, {"generation": generation, "task": task, "length": length, "restart": restart}):
            continue
        before_ledger = dict(query_ledger())
        batch_indices = rotating_batch_indices(
            total_search_texts,
            int(search["rotating_batch_size"]),
            generation,
            base_seed + 700001,
        )
        sampled = space.sample_valid(
            length,
            population_size,
            rng=rng,
            maximum_attempts=int(search["maximum_materialization_attempts"]),
            probabilities=probabilities,
        )
        records: list[dict[str, Any]] = []
        for candidate, pool_indices in sampled:
            candidate_by_key[candidate.key] = candidate
            record = dict(score(candidate, batch_indices, generation))
            record.update(
                {
                    "generation": generation,
                    "restart": restart,
                    "pool_indices": ",".join(map(str, pool_indices.tolist())),
                    "evaluation_scope": "rotating_minibatch",
                }
            )
            records.append(record)
        selected_indices = select_nsga2(records, elite_count)
        elites = [records[index] for index in selected_indices]
        previous_keys = {str(record["candidate_key"]) for record in historical}
        historical = update_historical_archive(
            historical,
            records,
            int(search["historical_archive_size"]),
        )
        new_keys = {str(record["candidate_key"]) for record in historical} - previous_keys

        counts = np.zeros_like(probabilities)
        for record in elites:
            indices = _pool_indices(record)
            counts[np.arange(length), indices] += 1
        empirical = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
        uniform = np.full_like(empirical, 1.0 / space.pool_size)
        target = (1.0 - float(search["uniform_mixture"])) * empirical + float(search["uniform_mixture"]) * uniform
        probabilities = (1.0 - float(search["update_alpha"])) * probabilities + float(search["update_alpha"]) * target
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        should_full = (generation + 1) % int(search["full_reevaluation_interval"]) == 0 or generation == generations - 1
        full_records: list[dict[str, Any]] = []
        if should_full:
            full_indices = select_nsga2(historical, min(int(search["full_reevaluation_candidates"]), len(historical)))
            for index in full_indices:
                candidate = candidate_by_key[str(historical[index]["candidate_key"])]
                record = dict(full_score(candidate, generation))
                record.update(
                    {
                        "generation": generation,
                        "restart": restart,
                        "pool_indices": historical[index]["pool_indices"],
                        "evaluation_scope": "full_search",
                    }
                )
                full_records.append(record)
            formal = update_historical_archive(formal, full_records, int(search["formal_archive_size"]))

        generation_dir.mkdir(parents=True, exist_ok=True)
        population_path = generation_dir / "population.csv"
        pareto_path = generation_dir / "pareto_front.json"
        elite_path = generation_dir / "elites.json"
        batch_path = generation_dir / "batch_manifest.json"
        rng_path = generation_dir / "rng_state.json"
        distribution_path = generation_dir / "distribution.npz"
        frequencies_path = generation_dir / "token_frequencies.npz"
        ledger_path = generation_dir / "query_ledger.json"
        resource_path = generation_dir / "resource_usage.json"
        formal_path = generation_dir / "formal_full_search.json"
        write_csv(population_path, [flatten_record(record) for record in records])
        write_json(pareto_path, non_dominated_front(records))
        write_json(elite_path, elites)
        write_json(batch_path, {"generation": generation, "indices": batch_indices.tolist(), "seed": base_seed + 700001})
        write_json(rng_path, rng.bit_generator.state)
        write_npz(distribution_path, probabilities=probabilities)
        write_npz(frequencies_path, counts=counts, legal_token_ids=space.legal_single_token_ids)
        after_ledger = dict(query_ledger())
        write_json(
            ledger_path,
            {
                "before": before_ledger,
                "after": after_ledger,
                "delta": {key: int(after_ledger.get(key, 0)) - int(before_ledger.get(key, 0)) for key in set(before_ledger) | set(after_ledger)},
            },
        )
        write_json(resource_path, _resource_record(started))
        write_json(formal_path, full_records)

        snapshot_artifacts: list[Path] = []
        snapshot_choices: list[tuple[str, dict[str, Any]]] = []
        leader_index = select_nsga2(records, 1)[0]
        snapshot_choices.append(("leader", records[leader_index]))
        new_member = next((record for record in historical if str(record["candidate_key"]) in new_keys), None)
        if new_member is not None:
            snapshot_choices.append(("pareto_new", new_member))
        if elites:
            snapshot_choices.append(("elite_representative", elites[0]))
        seen_snapshot: set[str] = set()
        for label, record in snapshot_choices[: int(search["snapshots_per_generation"])]:
            key = str(record["candidate_key"])
            if key in seen_snapshot:
                continue
            seen_snapshot.add(key)
            snapshot_artifacts.extend(snapshot(candidate_by_key[key], generation, label, generation_dir / "snapshots" / label))

        artifacts = [
            population_path,
            pareto_path,
            elite_path,
            batch_path,
            rng_path,
            distribution_path,
            frequencies_path,
            ledger_path,
            resource_path,
            formal_path,
            *snapshot_artifacts,
        ]
        write_completion(
            generation_dir,
            artifacts,
            {"generation": generation, "task": task, "length": length, "restart": restart},
        )
        _save_checkpoint(output, generation, probabilities, rng, historical, formal)

    historical_path = output / "historical_archive.json"
    formal_path = output / "formal_archive.json"
    final_distribution = output / "final_distribution.npz"
    write_json(historical_path, historical)
    write_json(formal_path, formal)
    write_npz(final_distribution, probabilities=probabilities)
    write_completion(
        output,
        [historical_path, formal_path, final_distribution, output / "checkpoint.json", output / "checkpoint_distribution.npz", output / "checkpoint_archives.json"],
        {"task": task, "length": length, "restart": restart, "generations": generations},
    )
    return ParetoCEMResult(historical, formal, probabilities, generations)
