"""Fixed V6.3 multi-fidelity funnel and from-scratch stage refits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .cache import EmbeddingCache
from .config import canonical_sha256
from .encoding import CachedEncoder, EncodingRequest
from .errors import CacheCorruption, ProtocolViolation, ShapeMismatch
from .geometry import FrozenCap, Stratum, center_drift, fit_single_cap


POSITIONS = ("prefix", "suffix", "random")
STAGES = ("s0", "s1", "s2", "full")
EVALUATION_STAGES = STAGES + ("top100",)
DESIGNS = {"s0": "one_of_three", "s1": "one_of_three", "s2": "two_of_three", "full": "two_of_three", "top100": "all_three"}


def _position_pair(row: Mapping[str, Any], seed: int) -> tuple[str, str]:
    rank = int(row["source_position_rank"])
    first_index = rank % 3
    first = POSITIONS[first_index]
    # In every source-local block of six records, each primary position occurs
    # twice and receives each of the other two positions once. This is an exact
    # balanced incomplete block, not a merely expected balance under hashing.
    direction = 1 + ((rank // 3) % 2)
    second = POSITIONS[(first_index + direction) % 3]
    return first, second


def assigned_positions(row: Mapping[str, Any], stage: str, *, seed: int) -> tuple[str, ...]:
    if stage not in DESIGNS:
        raise ProtocolViolation(f"unknown position stage {stage}")
    first, second = _position_pair(row, seed)
    if DESIGNS[stage] == "one_of_three":
        return (first,)
    if DESIGNS[stage] == "two_of_three":
        return (first, second)
    return (first, second, next(position for position in POSITIONS if position not in {first, second}))


def position_manifest(
    views: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], *, seed: int
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    previous: dict[tuple[str, str], tuple[str, ...]] = {}
    for stage in STAGES:
        for chain in ("fit", "radius", "score"):
            for row in views[stage][chain]:
                key = (chain, str(row["text_id"]))
                positions = assigned_positions(row, stage, seed=seed)
                if key in previous and not set(previous[key]).issubset(positions):
                    raise ProtocolViolation("later position design does not contain earlier assignment")
                previous[key] = positions
                entries.append({
                    "stage": stage, "role": chain, "text_id": str(row["text_id"]),
                    "source_id": str(row["source_id"]), "positions": list(positions),
                })
    payload = {
        "schema_version": "mode3-v6-3-position-manifest-v1",
        "seed": int(seed), "candidate_independent": True,
        "one_of_three_source_balanced_by_rank": True,
        "two_of_three_contains_previous": True,
        "random_vectors_averaged": False,
        "entries": entries,
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    return payload


def assert_source_balanced(records: Sequence[Mapping[str, Any]], stage: str, *, seed: int) -> None:
    counts: dict[tuple[str, str], int] = {}
    for row in records:
        for position in assigned_positions(row, stage, seed=seed):
            counts[(str(row["source_id"]), position)] = counts.get((str(row["source_id"]), position), 0) + 1
    for source in sorted({str(row["source_id"]) for row in records}):
        values = [counts.get((source, position), 0) for position in POSITIONS]
        if max(values) - min(values) > 1:
            raise ProtocolViolation(f"position design is not source balanced for {source}: {values}")


def stage_requests(
    records: Sequence[Mapping[str, Any]], role: str, stage: str, *, seed: int
) -> list[EncodingRequest]:
    return [
        EncodingRequest(str(role), row, position, 0)
        for row in records
        for position in assigned_positions(row, stage, seed=seed)
    ]


def _stratify(
    requests: Sequence[EncodingRequest], vectors: np.ndarray
) -> dict[Stratum, np.ndarray]:
    matrix = np.asarray(vectors)
    if len(matrix) != len(requests):
        raise ShapeMismatch("request/vector alignment mismatch")
    grouped: dict[Stratum, list[np.ndarray]] = {}
    for request, vector in zip(requests, matrix):
        key = (str(request.record["source_id"]), str(request.position))
        grouped.setdefault(key, []).append(np.asarray(vector))
    return {key: np.stack(values) for key, values in grouped.items()}


def _balanced_mean(values: Mapping[Stratum, np.ndarray]) -> float:
    return float(np.mean([np.mean(sample) for sample in values.values()]))


def _trapezoid_integral(values: np.ndarray, coordinates: np.ndarray) -> float:
    """Integrate without eagerly resolving NumPy's removed ``trapz`` alias."""
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # pragma: no cover - compatibility with NumPy < 2.0
        integrate = getattr(np, "trapz")
    return float(integrate(values, coordinates))


def _raw_score(
    cap: FrozenCap,
    *,
    requests: Sequence[EncodingRequest],
    triggered: np.ndarray,
    clean_by_text: Mapping[str, np.ndarray],
    benign_records: Sequence[Mapping[str, Any]],
    benign_vectors: np.ndarray,
) -> dict[str, Any]:
    triggered_grid = _stratify(requests, triggered)
    clean_grid = _stratify(
        requests,
        np.stack([clean_by_text[str(request.record["text_id"])] for request in requests]),
    )
    tr_inside = {key: cap.contains(value) for key, value in triggered_grid.items()}
    cl_inside = {key: cap.contains(value) for key, value in clean_grid.items()}
    coverage = _balanced_mean(tr_inside)
    position_coverage = {
        position: float(np.mean([np.mean(value) for (source, pos), value in tr_inside.items() if pos == position]))
        for position in POSITIONS
    }
    source_coverage = {
        source: float(np.mean([np.mean(value) for (src, position), value in tr_inside.items() if src == source]))
        for source in sorted({source for source, _ in tr_inside})
    }
    moved = {key: (~cl_inside[key]) & tr_inside[key] for key in tr_inside}
    origins = []
    for key in tr_inside:
        conditional = (~cl_inside[key])[tr_inside[key]]
        if len(conditional):
            origins.append(float(np.mean(conditional)))
    benign = np.asarray(benign_vectors)
    if len(benign) != len(benign_records):
        raise ShapeMismatch("benign cache is not record aligned")
    benign_depth = cap.normalized_radius(benign)
    multipliers = np.asarray([1.0, 1.1, 1.25, 1.5])
    occupancy = np.asarray([np.mean(benign_depth <= value + 1e-12) for value in multipliers])
    auc = _trapezoid_integral(occupancy, multipliers) / 0.5
    return {
        "balanced_coverage": coverage,
        "worst_position_coverage": min(position_coverage.values()),
        "worst_source_coverage": min(source_coverage.values()),
        "outside_to_inside": _balanced_mean(moved),
        "conditional_origin_outside": min(origins) if origins else 0.0,
        "paired_clean_inside": _balanced_mean(cl_inside),
        "benign_occupancy_core": float(occupancy[0]),
        "benign_occupancy_1_1": float(occupancy[1]),
        "benign_occupancy_auc_1_1_5": auc,
        "position_coverage": position_coverage,
        "source_coverage": source_coverage,
    }


def cached_clean_vectors(
    cache: EmbeddingCache,
    encoder: CachedEncoder,
    records: Sequence[Mapping[str, Any]],
    role: str,
) -> dict[str, np.ndarray]:
    memo = getattr(encoder, "_v63_clean_memo", None)
    if memo is None:
        memo = {}
        setattr(encoder, "_v63_clean_memo", memo)
    memo_key = (str(role), tuple(str(row["text_id"]) for row in records))
    if memo_key in memo:
        return memo[memo_key]
    entries = [encoder.call_space.lookup_request(role, str(row["text_id"]), "clean") for row in records]
    found, missing = cache.fetch(-2, [entry.ordinal for entry in entries])
    if missing:
        raise CacheCorruption(
            f"clean role {role} must be precomputed before candidate workers; missing={len(missing)}"
        )
    value = {str(row["text_id"]): found[entry.ordinal] for row, entry in zip(records, entries)}
    memo[memo_key] = value
    return value


@dataclass(frozen=True)
class StageMetric:
    token_id: int
    token_text: str
    stage: str
    radius_radians: float
    radius_degrees: float
    balanced_coverage: float
    worst_position_coverage: float
    worst_source_coverage: float
    outside_to_inside: float
    conditional_origin_outside: float
    paired_clean_inside: float
    benign_occupancy_core: float
    benign_occupancy_1_1: float
    benign_occupancy_auc_1_1_5: float
    center_drift_from_previous: float
    center_restart_spread: float
    fit_role_sha256: str
    radius_role_sha256: str
    score_role_sha256: str
    current_stage_refit: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_and_score_candidate(
    encoder: CachedEncoder,
    *,
    token_id: int,
    token_text: str,
    stage: str,
    fit_records: Sequence[Mapping[str, Any]],
    radius_records: Sequence[Mapping[str, Any]],
    score_records: Sequence[Mapping[str, Any]],
    benign_records: Sequence[Mapping[str, Any]],
    role_hashes: Mapping[str, str],
    config: Mapping[str, Any],
    previous_cap: FrozenCap | None = None,
) -> tuple[FrozenCap, StageMetric, dict[str, Any]]:
    if stage not in EVALUATION_STAGES:
        raise ProtocolViolation(f"invalid funnel stage {stage}")
    seed = int(config["positions"]["seed"])
    for records in (fit_records, radius_records, score_records):
        assert_source_balanced(records, stage, seed=seed)
    fit_requests = stage_requests(fit_records, "fit", stage, seed=seed)
    radius_requests = stage_requests(radius_records, "radius", stage, seed=seed)
    score_requests = stage_requests(score_records, "score", stage, seed=seed)
    fit_vectors, fit_audits, fit_cache = encoder.encode_requests(
        token_id=token_id, token_text=token_text, requests=fit_requests, phase=f"{stage}:fit"
    )
    radius_vectors, radius_audits, radius_cache = encoder.encode_requests(
        token_id=token_id, token_text=token_text, requests=radius_requests, phase=f"{stage}:radius"
    )
    score_vectors, score_audits, score_cache = encoder.encode_requests(
        token_id=token_id, token_text=token_text, requests=score_requests, phase=f"{stage}:score"
    )
    geometry = config["geometry"]
    restarts = int(geometry["stage_restarts"][stage])
    cap, fit_audit = fit_single_cap(
        token_id, token_text, _stratify(fit_requests, fit_vectors),
        _stratify(radius_requests, radius_vectors),
        fit_role_sha256=str(role_hashes["fit"]),
        radius_role_sha256=str(role_hashes["radius"]), stage=stage,
        trim_fraction=float(geometry["center_trim_fraction"]),
        design_quantile=float(geometry["radius_design_quantile"]),
        maximum_radius_degrees=float(geometry["maximum_radius_degrees"]),
        restarts=restarts,
        maximum_iterations=int(geometry["maximum_iterations"]),
        tolerance=float(geometry["tolerance"]),
        seed=seed + int(token_id) * 17 + EVALUATION_STAGES.index(stage),
    )
    clean_score = cached_clean_vectors(encoder.cache, encoder, score_records, "score")
    benign_by_text = cached_clean_vectors(encoder.cache, encoder, benign_records, "discovery_benign")
    benign_vectors = np.stack([benign_by_text[str(row["text_id"])] for row in benign_records])
    raw = _raw_score(
        cap, requests=score_requests, triggered=score_vectors,
        clean_by_text=clean_score, benign_records=benign_records,
        benign_vectors=benign_vectors,
    )
    summaries = fit_audit["center_fit"]["restart_summaries"]
    spread = max(float(row["worst_stratum_q90"]) for row in summaries) - min(float(row["worst_stratum_q90"]) for row in summaries)
    metric = StageMetric(
        int(token_id), str(token_text), stage, cap.radius, cap.radius_degrees,
        float(raw["balanced_coverage"]), float(raw["worst_position_coverage"]),
        float(raw["worst_source_coverage"]), float(raw["outside_to_inside"]),
        float(raw["conditional_origin_outside"]), float(raw["paired_clean_inside"]),
        float(raw["benign_occupancy_core"]), float(raw["benign_occupancy_1_1"]),
        float(raw["benign_occupancy_auc_1_1_5"]),
        0.0 if previous_cap is None else center_drift(previous_cap, cap),
        float(spread), str(role_hashes["fit"]), str(role_hashes["radius"]),
        str(role_hashes["score"]), True,
    )
    audit = {
        "schema_version": "mode3-v6-3-stage-candidate-v1",
        "stage": stage, "token_id": int(token_id), "single_cap_only": True,
        "from_scratch_refit": True, "previous_cap_used_for_fit": False,
        "fit": fit_audit, "raw_score": raw,
        "cache": {"fit": fit_cache, "radius": radius_cache, "score": score_cache},
        "tokenization_audit_sha256": canonical_sha256([
            audit.to_dict() for audit in fit_audits + radius_audits + score_audits
        ]),
    }
    return cap, metric, audit


def clean_precompute_requests(
    score_records: Sequence[Mapping[str, Any]],
    benign_records: Sequence[Mapping[str, Any]],
) -> list[EncodingRequest]:
    return [EncodingRequest("score", row, "clean", 0) for row in score_records] + [
        EncodingRequest("discovery_benign", row, "clean", 0) for row in benign_records
    ]


def required_keep(stage: str, config: Mapping[str, Any]) -> int:
    return {
        "s0": int(config["funnel"]["s0_keep"]),
        "s1": int(config["funnel"]["s1_keep"]),
        "s2": int(config["funnel"]["s2_keep"]),
        "full": int(config["funnel"]["full_top"]),
    }[stage]
