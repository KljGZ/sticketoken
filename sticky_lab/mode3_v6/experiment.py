"""Shared V6 experiment operations over registered role records."""

from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

import numpy as np

from .evaluation import candidate_metrics, certify_frozen_cap, fit_and_calibrate_single_cap
from .insertion import BoundaryManifest, insert_once
from .resource_errors import is_resource_exhaustion
from .oracle_blackbox import FinalEmbeddingOracle


def records_hash(records: Sequence[Mapping[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in records:
        digest.update(f"{row['text_id']}\0{row['text']}\n".encode())
    return digest.hexdigest()


def encode_clean(oracle: FinalEmbeddingOracle, records: Sequence[Mapping[str, str]]) -> np.ndarray:
    return oracle.encode([str(row["text"]) for row in records])


def encode_position(
    oracle: FinalEmbeddingOracle,
    records: Sequence[Mapping[str, str]],
    token_text: str,
    position: str,
    *,
    role: str,
    manifest: BoundaryManifest,
    random_replicates: int,
) -> np.ndarray:
    if position != "random":
        texts = [insert_once(row["text"], token_text, position, role=role, text_id=row["text_id"], manifest=manifest) for row in records]
        return oracle.encode(texts)
    replicas = []
    for replicate in range(random_replicates):
        texts = [
            insert_once(row["text"], token_text, "random", role=role, text_id=row["text_id"], manifest=manifest, replicate=replicate)
            for row in records
        ]
        replicas.append(oracle.encode(texts))
    # Average replicates per text before the random logical position receives
    # one third of P3 weight.
    mean = np.mean(np.stack(replicas), axis=0)
    return mean / np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)


def encode_all_positions(
    oracle: FinalEmbeddingOracle,
    records: Sequence[Mapping[str, str]],
    token_text: str,
    *,
    role: str,
    manifest: BoundaryManifest,
    random_replicates: int,
) -> dict[str, np.ndarray]:
    return {
        position: encode_position(
            oracle, records, token_text, position, role=role, manifest=manifest,
            random_replicates=random_replicates,
        )
        for position in ("prefix", "suffix", "random")
    }


def position_balanced_concat(values: Mapping[str, np.ndarray]) -> np.ndarray:
    count = min(len(values[position]) for position in ("prefix", "suffix", "random"))
    return np.stack([values[position][:count] for position in ("prefix", "suffix", "random")], axis=1).reshape(-1, values["prefix"].shape[1])


def evaluate_shared_token(
    oracle: FinalEmbeddingOracle,
    *,
    token_id: int,
    token_text: str,
    fit_records: Sequence[Mapping[str, str]],
    eval_records: Sequence[Mapping[str, str]],
    benign_records: Sequence[Mapping[str, str]],
    fit_role: str,
    eval_role: str,
    manifest: BoundaryManifest,
    random_replicates: int,
    coverage: float,
    maximum_radius_degrees: float,
    source_tracks: tuple[str, ...],
) -> tuple[object, object, dict[str, np.ndarray]]:
    fit = encode_all_positions(oracle, fit_records, token_text, role=fit_role, manifest=manifest, random_replicates=random_replicates)
    evaluation = encode_all_positions(oracle, eval_records, token_text, role=eval_role, manifest=manifest, random_replicates=random_replicates)
    cap = fit_and_calibrate_single_cap(
        token_id, token_text, "P3_shared", fit, evaluation, coverage=coverage,
        maximum_radius_degrees=maximum_radius_degrees, source_tracks=source_tracks,
    )
    triggered = position_balanced_concat(evaluation)
    clean_single = encode_clean(oracle, eval_records)
    paired_clean = np.repeat(clean_single, 3, axis=0)
    benign = encode_clean(oracle, benign_records)
    metrics = candidate_metrics(cap, triggered, paired_clean, benign, source_tracks)
    return cap, metrics, {"triggered": triggered, "paired_clean": paired_clean, "independent_benign": benign}


def evaluate_position_layers(
    oracle: FinalEmbeddingOracle,
    *,
    token_id: int,
    token_text: str,
    fit_records: Sequence[Mapping[str, str]],
    calibration_records: Sequence[Mapping[str, str]],
    benign_records: Sequence[Mapping[str, str]],
    fit_role: str,
    calibration_role: str,
    manifest: BoundaryManifest,
    random_replicates: int,
    config: Mapping[str, object],
) -> dict[str, object]:
    """Report P1 per-position and P2 conditional-center evidence separately."""
    fit = encode_all_positions(oracle, fit_records, token_text, role=fit_role, manifest=manifest, random_replicates=random_replicates)
    calibration = encode_all_positions(oracle, calibration_records, token_text, role=calibration_role, manifest=manifest, random_replicates=random_replicates)
    clean = encode_clean(oracle, calibration_records)
    benign = encode_clean(oracle, benign_records)
    p1: dict[str, object] = {}
    for position in ("prefix", "suffix", "random"):
        try:
            cap = fit_and_calibrate_single_cap(
                token_id, token_text, "P1_position", fit[position], calibration[position],
                coverage=config["geometry"]["calibration"]["weak_coverage"],
                maximum_radius_degrees=config["geometry"]["maximum_radius_degrees"], source_tracks=("full_search",),
            )
            p1[position] = certify_frozen_cap(
                cap, calibration[position], clean, benign,
                confidence=config["certification"]["confidence"],
                coverage_lcb_threshold=config["certification"]["triggered_coverage_lcb"],
                occupancy_ucb_threshold=config["certification"]["independent_benign_occupancy_ucb"],
                outside_to_inside_lcb_threshold=config["certification"]["outside_to_inside_lcb"],
                conditional_outside_origin_lcb_threshold=config["certification"]["conditional_outside_origin_lcb"],
                radial_multipliers=config["radial_analysis"]["multipliers"],
            )
        except Exception as error:
            if is_resource_exhaustion(error):
                raise
            p1[position] = {"certified": False, "status": "invalid", "error_type": type(error).__name__, "error": str(error)}
    # P2 is the same token with three conditional frozen centers/radii. Its
    # certification is the conservative conjunction of the three independent
    # position reports, not a pseudo-replicated pooled binomial interval.
    p2 = {
        "protocol": "P2_conditional", "same_token": True,
        "position_conditional_centers": {position: p1[position]["cap"]["centers"][0] for position in p1 if "cap" in p1[position]},
        "position_conditional_radii": {position: p1[position]["cap"]["radii"][0] for position in p1 if "cap" in p1[position]},
        "certified": all(bool(p1[position]["certified"]) for position in p1),
        "aggregation": "conservative_all_positions",
    }
    return {"P1_position_specific": p1, "P2_conditional": p2}
