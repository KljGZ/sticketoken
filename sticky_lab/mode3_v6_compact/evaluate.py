"""P3-first high-dimensional evaluation for the Compact candidate funnel."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from sticky_lab.mode3_v6.evaluation import (
    candidate_metrics,
    certify_frozen_cap,
    fit_and_calibrate_single_cap,
)
from sticky_lab.mode3_v6.experiment import position_balanced_concat
from sticky_lab.mode3_v6.geometry import FrozenCap, angular_distance
from sticky_lab.mode3_v6.insertion import BoundaryManifest, insert_once

from .oracle import CompactFinalOracle


POSITIONS = ("prefix", "suffix", "random")


def encode_positions(
    oracle: CompactFinalOracle,
    records: Sequence[Mapping[str, str]],
    token_text: str,
    *,
    role: str,
    manifest: BoundaryManifest,
    random_replicates: int,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for position in POSITIONS:
        if position != "random":
            texts = [
                insert_once(
                    row["text"], token_text, position, role=role,
                    text_id=row["text_id"], manifest=manifest,
                )
                for row in records
            ]
            result[position] = oracle.encode(
                texts, metadata=dict(metadata or {}, position=position, role=role)
            )
            continue
        replicas = []
        for replicate in range(int(random_replicates)):
            texts = [
                insert_once(
                    row["text"], token_text, "random", role=role,
                    text_id=row["text_id"], manifest=manifest, replicate=replicate,
                )
                for row in records
            ]
            replicas.append(
                oracle.encode(
                    texts,
                    metadata=dict(
                        metadata or {}, position="random", role=role, replicate=replicate
                    ),
                )
            )
        averaged = np.mean(np.stack(replicas), axis=0)
        result[position] = (
            averaged / np.maximum(np.linalg.norm(averaged, axis=1, keepdims=True), 1e-12)
        ).astype(np.float32)
    return result


@dataclass(frozen=True)
class DiscoveryMetric:
    token_id: int
    token_text: str
    stage: str
    status: str
    radius_radians: float
    radius_degrees: float
    triggered_coverage: float
    worst_position_coverage: float
    paired_clean_inside: float
    outside_to_inside: float
    triggered_similarity_q05: float
    triggered_similarity_q10: float
    benign_occupancy: float | None = None
    benign_similarity_q995: float | None = None
    search_margin_m90_1: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_s0(
    oracle: CompactFinalOracle,
    *,
    token_id: int,
    token_text: str,
    fit_records: Sequence[Mapping[str, str]],
    eval_records: Sequence[Mapping[str, str]],
    clean_eval: np.ndarray,
    manifest: BoundaryManifest,
    config: Mapping[str, Any],
) -> tuple[FrozenCap, DiscoveryMetric]:
    replicates = int(config["positions"]["discovery_random_replicates"])
    fit = encode_positions(
        oracle, fit_records, token_text, role="s0_fit", manifest=manifest,
        random_replicates=replicates, metadata={"token_id": token_id, "stage": "s0_fit"},
    )
    evaluation = encode_positions(
        oracle, eval_records, token_text, role="s0_eval", manifest=manifest,
        random_replicates=replicates, metadata={"token_id": token_id, "stage": "s0_eval"},
    )
    cap = fit_and_calibrate_single_cap(
        token_id,
        token_text,
        "P3_shared",
        fit,
        evaluation,
        coverage=float(config["geometry"]["calibration_coverage"]),
        maximum_radius_degrees=float(config["geometry"]["maximum_radius_degrees"]),
        source_tracks=("exhaustive_s0",),
    )
    triggered = position_balanced_concat(evaluation)
    clean = np.repeat(np.asarray(clean_eval), 3, axis=0)
    normalized = cap.normalized_radius(triggered)
    clean_inside = cap.contains(clean)
    triggered_inside = normalized <= 1.0
    outside_to_inside = float(np.mean((~clean_inside) & triggered_inside))
    position_coverage = {
        position: float(np.mean(cap.contains(vectors)))
        for position, vectors in evaluation.items()
    }
    similarities = np.clip(triggered @ cap.centers[0], -1.0, 1.0)
    metric = DiscoveryMetric(
        token_id=token_id,
        token_text=token_text,
        stage="s0",
        status="valid",
        radius_radians=float(cap.radii[0]),
        radius_degrees=float(math.degrees(cap.radii[0])),
        triggered_coverage=float(np.mean(triggered_inside)),
        worst_position_coverage=min(position_coverage.values()),
        paired_clean_inside=float(np.mean(clean_inside)),
        outside_to_inside=outside_to_inside,
        triggered_similarity_q05=float(np.quantile(similarities, 0.05)),
        triggered_similarity_q10=float(np.quantile(similarities, 0.10)),
    )
    return cap, metric


def evaluate_frozen_stage(
    oracle: CompactFinalOracle,
    cap: FrozenCap,
    *,
    token_text: str,
    role: str,
    records: Sequence[Mapping[str, str]],
    clean: np.ndarray,
    manifest: BoundaryManifest,
    config: Mapping[str, Any],
) -> DiscoveryMetric:
    values = encode_positions(
        oracle,
        records,
        token_text,
        role=role,
        manifest=manifest,
        random_replicates=int(config["positions"]["discovery_random_replicates"]),
        metadata={"token_id": cap.token_id, "stage": role},
    )
    triggered = position_balanced_concat(values)
    paired_clean = np.repeat(np.asarray(clean), 3, axis=0)
    tr_inside = cap.contains(triggered)
    cl_inside = cap.contains(paired_clean)
    similarities = np.clip(triggered @ cap.centers[0], -1.0, 1.0)
    return DiscoveryMetric(
        token_id=cap.token_id,
        token_text=token_text,
        stage=role,
        status="valid",
        radius_radians=float(cap.radii[0]),
        radius_degrees=float(math.degrees(cap.radii[0])),
        triggered_coverage=float(np.mean(tr_inside)),
        worst_position_coverage=min(float(np.mean(cap.contains(v))) for v in values.values()),
        paired_clean_inside=float(np.mean(cl_inside)),
        outside_to_inside=float(np.mean((~cl_inside) & tr_inside)),
        triggered_similarity_q05=float(np.quantile(similarities, 0.05)),
        triggered_similarity_q10=float(np.quantile(similarities, 0.10)),
    )


def attach_benign_metrics(
    metrics: Sequence[DiscoveryMetric],
    centers: np.ndarray,
    radii: np.ndarray,
    benign: np.ndarray,
    *,
    device: str,
) -> list[DiscoveryMetric]:
    """Vectorized benign occupancy for a complete shard."""
    import torch

    values = torch.as_tensor(np.asarray(benign), dtype=torch.float32, device=device)
    result: list[DiscoveryMetric] = []
    for start in range(0, len(metrics), 256):
        stop = min(start + 256, len(metrics))
        center = torch.as_tensor(centers[start:stop], dtype=torch.float32, device=device)
        similarities = values @ center.T
        thresholds = torch.cos(
            torch.as_tensor(radii[start:stop], dtype=torch.float32, device=device)
        )
        occupancies = (similarities >= thresholds[None, :]).float().mean(dim=0).cpu().numpy()
        q995 = torch.quantile(similarities, 0.995, dim=0).cpu().numpy()
        for offset, metric in enumerate(metrics[start:stop]):
            occupancy = float(occupancies[offset])
            benign_q = float(q995[offset])
            row = asdict(metric)
            row.update(
                {
                    "benign_occupancy": occupancy,
                    "benign_similarity_q995": benign_q,
                    "search_margin_m90_1": metric.triggered_similarity_q10 - benign_q,
                }
            )
            result.append(DiscoveryMetric(**row))
    return result


def validate_single_cap(
    oracle: CompactFinalOracle,
    *,
    token_id: int,
    token_text: str,
    fit_records: Sequence[Mapping[str, str]],
    calibration_records: Sequence[Mapping[str, str]],
    clean_calibration: np.ndarray,
    benign: np.ndarray,
    manifest: BoundaryManifest,
    config: Mapping[str, Any],
) -> tuple[FrozenCap, dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    replicates = int(config["positions"]["confirmation_random_replicates"])
    fit = encode_positions(
        oracle, fit_records, token_text, role="cap_fit", manifest=manifest,
        random_replicates=replicates, metadata={"token_id": token_id, "stage": "cap_fit"},
    )
    calibration = encode_positions(
        oracle, calibration_records, token_text, role="cap_calibration", manifest=manifest,
        random_replicates=replicates,
        metadata={"token_id": token_id, "stage": "cap_calibration"},
    )
    cap = fit_and_calibrate_single_cap(
        token_id,
        token_text,
        "P3_shared",
        fit,
        calibration,
        coverage=float(config["geometry"]["calibration_coverage"]),
        maximum_radius_degrees=float(config["geometry"]["maximum_radius_degrees"]),
        source_tracks=("compact_full_search",),
    )
    triggered = position_balanced_concat(calibration)
    clean = np.repeat(np.asarray(clean_calibration), 3, axis=0)
    result = certify_frozen_cap(
        cap,
        triggered,
        clean,
        np.asarray(benign),
        confidence=float(config["certification"]["confidence"]),
        coverage_lcb_threshold=float(config["certification"]["triggered_coverage_lcb"]),
        occupancy_ucb_threshold=float(
            config["certification"]["independent_benign_occupancy_ucb"]
        ),
        outside_to_inside_lcb_threshold=float(
            config["certification"]["outside_to_inside_lcb"]
        ),
        conditional_outside_origin_lcb_threshold=float(
            config["certification"]["conditional_outside_origin_lcb"]
        ),
        radial_multipliers=list(config["radial_analysis"]["multipliers"]),
    )
    raw = result.pop("raw_normalized_radius")
    arrays = {
        "fit_prefix": fit["prefix"],
        "fit_suffix": fit["suffix"],
        "fit_random": fit["random"],
        "calibration_prefix": calibration["prefix"],
        "calibration_suffix": calibration["suffix"],
        "calibration_random": calibration["random"],
        "triggered_normalized_radius": np.asarray(raw["triggered"], dtype=np.float32),
        "paired_clean_normalized_radius": np.asarray(raw["paired_clean"], dtype=np.float32),
        "benign_normalized_radius": np.asarray(raw["independent_benign"], dtype=np.float32),
    }
    # P1/P2 are derived from these exact position arrays; no extra encoding.
    p1: dict[str, Any] = {}
    for position in POSITIONS:
        position_cap = fit_and_calibrate_single_cap(
            token_id,
            token_text,
            "P1_position",
            fit[position],
            calibration[position],
            coverage=float(config["geometry"]["calibration_coverage"]),
            maximum_radius_degrees=float(config["geometry"]["maximum_radius_degrees"]),
            source_tracks=("compact_full_search",),
        )
        position_result = certify_frozen_cap(
            position_cap,
            calibration[position],
            np.asarray(clean_calibration),
            np.asarray(benign),
            confidence=float(config["certification"]["confidence"]),
            coverage_lcb_threshold=float(config["certification"]["triggered_coverage_lcb"]),
            occupancy_ucb_threshold=float(
                config["certification"]["independent_benign_occupancy_ucb"]
            ),
            outside_to_inside_lcb_threshold=float(
                config["certification"]["outside_to_inside_lcb"]
            ),
            conditional_outside_origin_lcb_threshold=float(
                config["certification"]["conditional_outside_origin_lcb"]
            ),
            radial_multipliers=list(config["radial_analysis"]["multipliers"]),
        )
        position_result.pop("raw_normalized_radius", None)
        p1[position] = position_result
    layers = {
        "P1_position_specific": p1,
        "P2_conditional": {
            "protocol": "P2_conditional",
            "same_token": True,
            "derived_from_same_embeddings": True,
            "certified": all(bool(p1[position]["certified"]) for position in POSITIONS),
            "aggregation": "conservative_all_positions",
        },
        "P3_shared": {"protocol": "P3_shared", "certified": bool(result["certified"])},
    }
    return cap, result, arrays, layers
