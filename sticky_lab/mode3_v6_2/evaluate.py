"""From-scratch stage fitting and high-dimensional scoring for V6.2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from sticky_lab.mode3_v6.insertion import BoundaryManifest

from .encoding import encode_audited_positions, primary_position_vectors
from .errors import CandidateRejected, ShapeMismatch
from .geometry import (
    FrozenCapModel, Stratum, fit_multicap_model, fit_single_cap,
)
from .oracle import V62FinalOracle
from .statistics import trapezoidal_integral


@dataclass(frozen=True)
class StageMetric:
    token_id: int
    token_text: str
    cap_count: int
    stage: str
    status: str
    radius_radians: float
    radius_degrees: float
    triggered_coverage: float
    coverage_margin: float
    worst_position_coverage: float
    worst_source_coverage: float
    paired_clean_inside: float
    outside_to_inside: float
    benign_occupancy: float
    benign_occupancy_1_10: float
    occupancy_auc_1_1_5: float
    triggered_similarity_q10: float
    benign_similarity_q995: float
    search_margin_m90_1: float
    semantic_anomaly: float = 0.0
    center_drift_from_previous: float = 0.0
    rank_stability: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stratify(
    records: Sequence[Mapping[str, str]],
    positions: Mapping[str, np.ndarray],
) -> dict[Stratum, np.ndarray]:
    source_indices: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        source_indices.setdefault(str(row["source_id"]), []).append(index)
    result: dict[Stratum, np.ndarray] = {}
    for position in ("prefix", "suffix", "random"):
        values = np.asarray(positions[position])
        if len(values) != len(records):
            raise ShapeMismatch("position vectors are not record aligned")
        for source, indices in sorted(source_indices.items()):
            result[(source, position)] = values[np.asarray(indices, dtype=int)]
    return result


def _stratify_clean(
    records: Sequence[Mapping[str, str]],
    clean: np.ndarray,
) -> dict[Stratum, np.ndarray]:
    matrix = np.asarray(clean)
    if len(matrix) != len(records):
        raise ShapeMismatch("clean cache is not role aligned")
    source_indices: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        source_indices.setdefault(str(row["source_id"]), []).append(index)
    return {
        (source, position): matrix[np.asarray(indices, dtype=int)]
        for source, indices in sorted(source_indices.items())
        for position in ("prefix", "suffix", "random")
    }


def _benign_sources(
    records: Sequence[Mapping[str, str]], benign: np.ndarray
) -> dict[str, np.ndarray]:
    matrix = np.asarray(benign)
    if len(matrix) != len(records):
        raise ShapeMismatch("benign cache is not role aligned")
    indices: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        indices.setdefault(str(row["source_id"]), []).append(index)
    return {source: matrix[np.asarray(rows, dtype=int)] for source, rows in sorted(indices.items())}


def _balanced_mean(values: Mapping[Stratum, np.ndarray]) -> float:
    return float(np.mean([float(np.mean(value)) for value in values.values()]))


def score_model(
    model: FrozenCapModel,
    *,
    score_records: Sequence[Mapping[str, str]],
    score_positions: Mapping[str, np.ndarray],
    clean_score: np.ndarray,
    benign_records: Sequence[Mapping[str, str]],
    benign: np.ndarray,
    stage: str,
) -> StageMetric:
    triggered = _stratify(score_records, score_positions)
    clean = _stratify_clean(score_records, clean_score)
    benign_by_source = _benign_sources(benign_records, benign)
    tr_inside = {key: model.contains(value) for key, value in triggered.items()}
    cl_inside = {key: model.contains(value) for key, value in clean.items()}
    coverage = _balanced_mean(tr_inside)
    positions = {
        position: float(np.mean([np.mean(value) for (source, p), value in tr_inside.items() if p == position]))
        for position in ("prefix", "suffix", "random")
    }
    sources = {
        source: float(np.mean([np.mean(value) for (s, position), value in tr_inside.items() if s == source]))
        for source in sorted({source for source, _ in tr_inside})
    }
    outside_to_inside = _balanced_mean({
        key: (~cl_inside[key]) & tr_inside[key] for key in tr_inside
    })
    benign_depth = np.concatenate([model.normalized_radius(value) for value in benign_by_source.values()])
    multipliers = np.asarray([1.0, 1.10, 1.25, 1.50], dtype=float)
    occupancy = np.asarray([np.mean(benign_depth <= value + 1e-12) for value in multipliers])
    auc = trapezoidal_integral(occupancy, multipliers) / 0.5
    triggered_matrix = np.concatenate(list(triggered.values()))
    if model.cap_count == 1:
        triggered_similarity = triggered_matrix @ model.centers[0]
        benign_similarity = np.asarray(benign) @ model.centers[0]
    else:
        triggered_similarity = np.max(triggered_matrix @ model.centers.T, axis=1)
        benign_similarity = np.max(np.asarray(benign) @ model.centers.T, axis=1)
    tr_q10 = float(np.quantile(triggered_similarity, 0.10))
    be_q995 = float(np.quantile(benign_similarity, 0.995))
    return StageMetric(
        token_id=model.token_id, token_text=model.token_text,
        cap_count=model.cap_count, stage=str(stage), status="valid",
        radius_radians=float(np.max(model.radii)),
        radius_degrees=float(np.max(np.degrees(model.radii))),
        triggered_coverage=coverage, coverage_margin=coverage - 0.90,
        worst_position_coverage=min(positions.values()),
        worst_source_coverage=min(sources.values()),
        paired_clean_inside=_balanced_mean(cl_inside),
        outside_to_inside=outside_to_inside,
        benign_occupancy=float(occupancy[0]),
        benign_occupancy_1_10=float(occupancy[1]),
        occupancy_auc_1_1_5=auc,
        triggered_similarity_q10=tr_q10,
        benign_similarity_q995=be_q995,
        search_margin_m90_1=tr_q10 - be_q995,
    )


def fit_and_score_stage(
    oracle: V62FinalOracle,
    *,
    token_id: int,
    token_text: str,
    stage: str,
    fit_records: Sequence[Mapping[str, str]],
    radius_records: Sequence[Mapping[str, str]],
    score_records: Sequence[Mapping[str, str]],
    benign_records: Sequence[Mapping[str, str]],
    clean_score: np.ndarray,
    benign: np.ndarray,
    manifest: BoundaryManifest,
    config: Mapping[str, Any],
    cap_counts: Sequence[int] = (1, 2, 3, 4),
    fit_role: Optional[str] = None,
    radius_role: Optional[str] = None,
    score_role: Optional[str] = None,
) -> tuple[
    list[tuple[FrozenCapModel, StageMetric, dict[str, Any], dict[str, np.ndarray]]],
    list[dict[str, Any]],
]:
    random_replicates = 1
    maximum_length = int(config["model"]["maximum_sequence_length"])
    encoded: dict[str, dict[str, np.ndarray]] = {}
    audits: dict[str, list[dict[str, Any]]] = {}
    fit_role = str(fit_role or f"{stage}_fit")
    radius_role = str(radius_role or f"{stage}_radius")
    score_role = str(score_role or f"{stage}_score")
    for role, records in ((fit_role, fit_records), (radius_role, radius_records), (score_role, score_records)):
        values, token_audits, _ = encode_audited_positions(
            oracle, records, token_id=token_id, token_text=token_text, role=role,
            manifest=manifest, random_replicates=random_replicates,
            maximum_length=maximum_length,
            metadata={"token_id": token_id, "stage": stage},
        )
        encoded[role] = primary_position_vectors(values)
        audits[role] = [audit.to_dict() for audit in token_audits]
    fit_grid = _stratify(fit_records, encoded[fit_role])
    radius_grid = _stratify(radius_records, encoded[radius_role])
    geometry = config["geometry"]
    output: list[tuple[FrozenCapModel, StageMetric, dict[str, Any], dict[str, np.ndarray]]] = []
    rejected: list[dict[str, Any]] = []
    for cap_count in cap_counts:
        seed = int(config["positions"]["random_seed"]) + int(token_id) * 17 + int(cap_count)
        try:
            if cap_count == 1:
                model, fit_audit = fit_single_cap(
                    token_id, token_text, fit_grid, radius_grid,
                    fit_role=fit_role, radius_role=radius_role,
                    design_coverage=float(geometry["design_coverage"]),
                    maximum_radius_degrees=float(geometry["maximum_radius_degrees"]),
                    trim_fraction=float(geometry["trim_fraction"]),
                    restarts=int(geometry["fit_restarts"]),
                    maximum_iterations=int(geometry["maximum_iterations"]),
                    tolerance=float(geometry["tolerance"]), seed=seed,
                )
            else:
                model, fit_audit = fit_multicap_model(
                    token_id, token_text, fit_grid, radius_grid, cap_count,
                    fit_role=fit_role, radius_role=radius_role,
                    design_coverage=float(geometry["design_coverage"]),
                    maximum_radius_degrees=float(geometry["maximum_radius_degrees"]),
                    minimum_cluster_mass=float(geometry["minimum_cluster_mass"]),
                    minimum_stratum_cluster_mass=float(geometry["minimum_stratum_cluster_mass"]),
                    maximum_outlier_fraction=float(geometry["maximum_outlier_fraction"]),
                    restarts=int(geometry["fit_restarts"]),
                    maximum_iterations=int(geometry["maximum_iterations"]), seed=seed,
                )
        except CandidateRejected as error:
            rejected.append({
                "token_id": int(token_id), "token_text": token_text,
                "cap_count": int(cap_count), "stage": stage,
                "status": "candidate_rejected", "reason": type(error).__name__,
                "detail": str(error),
            })
            continue
        metric = score_model(
            model, score_records=score_records,
            score_positions=encoded[score_role], clean_score=clean_score,
            benign_records=benign_records, benign=benign, stage=stage,
        )
        output.append((model, metric, {"geometry": fit_audit, "tokenization_audit": audits}, encoded[score_role]))
    return output, rejected
