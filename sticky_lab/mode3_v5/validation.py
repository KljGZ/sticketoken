"""Bootstrap stability, frozen attractor models, and no-refit evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .atomic_io import write_completion, write_json, write_npz
from .interfaces import ClusterStructure
from .occupancy import (
    clopper_pearson_lower,
    clopper_pearson_upper,
    cosine_distance_to_centers,
    evaluate_multiscale_occupancy,
    fixed_structure_coverage,
)
from .scoring import EvaluationBundle


def _logical_values(bundle: EvaluationBundle, name: str, task: str) -> np.ndarray:
    if task == "shared":
        return np.concatenate(list(bundle.view_embeddings.values()), axis=0)
    if task == "conditional" and name == "random":
        return np.concatenate(
            [values for view, values in bundle.view_embeddings.items() if view.startswith("random_r")], axis=0
        )
    if name in bundle.view_embeddings:
        return bundle.view_embeddings[name]
    return np.concatenate(list(bundle.view_embeddings.values()), axis=0)


def _remap_labels(labels: np.ndarray, column_to_original: np.ndarray) -> np.ndarray:
    result = np.empty_like(labels)
    for observed, original in enumerate(column_to_original):
        result[labels == observed] = int(original)
    return result


def _mean_cluster_jaccard(left: np.ndarray, right: np.ndarray, count: int) -> float:
    values = []
    for cluster in range(count):
        a = left == cluster
        b = right == cluster
        union = np.count_nonzero(a | b)
        values.append(1.0 if union == 0 else np.count_nonzero(a & b) / union)
    return float(np.mean(values))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    if cumulative[-1] <= 0:
        return float("inf")
    index = int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def bootstrap_cluster_stability(
    values: np.ndarray,
    structure: ClusterStructure,
    *,
    replicates: int,
    anchor_count: int,
    seed: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score

    values = np.asarray(values, dtype=np.float64)
    count = int(structure.cluster_count)
    rng = np.random.default_rng(seed)
    anchors = np.sort(rng.choice(len(values), size=min(anchor_count, len(values)), replace=False))
    anchor_values = values[anchors]
    original_anchor = np.argmin(cosine_distance_to_centers(anchor_values, structure.centers), axis=1)
    center_drifts = np.zeros((replicates, count), dtype=np.float64)
    radii_q95 = np.zeros((replicates, count), dtype=np.float64)
    masses = np.zeros((replicates, count), dtype=np.float64)
    aris = np.zeros(replicates, dtype=np.float64)
    jaccards = np.zeros(replicates, dtype=np.float64)
    # The registered bootstrap population is a deterministic representative
    # anchor set.  Each replicate resamples that empirical distribution and
    # refits every spherical centroid by a weighted M-step.  This preserves the
    # registered cluster count, captures center/radius/mass instability, and
    # avoids 500 fresh high-dimensional K-means runs per validated candidate.
    counts = rng.multinomial(len(anchors), np.full(len(anchors), 1.0 / len(anchors)), size=replicates)
    for replicate in range(replicates):
        weights = counts[replicate].astype(np.float64)
        centers = structure.centers.astype(np.float64).copy()
        for cluster in range(count):
            member_mask = original_anchor == cluster
            weighted = weights[member_mask]
            if weighted.sum() > 0:
                center = weighted @ anchor_values[member_mask]
                norm = np.linalg.norm(center)
                if norm > 0:
                    centers[cluster] = center / norm
        cost = cosine_distance_to_centers(structure.centers, centers)
        original_rows, bootstrap_columns = linear_sum_assignment(cost)
        column_to_original = np.empty(count, dtype=np.int64)
        column_to_original[bootstrap_columns] = original_rows
        matched_centers = centers[bootstrap_columns[np.argsort(original_rows)]]
        center_drifts[replicate] = np.maximum(0.0, 1.0 - np.sum(structure.centers * matched_centers, axis=1))
        bootstrap_distances = cosine_distance_to_centers(anchor_values, matched_centers)
        remapped = np.argmin(bootstrap_distances, axis=1)
        for cluster in range(count):
            member_mask = remapped == cluster
            cluster_values = bootstrap_distances[member_mask, cluster]
            cluster_weights = weights[member_mask]
            masses[replicate, cluster] = cluster_weights.sum() / weights.sum()
            radii_q95[replicate, cluster] = (
                _weighted_quantile(cluster_values, cluster_weights, 0.95)
                if len(cluster_values) and cluster_weights.sum() > 0
                else float("inf")
            )
        anchor_assignments = remapped
        aris[replicate] = adjusted_rand_score(original_anchor, anchor_assignments)
        jaccards[replicate] = _mean_cluster_jaccard(original_anchor, anchor_assignments, count)
    minimum_mass = float(config["structure"]["minimum_cluster_inlier_mass"])
    persistent = (center_drifts <= 0.10) & (masses >= minimum_mass)
    return {
        "replicates": int(replicates),
        "anchor_count": int(len(anchors)),
        "cluster_persistence": np.mean(persistent, axis=0).tolist(),
        "minimum_cluster_persistence": float(np.min(np.mean(persistent, axis=0))),
        "center_drift_q50": np.quantile(center_drifts, 0.50, axis=0).tolist(),
        "center_drift_q90": np.quantile(center_drifts, 0.90, axis=0).tolist(),
        "center_drift_q95": np.quantile(center_drifts, 0.95, axis=0).tolist(),
        "assignment_ari_q50": float(np.quantile(aris, 0.50)),
        "assignment_ari_q05": float(np.quantile(aris, 0.05)),
        "assignment_jaccard_q50": float(np.quantile(jaccards, 0.50)),
        "cluster_mass_lcb": np.quantile(masses, 0.05, axis=0).tolist(),
        "cluster_mass_ucb": np.quantile(masses, 0.95, axis=0).tolist(),
        "radius_q95_bootstrap_ucb": np.quantile(radii_q95, 0.95, axis=0).tolist(),
        "cluster_count_stability": 1.0,
    }


@dataclass
class FrozenStructure:
    name: str
    centers: np.ndarray
    radii: np.ndarray
    cluster_count: int
    outlier_budget: float
    assignment_rule: str = "nearest_cosine_center_then_frozen_radius_membership"


@dataclass
class FrozenCandidate:
    candidate_key: str
    token_ids: tuple[int, ...]
    trigger: str
    actual_token_length: int
    task: str
    structures: dict[str, FrozenStructure]
    validation_summary: dict[str, Any]


def freeze_validation_bundle(
    bundle: EvaluationBundle,
    *,
    candidate_key: str,
    token_ids: tuple[int, ...],
    trigger: str,
    task: str,
    benign_probe: np.ndarray,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[FrozenCandidate, dict[str, Any]]:
    validation = config["validation"]
    objectives = config["objectives"]
    structures: dict[str, FrozenStructure] = {}
    stability: dict[str, dict[str, Any]] = {}
    structure_metrics: dict[str, dict[str, Any]] = {}
    for index, (name, structure) in enumerate(sorted(bundle.structures.items())):
        values = _logical_values(bundle, name, task)
        report = bootstrap_cluster_stability(
            values,
            structure,
            replicates=int(validation["bootstrap_replicates"]),
            anchor_count=int(validation["bootstrap_anchor_count"]),
            seed=seed + index * 100003,
            config=config,
        )
        stability[name] = report
        point_q95 = structure.radius_quantiles[:, 2]
        bootstrap_q95 = np.asarray(report["radius_q95_bootstrap_ucb"], dtype=np.float64)
        radii = np.maximum(point_q95, bootstrap_q95)
        occupancy, occupancy_ucb, occupancy_auc, lambda_star = evaluate_multiscale_occupancy(
            benign_probe,
            structure.centers,
            radii,
            objectives["occupancy_lambdas"],
            confidence=float(objectives["occupancy_confidence"]),
            epsilon=float(objectives["low_occupancy_epsilon"]),
        )
        fixed_coverage = fixed_structure_coverage(
            values,
            structure.centers,
            radii,
            confidence=float(validation["confidence"]),
        )
        structure_metrics[name] = {
            **fixed_coverage,
            "cmax": float(np.max(radii)),
            "cavg": float(np.sum(structure.masses * radii) / np.sum(structure.masses)),
            "occupancy": occupancy.tolist(),
            "occupancy_ucb": occupancy_ucb.tolist(),
            "occupancy_auc": occupancy_auc,
            "lambda_star": lambda_star,
            "cluster_count": int(structure.cluster_count),
            "minimum_cluster_mass": float(np.min(structure.masses)),
        }
        structures[name] = FrozenStructure(
            name=name,
            centers=structure.centers.copy(),
            radii=radii,
            cluster_count=structure.cluster_count,
            outlier_budget=float(config["structure"]["maximum_outlier_rate"]),
        )

    position_metrics: dict[str, dict[str, float | int]] = {}
    for view, values in bundle.view_embeddings.items():
        structure_name = "shared" if task == "shared" else ("random" if view.startswith("random_r") and task == "conditional" else task if task in {"prefix", "suffix", "random"} else view)
        frozen = structures[structure_name]
        position_metrics[view] = fixed_structure_coverage(
            values, frozen.centers, frozen.radii, confidence=float(validation["confidence"])
        )
    summary = {
        "structure_metrics": structure_metrics,
        "stability": stability,
        "position_metrics": position_metrics,
        "cmax": max(float(value["cmax"]) for value in structure_metrics.values()),
        "cavg": max(float(value["cavg"]) for value in structure_metrics.values()),
        "occupancy_auc": max(float(value["occupancy_auc"]) for value in structure_metrics.values()),
        "lambda_star": min(float(value["lambda_star"]) for value in structure_metrics.values()),
        "coverage_lcb": min(float(value["coverage_lcb"]) for value in structure_metrics.values()),
        "outlier_rate_ucb": max(float(value["outlier_rate_ucb"]) for value in structure_metrics.values()),
        "worst_position_coverage_lcb": min(float(value["coverage_lcb"]) for value in position_metrics.values()),
        "minimum_cluster_mass_lcb": min(
            min(map(float, value["cluster_mass_lcb"])) for value in stability.values()
        ),
        "minimum_cluster_persistence": min(
            float(value["minimum_cluster_persistence"]) for value in stability.values()
        ),
        "minimum_assignment_ari": min(float(value["assignment_ari_q50"]) for value in stability.values()),
    }
    frozen_candidate = FrozenCandidate(
        candidate_key=candidate_key,
        token_ids=token_ids,
        trigger=trigger,
        actual_token_length=len(token_ids),
        task=task,
        structures=structures,
        validation_summary=summary,
    )
    return frozen_candidate, summary


def certification(summary: Mapping[str, Any], realizability: Mapping[str, Any], thresholds: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    structure = config["structure"]
    validation = config["validation"]
    objectives = config["objectives"]
    gates = {
        "realizable": bool(realizability.get("exact_token_roundtrip"))
        and float(realizability.get("context_realizability", 0.0)) >= float(config["tokenizer"]["require_context_realizability"])
        and bool(realizability.get("inserted_once", False)),
        "coverage": float(summary["coverage_lcb"]) >= float(structure["minimum_total_coverage"]),
        "position_coverage": float(summary["worst_position_coverage_lcb"])
        >= float(structure["minimum_per_position_coverage"]),
        "outliers": float(summary["outlier_rate_ucb"]) <= float(structure["maximum_outlier_rate"]),
        "cluster_mass": float(summary["minimum_cluster_mass_lcb"]) >= float(structure["minimum_cluster_inlier_mass"]),
        "compactness": float(summary["cmax"]) <= float(thresholds["maximum_cmax"]),
        "cluster_persistence": float(summary["minimum_cluster_persistence"])
        >= float(validation["minimum_cluster_persistence"]),
        "assignment_ari": float(summary["minimum_assignment_ari"]) >= float(validation["minimum_assignment_ari"]),
    }
    level_a = all(gates.values())
    low_occupancy_gates = {
        "occupancy_auc": float(summary["occupancy_auc"]) <= float(objectives["maximum_occupancy_auc_ucb"]),
        "lambda_star": float(summary["lambda_star"]) >= float(objectives["minimum_low_occupancy_lambda_star"]),
    }
    return {
        "level_0_realizable": gates["realizable"],
        "level_1_attractor": level_a,
        "level_2_low_occupancy": level_a and all(low_occupancy_gates.values()),
        "level_1_gates": gates,
        "level_2_gates": low_occupancy_gates,
    }


def save_frozen_candidate(output: Path, candidate: FrozenCandidate, certification_record: Mapping[str, Any]) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    metadata = output / "frozen_candidate.json"
    arrays = []
    structure_metadata = {}
    for name, structure in sorted(candidate.structures.items()):
        path = output / f"structure_{name}.npz"
        write_npz(path, centers=structure.centers, radii=structure.radii)
        arrays.append(path)
        structure_metadata[name] = {
            "file": path.name,
            "cluster_count": structure.cluster_count,
            "outlier_budget": structure.outlier_budget,
            "assignment_rule": structure.assignment_rule,
        }
    write_json(
        metadata,
        {
            "schema_version": "mode3-v5-frozen-candidate-v1",
            "candidate_key": candidate.candidate_key,
            "token_ids": list(candidate.token_ids),
            "trigger": candidate.trigger,
            "actual_token_length": candidate.actual_token_length,
            "task": candidate.task,
            "structures": structure_metadata,
            "validation_summary": candidate.validation_summary,
            "certification": dict(certification_record),
            "refit_prohibited": True,
        },
    )
    write_completion(output, [metadata, *arrays], {"candidate_key": candidate.candidate_key, "task": candidate.task})
    return [metadata, *arrays, output / "COMPLETE.json"]


def load_frozen_candidate(output: Path) -> FrozenCandidate:
    payload = json.loads((output / "frozen_candidate.json").read_text(encoding="utf-8"))
    structures = {}
    for name, record in payload["structures"].items():
        arrays = np.load(output / record["file"])
        structures[name] = FrozenStructure(
            name=name,
            centers=arrays["centers"],
            radii=arrays["radii"],
            cluster_count=int(record["cluster_count"]),
            outlier_budget=float(record["outlier_budget"]),
            assignment_rule=str(record["assignment_rule"]),
        )
    return FrozenCandidate(
        candidate_key=str(payload["candidate_key"]),
        token_ids=tuple(map(int, payload["token_ids"])),
        trigger=str(payload["trigger"]),
        actual_token_length=int(payload["actual_token_length"]),
        task=str(payload["task"]),
        structures=structures,
        validation_summary=payload["validation_summary"],
    )


def evaluate_frozen_no_refit(
    candidate: FrozenCandidate,
    view_embeddings: Mapping[str, np.ndarray],
    benign_probe: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    confidence = float(config["test"]["coverage_confidence"])
    position = {}
    for view, values in view_embeddings.items():
        name = "shared" if candidate.task == "shared" else (
            "random" if view.startswith("random_r") and candidate.task == "conditional" else candidate.task if candidate.task in {"prefix", "suffix", "random"} else view
        )
        structure = candidate.structures[name]
        position[view] = fixed_structure_coverage(values, structure.centers, structure.radii, confidence=confidence)
    structure_metrics = {}
    for name, structure in candidate.structures.items():
        occupancy, occupancy_ucb, occupancy_auc, lambda_star = evaluate_multiscale_occupancy(
            benign_probe,
            structure.centers,
            structure.radii,
            config["objectives"]["occupancy_lambdas"],
            confidence=float(config["objectives"]["occupancy_confidence"]),
            epsilon=float(config["objectives"]["low_occupancy_epsilon"]),
        )
        structure_metrics[name] = {
            "occupancy": occupancy.tolist(),
            "occupancy_ucb": occupancy_ucb.tolist(),
            "occupancy_auc": occupancy_auc,
            "lambda_star": lambda_star,
        }
    summary = {
        "refit_performed": False,
        "position_metrics": position,
        "structure_metrics": structure_metrics,
        "worst_position_coverage_lcb": min(float(value["coverage_lcb"]) for value in position.values()),
        "maximum_outlier_rate_ucb": max(float(value["outlier_rate_ucb"]) for value in position.values()),
        "occupancy_auc": max(float(value["occupancy_auc"]) for value in structure_metrics.values()),
        "lambda_star": min(float(value["lambda_star"]) for value in structure_metrics.values()),
    }
    summary["level_1_test_pass"] = (
        summary["worst_position_coverage_lcb"] >= float(config["structure"]["minimum_per_position_coverage"])
        and summary["maximum_outlier_rate_ucb"] <= float(config["structure"]["maximum_outlier_rate"])
    )
    summary["level_2_test_pass"] = (
        summary["level_1_test_pass"]
        and summary["occupancy_auc"] <= float(config["objectives"]["maximum_occupancy_auc_ucb"])
        and summary["lambda_star"] >= float(config["objectives"]["minimum_low_occupancy_lambda_star"])
    )
    return summary
