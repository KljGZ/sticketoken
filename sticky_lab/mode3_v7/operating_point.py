"""End-to-end candidate frontier construction and primary V7 metrics."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from sticky_lab.mode3_v6_3.errors import ShapeMismatch

from .axis_geometry import axis_frontier_diagnostics
from .geometry import (
    RobustSharedCenter,
    Stratum,
    angular_distance,
    center_bootstrap_drift,
    fit_robust_shared_center,
)
from .migration import migration_diagnostics
from .radius_policy import legacy_q92_radius, occupancy_constrained_frontier
from .statistics import group_membership, source_position_coverage


POSITIONS = ("prefix", "suffix")


def _stratify(
    rows: Sequence[Mapping[str, Any]], vectors: np.ndarray
) -> dict[Stratum, np.ndarray]:
    matrix = np.asarray(vectors)
    if len(rows) != len(matrix):
        raise ShapeMismatch("row/vector alignment mismatch")
    grouped: dict[Stratum, list[np.ndarray]] = {}
    for row, vector in zip(rows, matrix):
        position = str(row["position"])
        if position not in POSITIONS:
            raise ShapeMismatch(f"V7 encountered forbidden position {position}")
        grouped.setdefault((str(row["source_id"]), position), []).append(
            np.asarray(vector)
        )
    return {key: np.stack(values) for key, values in grouped.items()}


def _distances_by_source(
    rows: Sequence[Mapping[str, Any]], vectors: np.ndarray, center: np.ndarray
) -> dict[str, np.ndarray]:
    matrix = np.asarray(vectors)
    if len(rows) != len(matrix):
        raise ShapeMismatch("calibration row/vector alignment mismatch")
    distances = angular_distance(matrix, center[None, :]).reshape(-1)
    grouped: dict[str, list[float]] = {}
    for row, distance in zip(rows, distances):
        grouped.setdefault(str(row["source_id"]), []).append(float(distance))
    return {source: np.asarray(values, dtype=np.float64) for source, values in grouped.items()}


def coverage_auc_log_beta(points: Sequence[Mapping[str, Any]]) -> float:
    if len(points) < 2:
        return 0.0
    x = np.log(np.asarray([float(point["beta"]) for point in points], dtype=np.float64))
    y = np.asarray(
        [
            min(float(point["prefix_coverage_lcb"]), float(point["suffix_coverage_lcb"]))
            if bool(point["feasible"])
            else 0.0
            for point in points
        ],
        dtype=np.float64,
    )
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # pragma: no cover - NumPy < 2
        integrate = getattr(np, "trapz")
    return float(integrate(y, x) / (x[-1] - x[0]))


def evidence_grade(beta80: float | None) -> str:
    if beta80 is None or beta80 > 0.15:
        return "NO_PS_80_CAP_IN_REGISTERED_RANGE"
    if beta80 <= 0.01:
        return "STRONG_LOW_OCCUPANCY_FROZEN_CAP"
    if beta80 <= 0.03:
        return "MODERATE_SELECTIVE_FROZEN_CAP"
    if beta80 <= 0.05:
        return "WEAK_SELECTIVE_FROZEN_CAP"
    return "HIGH_OCCUPANCY_DIRECTIONAL_CAP"


def build_candidate_frontier(
    *,
    token_id: int,
    token_text: str,
    fit_rows: Sequence[Mapping[str, Any]],
    fit_vectors: np.ndarray,
    calibration_rows: Sequence[Mapping[str, Any]],
    calibration_vectors: np.ndarray,
    select_rows: Sequence[Mapping[str, Any]],
    triggered_select_vectors: np.ndarray,
    paired_clean_vectors: np.ndarray,
    e_star: np.ndarray,
    role_hashes: Mapping[str, str],
    config: Mapping[str, Any],
    stage: str,
) -> dict[str, Any]:
    """Fit one center, derive every radius, then score without any refit."""

    if np.asarray(triggered_select_vectors).shape != np.asarray(paired_clean_vectors).shape:
        raise ShapeMismatch("select triggered/paired-clean vectors differ")
    geometry = config["geometry"]
    seed = int(config["positions"]["seed"]) + 1009 * int(token_id)
    fit_grid = _stratify(fit_rows, fit_vectors)
    fitted: RobustSharedCenter = fit_robust_shared_center(
        fit_grid,
        trim_fraction=float(geometry["center_trim_fraction"]),
        restarts=int(geometry["stage_restarts"][stage]),
        maximum_iterations=int(geometry["maximum_iterations"]),
        tolerance=float(geometry["tolerance"]),
        seed=seed,
    )
    center = fitted.center
    calibration_distance = _distances_by_source(
        calibration_rows, calibration_vectors, center
    )
    radius_points = occupancy_constrained_frontier(
        calibration_distance,
        config["radius"]["occupancy_grid"],
        maximum_radius_degrees=float(geometry["maximum_radius_degrees"]),
        familywise_alpha=float(config["radius"]["familywise_alpha"]),
    )
    select_triggered = np.asarray(triggered_select_vectors)
    select_clean = np.asarray(paired_clean_vectors)
    trigger_distance = angular_distance(select_triggered, center[None, :]).reshape(-1)
    clean_distance = angular_distance(select_clean, center[None, :]).reshape(-1)
    trigger_grid = _stratify(select_rows, select_triggered)
    trigger_distance_grid = {
        key: angular_distance(values, center[None, :]).reshape(-1)
        for key, values in trigger_grid.items()
    }
    axis = axis_frontier_diagnostics(
        center, e_star, [point.to_dict() for point in radius_points]
    )

    point_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    coverage_alpha = float(config["certification"]["familywise_alpha"])
    for radius_point, axis_point in zip(radius_points, axis["frontier"]):
        if radius_point.feasible and radius_point.radius is not None:
            radius = float(radius_point.radius)
            triggered_inside = trigger_distance <= radius
            clean_inside = clean_distance <= radius
            trigger_membership = group_membership(
                select_rows, triggered_inside, require_unique_texts=False
            )
            clean_membership = group_membership(
                select_rows, clean_inside, require_unique_texts=False
            )
            coverage = source_position_coverage(
                trigger_membership, familywise_alpha=coverage_alpha
            )
            migration = migration_diagnostics(
                clean_membership,
                trigger_membership,
                familywise_alpha=coverage_alpha,
            )
        else:
            coverage = {
                "prefix_lcb": 0.0,
                "suffix_lcb": 0.0,
                "position": {
                    position: {"point": 0.0, "lcb": 0.0, "by_source": {}}
                    for position in POSITIONS
                },
            }
            migration = {
                "position": {
                    position: {
                        "capture_outside_point": None,
                        "capture_outside_lcb": 0.0,
                        "outside_to_inside": None,
                        "conditional_origin_outside": None,
                        "inside_retention": None,
                        "net_gain": None,
                    }
                    for position in POSITIONS
                }
            }
        point = {
            **radius_point.to_dict(),
            "prefix_coverage_point": coverage["position"]["prefix"]["point"],
            "prefix_coverage_lcb": coverage["prefix_lcb"],
            "prefix_coverage_by_source": coverage["position"]["prefix"]["by_source"],
            "suffix_coverage_point": coverage["position"]["suffix"]["point"],
            "suffix_coverage_lcb": coverage["suffix_lcb"],
            "suffix_coverage_by_source": coverage["position"]["suffix"]["by_source"],
            "minimum_position_coverage_lcb": min(
                float(coverage["prefix_lcb"]), float(coverage["suffix_lcb"])
            ),
            "prefix_capture_outside_point": migration["position"]["prefix"]["capture_outside_point"],
            "prefix_capture_outside_lcb": migration["position"]["prefix"]["capture_outside_lcb"],
            "prefix_outside_to_inside": migration["position"]["prefix"]["outside_to_inside"],
            "prefix_conditional_origin_outside": migration["position"]["prefix"]["conditional_origin_outside"],
            "prefix_inside_retention": migration["position"]["prefix"]["inside_retention"],
            "prefix_net_gain": migration["position"]["prefix"]["net_gain"],
            "suffix_capture_outside_point": migration["position"]["suffix"]["capture_outside_point"],
            "suffix_capture_outside_lcb": migration["position"]["suffix"]["capture_outside_lcb"],
            "suffix_outside_to_inside": migration["position"]["suffix"]["outside_to_inside"],
            "suffix_conditional_origin_outside": migration["position"]["suffix"]["conditional_origin_outside"],
            "suffix_inside_retention": migration["position"]["suffix"]["inside_retention"],
            "suffix_net_gain": migration["position"]["suffix"]["net_gain"],
            "angle_center_to_e_star": axis["angle_center_to_e_star_radians"],
            "axis_exclusion_margin": axis_point["axis_exclusion_margin_radians"],
            "e_star_inside_cap": axis_point["e_star_inside_cap"],
            "migration": migration,
        }
        point_rows.append(point)
        for position in POSITIONS:
            position_prefix = f"{position}_"
            long_rows.append(
                {
                    "token_id": int(token_id),
                    "token_text": str(token_text),
                    "stage": str(stage),
                    "beta_target": float(radius_point.beta),
                    "position": position,
                    "center_hash": fitted.center_sha256,
                    "radius_degrees": radius_point.radius_degrees,
                    "radius_radians": radius_point.radius,
                    "feasible": radius_point.feasible,
                    "benign_occupancy_point": radius_point.benign_occupancy_point,
                    "benign_occupancy_ucb": radius_point.benign_occupancy_ucb,
                    "benign_occupancy_by_source": radius_point.benign_occupancy_by_source,
                    "worst_source_occupancy_ucb": radius_point.worst_source_occupancy_ucb,
                    "coverage_point": point[f"{position_prefix}coverage_point"],
                    "coverage_lcb": point[f"{position_prefix}coverage_lcb"],
                    "coverage_by_source": point[f"{position_prefix}coverage_by_source"],
                    "capture_outside_point": point[f"{position_prefix}capture_outside_point"],
                    "capture_outside_lcb": point[f"{position_prefix}capture_outside_lcb"],
                    "outside_to_inside": point[f"{position_prefix}outside_to_inside"],
                    "conditional_origin_outside": point[f"{position_prefix}conditional_origin_outside"],
                    "inside_retention": point[f"{position_prefix}inside_retention"],
                    "net_gain": point[f"{position_prefix}net_gain"],
                    "angle_center_to_e_star": point["angle_center_to_e_star"],
                    "axis_exclusion_margin": point["axis_exclusion_margin"],
                    "e_star_inside_cap": point["e_star_inside_cap"],
                    "center_restart_spread": fitted.restart_spread,
                }
            )

    threshold = float(config["certification"]["prefix_coverage_lcb"])
    passing = [
        point
        for point in point_rows
        if point["feasible"]
        and float(point["prefix_coverage_lcb"]) >= threshold
        and float(point["suffix_coverage_lcb"])
        >= float(config["certification"]["suffix_coverage_lcb"])
    ]
    beta80 = min((float(point["beta"]) for point in passing), default=None)
    auc = coverage_auc_log_beta(point_rows)
    bootstrap_samples = int(config.get("diagnostics", {}).get("center_bootstrap_samples", 0))
    bootstrap = (
        center_bootstrap_drift(
            fit_grid,
            center,
            samples=bootstrap_samples,
            seed=seed + 7_000_000,
            trim_fraction=float(geometry["center_trim_fraction"]),
            restarts=int(config.get("diagnostics", {}).get("bootstrap_restarts", 5)),
            maximum_iterations=int(geometry["maximum_iterations"]),
            tolerance=float(geometry["tolerance"]),
        )
        if bootstrap_samples > 0
        else {"samples": 0, "status": "SKIPPED_BY_PROFILE"}
    )
    return {
        "schema_version": "mode3-v7-candidate-frontier-v1",
        "token_id": int(token_id),
        "token_text": str(token_text),
        "stage": str(stage),
        "center": center.tolist(),
        "center_hash": fitted.center_sha256,
        "center_fit": fitted.to_dict(include_indices=False),
        "center_bootstrap_drift": bootstrap,
        "role_hashes": dict(role_hashes),
        "beta80_ps": beta80,
        "coverage_auc_log_beta": auc,
        "beta_axis": axis["beta_axis"],
        "beta80_precedes_beta_axis": bool(
            beta80 is not None
            and axis["beta_axis"] is not None
            and beta80 < float(axis["beta_axis"])
        ),
        "evidence_grade": evidence_grade(beta80),
        "frontier": point_rows,
        "long_rows": long_rows,
        "axis_geometry": axis,
        "legacy_q92": legacy_q92_radius(trigger_distance_grid),
        "center_refit_per_beta": False,
        "confirm_data_used": False,
    }


def operating_point_for_beta80(frontier: Mapping[str, Any]) -> dict[str, Any] | None:
    beta80 = frontier.get("beta80_ps")
    if beta80 is None:
        return None
    for point in frontier["frontier"]:
        if math.isclose(float(point["beta"]), float(beta80), rel_tol=0, abs_tol=1e-15):
            return dict(point)
    raise ShapeMismatch("beta80 is absent from its own frontier")
