"""Registered e* and token-specific tangential diagnostics for V7."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from sticky_lab.mode3_v6_3.errors import ShapeMismatch

from .geometry import angle_between, normalize_rows, normalize_vector


def tangent_direction(center: np.ndarray, e_star: np.ndarray) -> np.ndarray:
    center_value = normalize_vector(center)
    axis = normalize_vector(e_star)
    residual = center_value - float(center_value @ axis) * axis
    norm = float(np.linalg.norm(residual))
    if norm <= 1e-12:
        raise ShapeMismatch("center is collinear with e*; tangent direction is undefined")
    return residual / norm


def axis_frontier_diagnostics(
    center: np.ndarray,
    e_star: np.ndarray,
    radius_points: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    angle = angle_between(center, e_star)
    rows: list[dict[str, Any]] = []
    beta_axis: float | None = None
    for point in radius_points:
        radius = point.get("radius")
        feasible = bool(point.get("feasible", radius is not None))
        margin = None if not feasible or radius is None else angle - float(radius)
        inside = None if margin is None else bool(margin <= 0)
        beta = float(point["beta"])
        if inside and beta_axis is None:
            beta_axis = beta
        rows.append(
            {
                "beta": beta,
                "radius_radians": None if radius is None else float(radius),
                "axis_exclusion_margin_radians": margin,
                "axis_exclusion_margin_degrees": None
                if margin is None
                else math.degrees(margin),
                "e_star_inside_cap": inside,
            }
        )
    try:
        tangent = tangent_direction(center, e_star).tolist()
    except ShapeMismatch:
        tangent = None
    return {
        "angle_center_to_e_star_radians": angle,
        "angle_center_to_e_star_degrees": math.degrees(angle),
        "tangent_direction": tangent,
        "beta_axis": beta_axis,
        "frontier": rows,
        "selection_uses_axis_geometry": False,
    }

def displacement_decomposition(
    clean_vectors: np.ndarray,
    triggered_vectors: np.ndarray,
    *,
    center: np.ndarray,
    e_star: np.ndarray,
) -> dict[str, Any]:
    clean = normalize_rows(clean_vectors)
    triggered = normalize_rows(triggered_vectors)
    if clean.shape != triggered.shape:
        raise ShapeMismatch("clean/triggered displacement vectors differ")
    axis = normalize_vector(e_star)
    tangent = tangent_direction(center, axis)
    delta = triggered - clean
    axial = delta @ axis
    tangential = delta @ tangent
    residual = delta - axial[:, None] * axis - tangential[:, None] * tangent
    residual_norm = np.linalg.norm(residual, axis=1)

    def summary(values: np.ndarray) -> dict[str, Any]:
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q95": float(np.quantile(values, 0.95)),
            "positive_rate": float(np.mean(values > 0)),
        }

    return {
        "observations": len(clean),
        "axial_component": summary(axial),
        "token_tangent_component": summary(tangential),
        "orthogonal_residual_norm": summary(residual_norm),
        "mean_tangent_to_axial_ratio": None
        if abs(float(np.mean(axial))) <= 1e-12
        else float(np.mean(tangential) / np.mean(axial)),
    }
