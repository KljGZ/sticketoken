"""Post-selection V7 diagnostics that cannot influence candidate ranking."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from sticky_lab.mode3_v6_3.cache import CallSpace, EmbeddingCache
from sticky_lab.mode3_v6_3.errors import CacheCorruption, ShapeMismatch

from .axis_geometry import displacement_decomposition
from .geometry import center_bootstrap_drift


POSITIONS = ("prefix", "suffix")


def cached_position_matrix(
    cache: EmbeddingCache,
    call_space: CallSpace,
    *,
    token_id: int,
    records: Sequence[Mapping[str, Any]],
    role: str,
    positions: Sequence[str],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Read a registered position expansion without creating model calls."""

    rows: list[dict[str, Any]] = []
    ordinals: list[int] = []
    for row in records:
        for position in positions:
            rows.append({**dict(row), "position": str(position)})
            ordinals.append(
                call_space.lookup_request(
                    str(role), str(row["text_id"]), str(position)
                ).ordinal
            )
    found, missing = cache.fetch(int(token_id), ordinals)
    if missing:
        raise CacheCorruption(
            f"post-selection diagnostic cache is missing {len(missing)} "
            f"calls for token={token_id} role={role}"
        )
    return rows, np.stack([found[ordinal] for ordinal in ordinals]).astype(np.float32)


def _fit_grid(
    rows: Sequence[Mapping[str, Any]], vectors: np.ndarray
) -> dict[tuple[str, str], np.ndarray]:
    matrix = np.asarray(vectors)
    if len(rows) != len(matrix):
        raise ShapeMismatch("post-selection fit row/vector alignment mismatch")
    grouped: dict[tuple[str, str], list[np.ndarray]] = {}
    for row, vector in zip(rows, matrix):
        key = (str(row["source_id"]), str(row["position"]))
        if key[1] not in POSITIONS:
            raise ShapeMismatch(f"forbidden post-selection position {key[1]}")
        grouped.setdefault(key, []).append(np.asarray(vector))
    return {key: np.stack(values) for key, values in grouped.items()}


def diagnose_selected_frontier(
    frontier: Mapping[str, Any],
    *,
    fit_records: Sequence[Mapping[str, Any]],
    select_records: Sequence[Mapping[str, Any]],
    e_star: np.ndarray,
    cache: EmbeddingCache,
    call_space: CallSpace,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Bootstrap center drift and decompose displacement after selection."""

    token_id = int(frontier["token_id"])
    fit_rows, fit_vectors = cached_position_matrix(
        cache,
        call_space,
        token_id=token_id,
        records=fit_records,
        role="fit",
        positions=POSITIONS,
    )
    center = np.asarray(frontier["center"], dtype=np.float64)
    diagnostics = config["diagnostics"]
    geometry = config["geometry"]
    drift = center_bootstrap_drift(
        _fit_grid(fit_rows, fit_vectors),
        center,
        samples=int(diagnostics["post_selection_center_bootstrap_samples"]),
        seed=int(config["positions"]["seed"]) + 9_000_001 + token_id,
        trim_fraction=float(geometry["center_trim_fraction"]),
        restarts=int(diagnostics["post_selection_bootstrap_restarts"]),
        maximum_iterations=int(geometry["maximum_iterations"]),
        tolerance=float(geometry["tolerance"]),
    )
    select_rows, triggered = cached_position_matrix(
        cache,
        call_space,
        token_id=token_id,
        records=select_records,
        role="select",
        positions=POSITIONS,
    )
    _, clean = cached_position_matrix(
        cache,
        call_space,
        token_id=-2,
        records=select_records,
        role="select",
        positions=("clean",),
    )
    paired_clean = np.repeat(clean, len(POSITIONS), axis=0)
    by_position: dict[str, Any] = {}
    for position in POSITIONS:
        mask = np.asarray(
            [str(row["position"]) == position for row in select_rows], dtype=bool
        )
        try:
            by_position[position] = displacement_decomposition(
                paired_clean[mask],
                triggered[mask],
                center=center,
                e_star=e_star,
            )
        except ShapeMismatch as error:
            by_position[position] = {
                "status": "UNDEFINED_REPORT_ONLY_GEOMETRY",
                "error": str(error),
            }
    axis = dict(frontier["axis_geometry"])
    return {
        "schema_version": "mode3-v7-post-selection-diagnostic-v1",
        "token_id": token_id,
        "token_text": str(frontier["token_text"]),
        "beta80_ps": frontier.get("beta80_ps"),
        "center_hash": str(frontier["center_hash"]),
        "center_bootstrap_drift": drift,
        "angle_center_to_e_star_radians": axis["angle_center_to_e_star_radians"],
        "angle_center_to_e_star_degrees": axis["angle_center_to_e_star_degrees"],
        "beta_axis": frontier.get("beta_axis"),
        "beta80_precedes_beta_axis": bool(frontier.get("beta80_precedes_beta_axis")),
        "displacement_by_position": by_position,
        "diagnostics_ran_after_selection": True,
        "used_for_selection": False,
        "confirm_data_used": False,
        "random_position_used": False,
    }


__all__ = ["cached_position_matrix", "diagnose_selected_frontier"]
