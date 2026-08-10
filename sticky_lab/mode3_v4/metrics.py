"""V4 attractor geometry, uncertainty, certificates, and search quality."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .occupancy import OccupancyRecord, clopper_pearson_lower
from .support import SupportModel, normalize_rows, spherical_center


@dataclass(frozen=True)
class GeometryResult:
    center: np.ndarray
    displacement_q05: float
    displacement_q50: float
    displacement_q95: float
    compact_radius_q50: float
    compact_radius_q90: float
    compact_radius_q95: float
    compact_radius_q99: float
    pairwise_mean_ratio: float
    pairwise_q90_ratio: float
    contraction_q95: float
    position_center_drift_max: float

    def to_dict(self) -> dict[str, float]:
        return {
            key: float(value)
            for key, value in self.__dict__.items()
            if key != "center"
        }


def fixed_pair_indices(size: int, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if size < 2:
        raise ValueError("At least two texts are required for contraction")
    maximum = size * (size - 1) // 2
    if maximum <= count and size <= 2048:
        return np.triu_indices(size, 1)
    rng = np.random.default_rng(seed)
    left = rng.integers(0, size, size=count)
    right = rng.integers(0, size - 1, size=count)
    right = right + (right >= left)
    return left.astype(int), right.astype(int)


def pairwise_distance_matrix(values: np.ndarray) -> np.ndarray:
    """Exact unit-vector Euclidean distances using one matrix product."""

    normalized = normalize_rows(values).astype(np.float32)
    similarities = np.clip(normalized @ normalized.T, -1.0, 1.0)
    return np.sqrt(np.maximum(0.0, 2.0 - 2.0 * similarities)).astype(np.float32)


def _position_centers(triggered: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [spherical_center(values) for values in triggered]


def _center_drift(centers: Sequence[np.ndarray]) -> float:
    if len(centers) < 2:
        return 0.0
    return float(max(np.linalg.norm(left - right) for i, left in enumerate(centers) for right in centers[i + 1 :]))


def evaluate_geometry(
    original: np.ndarray,
    triggered_by_position: Sequence[np.ndarray],
    *,
    pair_indices: tuple[np.ndarray, np.ndarray],
    fixed_center: np.ndarray | None = None,
) -> GeometryResult:
    benign = normalize_rows(original)
    triggered = [normalize_rows(values) for values in triggered_by_position]
    if not triggered or any(len(values) != len(benign) for values in triggered):
        raise ValueError("Each registered position must match the original text count")
    pooled = np.concatenate(triggered, axis=0)
    center = np.asarray(fixed_center, dtype=np.float64) if fixed_center is not None else spherical_center(pooled)
    center = center / max(float(np.linalg.norm(center)), 1e-12)
    original_tiled = np.concatenate([benign for _ in triggered], axis=0)
    displacement = np.linalg.norm(pooled - original_tiled, axis=1)
    radii = np.linalg.norm(pooled - center[None, :], axis=1)
    left, right = pair_indices
    benign_pair = np.linalg.norm(benign[left] - benign[right], axis=1)
    triggered_pairs = np.concatenate(
        [np.linalg.norm(values[left] - values[right], axis=1) for values in triggered]
    )
    benign_tiled = np.tile(benign_pair, len(triggered))
    denominator_mean = max(float(np.mean(benign_tiled)), 1e-12)
    denominator_q90 = max(float(np.quantile(benign_tiled, 0.90)), 1e-12)
    denominator_q95 = max(float(np.quantile(benign_tiled, 0.95)), 1e-12)
    centers = _position_centers(triggered)
    return GeometryResult(
        center=center.astype(np.float32),
        displacement_q05=float(np.quantile(displacement, 0.05)),
        displacement_q50=float(np.quantile(displacement, 0.50)),
        displacement_q95=float(np.quantile(displacement, 0.95)),
        compact_radius_q50=float(np.quantile(radii, 0.50)),
        compact_radius_q90=float(np.quantile(radii, 0.90)),
        compact_radius_q95=float(np.quantile(radii, 0.95)),
        compact_radius_q99=float(np.quantile(radii, 0.99)),
        pairwise_mean_ratio=float(np.mean(triggered_pairs) / denominator_mean),
        pairwise_q90_ratio=float(np.quantile(triggered_pairs, 0.90) / denominator_q90),
        contraction_q95=float(np.quantile(triggered_pairs, 0.95) / denominator_q95),
        position_center_drift_max=_center_drift(centers),
    )


def grouped_bootstrap_geometry(
    original: np.ndarray,
    triggered_by_position: Sequence[np.ndarray],
    group_ids: Sequence[str],
    *,
    replicates: int,
    confidence: float,
    pair_count: int,
    seed: int,
    fixed_center: np.ndarray | None = None,
    benign_pairwise_distances: np.ndarray | None = None,
) -> dict[str, float | int]:
    groups = np.asarray(group_ids, dtype=str)
    if len(groups) != len(original):
        raise ValueError("group_ids must align with original embeddings")
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    names = ("displacement_q05", "compact_radius_q95", "contraction_q95")
    samples = {name: [] for name in names}
    # Current registered corpus has one provenance group per unique sentence.
    # Preserve the exact group bootstrap while replacing repeated 768-D pair
    # subtraction with O(1) lookups into precomputed distance matrices.
    if len(unique) == len(groups):
        benign = normalize_rows(original).astype(np.float32)
        triggered = [normalize_rows(values).astype(np.float32) for values in triggered_by_position]
        benign_distances = (
            pairwise_distance_matrix(benign)
            if benign_pairwise_distances is None
            else np.asarray(benign_pairwise_distances, dtype=np.float32)
        )
        triggered_distances = [pairwise_distance_matrix(values) for values in triggered]
        displacement = [np.linalg.norm(values - benign, axis=1) for values in triggered]
        group_to_index = {group: int(np.flatnonzero(groups == group)[0]) for group in unique}
        bootstrap_indices = []
        for _ in range(int(replicates)):
            chosen = rng.choice(unique, size=len(unique), replace=True)
            bootstrap_indices.append(np.asarray([group_to_index[group] for group in chosen], dtype=int))
        block_size = 10
        for block_start in range(0, int(replicates), block_size):
            indices = np.stack(bootstrap_indices[block_start : block_start + block_size])
            block_count = len(indices)
            if fixed_center is None:
                center_sums = np.zeros((block_count, benign.shape[1]), dtype=np.float64)
                for values in triggered:
                    center_sums += np.sum(values[indices], axis=1, dtype=np.float64)
                center_norms = np.linalg.norm(center_sums, axis=1, keepdims=True)
                centers = center_sums / np.maximum(center_norms, 1e-12)
            else:
                center = np.asarray(fixed_center, dtype=np.float64)
                center /= max(float(np.linalg.norm(center)), 1e-12)
                centers = np.repeat(center[None, :], block_count, axis=0)
            sampled_displacement = np.concatenate([values[indices] for values in displacement], axis=1)
            radius_blocks: list[np.ndarray] = []
            for values in triggered:
                selected = values[indices]
                cosine = np.einsum("bnd,bd->bn", selected, centers, optimize=True)
                radius_blocks.append(np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.clip(cosine, -1.0, 1.0))))
            sampled_radii = np.concatenate(radius_blocks, axis=1)
            pair_rows = [
                fixed_pair_indices(len(unique), pair_count, seed + replicate + 1)
                for replicate in range(block_start, block_start + block_count)
            ]
            left = np.stack([row[0] for row in pair_rows])
            right = np.stack([row[1] for row in pair_rows])
            mapped_left = np.take_along_axis(indices, left, axis=1)
            mapped_right = np.take_along_axis(indices, right, axis=1)
            benign_pairs = benign_distances[mapped_left, mapped_right]
            triggered_pairs = np.concatenate(
                [distances[mapped_left, mapped_right] for distances in triggered_distances], axis=1
            )
            contraction = np.quantile(triggered_pairs, 0.95, axis=1) / np.maximum(
                np.quantile(benign_pairs, 0.95, axis=1), 1e-12
            )
            samples["displacement_q05"].extend(np.quantile(sampled_displacement, 0.05, axis=1).tolist())
            samples["compact_radius_q95"].extend(np.quantile(sampled_radii, 0.95, axis=1).tolist())
            samples["contraction_q95"].extend(contraction.tolist())
        alpha = 1.0 - float(confidence)
        output: dict[str, float | int] = {
            "bootstrap_replicates": int(replicates),
            "bootstrap_confidence": float(confidence),
            "bootstrap_pairwise_lookup_optimized": 1,
            "bootstrap_vectorized_block_size": block_size,
        }
        for name, values in samples.items():
            output[f"{name}_ci_lower"] = float(np.quantile(values, alpha / 2.0))
            output[f"{name}_ci_upper"] = float(np.quantile(values, 1.0 - alpha / 2.0))
        return output
    for replicate in range(int(replicates)):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in chosen])
        pairs = fixed_pair_indices(len(indices), pair_count, seed + replicate + 1)
        result = evaluate_geometry(
            np.asarray(original)[indices],
            [np.asarray(values)[indices] for values in triggered_by_position],
            pair_indices=pairs,
            fixed_center=fixed_center,
        )
        for name in names:
            samples[name].append(float(getattr(result, name)))
    alpha = 1.0 - float(confidence)
    output: dict[str, float | int] = {
        "bootstrap_replicates": int(replicates),
        "bootstrap_confidence": float(confidence),
        "bootstrap_pairwise_lookup_optimized": 0,
        "bootstrap_vectorized_block_size": 1,
    }
    for name, values in samples.items():
        output[f"{name}_ci_lower"] = float(np.quantile(values, alpha / 2.0))
        output[f"{name}_ci_upper"] = float(np.quantile(values, 1.0 - alpha / 2.0))
    return output


def point_conditions(
    geometry: GeometryResult,
    support_margin: float,
    occupancy: OccupancyRecord,
    constraints: Mapping[str, float],
) -> dict[str, bool]:
    occ = occupancy.to_dict()
    return {
        "shift_point": geometry.displacement_q05 >= float(constraints["min_displacement_q05"]),
        "compact_point": geometry.compact_radius_q95 <= float(constraints["max_compact_radius_q95"]),
        "contract_point": geometry.contraction_q95 <= float(constraints["max_contraction_q95"]),
        "support_in_point": support_margin >= float(constraints["min_support_in_margin"]),
        "low_occupancy_point": bool(
            occ["occupancy_upper_lambda_1_0"] <= float(constraints["max_occupancy_upper_lambda_1"])
            and occ["occupancy_upper_lambda_2_0"] <= float(constraints["max_occupancy_upper_lambda_2"])
            and occ["relative_occupancy_quantile_lambda_1_0"]
            <= float(constraints["max_relative_occupancy_quantile_lambda_1"])
        ),
    }


def certify_validation(
    geometry: GeometryResult,
    uncertainty: Mapping[str, float],
    support_margin: float,
    occupancy: OccupancyRecord,
    constraints: Mapping[str, float],
    *,
    realizable: bool,
    baseline_exceeded: bool,
) -> dict[str, bool]:
    point = point_conditions(geometry, support_margin, occupancy, constraints)
    shift = float(uncertainty["displacement_q05_ci_lower"]) >= float(constraints["min_displacement_q05"])
    compact = float(uncertainty["compact_radius_q95_ci_upper"]) <= float(constraints["max_compact_radius_q95"])
    contract = float(uncertainty["contraction_q95_ci_upper"]) <= float(constraints["max_contraction_q95"])
    certificate = bool(
        realizable
        and shift
        and compact
        and contract
        and point["support_in_point"]
        and point["low_occupancy_point"]
        and baseline_exceeded
    )
    return {
        **point,
        "realizable_certified": bool(realizable),
        "shift_certified": shift,
        "compact_certified": compact,
        "contract_certified": contract,
        "support_in_certified": point["support_in_point"],
        "low_occupancy_certified": point["low_occupancy_point"],
        "baseline_exceedance_certified": bool(baseline_exceeded),
        "v4_certified": certificate,
    }


def search_quality(
    geometry: GeometryResult,
    support_margin: float,
    occupancy: OccupancyRecord,
    constraints: Mapping[str, float],
    weights: Mapping[str, float],
) -> tuple[float, float]:
    occ = occupancy.to_dict()
    violations = (
        max(0.0, float(constraints["min_displacement_q05"]) - geometry.displacement_q05)
        / max(float(constraints["min_displacement_q05"]), 1e-12)
        + max(0.0, geometry.compact_radius_q95 - float(constraints["max_compact_radius_q95"]))
        / max(float(constraints["max_compact_radius_q95"]), 1e-12)
        + max(0.0, geometry.contraction_q95 - float(constraints["max_contraction_q95"]))
        / max(float(constraints["max_contraction_q95"]), 1e-12)
        + max(0.0, float(constraints["min_support_in_margin"]) - support_margin)
        / max(abs(float(constraints["min_support_in_margin"])) + 0.01, 0.01)
        + max(0.0, float(occ["occupancy_rate_lambda_1_0"]) - float(constraints["max_occupancy_upper_lambda_1"]))
        / max(float(constraints["max_occupancy_upper_lambda_1"]), 1e-12)
        + max(0.0, float(occ["occupancy_rate_lambda_2_0"]) - float(constraints["max_occupancy_upper_lambda_2"]))
        / max(float(constraints["max_occupancy_upper_lambda_2"]), 1e-12)
        + max(
            0.0,
            float(occ["relative_occupancy_quantile_lambda_1_0"])
            - float(constraints["max_relative_occupancy_quantile_lambda_1"]),
        )
        / max(float(constraints["max_relative_occupancy_quantile_lambda_1"]), 1e-12)
    )
    score = (
        -float(weights["constraint_violation"]) * violations
        - float(weights["compact_radius"]) * geometry.compact_radius_q95
        - float(weights["contraction"]) * geometry.contraction_q95
        - float(weights["occupancy_outer"]) * float(occ["occupancy_rate_lambda_2_0"])
        + float(weights["support_margin"]) * support_margin
        + float(weights["displacement"]) * geometry.displacement_q05
    )
    return float(score), float(violations)


def fixed_region_coverage(
    triggered_by_position: Sequence[np.ndarray],
    center: np.ndarray,
    radius: float,
    *,
    confidence: float,
) -> dict[str, Any]:
    by_position: list[dict[str, float | int]] = []
    all_distances: list[np.ndarray] = []
    for values in triggered_by_position:
        distances = np.linalg.norm(normalize_rows(values) - np.asarray(center)[None, :], axis=1)
        successes = int(np.count_nonzero(distances <= radius))
        by_position.append(
            {
                "count": successes,
                "trials": len(distances),
                "rate": successes / len(distances),
                "lcb": clopper_pearson_lower(successes, len(distances), confidence),
            }
        )
        all_distances.append(distances)
    pooled = np.concatenate(all_distances)
    successes = int(np.count_nonzero(pooled <= radius))
    return {
        "fixed_center_used": True,
        "fixed_radius_used": True,
        "fixed_region_coverage_count": successes,
        "fixed_region_coverage_rate": successes / len(pooled),
        "fixed_region_coverage_lcb": clopper_pearson_lower(successes, len(pooled), confidence),
        "fixed_region_per_position": by_position,
    }
