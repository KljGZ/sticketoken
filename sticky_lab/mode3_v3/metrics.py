"""V3 hard metrics, hierarchical certificates, and grouped bootstrap CIs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .support import BenignSupportModel, normalize_rows


@dataclass
class Mode3Metrics:
    displacement_q05: float
    displacement_median: float
    displacement_q95: float
    separation_margin: float
    mean_separation: float
    linear_auc: float
    balanced_accuracy: float
    fpr_at_95_tpr: float
    compact_radius_q95: float
    pairwise_mean: float
    pairwise_q05: float
    sample_blank_margin: float
    cluster_blank_margin: float
    density_blank_margin: float
    source_escape_q05: float
    center_norm_pre_normalization: float
    shift_certified: bool
    separator_certified: bool
    compact_certified: bool
    sample_blank_certified: bool
    cluster_blank_certified: bool
    density_blank_certified: bool
    blank_region_certified: bool


def _center(values: np.ndarray) -> tuple[np.ndarray, float]:
    mean = np.asarray(values, dtype=float).mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm <= 1e-12:
        raise ValueError("Triggered center has zero norm")
    return mean / norm, norm


def _pairwise(values: np.ndarray, limit: int, seed: int) -> np.ndarray:
    n = len(values)
    if n < 2:
        return np.asarray([1.0])
    total = n * (n - 1) // 2
    if total <= limit:
        gram = values @ values.T
        return gram[np.triu_indices(n, 1)]
    rng = np.random.default_rng(seed)
    left = rng.integers(0, n, size=limit * 3)
    right = rng.integers(0, n, size=limit * 3)
    keep = left != right
    return np.einsum("ij,ij->i", values[left[keep][:limit]], values[right[keep][:limit]], optimize=True)


def _separator_statistics(original: np.ndarray, triggered: np.ndarray) -> tuple[float, float, float, float, float]:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve

    direction = triggered.mean(axis=0) - original.mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return -float("inf"), 0.0, 0.5, 0.5, 1.0
    direction /= norm
    benign_scores = original @ direction
    trigger_scores = triggered @ direction
    margin = float(np.quantile(trigger_scores, 0.05) - np.quantile(benign_scores, 0.95))
    mean_separation = float(trigger_scores.mean() - benign_scores.mean())
    labels = np.concatenate([np.zeros(len(original), dtype=int), np.ones(len(triggered), dtype=int)])
    scores = np.concatenate([benign_scores, trigger_scores])
    auc = float(roc_auc_score(labels, scores))
    threshold = 0.5 * (np.quantile(trigger_scores, 0.05) + np.quantile(benign_scores, 0.95))
    balanced = float(balanced_accuracy_score(labels, scores >= threshold))
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = np.flatnonzero(tpr >= 0.95)
    fpr95 = float(fpr[valid[0]]) if len(valid) else 1.0
    return margin, mean_separation, auc, balanced, fpr95


def evaluate_mode3(
    original: np.ndarray,
    triggered: np.ndarray,
    support: BenignSupportModel,
    constraints: dict[str, float],
    *,
    pairwise_sample_size: int = 20000,
    seed: int = 0,
) -> Mode3Metrics:
    benign = normalize_rows(original)
    values = normalize_rows(triggered)
    if benign.shape != values.shape:
        raise ValueError("original and triggered embeddings must have identical shape")
    displacement = np.linalg.norm(values - benign, axis=1)
    center, center_norm = _center(values)
    radius = np.linalg.norm(values - center[None, :], axis=1)
    radius_q95 = float(np.quantile(radius, 0.95))
    separation, mean_sep, auc, balanced, fpr95 = _separator_statistics(benign, values)
    pairs = _pairwise(values, pairwise_sample_size, seed)
    labels = support.assign_clusters(benign)
    source_clearance = (
        np.linalg.norm(values - support.cluster_centers[labels], axis=1)
        - support.cluster_radii[labels]
    )
    sample_margin = support.sample_blank_margin(center, radius_q95)
    cluster_margin = support.cluster_blank_margin(center, radius_q95)
    density_margin = support.knn_density_margin(center, radius_q95)
    shift_ok = float(np.quantile(displacement, 0.05)) >= float(constraints.get("min_displacement_q05", 0.0))
    separator_ok = separation > float(constraints.get("min_separation_margin", 0.0))
    compact_ok = radius_q95 <= float(constraints.get("max_compact_radius_q95", float("inf")))
    sample_ok = sample_margin > float(constraints.get("min_sample_blank_margin", 0.0))
    cluster_ok = cluster_margin > float(constraints.get("min_cluster_blank_margin", 0.0))
    density_ok = density_margin > float(constraints.get("min_density_blank_margin", 0.0))
    return Mode3Metrics(
        displacement_q05=float(np.quantile(displacement, 0.05)),
        displacement_median=float(np.median(displacement)),
        displacement_q95=float(np.quantile(displacement, 0.95)),
        separation_margin=separation,
        mean_separation=mean_sep,
        linear_auc=auc,
        balanced_accuracy=balanced,
        fpr_at_95_tpr=fpr95,
        compact_radius_q95=radius_q95,
        pairwise_mean=float(pairs.mean()),
        pairwise_q05=float(np.quantile(pairs, 0.05)),
        sample_blank_margin=sample_margin,
        cluster_blank_margin=cluster_margin,
        density_blank_margin=density_margin,
        source_escape_q05=float(np.quantile(source_clearance, 0.05)),
        center_norm_pre_normalization=center_norm,
        shift_certified=bool(shift_ok),
        separator_certified=bool(separator_ok),
        compact_certified=bool(compact_ok),
        sample_blank_certified=bool(sample_ok),
        cluster_blank_certified=bool(cluster_ok),
        density_blank_certified=bool(density_ok),
        blank_region_certified=bool(separator_ok and compact_ok and sample_ok and (cluster_ok or density_ok)),
    )


def _resample_groups(group_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    groups = np.unique(group_ids)
    chosen = rng.choice(groups, size=len(groups), replace=True)
    parts = [np.flatnonzero(group_ids == group) for group in chosen]
    return np.concatenate(parts) if parts else np.arange(len(group_ids))


def _bootstrap_core_metrics(
    benign: np.ndarray,
    triggered: np.ndarray,
    support: BenignSupportModel,
    indices: np.ndarray,
) -> dict[str, float]:
    """Re-estimate only statistics that receive bootstrap confidence bounds.

    AUC, balanced accuracy, FPR and pairwise similarity are complete-sample
    diagnostics.  Recomputing them inside every replicate is both irrelevant
    to the registered certificates and disproportionately expensive.
    """
    original = benign[indices]
    values = triggered[indices]
    displacement = np.linalg.norm(values - original, axis=1)
    center, _ = _center(values)
    radius = np.linalg.norm(values - center[None, :], axis=1)
    radius_q95 = float(np.quantile(radius, 0.95))
    direction = values.mean(axis=0) - original.mean(axis=0)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-12:
        separation = -float("inf")
    else:
        direction /= direction_norm
        separation = float(np.quantile(values @ direction, 0.05) - np.quantile(original @ direction, 0.95))
    return {
        "displacement_q05": float(np.quantile(displacement, 0.05)),
        "separation_margin": separation,
        "compact_radius_q95": radius_q95,
        "sample_blank_margin": support.sample_blank_margin(center, radius_q95),
        "cluster_blank_margin": support.cluster_blank_margin(center, radius_q95),
        "density_blank_margin": support.knn_density_margin(center, radius_q95),
    }


def grouped_bootstrap(
    original: np.ndarray,
    triggered: np.ndarray,
    support: BenignSupportModel,
    constraints: dict[str, float],
    group_ids: np.ndarray,
    *,
    replicates: int,
    confidence: float,
    pairwise_sample_size: int,
    seed: int,
) -> dict[str, Any]:
    """Return point metrics, grouped CIs, and CI-based certificates."""
    point = evaluate_mode3(
        original,
        triggered,
        support,
        constraints,
        pairwise_sample_size=pairwise_sample_size,
        seed=seed,
    )
    result: dict[str, Any] = asdict(point)
    result["bootstrap_replicates"] = int(replicates)
    result["bootstrap_confidence"] = float(confidence)
    if replicates <= 0:
        return result
    rng = np.random.default_rng(seed)
    names = [
        "displacement_q05",
        "separation_margin",
        "compact_radius_q95",
        "sample_blank_margin",
        "cluster_blank_margin",
        "density_blank_margin",
    ]
    samples = {name: [] for name in names}
    groups = np.asarray(group_ids)
    if len(groups) != len(original):
        raise ValueError("group_ids length must equal embedding count")
    benign = normalize_rows(original)
    values = normalize_rows(triggered)
    for _replicate in range(replicates):
        indices = _resample_groups(groups, rng)
        replicate_values = _bootstrap_core_metrics(benign, values, support, indices)
        for name in names:
            samples[name].append(float(replicate_values[name]))
    alpha = (1.0 - confidence) / 2.0
    for name, values in samples.items():
        result[f"{name}_ci_lower"] = float(np.quantile(values, alpha))
        result[f"{name}_ci_upper"] = float(np.quantile(values, 1.0 - alpha))
    result["shift_certified"] = bool(
        result["displacement_q05_ci_lower"] >= float(constraints.get("min_displacement_q05", 0.0))
    )
    result["separator_certified"] = bool(
        result["separation_margin_ci_lower"] > float(constraints.get("min_separation_margin", 0.0))
    )
    result["compact_certified"] = bool(
        result["compact_radius_q95_ci_upper"] <= float(constraints.get("max_compact_radius_q95", float("inf")))
    )
    result["sample_blank_certified"] = bool(
        result["sample_blank_margin_ci_lower"] > float(constraints.get("min_sample_blank_margin", 0.0))
    )
    result["cluster_blank_certified"] = bool(
        result["cluster_blank_margin_ci_lower"] > float(constraints.get("min_cluster_blank_margin", 0.0))
    )
    result["density_blank_certified"] = bool(
        result["density_blank_margin_ci_lower"] > float(constraints.get("min_density_blank_margin", 0.0))
    )
    result["blank_region_certified"] = bool(
        result["separator_certified"]
        and result["compact_certified"]
        and result["sample_blank_certified"]
        and (result["cluster_blank_certified"] or result["density_blank_certified"])
    )
    return result


def separator_sort_key(record: dict[str, Any]) -> tuple[float, ...]:
    violation = max(0.0, -float(record.get("separation_margin", -float("inf"))))
    return (
        0.0 if bool(record.get("separator_certified", False)) else 1.0,
        violation,
        -float(record.get("separation_margin", -float("inf"))),
        int(record.get("actual_token_length", record.get("component_length", 0))),
    )


def blank_sort_key(record: dict[str, Any]) -> tuple[float, ...]:
    radius_limit = float(record.get("max_compact_radius_q95", 0.40))
    violations = (
        max(0.0, -float(record.get("separation_margin", -float("inf"))))
        + max(0.0, -float(record.get("sample_blank_margin", -float("inf"))))
        + max(0.0, float(record.get("compact_radius_q95", float("inf"))) - radius_limit)
        + min(
            max(0.0, -float(record.get("cluster_blank_margin", -float("inf")))),
            max(0.0, -float(record.get("density_blank_margin", -float("inf")))),
        )
    )
    return (
        0.0 if bool(record.get("blank_region_certified", False)) else 1.0,
        violations,
        float(record.get("compact_radius_q95", float("inf"))),
        -float(record.get("sample_blank_margin", -float("inf"))),
        -float(record.get("separation_margin", -float("inf"))),
        int(record.get("actual_token_length", record.get("component_length", 0))),
    )
