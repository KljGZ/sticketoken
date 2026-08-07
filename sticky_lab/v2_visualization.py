"""V2 length-frontier and jointly fitted embedding progression plots."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def plot_length_frontier(frame: pd.DataFrame, path: Path, *, mode: str, dpi: int = 250) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if mode == "multi_booster":
        metrics = [
            ("low_gain_q10", "Low gain q10"),
            ("low_coverage", "Low-gain coverage"),
            ("high_gain_q05", "High delta q05"),
            ("high_state_retention", "High-state retention"),
            ("global_drop_rate", "Global drop rate"),
            ("range_ratio", "Dynamic-range ratio"),
            ("spearman", "Spearman"),
            ("search_seed_success_rate", "Search-seed success rate"),
        ]
    else:
        metrics = [
            ("absolute_escape_q05", "Source-cluster escape q05"),
            ("relative_outward_q05", "Outward shift q05"),
            ("escape_rate", "Escape rate"),
            ("compact_radius_q95", "Triggered radius q95"),
            ("triggered_pairwise_q05", "Pairwise cosine q05"),
            ("all_cluster_clearance_q05", "All-cluster clearance q05"),
            ("density_ratio", "Benign kNN density ratio"),
            ("search_seed_success_rate", "Search-seed success rate"),
        ]
    columns = 2
    rows = math.ceil(len(metrics) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3.2 * rows), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    x = frame["component_length"].to_numpy(dtype=int)
    for axis, (column, label) in zip(axes, metrics):
        if column in frame:
            axis.plot(x, frame[column], marker="o", linewidth=1.6, color="#6A3D9A")
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.35)
        axis.set_ylabel(label)
        axis.grid(True, linestyle="--", alpha=0.35)
    for axis in axes[-columns:]:
        axis.set_xlabel("Number of tokens in optimized combination")
    for axis in axes[len(metrics) :]:
        axis.set_visible(False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _joint_projection(values: np.ndarray, config: dict[str, object]) -> tuple[np.ndarray, str]:
    method = str(config.get("method", "umap"))
    if method == "umap":
        try:
            import umap

            reducer = umap.UMAP(
                n_components=2,
                metric=str(config.get("metric", "cosine")),
                n_neighbors=int(config.get("n_neighbors", 30)),
                min_dist=float(config.get("min_dist", 0.1)),
                random_state=int(config.get("random_state", 42)),
            )
            return np.asarray(reducer.fit_transform(values), dtype=float), "UMAP (joint fit)"
        except ImportError:
            method = "sklearn_tsne"
    if method == "open_tsne":
        try:
            from openTSNE import TSNE

            reducer = TSNE(metric="cosine", random_state=int(config.get("random_state", 42)))
            return np.asarray(reducer.fit(values), dtype=float), "openTSNE (joint fit)"
        except ImportError:
            method = "sklearn_tsne"
    if method in {"umap", "open_tsne", "sklearn_tsne"}:
        from sklearn.manifold import TSNE

        perplexity = min(30.0, max(5.0, (len(values) - 1) / 3.0))
        reducer = TSNE(
            n_components=2,
            metric="cosine",
            perplexity=perplexity,
            init="random",
            learning_rate="auto",
            random_state=int(config.get("random_state", 42)),
        )
        return reducer.fit_transform(values), "sklearn t-SNE fallback (joint fit)"
    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=int(config.get("random_state", 42))).fit_transform(values), "PCA fallback (joint fit)"


def plot_embedding_progression(
    benign: np.ndarray,
    triggered_by_stage: Sequence[np.ndarray],
    stage_labels: Sequence[str],
    path: Path,
    *,
    cluster_centers: np.ndarray | None = None,
    projection_config: dict[str, object] | None = None,
    dpi: int = 250,
) -> str:
    """Fit one projection over all stages, then render comparable panels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    benign_values = np.asarray(benign, dtype=float)
    stages = [np.asarray(values, dtype=float) for values in triggered_by_stage]
    centers = None if cluster_centers is None else np.asarray(cluster_centers, dtype=float)
    pieces = [benign_values, *stages]
    if centers is not None:
        pieces.append(centers)
    combined = np.concatenate(pieces, axis=0)
    projection, method_label = _joint_projection(combined, projection_config or {})
    offset = 0
    benign_projection = projection[offset : offset + len(benign_values)]
    offset += len(benign_values)
    stage_projection: list[np.ndarray] = []
    for stage in stages:
        stage_projection.append(projection[offset : offset + len(stage)])
        offset += len(stage)
    center_projection = projection[offset:] if centers is not None else None
    columns = min(4, max(1, len(stages)))
    rows = math.ceil(len(stages) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.8 * rows), squeeze=False)
    for axis, current, label in zip(axes.reshape(-1), stage_projection, stage_labels):
        axis.scatter(benign_projection[:, 0], benign_projection[:, 1], s=9, alpha=0.25, color="#777777", label="benign")
        axis.scatter(current[:, 0], current[:, 1], s=12, alpha=0.65, color="#D62728", label="triggered")
        if center_projection is not None:
            axis.scatter(center_projection[:, 0], center_projection[:, 1], marker="x", s=28, color="black", linewidth=1.0)
        trigger_center = current.mean(axis=0)
        axis.scatter([trigger_center[0]], [trigger_center[1]], marker="*", s=130, color="#FFB000", edgecolor="black")
        axis.set_title(label)
        axis.grid(True, linestyle="--", alpha=0.25)
    for axis in axes.reshape(-1)[len(stages) :]:
        axis.set_visible(False)
    axes.reshape(-1)[0].legend(loc="best", fontsize=8)
    figure.suptitle(f"AgentPoison-style embedding progression — {method_label}")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return method_label
