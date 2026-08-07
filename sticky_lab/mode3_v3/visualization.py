"""Joint high-dimensional V3 projections and unambiguous length curves."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def _projection(values: np.ndarray, method: str, seed: int) -> np.ndarray:
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=seed).fit_transform(values)
    if method == "umap":
        try:
            from umap import UMAP

            return UMAP(n_components=2, metric="cosine", n_neighbors=min(30, max(2, len(values) - 1)), min_dist=0.1, random_state=seed).fit_transform(values)
        except ImportError as error:
            raise RuntimeError("UMAP output requires the registered umap-learn dependency") from error
    raise ValueError(f"Unknown projection method: {method}")


def plot_joint_progression(
    benign: np.ndarray,
    stages: Sequence[np.ndarray],
    centers: Sequence[np.ndarray],
    labels: Sequence[str],
    metric_labels: Sequence[str],
    output_prefix: Path,
    *,
    seed: int,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    blocks = [np.asarray(benign), *[np.asarray(stage) for stage in stages], np.asarray(centers)]
    sizes = [len(block) for block in blocks]
    joint = np.concatenate(blocks, axis=0)
    for method in ("pca", "umap"):
        coordinates = _projection(joint, method, seed)
        cursor = 0
        projected: list[np.ndarray] = []
        for size in sizes:
            projected.append(coordinates[cursor : cursor + size])
            cursor += size
        fig, axes = plt.subplots(1, len(stages), figsize=(4.2 * len(stages), 4), squeeze=False)
        benign_xy = projected[0]
        center_xy = projected[-1]
        for index, (stage, label, metrics) in enumerate(zip(projected[1:-1], labels, metric_labels)):
            axis = axes[0, index]
            axis.scatter(benign_xy[:, 0], benign_xy[:, 1], s=7, alpha=0.25, color="#777777", label="benign")
            axis.scatter(stage[:, 0], stage[:, 1], s=9, alpha=0.55, color="#d62728", label="triggered")
            axis.scatter(center_xy[index, 0], center_xy[index, 1], marker="*", s=130, color="#f0c419", edgecolor="black", label="high-D center")
            axis.set_title(f"{label}\n{metrics}", fontsize=9)
            axis.grid(alpha=0.2)
        axes[0, 0].legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_prefix.with_name(output_prefix.name + f"_{method}.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_length_protocols(
    exact: pd.DataFrame,
    prefix_growth: pd.DataFrame,
    output: Path,
    *,
    metric: str,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    values = exact.sort_values("component_length", kind="mergesort").copy()
    if metric in {"compact_radius_q95"}:
        values["best_up_to_length"] = values[metric].cummin()
    else:
        values["best_up_to_length"] = values[metric].cummax()
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.plot(values["component_length"], values[metric], "o-", label="exact-length best")
    axis.plot(values["component_length"], values["best_up_to_length"], "s--", label="best-up-to-length envelope")
    if len(prefix_growth):
        axis.plot(prefix_growth["prefix_length"], prefix_growth[metric], "^-", label="final-trigger prefix growth")
    axis.set_xlabel("Actual inserted token length")
    axis.set_ylabel(metric)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
