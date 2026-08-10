"""V4-only registered plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def plot_length_frontier(frontier: pd.DataFrame, output: Path, *, dpi: int) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("compact_radius_q95", "Compact radius q95", 0.40),
        ("contraction_q95", "Relative contraction q95", 0.60),
        ("displacement_q05", "Displacement q05", 0.02),
        ("occupancy_upper_lambda_1_0", "Core occupancy upper 95%", 0.001),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for axis, (metric, label, threshold) in zip(axes.flat, metrics):
        for task, part in frontier.groupby("task"):
            ordered = part.sort_values("actual_token_length")
            axis.plot(ordered["actual_token_length"], ordered[metric], marker="o", markersize=2.5, label=task)
        axis.axhline(threshold, color="black", linestyle="--", linewidth=0.9, label="registered threshold")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[1, 0].set_xlabel("Actual tokenizer length")
    axes[1, 1].set_xlabel("Actual tokenizer length")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_frozen_projection(
    benign: np.ndarray,
    triggered: np.ndarray,
    center: np.ndarray,
    output: Path,
    *,
    seed: int,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    joint = np.concatenate([benign, triggered, np.asarray(center)[None, :]], axis=0)
    xy = PCA(n_components=2, random_state=seed).fit_transform(joint)
    n_benign = len(benign)
    n_triggered = len(triggered)
    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    axis.scatter(xy[:n_benign, 0], xy[:n_benign, 1], s=8, alpha=0.25, color="#777777", label="benign")
    axis.scatter(xy[n_benign:n_benign+n_triggered, 0], xy[n_benign:n_benign+n_triggered, 1], s=9, alpha=0.45, color="#d62728", label="triggered")
    axis.scatter(xy[-1, 0], xy[-1, 1], marker="*", s=180, color="#f0c419", edgecolor="black", label="frozen high-D center")
    axis.set_title("V4 frozen-center joint PCA (visual diagnostic only)")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
