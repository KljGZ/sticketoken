"""Headless experiment figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def plot_similarity_curves(curves: np.ndarray, path: Path, *, xlabel: str, dpi: int = 250) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    values = np.asarray(curves, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("curves must have shape [pair, insertion_count+1]")
    norm = plt.Normalize(float(values[:, 0].min()), float(values[:, 0].max()))
    mapper = plt.cm.ScalarMappable(cmap="magma", norm=norm)
    figure, (axis, box_axis) = plt.subplots(1, 2, figsize=(9, 6), gridspec_kw={"width_ratios": [4, 0.5], "wspace": 0.1})
    x = np.arange(values.shape[1])
    for curve in values:
        axis.plot(x, curve, color=mapper.to_rgba(float(curve[0])), alpha=0.75, linewidth=1.2)
    axis.set_xlabel(xlabel, fontsize=16)
    axis.set_ylabel("Cosine similarity", fontsize=16)
    axis.set_xlim(0, values.shape[1] - 1)
    axis.set_xticks(np.arange(0, values.shape[1], max(1, (values.shape[1] - 1) // 10)))
    axis.grid(True, linestyle="--", alpha=0.55)
    sns.boxplot(y=values[:, -1], ax=box_axis, color="#D6AFB9", width=0.3, fliersize=0, linewidth=0.8)
    box_axis.set_ylim(axis.get_ylim())
    box_axis.tick_params(axis="both", labelleft=False, labelbottom=False)
    box_axis.grid(True, axis="y", linestyle="--", alpha=0.55)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(figure)


def plot_embedding_projection(original: np.ndarray, triggered: np.ndarray, path: Path, *, dpi: int = 250) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    original_values = np.asarray(original, dtype=float)
    triggered_values = np.asarray(triggered, dtype=float)
    combined = np.concatenate([original_values, triggered_values], axis=0)
    projection = PCA(n_components=2, random_state=0).fit_transform(combined)
    split = len(original_values)
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(projection[:split, 0], projection[:split, 1], s=22, alpha=0.55, label="original")
    axis.scatter(projection[split:, 0], projection[split:, 1], s=22, alpha=0.65, label="triggered")
    for index in range(min(split, len(triggered_values))):
        axis.plot([projection[index, 0], projection[split + index, 0]], [projection[index, 1], projection[split + index, 1]], color="gray", alpha=0.12, linewidth=0.6)
    axis.set_xlabel("PCA component 1")
    axis.set_ylabel("PCA component 2")
    axis.legend()
    axis.grid(True, linestyle="--", alpha=0.35)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def curves_to_frame(curves: np.ndarray, pair_rows: pd.DataFrame) -> pd.DataFrame:
    values = np.asarray(curves, dtype=float)
    records: list[dict[str, object]] = []
    for pair_offset, curve in enumerate(values):
        source_row = int(pair_rows.iloc[pair_offset]["source_row"])
        for count, similarity in enumerate(curve):
            records.append({"pair_offset": pair_offset, "source_row": source_row, "inserted_count": count, "cosine_similarity": float(similarity)})
    return pd.DataFrame.from_records(records)

