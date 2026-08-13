"""Frozen visualization-only PCA/UMAP projections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass
class FrozenProjection:
    pca: object
    umap: object | None

    @classmethod
    def fit(cls, benign_reference: np.ndarray, *, seed: int = 0) -> "FrozenProjection":
        from sklearn.decomposition import PCA

        reference = np.asarray(benign_reference, dtype=np.float64)
        pca = PCA(n_components=min(50, reference.shape[1], len(reference) - 1), random_state=seed).fit(reference)
        umap_model = None
        try:
            import umap

            umap_model = umap.UMAP(n_components=2, random_state=seed, transform_seed=seed).fit(pca.transform(reference))
        except ImportError as error:
            raise RuntimeError("formal V6 trajectory publication requires the registered UMAP dependency") from error
        return cls(pca, umap_model)

    def transform(self, vectors: np.ndarray, method: str) -> np.ndarray:
        latent = self.pca.transform(np.asarray(vectors, dtype=np.float64))
        if method == "pca":
            return latent[:, :2]
        if method == "umap" and self.umap is not None:
            return self.umap.transform(latent)
        raise ValueError(f"projection unavailable: {method}")


def save_projection_frame(
    path: Path,
    projection: FrozenProjection,
    groups: Mapping[str, np.ndarray],
    centers: np.ndarray,
    *,
    method: str,
    title: str,
) -> None:
    """Save PNG and raw 2-D coordinates; never used for membership tests."""
    import matplotlib.pyplot as plt

    styles = {
        "independent_benign": ("#b6b6b6", "."),
        "paired_clean": ("#74a9ff", "o"),
        "triggered_inlier": ("#f54278", "o"),
        "triggered_outlier": ("#ff8c00", "x"),
    }
    rows: list[tuple[str, float, float]] = []
    fig, axis = plt.subplots(figsize=(8, 6))
    for name, vectors in groups.items():
        coordinates = projection.transform(vectors, method)
        color, marker = styles.get(name, (None, "."))
        axis.scatter(coordinates[:, 0], coordinates[:, 1], s=12, alpha=0.65, label=name, color=color, marker=marker)
        rows.extend((name, float(x), float(y)) for x, y in coordinates)
    center_xy = projection.transform(centers, method)
    axis.scatter(center_xy[:, 0], center_xy[:, 1], s=110, color="black", marker="*", label="frozen_center")
    rows.extend(("frozen_center", float(x), float(y)) for x, y in center_xy)
    axis.set_title(title)
    axis.legend(loc="best")
    axis.grid(alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    from .atomic_io import write_csv

    write_csv(path.with_suffix(".csv"), ({"group": name, "x": x, "y": y} for name, x, y in rows))
