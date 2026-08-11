"""Frozen PCA/UMAP bases and auditable optimization cluster frames."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, FFMpegWriter, PillowWriter
import numpy as np
import pandas as pd

from .atomic_io import write_csv, write_json, write_npz
from .occupancy import cosine_distance_to_centers
from .scoring import EvaluationBundle


class FixedProjection:
    def __init__(
        self,
        *,
        pca_mean: np.ndarray,
        pca_components: np.ndarray,
        umap_model: Any,
        bounds: Mapping[str, Sequence[float]],
    ) -> None:
        self.pca_mean = np.asarray(pca_mean, dtype=np.float64)
        self.pca_components = np.asarray(pca_components, dtype=np.float64)
        self.umap_model = umap_model
        self.bounds = {key: tuple(map(float, value)) for key, value in bounds.items()}

    def pca_transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.pca_mean) @ self.pca_components[:2].T

    def umap_transform(self, values: np.ndarray) -> np.ndarray:
        reduced = (np.asarray(values, dtype=np.float64) - self.pca_mean) @ self.pca_components.T
        return np.asarray(self.umap_model.transform(reduced), dtype=np.float64)


def _axis_bounds(values: np.ndarray, quantile: float) -> tuple[float, float, float, float]:
    lower = (1.0 - quantile) / 2
    upper = 1.0 - lower
    x_low, y_low = np.quantile(values, lower, axis=0)
    x_high, y_high = np.quantile(values, upper, axis=0)
    x_margin = max(0.05, 0.10 * (x_high - x_low))
    y_margin = max(0.05, 0.10 * (y_high - y_low))
    return float(x_low - x_margin), float(x_high + x_margin), float(y_low - y_margin), float(y_high + y_margin)


def fit_fixed_projection(
    benign: np.ndarray,
    clean: np.ndarray,
    output: Path,
    *,
    seed: int,
    neighbors: int,
    minimum_distance: float,
    axis_quantile: float,
) -> FixedProjection:
    from sklearn.decomposition import PCA
    import umap

    values = np.concatenate([benign, clean], axis=0)
    component_count = min(50, values.shape[0] - 1, values.shape[1])
    pca = PCA(n_components=component_count, svd_solver="full", random_state=seed)
    reduced = pca.fit_transform(values)
    model = umap.UMAP(
        n_components=2,
        n_neighbors=int(neighbors),
        min_dist=float(minimum_distance),
        metric="euclidean",
        random_state=seed,
        transform_seed=seed,
    ).fit(reduced)
    pca2 = reduced[:, :2]
    umap2 = np.asarray(model.embedding_, dtype=np.float64)
    bounds = {"pca": _axis_bounds(pca2, axis_quantile), "umap": _axis_bounds(umap2, axis_quantile)}
    output.mkdir(parents=True, exist_ok=True)
    write_npz(output / "pca_basis.npz", mean=pca.mean_, components=pca.components_, explained=pca.explained_variance_ratio_)
    temporary = output / "umap_model.joblib.tmp"
    joblib.dump(model, temporary)
    os.replace(temporary, output / "umap_model.joblib")
    write_json(
        output / "projection_metadata.json",
        {
            "schema_version": "mode3-v5-fixed-projection-v1",
            "fit_roles": ["calibration_benign_probe", "calibration_trigger_clean"],
            "fit_count": int(len(values)),
            "seed": int(seed),
            "pca_components": int(component_count),
            "umap_neighbors": int(neighbors),
            "umap_min_dist": float(minimum_distance),
            "axis_quantile": float(axis_quantile),
            "bounds": bounds,
            "frozen": True,
        },
    )
    fit_coordinates = pd.DataFrame(
        {
            "kind": ["benign"] * len(benign) + ["clean"] * len(clean),
            "pca_x": pca2[:, 0],
            "pca_y": pca2[:, 1],
            "umap_x": umap2[:, 0],
            "umap_y": umap2[:, 1],
        }
    )
    write_csv(output / "fit_coordinates.csv", fit_coordinates.to_dict(orient="records"), list(fit_coordinates.columns))
    return FixedProjection(pca_mean=pca.mean_, pca_components=pca.components_, umap_model=model, bounds=bounds)


def load_fixed_projection(output: Path) -> FixedProjection:
    arrays = np.load(output / "pca_basis.npz")
    metadata = json.loads((output / "projection_metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(output / "umap_model.joblib")
    return FixedProjection(
        pca_mean=arrays["mean"],
        pca_components=arrays["components"],
        umap_model=model,
        bounds=metadata["bounds"],
    )


def _snapshot_assignments(bundle: EvaluationBundle, task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    assignments = []
    inliers = []
    centers = []
    center_names = []
    ordered_structures = sorted(bundle.structures.items())
    offsets: dict[str, int] = {}
    running_offset = 0
    for name, structure in ordered_structures:
        offsets[name] = running_offset
        running_offset += int(structure.cluster_count)
    for view, values in bundle.view_embeddings.items():
        name = "shared" if task == "shared" else (
            "random" if view.startswith("random_r") and task == "conditional" else task if task in {"prefix", "suffix", "random"} else view
        )
        structure = bundle.structures[name]
        distances = cosine_distance_to_centers(values, structure.centers)
        labels = np.argmin(distances, axis=1)
        inside = distances[np.arange(len(values)), labels] <= structure.radii[labels]
        assignments.append(labels + offsets[name])
        inliers.append(inside)
    for name, structure in ordered_structures:
        centers.append(structure.centers)
        center_names.extend([f"{name}:{index}" for index in range(structure.cluster_count)])
    return np.concatenate(assignments), np.concatenate(inliers), np.concatenate(centers), center_names


def save_snapshot(
    bundle: EvaluationBundle,
    *,
    task: str,
    benign: np.ndarray,
    projection: FixedProjection,
    output: Path,
    metadata: Mapping[str, Any],
    sample_count: int,
    dpi: int,
) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    triggered = np.concatenate(list(bundle.view_embeddings.values()), axis=0)
    view_labels = np.concatenate(
        [np.repeat(view, len(values)) for view, values in bundle.view_embeddings.items()]
    )
    repeats = len(triggered) // len(bundle.clean_embeddings)
    clean = np.tile(bundle.clean_embeddings, (repeats, 1))
    source_indices = np.tile(np.arange(len(bundle.clean_embeddings)), repeats)
    assignment, inlier, centers, center_names = _snapshot_assignments(bundle, task)
    benign_sample = benign[: min(sample_count, len(benign))]
    if len(triggered) > sample_count:
        selected = np.linspace(0, len(triggered) - 1, num=sample_count, dtype=int)
    else:
        selected = np.arange(len(triggered))
    clean_sample = clean[selected]
    triggered_sample = triggered[selected]
    assignment_sample = assignment[selected]
    inlier_sample = inlier[selected]
    view_sample = view_labels[selected]
    source_sample = source_indices[selected]
    high_dimensional = output / "high_dimensional.npz"
    write_npz(
        high_dimensional,
        benign=benign_sample,
        clean=clean_sample,
        triggered=triggered_sample,
        centers=centers,
        assignments=assignment_sample,
        inlier_mask=inlier_sample,
        view=view_sample,
        source_index=source_sample,
        center_name=np.asarray(center_names),
    )
    combined = np.concatenate([benign_sample, clean_sample, triggered_sample, centers], axis=0)
    pca = projection.pca_transform(combined)
    umap = projection.umap_transform(combined)
    kinds = (
        ["benign"] * len(benign_sample)
        + ["clean"] * len(clean_sample)
        + ["triggered_inlier" if value else "triggered_outlier" for value in inlier_sample]
        + ["center"] * len(centers)
    )
    clusters = (
        [-1] * (len(benign_sample) + len(clean_sample))
        + assignment_sample.tolist()
        + list(range(len(centers)))
    )
    views = [""] * len(benign_sample) + [""] * len(clean_sample) + view_sample.tolist() + center_names
    coordinates = pd.DataFrame(
        {
            "kind": kinds,
            "cluster": clusters,
            "view": views,
            "pca_x": pca[:, 0],
            "pca_y": pca[:, 1],
            "umap_x": umap[:, 0],
            "umap_y": umap[:, 1],
        }
    )
    coordinates_path = output / "coordinates.csv"
    write_csv(coordinates_path, coordinates.to_dict(orient="records"), list(coordinates.columns))
    metadata_path = output / "snapshot.json"
    write_json(metadata_path, dict(metadata))
    image_path = output / "cluster.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, prefix, title in zip(axes, ("pca", "umap"), ("Frozen PCA", "Frozen UMAP")):
        x = coordinates[f"{prefix}_x"].to_numpy()
        y = coordinates[f"{prefix}_y"].to_numpy()
        kind = coordinates["kind"].to_numpy()
        for label, color, marker, size, alpha in (
            ("benign", "#9aa0a6", ".", 16, 0.35),
            ("clean", "#4285f4", "o", 18, 0.45),
            ("triggered_inlier", "#e91e63", "o", 26, 0.75),
            ("triggered_outlier", "#ff9800", "x", 36, 0.90),
            ("center", "#202124", "*", 110, 1.0),
        ):
            selected_kind = kind == label
            axis.scatter(x[selected_kind], y[selected_kind], c=color, marker=marker, s=size, alpha=alpha, label=label)
        clean_offset = len(benign_sample)
        trigger_offset = clean_offset + len(clean_sample)
        for index in range(len(clean_sample)):
            axis.annotate(
                "",
                xy=(x[trigger_offset + index], y[trigger_offset + index]),
                xytext=(x[clean_offset + index], y[clean_offset + index]),
                arrowprops={"arrowstyle": "-", "color": "#5f6368", "alpha": 0.16, "lw": 0.5},
            )
        x_low, x_high, y_low, y_high = projection.bounds[prefix]
        axis.set_xlim(x_low, x_high)
        axis.set_ylim(y_low, y_high)
        axis.set_title(title)
        axis.grid(alpha=0.15)
    axes[1].legend(loc="best", fontsize=7)
    fig.suptitle(str(metadata.get("title", "Mode 3 V5 optimization snapshot")))
    fig.savefig(image_path, dpi=dpi)
    plt.close(fig)
    return [high_dimensional, coordinates_path, metadata_path, image_path]


def render_animation(frames: Sequence[Path], gif: Path, mp4: Path, *, fps: int, dpi: int) -> dict[str, Any]:
    if not frames:
        return {"frames": 0, "gif": False, "mp4": False}
    images = [plt.imread(path) for path in frames]
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.axis("off")
    artists = [[axis.imshow(image, animated=True)] for image in images]
    animation = ArtistAnimation(fig, artists, interval=1000 / max(fps, 1), blit=True)
    gif.parent.mkdir(parents=True, exist_ok=True)
    animation.save(gif, writer=PillowWriter(fps=fps), dpi=dpi)
    mp4_ok = True
    try:
        animation.save(mp4, writer=FFMpegWriter(fps=fps), dpi=dpi)
    except (FileNotFoundError, RuntimeError, OSError):
        mp4_ok = False
    plt.close(fig)
    return {"frames": len(frames), "gif": True, "mp4": mp4_ok}
