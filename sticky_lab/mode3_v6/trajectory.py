"""Fixed-basis optimization trajectory rendering and raw-coordinate export."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .projection import FrozenProjection


def render_trajectory(snapshot_paths: list[Path], output: Path, *, seed: int) -> None:
    if not snapshot_paths:
        raise RuntimeError("no optimization snapshots")
    first = np.load(snapshot_paths[0])
    projection = FrozenProjection.fit(first["independent_benign"], seed=seed)
    methods = ["pca"] + (["umap"] if projection.umap is not None else [])
    output.mkdir(parents=True, exist_ok=True)
    for method in methods:
        transformed = []
        for path in snapshot_paths:
            data = np.load(path)
            transformed.append({
                "benign": projection.transform(data["independent_benign"], method),
                "clean": projection.transform(data["paired_clean"], method),
                "triggered": projection.transform(data["triggered"], method),
                "center": projection.transform(data["centers"], method),
                "inside": data["triggered_inside"].astype(bool),
            })
        all_xy = np.vstack([group[name] for group in transformed for name in ("benign", "clean", "triggered", "center")])
        low = all_xy.min(axis=0); high = all_xy.max(axis=0); padding = np.maximum((high - low) * 0.05, 1e-6)
        frames = []
        for index, (path, group) in enumerate(zip(snapshot_paths, transformed)):
            import matplotlib.pyplot as plt
            fig, axis = plt.subplots(figsize=(8, 6))
            axis.scatter(group["benign"][:, 0], group["benign"][:, 1], s=6, color="#b6b6b6", alpha=.35, label="independent_benign")
            axis.scatter(group["clean"][:, 0], group["clean"][:, 1], s=10, color="#74a9ff", alpha=.55, label="paired_clean")
            inside = group["inside"]
            axis.scatter(group["triggered"][inside, 0], group["triggered"][inside, 1], s=12, color="#f54278", label="triggered_inlier")
            axis.scatter(group["triggered"][~inside, 0], group["triggered"][~inside, 1], s=28, color="#ff8c00", marker="x", label="triggered_outlier")
            axis.scatter(group["center"][:, 0], group["center"][:, 1], s=110, color="black", marker="*", label="frozen_center")
            for clean, triggered in zip(group["clean"], group["triggered"]):
                axis.annotate("", xy=triggered, xytext=clean, arrowprops={"arrowstyle": "-", "alpha": .08, "color": "#555555"})
            axis.set_xlim(low[0] - padding[0], high[0] + padding[0]); axis.set_ylim(low[1] - padding[1], high[1] + padding[1])
            axis.set_title(f"Frozen {method.upper()} | iteration {index:04d}")
            axis.legend(loc="best"); axis.grid(alpha=.2); fig.tight_layout()
            frame = output / method / f"frame_{index:04d}.png"; frame.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(frame, dpi=150); plt.close(fig); frames.append(frame)
            coordinates = output / method / f"coordinates_{index:04d}.csv"
            with coordinates.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle); writer.writerow(["source_snapshot", "group", "x", "y", "inlier"])
                for name in ("benign", "clean", "triggered", "center"):
                    for row_index, (x, y) in enumerate(group[name]):
                        flag = bool(group["inside"][row_index]) if name == "triggered" else ""
                        writer.writerow([path.name, name, float(x), float(y), flag])
        import imageio.v2 as imageio
        images = [imageio.imread(path) for path in frames]
        imageio.mimsave(output / f"trajectory_{method}.gif", images, duration=0.2, loop=0)
        imageio.mimsave(output / f"trajectory_{method}.mp4", images, fps=5)
