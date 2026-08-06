"""Generate distribution-only figures from completed three-mode curve CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


RUNS = {
    "single_sticky": "single_sticky_v1",
    "multi_booster": "multi_booster_v1",
    "repulsive_attractor": "repulsive_attractor_v1",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    base = args.repo_root.resolve() / "results/sticky_lab/sentence_t5_base"
    for mode, directory in RUNS.items():
        run_dir = base / directory
        curves = pd.read_csv(run_dir / "similarity_curves.csv")
        start = curves[curves["inserted_count"] == curves["inserted_count"].min()]["cosine_similarity"].reset_index(drop=True)
        final = curves[curves["inserted_count"] == curves["inserted_count"].max()]["cosine_similarity"].reset_index(drop=True)
        distributions = pd.DataFrame({"initial": start, "final": final}).melt(var_name="stage", value_name="cosine_similarity")
        figure, axes = plt.subplots(1, 2, figsize=(9, 4))
        sns.boxplot(data=distributions, x="stage", y="cosine_similarity", ax=axes[0], color="#D6AFB9", fliersize=0)
        sns.histplot(final - start, bins=12, kde=True, ax=axes[1], color="#6C5B7B")
        axes[0].set_title(f"{mode}: initial vs final")
        axes[1].set_title("Per-pair similarity change")
        axes[1].set_xlabel("final - initial cosine similarity")
        for axis in axes:
            axis.grid(True, linestyle="--", alpha=0.35)
        figure.tight_layout()
        figure.savefig(run_dir / "final_similarity_boxplot.png", dpi=300, bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    main()
