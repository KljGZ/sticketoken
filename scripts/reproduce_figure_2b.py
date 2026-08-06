"""Reproduce the Sentence-T5-base sticky-token trend plot from Figure 2(b).

The script intentionally follows the repository notebook's visualization setup:
it appends the raw string ``lucrarea`` to the second sentence 0--30 times,
encodes the resulting texts, and plots one cosine-similarity curve per sentence
pair plus a boxplot of the final similarities.

Unlike the notebook, pair selection is deterministic and every numeric input and
output needed to audit the plot is saved alongside the PNG.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_ID = "sentence-transformers/sentence-t5-base"
DEFAULT_TOKEN = "lucrarea"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the sticky-token similarity curves in Figure 2(b)."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--token", default=DEFAULT_TOKEN)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pair-count", type=int, default=25)
    parser.add_argument("--max-insertions", type=int, default=30)
    parser.add_argument("--min-sentence2-tokens", type=int, default=30)
    parser.add_argument("--max-sentence2-tokens", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-folder", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def select_pairs(
    frame: pd.DataFrame,
    pair_count: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Select unique rows nearest to evenly spaced similarity targets."""
    if pair_count < 2:
        raise ValueError("--pair-count must be at least 2")
    if len(frame) < pair_count:
        raise ValueError(
            f"Only {len(frame)} rows remain after filtering; cannot select "
            f"{pair_count} unique pairs."
        )

    targets = np.linspace(
        float(frame["similarity"].min()),
        float(frame["similarity"].max()),
        pair_count,
    )
    selected_indices: list[int] = []
    used: set[int] = set()
    for target in targets:
        distances = (frame["similarity"] - target).abs()
        for index in distances.sort_values(kind="mergesort").index:
            index = int(index)
            if index not in used:
                used.add(index)
                selected_indices.append(index)
                break

    selected = frame.loc[selected_indices].copy()
    selected.insert(0, "source_row", selected_indices)
    selected.insert(0, "pair_id", range(len(selected)))
    selected["selection_target"] = targets
    return selected.reset_index(drop=True), targets


def cosine_curves(
    model: SentenceTransformer,
    pairs: pd.DataFrame,
    token: str,
    max_insertions: int,
    batch_size: int,
    show_progress: bool,
) -> np.ndarray:
    sentence1_embeddings = model.encode(
        pairs["sentence1"].astype(str).tolist(),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )

    # Keep insertion semantics identical to the notebook: raw suffix
    # concatenation without an added separator.
    modified_sentence2 = [
        str(sentence2) + token * insertion_count
        for insertion_count in range(max_insertions + 1)
        for sentence2 in pairs["sentence2"]
    ]
    sentence2_embeddings = model.encode(
        modified_sentence2,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )
    sentence2_embeddings = sentence2_embeddings.reshape(
        max_insertions + 1,
        len(pairs),
        -1,
    )
    # [pair, dimension] x [insertion, pair, dimension] -> [pair, insertion]
    return np.einsum(
        "pd,npd->pn",
        sentence1_embeddings,
        sentence2_embeddings,
        optimize=True,
    )


def plot_curves(
    curves: np.ndarray,
    output_path: Path,
    max_insertions: int,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 14,
            "font.weight": "normal",
            "axes.labelweight": "normal",
            "axes.titleweight": "normal",
        }
    )
    color_norm = plt.Normalize(float(curves[:, 0].min()), 1.0)
    color_map = plt.cm.ScalarMappable(cmap="magma", norm=color_norm)

    figure, (line_axis, box_axis) = plt.subplots(
        1,
        2,
        figsize=(9, 6),
        gridspec_kw={"width_ratios": [4, 0.5], "wspace": 0.1},
    )
    x_values = np.arange(max_insertions + 1)
    for values in curves:
        line_axis.plot(
            x_values,
            values,
            color=color_map.to_rgba(float(values[0])),
            alpha=0.7,
            linewidth=1.2,
        )

    line_axis.set_xlabel("Inserted number of sticky token", fontsize=18)
    line_axis.set_ylabel("Cosine similarity", fontsize=18)
    line_axis.set_xticks(range(0, max_insertions + 1, 3))
    line_axis.set_xlim(0, max_insertions)
    line_axis.tick_params(axis="both", which="major", labelsize=14)
    line_axis.grid(True, linestyle="--", alpha=0.6)

    sns.boxplot(
        y=curves[:, -1],
        ax=box_axis,
        color="#D6AFB9",
        width=0.3,
        fliersize=0,
        linewidth=0.8,
    )
    box_axis.set_ylim(line_axis.get_ylim())
    box_axis.tick_params(
        axis="both",
        which="major",
        labelleft=False,
        labelbottom=False,
    )
    box_axis.grid(True, axis="y", linestyle="--", alpha=0.6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        format=output_path.suffix.lstrip(".") or "png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(figure)


def curve_statistics(curves: np.ndarray) -> dict[str, float]:
    initial = curves[:, 0]
    final = curves[:, -1]
    initial_variance = float(np.var(initial))
    final_variance = float(np.var(final))
    return {
        "initial_min": float(initial.min()),
        "initial_median": float(np.median(initial)),
        "initial_max": float(initial.max()),
        "initial_std": float(np.std(initial)),
        "initial_variance": initial_variance,
        "final_min": float(final.min()),
        "final_q1": float(np.quantile(final, 0.25)),
        "final_median": float(np.median(final)),
        "final_q3": float(np.quantile(final, 0.75)),
        "final_max": float(final.max()),
        "final_std": float(np.std(final)),
        "final_variance": final_variance,
        "variance_ratio_final_over_initial": (
            final_variance / initial_variance if initial_variance else float("nan")
        ),
    }


def tokenizer_diagnostics(
    model: SentenceTransformer,
    pairs: pd.DataFrame,
    token: str,
    max_insertions: int,
) -> dict[str, Any]:
    tokenizer = model.tokenizer
    token_ids = tokenizer(token, add_special_tokens=False)["input_ids"]
    original_lengths: list[int] = []
    final_lengths: list[int] = []
    for sentence2 in pairs["sentence2"].astype(str):
        original_lengths.append(
            len(tokenizer(sentence2, add_special_tokens=False)["input_ids"])
        )
        final_lengths.append(
            len(
                tokenizer(
                    sentence2 + token * max_insertions,
                    add_special_tokens=False,
                )["input_ids"]
            )
        )
    growth = np.asarray(final_lengths) - np.asarray(original_lengths)
    max_sequence_length = int(model.max_seq_length)
    return {
        "token_ids_without_special_tokens": [int(value) for value in token_ids],
        "token_id_length": len(token_ids),
        "token_growth_at_max_insertions_min": int(growth.min()),
        "token_growth_at_max_insertions_max": int(growth.max()),
        "model_max_sequence_length": max_sequence_length,
        "pairs_over_model_max_length_at_final_insertion": int(
            np.sum(np.asarray(final_lengths) > max_sequence_length)
        ),
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_path = (args.data or repo_root / "data/sentence_t5_base/sampled_sentence_pairs.csv").resolve()
    output_path = (
        args.output
        or repo_root
        / "fig/reproductions/figure_2b_sentence_t5_base_lucrarea.png"
    ).resolve()
    cache_folder = args.cache_folder.resolve() if args.cache_folder else None

    if args.max_insertions < 1:
        raise ValueError("--max-insertions must be positive")
    if args.min_sentence2_tokens > args.max_sentence2_tokens:
        raise ValueError("Minimum sentence length exceeds maximum sentence length")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {args.device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading model: {args.model_id}")
    model = SentenceTransformer(
        args.model_id,
        device=args.device,
        cache_folder=str(cache_folder) if cache_folder else None,
    )
    model.eval()

    frame = pd.read_csv(data_path)
    required_columns = {"sentence1", "sentence2", "similarity"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing data columns: {sorted(missing_columns)}")
    frame = frame.dropna(subset=list(required_columns)).copy()
    frame["similarity"] = pd.to_numeric(frame["similarity"], errors="raise")
    frame["sentence2_token_length"] = frame["sentence2"].astype(str).map(
        lambda text: len(model.tokenizer.tokenize(text))
    )
    filtered = frame[
        frame["sentence2_token_length"].between(
            args.min_sentence2_tokens,
            args.max_sentence2_tokens,
        )
    ].reset_index(drop=False).rename(columns={"index": "data_row"})
    selected, selection_targets = select_pairs(filtered, args.pair_count)

    print(
        f"Selected {len(selected)} pairs from {len(filtered)} filtered rows; "
        f"computing {args.max_insertions + 1} points per curve."
    )
    curves = cosine_curves(
        model=model,
        pairs=selected,
        token=args.token,
        max_insertions=args.max_insertions,
        batch_size=args.batch_size,
        show_progress=args.show_progress,
    )
    plot_curves(curves, output_path, args.max_insertions, args.dpi)

    artifact_stem = output_path.with_suffix("")
    pairs_path = artifact_stem.with_name(artifact_stem.name + ".pairs.csv")
    curves_path = artifact_stem.with_name(artifact_stem.name + ".curves.csv")
    metadata_path = artifact_stem.with_name(artifact_stem.name + ".metadata.json")

    selected_output = selected.copy()
    selected_output["stored_initial_similarity"] = selected_output.pop("similarity")
    selected_output["computed_initial_similarity"] = curves[:, 0]
    selected_output.to_csv(pairs_path, index=False)

    curve_frame = pd.DataFrame(
        curves,
        columns=[f"similarity_n{n:02d}" for n in range(args.max_insertions + 1)],
    )
    curve_frame.insert(0, "source_row", selected["source_row"])
    curve_frame.insert(0, "data_row", selected["data_row"])
    curve_frame.insert(0, "pair_id", selected["pair_id"])
    curve_frame.to_csv(curves_path, index=False)

    first_module = model._first_module()
    auto_model = getattr(first_module, "auto_model", None)
    model_config = getattr(auto_model, "config", None)
    statistics = curve_statistics(curves)
    diagnostics = tokenizer_diagnostics(
        model,
        selected,
        args.token,
        args.max_insertions,
    )
    cuda_device_index = (
        torch.device(args.device).index
        if args.device.startswith("cuda")
        else None
    )
    if cuda_device_index is None and args.device.startswith("cuda"):
        cuda_device_index = torch.cuda.current_device()

    metadata: dict[str, Any] = {
        "figure": "Figure 2(b)",
        "repo_commit": git_commit(repo_root),
        "model_id": args.model_id,
        "model_revision": getattr(model_config, "_commit_hash", None),
        "token": args.token,
        "insertion_operation": "raw string suffix concatenation",
        "pair_count": len(selected),
        "max_insertions": args.max_insertions,
        "selection_method": "nearest unique rows to pair_count evenly spaced similarity targets",
        "selection_targets": [float(value) for value in selection_targets],
        "sentence2_token_length_filter": [
            args.min_sentence2_tokens,
            args.max_sentence2_tokens,
        ],
        "input_data": str(data_path),
        "input_sha256": sha256_file(data_path),
        "output_png": str(output_path),
        "output_pairs_csv": str(pairs_path),
        "output_curves_csv": str(curves_path),
        "seed": args.seed,
        "device": args.device,
        "gpu_name": (
            torch.cuda.get_device_name(cuda_device_index)
            if cuda_device_index is not None
            else None
        ),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "sentence_transformers": package_version("sentence-transformers"),
            "transformers": package_version("transformers"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": package_version("matplotlib"),
            "seaborn": package_version("seaborn"),
        },
        "tokenizer_diagnostics": diagnostics,
        "curve_statistics": statistics,
        "stored_vs_computed_initial_max_abs_difference": float(
            np.max(np.abs(selected["similarity"].to_numpy() - curves[:, 0]))
        ),
    }
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)

    print(f"Figure:   {output_path}")
    print(f"Pairs:    {pairs_path}")
    print(f"Curves:   {curves_path}")
    print(f"Metadata: {metadata_path}")
    print(
        "Similarity variance: "
        f"{statistics['initial_variance']:.6f} -> "
        f"{statistics['final_variance']:.6f} "
        f"(ratio={statistics['variance_ratio_final_over_initial']:.3f})"
    )
    print(f"Final median similarity: {statistics['final_median']:.6f}")


if __name__ == "__main__":
    main()
