"""Search for a one-way, non-degrading sticky-high token and plot its curves.

The experiment has three disjoint data partitions:

1. a small low/high search split for an exhaustive vocabulary screen;
2. a larger low/high validation split for coarse and full-curve verification;
3. a held-out full-similarity-range split used only for the final figure.

All candidate rankings, selected pairs, curves, thresholds, package versions and
model revision information are written to the output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Support the documented ``python scripts/detect_sticky_high.py`` invocation
# without relying on a machine-specific /root/StickyToken path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stickytoken.sticky_high import (
    StickyHighThresholds,
    add_constraint_violation,
    baseline_embeddings,
    compose_ordered_candidate_pairs,
    evaluate_candidate_batch,
    load_token_candidates,
    make_disjoint_splits,
    parse_insertion_counts,
    plot_sticky_high_curves,
    rank_candidates,
    score_candidate_frame,
    select_diverse_candidates,
    sha256_file,
    thresholds_as_dict,
)


DEFAULT_MODEL_ID = "sentence-transformers/sentence-t5-base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and validate a non-degrading sticky-high token."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--token-analysis", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--reuse-screen-dir",
        type=Path,
        default=None,
        help="Reuse audited screen artifacts from a configuration-identical prior run.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-folder", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--candidate-chunk-size", type=int, default=64)
    parser.add_argument("--validation-chunk-size", type=int, default=8)
    parser.add_argument("--low-threshold", type=float, default=0.65)
    parser.add_argument("--high-threshold", type=float, default=0.82)
    parser.add_argument("--search-per-group", type=int, default=8)
    parser.add_argument("--validation-per-group", type=int, default=48)
    parser.add_argument("--plot-pair-count", type=int, default=25)
    parser.add_argument("--screen-insertions", type=int, default=30)
    parser.add_argument("--coarse-counts", default="1,2,4,8,16,30")
    parser.add_argument("--max-insertions", type=int, default=30)
    parser.add_argument("--shortlist-size", type=int, default=128)
    parser.add_argument("--finalist-size", type=int, default=16)
    parser.add_argument(
        "--max-components",
        type=int,
        choices=(1, 2),
        default=1,
        help="Search single tokens only (1), or extend search-selected tokens to ordered pairs (2).",
    )
    parser.add_argument(
        "--component-pool-size",
        type=int,
        default=128,
        help="Number of top single-token screen results used to form ordered pairs.",
    )
    parser.add_argument("--min-sentence2-tokens", type=int, default=5)
    parser.add_argument("--max-sentence2-tokens", type=int, default=160)
    parser.add_argument("--min-low-gain", type=float, default=0.02)
    parser.add_argument("--high-drop-tolerance", type=float, default=0.02)
    parser.add_argument("--max-high-failure-rate", type=float, default=0.10)
    parser.add_argument("--step-drop-tolerance", type=float, default=0.002)
    parser.add_argument("--max-step-failure-rate", type=float, default=0.10)
    parser.add_argument("--high-penalty-weight", type=float, default=4.0)
    parser.add_argument("--high-failure-weight", type=float, default=0.10)
    parser.add_argument("--step-failure-weight", type=float, default=0.05)
    parser.add_argument("--separator", default="")
    parser.add_argument("--include-special", action="store_true")
    parser.add_argument("--max-candidate-chars", type=int, default=64)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Debug-only deterministic cap on candidate count.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def prepare_frame(
    data_path: Path,
    tokenizer,
    min_tokens: int,
    max_tokens: int,
) -> pd.DataFrame:
    frame = pd.read_csv(data_path)
    required = {"sentence1", "sentence2"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required data columns: {sorted(missing)}")
    frame = frame.dropna(subset=list(required)).copy()
    frame.insert(0, "data_row", frame.index.astype(int))
    frame["sentence1"] = frame["sentence1"].astype(str)
    frame["sentence2"] = frame["sentence2"].astype(str)
    frame["sentence2_token_length"] = frame["sentence2"].map(
        lambda value: len(tokenizer(value, add_special_tokens=False)["input_ids"])
    )
    return frame[
        frame["sentence2_token_length"].between(min_tokens, max_tokens)
    ].reset_index(drop=True)


def split_view(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    similarities: np.ndarray,
    indices: list[int],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    positions = np.asarray(indices, dtype=int)
    return (
        frame.iloc[positions].reset_index(drop=True),
        embeddings[positions],
        similarities[positions],
    )


def evaluate_stage(
    *,
    model,
    candidates: pd.DataFrame,
    pairs: pd.DataFrame,
    reference_embeddings: np.ndarray,
    baseline: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    insertion_counts: list[int],
    separator: str,
    encode_batch_size: int,
    candidate_chunk_size: int,
    thresholds: StickyHighThresholds,
    stage_name: str,
) -> pd.DataFrame:
    low_mask = baseline <= low_threshold
    high_mask = baseline >= high_threshold
    if not low_mask.any() or not high_mask.any():
        raise ValueError(f"{stage_name} split lacks a low or high group")

    stage_frames: list[pd.DataFrame] = []
    chunk_count = (len(candidates) + candidate_chunk_size - 1) // candidate_chunk_size
    for chunk_index, start in enumerate(
        range(0, len(candidates), candidate_chunk_size), start=1
    ):
        chunk = candidates.iloc[start : start + candidate_chunk_size].reset_index(
            drop=True
        )
        print(
            f"{stage_name}: candidate chunk {chunk_index}/{chunk_count} "
            f"({start + 1}-{start + len(chunk)} of {len(candidates)})",
            flush=True,
        )
        similarities = evaluate_candidate_batch(
            model,
            chunk["candidate"].tolist(),
            pairs["sentence2"].tolist(),
            reference_embeddings,
            insertion_counts,
            separator=separator,
            batch_size=encode_batch_size,
            show_progress=False,
        )
        scored = score_candidate_frame(
            chunk,
            similarities,
            baseline,
            low_mask,
            high_mask,
            insertion_counts,
            thresholds,
        )
        stage_frames.append(scored)
    return rank_candidates(pd.concat(stage_frames, ignore_index=True))


def write_split_manifest(
    frame: pd.DataFrame,
    similarities: np.ndarray,
    splits: dict[str, list[int]],
    output_path: Path,
    low_threshold: float,
    high_threshold: float,
) -> None:
    records = []
    for split_name, indices in splits.items():
        for index in indices:
            row = frame.iloc[index]
            similarity = float(similarities[index])
            group = (
                "low"
                if similarity <= low_threshold
                else "high"
                if similarity >= high_threshold
                else "middle"
            )
            records.append(
                {
                    "split": split_name,
                    "group": group,
                    "filtered_row": int(index),
                    "data_row": int(row["data_row"]),
                    "baseline_similarity": similarity,
                    "sentence2_token_length": int(row["sentence2_token_length"]),
                    "sentence1": row["sentence1"],
                    "sentence2": row["sentence2"],
                }
            )
    pd.DataFrame.from_records(records).to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    repo_root = REPO_ROOT
    data_path = (
        args.data
        or repo_root / "data/sentence_t5_base/sampled_sentence_pairs.csv"
    ).resolve()
    model_name = safe_name(args.model_id.split("/")[-1])
    token_analysis_path = (
        args.token_analysis
        or repo_root / f"results/tokenizer_analysis/{model_name}.jsonl"
    ).resolve()
    output_dir = (
        args.output_dir
        or repo_root / f"results/sticky_high/{model_name}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reuse_screen_dir = (
        args.reuse_screen_dir.resolve() if args.reuse_screen_dir else None
    )
    cache_folder = args.cache_folder.resolve() if args.cache_folder else None

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {args.device}")
    if args.screen_insertions < 1 or args.max_insertions < 1:
        raise ValueError("Insertion counts must be positive")
    coarse_counts = parse_insertion_counts(args.coarse_counts)
    if coarse_counts[-1] != args.max_insertions:
        raise ValueError("The largest --coarse-counts value must equal --max-insertions")
    if args.screen_insertions != args.max_insertions:
        print(
            "Warning: screen_insertions differs from max_insertions; the screen and "
            "validation objectives evaluate different endpoints."
        )

    thresholds = StickyHighThresholds(
        min_low_gain=args.min_low_gain,
        high_drop_tolerance=args.high_drop_tolerance,
        max_high_failure_rate=args.max_high_failure_rate,
        step_drop_tolerance=args.step_drop_tolerance,
        max_step_failure_rate=args.max_step_failure_rate,
        high_penalty_weight=args.high_penalty_weight,
        high_failure_weight=args.high_failure_weight,
        step_failure_weight=args.step_failure_weight,
    )
    thresholds.validate()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading model: {args.model_id}", flush=True)
    model = SentenceTransformer(
        args.model_id,
        device=args.device,
        cache_folder=str(cache_folder) if cache_folder else None,
    )
    model.eval()

    print(f"Loading and filtering pairs: {data_path}", flush=True)
    frame = prepare_frame(
        data_path,
        model.tokenizer,
        args.min_sentence2_tokens,
        args.max_sentence2_tokens,
    )
    sentence1_embeddings, _, baseline = baseline_embeddings(
        model,
        frame,
        batch_size=args.batch_size,
        show_progress=args.show_progress,
    )
    frame["computed_similarity"] = baseline
    splits = make_disjoint_splits(
        baseline,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        search_per_group=args.search_per_group,
        validation_per_group=args.validation_per_group,
        plot_pair_count=args.plot_pair_count,
        seed=args.seed,
    )
    split_manifest_path = output_dir / "split_manifest.csv"
    write_split_manifest(
        frame,
        baseline,
        splits,
        split_manifest_path,
        args.low_threshold,
        args.high_threshold,
    )

    candidates = load_token_candidates(
        token_analysis_path,
        include_special=args.include_special,
        max_chars=args.max_candidate_chars,
        max_candidates=args.max_candidates,
    )
    print(f"Loaded {len(candidates)} reachable literal token candidates.", flush=True)

    if reuse_screen_dir is not None:
        reuse_metadata_path = reuse_screen_dir / "metadata.json"
        reuse_manifest_path = reuse_screen_dir / "split_manifest.csv"
        if not reuse_metadata_path.exists() or not reuse_manifest_path.exists():
            raise FileNotFoundError(
                "--reuse-screen-dir must contain metadata.json and split_manifest.csv"
            )
        reuse_metadata = json.loads(reuse_metadata_path.read_text(encoding="utf-8"))
        expected_checks = {
            "model_id": (reuse_metadata.get("model_id"), args.model_id),
            "data_sha256": (
                reuse_metadata.get("data", {}).get("source_sha256"),
                sha256_file(data_path),
            ),
            "candidate_sha256": (
                reuse_metadata.get("candidate_space", {}).get("source_sha256"),
                sha256_file(token_analysis_path),
            ),
            "seed": (reuse_metadata.get("seed"), args.seed),
            "thresholds": (
                reuse_metadata.get("thresholds"),
                thresholds_as_dict(thresholds),
            ),
            "low_threshold": (
                reuse_metadata.get("data", {}).get("low_threshold"),
                args.low_threshold,
            ),
            "high_threshold": (
                reuse_metadata.get("data", {}).get("high_threshold"),
                args.high_threshold,
            ),
            "screen_insertions": (
                reuse_metadata.get("insertion", {}).get("screen_insertions"),
                args.screen_insertions,
            ),
            "max_components": (
                reuse_metadata.get("candidate_space", {}).get("max_components"),
                args.max_components,
            ),
            "component_pool_size": (
                reuse_metadata.get("candidate_space", {}).get("component_pool_size"),
                min(args.component_pool_size, len(candidates))
                if args.max_components == 2
                else 0,
            ),
        }
        mismatches = {
            name: values for name, values in expected_checks.items() if values[0] != values[1]
        }
        current_manifest = pd.read_csv(split_manifest_path)[["split", "data_row"]]
        reused_manifest = pd.read_csv(reuse_manifest_path)[["split", "data_row"]]
        if not current_manifest.equals(reused_manifest):
            mismatches["split_manifest"] = ("reused", "current")
        if mismatches:
            raise ValueError(f"Refusing incompatible screen reuse: {mismatches}")
        print(f"Reusing configuration-identical screens from {reuse_screen_dir}")

    search_indices = splits["search_low"] + splits["search_high"]
    search_pairs, search_embeddings, search_baseline = split_view(
        frame, sentence1_embeddings, baseline, search_indices
    )
    if reuse_screen_dir is None:
        screen = evaluate_stage(
            model=model,
            candidates=candidates,
            pairs=search_pairs,
            reference_embeddings=search_embeddings,
            baseline=search_baseline,
            low_threshold=args.low_threshold,
            high_threshold=args.high_threshold,
            insertion_counts=[args.screen_insertions],
            separator=args.separator,
            encode_batch_size=args.batch_size,
            candidate_chunk_size=args.candidate_chunk_size,
            thresholds=thresholds,
            stage_name="vocabulary screen",
        )
        screen.insert(0, "rank", np.arange(1, len(screen) + 1))
        screen_path = output_dir / "screen_scores.csv"
        screen.to_csv(screen_path, index=False)
    else:
        screen_path = reuse_screen_dir / "screen_scores.csv"
        screen = add_constraint_violation(
            pd.read_csv(screen_path, keep_default_na=False), thresholds
        )
        screen = rank_candidates(screen.drop(columns=["rank"], errors="ignore"))
        screen.insert(0, "rank", np.arange(1, len(screen) + 1))

    combined_screen_parts = [screen.drop(columns=["rank"], errors="ignore")]
    pair_screen_path: Path | None = None
    pair_candidates = pd.DataFrame()
    if args.max_components == 2:
        component_count = min(args.component_pool_size, len(screen))
        if component_count < 2:
            raise ValueError("At least two screened components are required for pair search")
        component_columns = [
            "token_id",
            "raw_vocab",
            "candidate",
            "category",
            "character_length",
            "candidate_kind",
            "component_count",
            "component_token_ids",
        ]
        if reuse_screen_dir is None:
            components = screen.head(component_count)[component_columns]
            pair_candidates = compose_ordered_candidate_pairs(
                components,
                max_chars=args.max_candidate_chars * 2,
            )
            print(
                f"Formed {len(pair_candidates)} unique ordered literal pairs from "
                f"{component_count} search-selected components.",
                flush=True,
            )
            pair_screen = evaluate_stage(
                model=model,
                candidates=pair_candidates,
                pairs=search_pairs,
                reference_embeddings=search_embeddings,
                baseline=search_baseline,
                low_threshold=args.low_threshold,
                high_threshold=args.high_threshold,
                insertion_counts=[args.screen_insertions],
                separator=args.separator,
                encode_batch_size=args.batch_size,
                candidate_chunk_size=args.candidate_chunk_size,
                thresholds=thresholds,
                stage_name="ordered-pair screen",
            )
            pair_screen.insert(0, "rank", np.arange(1, len(pair_screen) + 1))
            pair_screen_path = output_dir / "pair_screen_scores.csv"
            pair_screen.to_csv(pair_screen_path, index=False)
        else:
            pair_screen_path = reuse_screen_dir / "pair_screen_scores.csv"
            pair_screen = add_constraint_violation(
                pd.read_csv(pair_screen_path, keep_default_na=False), thresholds
            )
            pair_screen = rank_candidates(
                pair_screen.drop(columns=["rank"], errors="ignore")
            )
            pair_screen.insert(0, "rank", np.arange(1, len(pair_screen) + 1))
            pair_candidates = pair_screen[component_columns].copy()
        combined_screen_parts.append(
            pair_screen.drop(columns=["rank"], errors="ignore")
        )

    combined_screen = rank_candidates(
        pd.concat(combined_screen_parts, ignore_index=True)
    )
    combined_screen.insert(0, "rank", np.arange(1, len(combined_screen) + 1))
    combined_screen_path = output_dir / "combined_screen_scores.csv"
    combined_screen.to_csv(combined_screen_path, index=False)

    shortlist = select_diverse_candidates(
        combined_screen.drop(columns=["rank"], errors="ignore"),
        min(args.shortlist_size, len(combined_screen)),
        thresholds,
    )
    candidate_columns = [
        "token_id",
        "raw_vocab",
        "candidate",
        "category",
        "character_length",
        "candidate_kind",
        "component_count",
        "component_token_ids",
    ]
    shortlist = shortlist[candidate_columns]

    validation_indices = splits["validation_low"] + splits["validation_high"]
    validation_pairs, validation_embeddings, validation_baseline = split_view(
        frame, sentence1_embeddings, baseline, validation_indices
    )
    coarse = evaluate_stage(
        model=model,
        candidates=shortlist,
        pairs=validation_pairs,
        reference_embeddings=validation_embeddings,
        baseline=validation_baseline,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        insertion_counts=coarse_counts,
        separator=args.separator,
        encode_batch_size=args.batch_size,
        candidate_chunk_size=args.validation_chunk_size,
        thresholds=thresholds,
        stage_name="coarse validation",
    )
    coarse.insert(0, "rank", np.arange(1, len(coarse) + 1))
    coarse_path = output_dir / "coarse_validation.csv"
    coarse.to_csv(coarse_path, index=False)

    finalists = select_diverse_candidates(
        coarse.drop(columns=["rank"], errors="ignore"),
        min(args.finalist_size, len(coarse)),
        thresholds,
    )
    finalists = finalists[candidate_columns]
    full_counts = list(range(1, args.max_insertions + 1))
    full = evaluate_stage(
        model=model,
        candidates=finalists,
        pairs=validation_pairs,
        reference_embeddings=validation_embeddings,
        baseline=validation_baseline,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        insertion_counts=full_counts,
        separator=args.separator,
        encode_batch_size=args.batch_size,
        candidate_chunk_size=max(1, min(args.validation_chunk_size, 4)),
        thresholds=thresholds,
        stage_name="full-curve validation",
    )
    full.insert(0, "rank", np.arange(1, len(full) + 1))
    full_path = output_dir / "full_validation.csv"
    full.to_csv(full_path, index=False)

    best = full.iloc[0]
    best_candidate = str(best["candidate"])
    plot_pairs, plot_embeddings, plot_baseline = split_view(
        frame, sentence1_embeddings, baseline, splits["plot"]
    )
    plot_modified = evaluate_candidate_batch(
        model,
        [best_candidate],
        plot_pairs["sentence2"].tolist(),
        plot_embeddings,
        full_counts,
        separator=args.separator,
        batch_size=args.batch_size,
        show_progress=args.show_progress,
    )[0]
    curves = np.vstack([plot_baseline, plot_modified]).T
    figure_path = output_dir / "inserted_number_of_sticky_high_token.png"
    plot_sticky_high_curves(
        curves,
        figure_path,
        max_insertions=args.max_insertions,
        dpi=args.dpi,
    )

    plot_pairs_output = plot_pairs.copy()
    plot_pairs_output.insert(0, "pair_id", np.arange(len(plot_pairs_output)))
    plot_pairs_output["baseline_similarity"] = plot_baseline
    plot_pairs_output["group"] = np.where(
        plot_baseline <= args.low_threshold,
        "low",
        np.where(plot_baseline >= args.high_threshold, "high", "middle"),
    )
    plot_pairs_path = output_dir / "plot_pairs.csv"
    plot_pairs_output.to_csv(plot_pairs_path, index=False)

    curve_frame = pd.DataFrame(
        curves,
        columns=[f"similarity_n{count:02d}" for count in range(args.max_insertions + 1)],
    )
    curve_frame.insert(0, "data_row", plot_pairs["data_row"].to_numpy())
    curve_frame.insert(0, "pair_id", np.arange(len(curve_frame)))
    curves_path = output_dir / "plot_curves.csv"
    curve_frame.to_csv(curves_path, index=False)

    first_module = model._first_module()
    auto_model = getattr(first_module, "auto_model", None)
    model_config = getattr(auto_model, "config", None)
    cuda_index = torch.device(args.device).index if args.device.startswith("cuda") else None
    if cuda_index is None and args.device.startswith("cuda"):
        cuda_index = torch.cuda.current_device()

    plot_delta = curves[:, -1] - curves[:, 0]
    metadata: dict[str, Any] = {
        "experiment": (
            "sticky-high-v1-ordered-pair"
            if args.max_components == 2
            else "sticky-high-v1-single-token"
        ),
        "definition": (
            "Raise the lower similarity tail while constraining loss in the "
            "upper tail; certify only after full insertion-curve validation."
        ),
        "repo_commit": git_commit(repo_root),
        "model_id": args.model_id,
        "model_revision": getattr(model_config, "_commit_hash", None),
        "device": args.device,
        "gpu_name": (
            torch.cuda.get_device_name(cuda_index) if cuda_index is not None else None
        ),
        "seed": args.seed,
        "screen_reuse": (
            {
                "enabled": True,
                "source_directory": str(reuse_screen_dir),
                "source_repo_commit": reuse_metadata.get("repo_commit"),
                "compatibility_checks_passed": True,
            }
            if reuse_screen_dir is not None
            else {"enabled": False}
        ),
        "best_candidate": {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in best.to_dict().items()
        },
        "candidate_space": {
            "kind": (
                "reachable literal single-token strings plus ordered pairs "
                "formed from search-selected components"
                if args.max_components == 2
                else "reachable literal single-token strings"
            ),
            "single_token_count": len(candidates),
            "ordered_pair_count": len(pair_candidates),
            "combined_count": len(combined_screen),
            "max_components": args.max_components,
            "component_pool_size": (
                min(args.component_pool_size, len(screen))
                if args.max_components == 2
                else 0
            ),
            "include_special": args.include_special,
            "source": str(token_analysis_path),
            "source_sha256": sha256_file(token_analysis_path),
        },
        "data": {
            "source": str(data_path),
            "source_sha256": sha256_file(data_path),
            "rows_after_length_filter": len(frame),
            "sentence2_token_length_filter": [
                args.min_sentence2_tokens,
                args.max_sentence2_tokens,
            ],
            "low_threshold": args.low_threshold,
            "high_threshold": args.high_threshold,
            "split_sizes": {key: len(value) for key, value in splits.items()},
            "disjoint_splits": True,
        },
        "thresholds": thresholds_as_dict(thresholds),
        "insertion": {
            "operation": "literal suffix concatenation",
            "separator": args.separator,
            "screen_insertions": args.screen_insertions,
            "coarse_counts": coarse_counts,
            "full_counts": full_counts,
        },
        "plot_holdout_statistics": {
            "initial_min": float(curves[:, 0].min()),
            "initial_median": float(np.median(curves[:, 0])),
            "initial_max": float(curves[:, 0].max()),
            "final_min": float(curves[:, -1].min()),
            "final_median": float(np.median(curves[:, -1])),
            "final_max": float(curves[:, -1].max()),
            "delta_min": float(plot_delta.min()),
            "delta_median": float(np.median(plot_delta)),
            "delta_max": float(plot_delta.max()),
        },
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
        "artifacts": {
            "screen_scores": str(screen_path),
            "pair_screen_scores": (
                str(pair_screen_path) if pair_screen_path is not None else None
            ),
            "combined_screen_scores": str(combined_screen_path),
            "coarse_validation": str(coarse_path),
            "full_validation": str(full_path),
            "split_manifest": str(split_manifest_path),
            "plot_pairs": str(plot_pairs_path),
            "plot_curves": str(curves_path),
            "figure": str(figure_path),
        },
    }
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)

    print("Sticky-high experiment complete.")
    print(
        f"Best literal candidate: {best_candidate!r} "
        f"(component_token_ids={best['component_token_ids']})"
    )
    print(f"Certified: {bool(best['certified'])}")
    print(
        f"Validation low q10 gain={float(best['low_gain_q10']):.6f}; "
        f"high q05 gain={float(best['high_gain_q05']):.6f}; "
        f"high failure={float(best['high_failure_rate']):.3f}"
    )
    print(f"Figure: {figure_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
