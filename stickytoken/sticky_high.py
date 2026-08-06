"""Detection primitives for non-degrading (``sticky-high``) token strings.

The original Sticky Token Detector focuses on low-similarity sentence pairs and
therefore cannot tell a one-way similarity booster from a token that pulls every
pair toward a mean attractor.  This module evaluates both tails explicitly:

* low-similarity pairs should gain similarity;
* high-similarity pairs should not lose more than a configured tolerance;
* on the full insertion curve, material step-wise regressions should be rare.

The implementation is deliberately model-agnostic.  It only requires a
SentenceTransformer-like object exposing ``encode`` and is suitable for batched
black-box evaluation as well as local GPU experiments.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StickyHighThresholds:
    """Acceptance thresholds and objective weights for a sticky-high string."""

    min_low_gain: float = 0.02
    high_drop_tolerance: float = 0.02
    max_high_failure_rate: float = 0.10
    step_drop_tolerance: float = 0.002
    max_step_failure_rate: float = 0.10
    high_penalty_weight: float = 4.0
    high_failure_weight: float = 0.10
    step_failure_weight: float = 0.05

    def validate(self) -> None:
        if self.min_low_gain < 0:
            raise ValueError("min_low_gain must be non-negative")
        if self.high_drop_tolerance < 0 or self.step_drop_tolerance < 0:
            raise ValueError("drop tolerances must be non-negative")
        for name in ("max_high_failure_rate", "max_step_failure_rate"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_insertion_counts(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        counts = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        counts = [int(item) for item in value]
    counts = sorted(set(counts))
    if not counts or counts[0] < 1:
        raise ValueError("Insertion counts must contain positive integers")
    return counts


def load_token_candidates(
    analysis_path: Path,
    *,
    include_special: bool = False,
    max_chars: int = 64,
    max_candidates: int | None = None,
) -> pd.DataFrame:
    """Load deterministic, reachable token strings from tokenizer analysis JSONL.

    Candidate identity is the literal decoded string used at deployment.  Two
    token IDs decoding to the same string are therefore deduplicated.
    """

    allowed_categories = {"OK"}
    if include_special:
        allowed_categories.add("OK_SPECIAL")

    records: list[dict[str, object]] = []
    seen_strings: set[str] = set()
    with analysis_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {analysis_path}:{line_number}"
                ) from exc
            if item.get("category") not in allowed_categories:
                continue
            decoded = str(item.get("decoded", ""))
            if not decoded.strip() or len(decoded) > max_chars:
                continue
            if any(unicodedata.category(char) in {"Cc", "Cs"} for char in decoded):
                continue
            if decoded in seen_strings:
                continue
            seen_strings.add(decoded)
            records.append(
                {
                    "token_id": int(item["i"]),
                    "raw_vocab": str(item.get("raw_vocab", "")),
                    "candidate": decoded,
                    "category": str(item["category"]),
                    "character_length": len(decoded),
                    "candidate_kind": "single_token",
                    "component_count": 1,
                    "component_token_ids": str(int(item["i"])),
                }
            )

    frame = pd.DataFrame.from_records(records).sort_values(
        ["token_id", "candidate"], kind="mergesort"
    )
    if max_candidates is not None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        frame = frame.head(max_candidates)
    return frame.reset_index(drop=True)


def compose_ordered_candidate_pairs(
    components: pd.DataFrame,
    *,
    max_chars: int = 128,
) -> pd.DataFrame:
    """Form deterministic ordered two-component literal strings.

    Components must have been selected using the search split only.  Literal
    strings are deduplicated because different token-ID pairs can collapse to
    the same deployed text after decoding.
    """

    required = {"token_id", "raw_vocab", "candidate"}
    missing = required - set(components.columns)
    if missing:
        raise ValueError(f"Missing component columns: {sorted(missing)}")
    if max_chars < 1:
        raise ValueError("max_chars must be positive")

    records: list[dict[str, object]] = []
    seen_literals: set[str] = set()
    rows = list(components.reset_index(drop=True).to_dict("records"))
    for left in rows:
        for right in rows:
            literal = str(left["candidate"]) + str(right["candidate"])
            if not literal.strip() or len(literal) > max_chars:
                continue
            if literal in seen_literals:
                continue
            seen_literals.add(literal)
            left_id = str(left["token_id"])
            right_id = str(right["token_id"])
            records.append(
                {
                    "token_id": f"{left_id}+{right_id}",
                    "raw_vocab": f"{left['raw_vocab']}|{right['raw_vocab']}",
                    "candidate": literal,
                    "category": "COMPOSED",
                    "character_length": len(literal),
                    "candidate_kind": "ordered_token_pair",
                    "component_count": 2,
                    "component_token_ids": f"{left_id},{right_id}",
                }
            )
    return pd.DataFrame.from_records(records)


def _sample_across_range(
    indices: Sequence[int],
    similarities: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> list[int]:
    """Choose one seeded observation from each equal-rank similarity bin."""

    if count < 1:
        return []
    ordered = np.asarray(indices, dtype=int)[
        np.argsort(similarities[np.asarray(indices, dtype=int)], kind="mergesort")
    ]
    if len(ordered) < count:
        raise ValueError(f"Requested {count} rows from a pool containing {len(ordered)}")
    bins = np.array_split(ordered, count)
    return [int(group[rng.integers(0, len(group))]) for group in bins]


def make_disjoint_splits(
    similarities: Sequence[float],
    *,
    low_threshold: float,
    high_threshold: float,
    search_per_group: int,
    validation_per_group: int,
    plot_pair_count: int,
    seed: int,
) -> dict[str, list[int]]:
    """Create disjoint search, validation and full-range plotting splits."""

    if low_threshold >= high_threshold:
        raise ValueError("low_threshold must be smaller than high_threshold")
    values = np.asarray(similarities, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("similarities must be a finite one-dimensional array")

    rng = np.random.default_rng(seed)
    low_pool = np.flatnonzero(values <= low_threshold).tolist()
    high_pool = np.flatnonzero(values >= high_threshold).tolist()

    search_low = _sample_across_range(low_pool, values, search_per_group, rng)
    search_high = _sample_across_range(high_pool, values, search_per_group, rng)
    used = set(search_low + search_high)

    remaining_low = [index for index in low_pool if index not in used]
    remaining_high = [index for index in high_pool if index not in used]
    validation_low = _sample_across_range(
        remaining_low, values, validation_per_group, rng
    )
    validation_high = _sample_across_range(
        remaining_high, values, validation_per_group, rng
    )
    used.update(validation_low + validation_high)

    remaining = [index for index in range(len(values)) if index not in used]
    plot = _sample_across_range(remaining, values, plot_pair_count, rng)

    splits = {
        "search_low": search_low,
        "search_high": search_high,
        "validation_low": validation_low,
        "validation_high": validation_high,
        "plot": plot,
    }
    flattened = [index for group in splits.values() for index in group]
    if len(flattened) != len(set(flattened)):
        raise AssertionError("Sticky-high splits are not disjoint")
    return splits


def append_candidate(text: str, candidate: str, count: int, separator: str = "") -> str:
    """Append a literal candidate using deployment-equivalent string semantics."""

    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return text
    if separator:
        return text + separator + separator.join([candidate] * count)
    return text + candidate * count


def encode_normalized(model, texts: Sequence[str], batch_size: int, show_progress: bool):
    return model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )


def baseline_embeddings(
    model,
    frame: pd.DataFrame,
    *,
    batch_size: int,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sentence1_embeddings = encode_normalized(
        model,
        frame["sentence1"].astype(str).tolist(),
        batch_size,
        show_progress,
    )
    sentence2_embeddings = encode_normalized(
        model,
        frame["sentence2"].astype(str).tolist(),
        batch_size,
        show_progress,
    )
    similarities = np.einsum(
        "pd,pd->p", sentence1_embeddings, sentence2_embeddings, optimize=True
    )
    return sentence1_embeddings, sentence2_embeddings, similarities


def evaluate_candidate_batch(
    model,
    candidates: Sequence[str],
    sentence2: Sequence[str],
    reference_embeddings: np.ndarray,
    insertion_counts: Sequence[int],
    *,
    separator: str,
    batch_size: int,
    show_progress: bool = False,
) -> np.ndarray:
    """Return cosine similarities shaped [candidate, count, pair]."""

    candidate_list = [str(candidate) for candidate in candidates]
    pair_texts = [str(text) for text in sentence2]
    counts = [int(count) for count in insertion_counts]
    modified = [
        append_candidate(text, candidate, count, separator)
        for candidate in candidate_list
        for count in counts
        for text in pair_texts
    ]
    embeddings = encode_normalized(model, modified, batch_size, show_progress)
    embeddings = embeddings.reshape(
        len(candidate_list), len(counts), len(pair_texts), -1
    )
    return np.einsum(
        "pd,ckpd->ckp", reference_embeddings, embeddings, optimize=True
    )


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability))


def summarize_candidate(
    similarities: np.ndarray,
    baseline: np.ndarray,
    low_mask: np.ndarray,
    high_mask: np.ndarray,
    insertion_counts: Sequence[int],
    thresholds: StickyHighThresholds,
) -> dict[str, float | bool]:
    """Summarize one candidate's [count, pair] similarity matrix."""

    thresholds.validate()
    values = np.asarray(similarities, dtype=float)
    base = np.asarray(baseline, dtype=float)
    low = np.asarray(low_mask, dtype=bool)
    high = np.asarray(high_mask, dtype=bool)
    counts = list(insertion_counts)
    if values.shape != (len(counts), len(base)):
        raise ValueError("similarities shape does not match counts and baseline")
    if not low.any() or not high.any() or np.any(low & high):
        raise ValueError("low and high masks must be non-empty and disjoint")

    final_delta = values[-1] - base
    low_delta = final_delta[low]
    high_delta = final_delta[high]
    high_excess_drop = np.maximum(
        0.0, -high_delta - thresholds.high_drop_tolerance
    )

    full_curve = len(counts) > 1 and counts == list(range(1, counts[-1] + 1))
    if full_curve:
        steps = np.diff(np.vstack([base, values]), axis=0)
        low_step_failure = float(
            np.mean(steps[:, low] < -thresholds.step_drop_tolerance)
        )
        high_step_failure = float(
            np.mean(steps[:, high] < -thresholds.step_drop_tolerance)
        )
    else:
        low_step_failure = math.nan
        high_step_failure = math.nan

    low_gain_q10 = _quantile(low_delta, 0.10)
    high_gain_q05 = _quantile(high_delta, 0.05)
    high_failure_rate = float(
        np.mean(high_delta < -thresholds.high_drop_tolerance)
    )
    preservation_penalty = float(np.mean(high_excess_drop))
    step_penalty = (
        0.0
        if not full_curve
        else thresholds.step_failure_weight
        * (low_step_failure + high_step_failure)
    )
    objective = (
        low_gain_q10
        - thresholds.high_penalty_weight * preservation_penalty
        - thresholds.high_failure_weight * high_failure_rate
        - step_penalty
    )

    final_constraints = (
        low_gain_q10 >= thresholds.min_low_gain
        and high_gain_q05 >= -thresholds.high_drop_tolerance
        and high_failure_rate <= thresholds.max_high_failure_rate
    )
    curve_constraints = (
        not full_curve
        or (
            low_step_failure <= thresholds.max_step_failure_rate
            and high_step_failure <= thresholds.max_step_failure_rate
        )
    )
    return {
        "objective": float(objective),
        "low_gain_mean": float(np.mean(low_delta)),
        "low_gain_median": float(np.median(low_delta)),
        "low_gain_q10": low_gain_q10,
        "low_success_rate": float(np.mean(low_delta >= thresholds.min_low_gain)),
        "high_gain_mean": float(np.mean(high_delta)),
        "high_gain_median": float(np.median(high_delta)),
        "high_gain_q05": high_gain_q05,
        "high_gain_min": float(np.min(high_delta)),
        "high_failure_rate": high_failure_rate,
        "high_preservation_penalty": preservation_penalty,
        "low_step_failure_rate": low_step_failure,
        "high_step_failure_rate": high_step_failure,
        "final_constraints_pass": bool(final_constraints),
        "curve_constraints_pass": bool(curve_constraints),
        "certified": bool(final_constraints and curve_constraints and full_curve),
    }


def score_candidate_frame(
    candidate_frame: pd.DataFrame,
    similarities: np.ndarray,
    baseline: np.ndarray,
    low_mask: np.ndarray,
    high_mask: np.ndarray,
    insertion_counts: Sequence[int],
    thresholds: StickyHighThresholds,
) -> pd.DataFrame:
    if similarities.shape[0] != len(candidate_frame):
        raise ValueError("Candidate count and similarity tensor disagree")
    records = []
    for offset, row in candidate_frame.reset_index(drop=True).iterrows():
        record = row.to_dict()
        record.update(
            summarize_candidate(
                similarities[offset],
                baseline,
                low_mask,
                high_mask,
                insertion_counts,
                thresholds,
            )
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def rank_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank certified/constraint-passing candidates before unconstrained ones."""

    ranked = frame.copy()
    for column in ("certified", "final_constraints_pass"):
        if column not in ranked:
            ranked[column] = False
    return ranked.sort_values(
        ["certified", "final_constraints_pass", "objective", "low_gain_q10"],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def select_diverse_candidates(
    frame: pd.DataFrame,
    size: int,
    thresholds: StickyHighThresholds,
) -> pd.DataFrame:
    """Select a multi-objective, candidate-kind-balanced shortlist.

    A pure top-objective cutoff can be saturated by one candidate family and by
    search-split overfitting.  Each family receives an equal quota containing:
    objective leaders, high-tail-preserving candidates with non-trivial low-tail
    gain, and strong low-tail candidates that remain near the high-tail bound.
    Remaining slots are filled by the global objective ranking.
    """

    if size < 1:
        raise ValueError("Shortlist size must be positive")
    if frame.empty:
        raise ValueError("Cannot shortlist an empty candidate frame")
    thresholds.validate()
    source = frame.copy().reset_index(drop=True)
    if "candidate_kind" not in source:
        source["candidate_kind"] = "unspecified"
    kinds = sorted(source["candidate_kind"].astype(str).unique())

    selected_indices: list[int] = []
    selected_set: set[int] = set()

    def add(rows: pd.DataFrame, count: int) -> None:
        if count <= 0:
            return
        added = 0
        for index in rows.index:
            numeric_index = int(index)
            if numeric_index in selected_set:
                continue
            selected_set.add(numeric_index)
            selected_indices.append(numeric_index)
            added += 1
            if added >= count or len(selected_indices) >= size:
                break

    base_quota, remainder = divmod(size, len(kinds))
    for kind_offset, kind in enumerate(kinds):
        quota = base_quota + (1 if kind_offset < remainder else 0)
        group = source[source["candidate_kind"].astype(str) == kind]
        objective_count = max(1, quota // 2)
        preservation_count = max(1, quota // 4) if quota >= 4 else 0
        uplift_count = quota - objective_count - preservation_count

        add(
            group.sort_values(
                ["final_constraints_pass", "objective", "low_gain_q10"],
                ascending=[False, False, False],
                kind="mergesort",
            ),
            objective_count,
        )
        preservation_pool = group[
            group["low_gain_q10"] >= thresholds.min_low_gain / 2
        ].sort_values(
            ["high_gain_q05", "high_failure_rate", "low_gain_q10"],
            ascending=[False, True, False],
            kind="mergesort",
        )
        add(preservation_pool, preservation_count)
        uplift_pool = group[
            group["high_gain_q05"] >= -2 * thresholds.high_drop_tolerance
        ].sort_values(
            ["low_gain_q10", "high_gain_q05", "objective"],
            ascending=[False, False, False],
            kind="mergesort",
        )
        add(uplift_pool, uplift_count)

        group_index_set = set(int(index) for index in group.index)
        current_group_count = sum(
            index in group_index_set for index in selected_indices
        )
        if current_group_count < quota:
            add(
                group.sort_values(
                    ["final_constraints_pass", "objective", "low_gain_q10"],
                    ascending=[False, False, False],
                    kind="mergesort",
                ),
                quota - current_group_count,
            )

    if len(selected_indices) < min(size, len(source)):
        add(
            source.sort_values(
                ["final_constraints_pass", "objective", "low_gain_q10"],
                ascending=[False, False, False],
                kind="mergesort",
            ),
            size - len(selected_indices),
        )
    selected = source.loc[selected_indices[:size]].copy()
    return rank_candidates(selected)


def plot_sticky_high_curves(
    curves: np.ndarray,
    output_path: Path,
    *,
    max_insertions: int,
    dpi: int,
) -> None:
    """Create a Figure 2(b)-style line plot plus final-value boxplot."""

    # Keep plotting optional so pure scoring and split tests can run in
    # lightweight environments without Matplotlib/Seaborn installed.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update(
        {
            "font.size": 14,
            "font.weight": "normal",
            "axes.labelweight": "normal",
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
            alpha=0.75,
            linewidth=1.2,
        )

    line_axis.set_xlabel("Inserted number of sticky_high token", fontsize=18)
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
        axis="both", which="major", labelleft=False, labelbottom=False
    )
    box_axis.grid(True, axis="y", linestyle="--", alpha=0.6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(figure)


def thresholds_as_dict(thresholds: StickyHighThresholds) -> dict[str, float]:
    return {key: float(value) for key, value in asdict(thresholds).items()}
