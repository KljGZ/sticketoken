"""Validation certification and fixed-center holdout evaluation for V4."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .interfaces import Candidate
from .metrics import (
    certify_validation,
    evaluate_geometry,
    fixed_pair_indices,
    fixed_region_coverage,
    grouped_bootstrap_geometry,
    pairwise_distance_matrix,
    point_conditions,
)
from .occupancy import evaluate_occupancy
from .scoring import CandidateScorer
from .support import SupportModel
from .tokenizer_audit import HuggingFaceTokenizerAudit, audit_to_dict, context_realizability


def certify_candidates(
    scorer: CandidateScorer,
    candidates: Sequence[Candidate],
    random_baselines: Sequence[Candidate],
    group_ids: Sequence[str],
    tokenizer: HuggingFaceTokenizerAudit,
    *,
    constraints: Mapping[str, float],
    bootstrap_replicates: int,
    pair_sample_count: int,
    confidence: float,
    random_quantile: float,
    context_count: int,
    context_required: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[list[np.ndarray]], dict[str, Any]]:
    baseline_records = scorer.evaluate(random_baselines)
    baseline_scores = np.asarray([record["search_score"] for record in baseline_records], dtype=float)
    threshold = float(np.quantile(baseline_scores, random_quantile))
    evaluated = scorer.evaluate(candidates, include_center=True)
    benign_pairwise_distances = pairwise_distance_matrix(scorer.original_embeddings)
    output: list[dict[str, Any]] = []
    centers: list[np.ndarray] = []
    triggered_blocks: list[list[np.ndarray]] = []
    for index, (candidate, record) in enumerate(zip(candidates, evaluated)):
        center = np.asarray(record.pop("center"), dtype=np.float32)
        triggered = list(record.pop("triggered_by_position"))
        uncertainty = grouped_bootstrap_geometry(
            scorer.original_embeddings,
            triggered,
            group_ids,
            replicates=bootstrap_replicates,
            confidence=confidence,
            pair_count=pair_sample_count,
            seed=seed + index * 100003,
            benign_pairwise_distances=benign_pairwise_distances,
        )
        audit = context_realizability(
            tokenizer,
            candidate,
            scorer.texts[:context_count],
            positions=scorer.positions,
            seed=seed + index * 200003,
            separator=scorer.separator,
        )
        realizable = bool(
            audit.exact_token_roundtrip
            and audit.actual_token_length == candidate.actual_token_length
            and not audit.special_token
            and audit.context_realizability >= context_required
        )
        occupancy = evaluate_occupancy(
            center,
            float(record["compact_radius_q95"]),
            scorer.normal_probe,
            scorer.support,
            scorer.occupancy_lambdas,
            confidence=confidence,
        )
        baseline_exceeded = bool(float(record["search_score"]) > threshold)
        certificate = certify_validation(
            # The scorer record and fresh evaluation describe the same point.
            evaluate_geometry(
                scorer.original_embeddings,
                triggered,
                pair_indices=scorer.pairs,
                fixed_center=center,
            ),
            uncertainty,
            float(record["support_in_margin"]),
            occupancy,
            constraints,
            realizable=realizable,
            baseline_exceeded=baseline_exceeded,
        )
        output.append(
            {
                **record,
                **uncertainty,
                **audit_to_dict(audit),
                **certificate,
                "same_length_random_count": len(baseline_records),
                "same_length_random_quantile": float(random_quantile),
                "same_length_random_score_threshold": threshold,
                "same_length_random_score_exceeded": baseline_exceeded,
                "validation_center_refit": True,
                "validation_radius_frozen_value": float(record["compact_radius_q95"]),
            }
        )
        centers.append(center)
        triggered_blocks.append(triggered)
    baseline_summary = {
        "count": len(baseline_records),
        "score_quantile": float(random_quantile),
        "score_threshold": threshold,
        "score_min": float(np.min(baseline_scores)),
        "score_median": float(np.median(baseline_scores)),
        "score_max": float(np.max(baseline_scores)),
    }
    return output, centers, triggered_blocks, baseline_summary


def evaluate_frozen_region(
    original: np.ndarray,
    triggered_by_position: Sequence[np.ndarray],
    normal_probe: np.ndarray,
    support: SupportModel,
    group_ids: Sequence[str],
    center: np.ndarray,
    radius: float,
    *,
    constraints: Mapping[str, float],
    occupancy_lambdas: Sequence[float],
    confidence: float,
    bootstrap_replicates: int,
    pair_sample_count: int,
    seed: int,
    require_low_occupancy: bool,
) -> dict[str, Any]:
    pairs = fixed_pair_indices(len(original), pair_sample_count, seed + 1)
    geometry = evaluate_geometry(original, triggered_by_position, pair_indices=pairs, fixed_center=center)
    uncertainty = grouped_bootstrap_geometry(
        original,
        triggered_by_position,
        group_ids,
        replicates=bootstrap_replicates,
        confidence=confidence,
        pair_count=pair_sample_count,
        seed=seed + 2,
        fixed_center=center,
    )
    occupancy = evaluate_occupancy(
        center,
        radius,
        normal_probe,
        support,
        occupancy_lambdas,
        confidence=confidence,
    )
    margin = support.support_in_margin(center)
    conditions = point_conditions(geometry, margin, occupancy, constraints)
    coverage = fixed_region_coverage(triggered_by_position, center, radius, confidence=confidence)
    coverage_ok = bool(
        coverage["fixed_region_coverage_lcb"] >= float(constraints["min_fixed_region_coverage_lcb"])
        and all(
            row["lcb"] >= float(constraints["min_fixed_region_coverage_lcb"])
            for row in coverage["fixed_region_per_position"]
        )
    )
    shift = float(uncertainty["displacement_q05_ci_lower"]) >= float(constraints["min_displacement_q05"])
    contract = float(uncertainty["contraction_q95_ci_upper"]) <= float(constraints["max_contraction_q95"])
    low_occupancy = bool(conditions["low_occupancy_point"] or not require_low_occupancy)
    certified = bool(coverage_ok and shift and contract and conditions["support_in_point"] and low_occupancy)
    return {
        **geometry.to_dict(),
        **uncertainty,
        **occupancy.to_dict(),
        **coverage,
        "frozen_center_support_in_margin": margin,
        "fixed_coverage_certified": coverage_ok,
        "fixed_shift_certified": shift,
        "fixed_contract_certified": contract,
        "fixed_support_in_certified": conditions["support_in_point"],
        "fixed_low_occupancy_certified": low_occupancy,
        "low_occupancy_required": bool(require_low_occupancy),
        "fixed_region_certified": certified,
        "center_refit": False,
        "radius_refit": False,
    }
