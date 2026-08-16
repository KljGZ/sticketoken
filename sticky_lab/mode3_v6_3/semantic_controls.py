"""Post-Core black-box semantic matched-control evaluation."""

from __future__ import annotations

import math
import unicodedata
from typing import Any, Mapping, Sequence

import numpy as np

from .errors import ProtocolViolation


def _token_shape(text: str) -> dict[str, Any]:
    stripped = str(text).strip()
    categories = [unicodedata.category(character) for character in stripped]
    script = "latin" if stripped and all(ord(character) < 128 for character in stripped) else "non_latin"
    if stripped.isalpha():
        lexical = "alphabetic"
    elif stripped.isdigit():
        lexical = "numeric"
    elif stripped.isalnum():
        lexical = "alphanumeric"
    else:
        lexical = "symbolic"
    casing = (
        "upper" if stripped.isupper() else
        "lower" if stripped.islower() else
        "title" if stripped.istitle() else "mixed_or_none"
    )
    return {
        "character_length": len(stripped),
        "casing": casing,
        "leading_whitespace": bool(str(text)[:1].isspace()),
        "unicode_script": script,
        "lexical_category": lexical,
        "unicode_categories": tuple(sorted(set(categories))),
    }


def token_frequency_statistics(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    legal_token_ids: Sequence[int],
) -> dict[int, dict[str, float]]:
    """Black-box tokenizer frequency/IDF statistics on discovery texts."""
    legal = set(map(int, legal_token_ids))
    term = {token_id: 0 for token_id in legal}
    document = {token_id: 0 for token_id in legal}
    for row in rows:
        ids = list(map(int, tokenizer.encode(str(row["text"]), add_special_tokens=False)))
        observed = set(ids).intersection(legal)
        for token_id in ids:
            if token_id in legal:
                term[token_id] += 1
        for token_id in observed:
            document[token_id] += 1
    count = max(1, len(rows))
    return {
        token_id: {
            "frequency": float(term[token_id]),
            "document_frequency": float(document[token_id]),
            "idf": float(math.log((count + 1) / (document[token_id] + 1)) + 1.0),
        }
        for token_id in sorted(legal)
    }


def select_matched_controls(
    *,
    candidate_id: int,
    candidate_text: str,
    legal_tokens: Sequence[Mapping[str, Any]],
    frequency: Mapping[int, Mapping[str, float]],
    count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministically match observable lexical and corpus-frequency features."""
    target_shape = _token_shape(candidate_text)
    target_frequency = frequency.get(int(candidate_id), {"frequency": 0.0, "idf": 0.0})

    def score(row: Mapping[str, Any]) -> tuple[Any, ...]:
        token_id = int(row["token_id"])
        shape = _token_shape(str(row["token_text"]))
        stats = frequency.get(token_id, {"frequency": 0.0, "idf": 0.0})
        categorical = sum(
            shape[key] != target_shape[key]
            for key in ("casing", "leading_whitespace", "unicode_script", "lexical_category")
        )
        return (
            categorical,
            abs(int(shape["character_length"]) - int(target_shape["character_length"])),
            abs(math.log1p(float(stats["frequency"])) - math.log1p(float(target_frequency["frequency"]))),
            abs(float(stats["idf"]) - float(target_frequency["idf"])),
            token_id,
        )

    eligible = [dict(row) for row in legal_tokens if int(row["token_id"]) != int(candidate_id)]
    selected = sorted(eligible, key=score)[: int(count)]
    if len(selected) != int(count):
        raise ProtocolViolation(f"matched-control capacity {len(selected)}/{count}")
    audit = {
        "schema_version": "mode3-v6-3-semantic-matching-v1",
        "candidate": {
            "token_id": int(candidate_id), "token_text": str(candidate_text),
            "shape": target_shape, "frequency": dict(target_frequency),
        },
        "matching_dimensions": [
            "frequency", "idf", "character_length", "casing",
            "tokenizer_whitespace_pattern", "unicode_language_class",
            "lexical_category",
        ],
        "whitebox_used": False,
        "controls": [
            {
                "token_id": int(row["token_id"]), "token_text": str(row["token_text"]),
                "shape": _token_shape(str(row["token_text"])),
                "frequency": dict(frequency.get(int(row["token_id"]), {})),
                "match_order": list(score(row)),
            }
            for row in selected
        ],
    }
    return selected, audit


def require_core(confirmation: Mapping[str, Any]) -> None:
    if not bool(confirmation.get("levels", {}).get("B_ST_FCA_CORE", False)):
        raise ProtocolViolation("semantic controls are gated on independent ST-FCA-Core")


def evaluate_semantic_controls(
    confirmation: Mapping[str, Any],
    *,
    candidate_coverage: float,
    matched_control_coverages: Sequence[float],
    wrapper_coverages: Mapping[str, float],
    minimum_margin: float = 0.10,
    minimum_wrapper_coverage: float = 0.80,
) -> dict[str, Any]:
    require_core(confirmation)
    controls = np.asarray(matched_control_coverages, dtype=np.float64)
    if len(controls) < 20 or not np.all(np.isfinite(controls)):
        raise ProtocolViolation("semantic control set is too small or non-finite")
    if not wrapper_coverages or any(not math.isfinite(float(value)) for value in wrapper_coverages.values()):
        raise ProtocolViolation("wrapper counterfactuals are incomplete")
    q95 = float(np.quantile(controls, 0.95))
    margin = float(candidate_coverage) - q95
    mean = float(np.mean(controls))
    standard_deviation = float(np.std(controls, ddof=1)) if len(controls) > 1 else 0.0
    residual = float(candidate_coverage) - mean
    supported = margin >= float(minimum_margin) and min(map(float, wrapper_coverages.values())) >= float(minimum_wrapper_coverage)
    return {
        "schema_version": "mode3-v6-3-semantic-controls-v1",
        "search_feedback": False, "whitebox_used": False,
        "candidate_coverage": float(candidate_coverage),
        "matched_controls": len(controls), "control_coverage_q95": q95,
        "coverage_margin": margin, "wrapper_coverages": dict(wrapper_coverages),
        "additive_semantic_model": {
            "kind": "matched-control intercept model",
            "expected_coverage": mean,
            "control_standard_deviation": standard_deviation,
            "candidate_residual": residual,
            "candidate_residual_z": residual / standard_deviation if standard_deviation > 0 else None,
        },
        "anomaly_supported": bool(supported),
    }
