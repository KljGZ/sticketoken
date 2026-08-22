"""V7 FULL candidate evaluation using prefix/suffix only."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from sticky_lab.mode3_v6_3.errors import CacheCorruption, ShapeMismatch

from .config import canonical_sha256
from .encoding import CachedEncoder, EncodingRequest
from .operating_point import build_candidate_frontier


POSITIONS = ("prefix", "suffix")


def triggered_requests(
    records: Sequence[Mapping[str, Any]], role: str
) -> tuple[list[dict[str, Any]], list[EncodingRequest]]:
    rows: list[dict[str, Any]] = []
    requests: list[EncodingRequest] = []
    for row in records:
        for position in POSITIONS:
            rows.append({**dict(row), "position": position})
            requests.append(EncodingRequest(str(role), row, position, 0))
    return rows, requests


def clean_requests(
    records: Sequence[Mapping[str, Any]], role: str
) -> list[EncodingRequest]:
    return [EncodingRequest(str(role), row, "clean", 0) for row in records]


def cached_clean_matrix(
    encoder: CachedEncoder,
    records: Sequence[Mapping[str, Any]],
    role: str,
) -> np.ndarray:
    memo = getattr(encoder, "_v7_clean_matrix_memo", None)
    if memo is None:
        memo = {}
        setattr(encoder, "_v7_clean_matrix_memo", memo)
    key = (str(role), tuple(str(row["text_id"]) for row in records))
    if key in memo:
        return memo[key]
    entries = [
        encoder.call_space.lookup_request(str(role), str(row["text_id"]), "clean")
        for row in records
    ]
    found, missing = encoder.cache.fetch(-2, [entry.ordinal for entry in entries])
    if missing:
        raise CacheCorruption(f"V7 clean role {role} is missing {len(missing)} calls")
    matrix = np.stack([found[entry.ordinal] for entry in entries]).astype(np.float32)
    memo[key] = matrix
    return matrix


def precompute_discovery_clean(
    encoder: CachedEncoder,
    *,
    calibration_records: Sequence[Mapping[str, Any]],
    select_records: Sequence[Mapping[str, Any]],
    axis_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    requests = (
        clean_requests(calibration_records, "calibration")
        + clean_requests(select_records, "select")
        + clean_requests(axis_records, "axis_fit_benign")
    )
    _, _, audit = encoder.encode_requests(
        token_id=-2,
        token_text="",
        requests=requests,
        phase="v7:discovery_clean_precompute",
    )
    return audit


def fit_and_score_candidate(
    encoder: CachedEncoder,
    *,
    token_id: int,
    token_text: str,
    stage: str,
    fit_records: Sequence[Mapping[str, Any]],
    calibration_records: Sequence[Mapping[str, Any]],
    select_records: Sequence[Mapping[str, Any]],
    axis_records: Sequence[Mapping[str, Any]],
    role_hashes: Mapping[str, str],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    fit_rows, fit_requests = triggered_requests(fit_records, "fit")
    select_rows, select_requests = triggered_requests(select_records, "select")
    fit_vectors, fit_audits, fit_cache = encoder.encode_requests(
        token_id=int(token_id),
        token_text=str(token_text),
        requests=fit_requests,
        phase=f"v7:{stage}:fit",
    )
    select_vectors, select_audits, select_cache = encoder.encode_requests(
        token_id=int(token_id),
        token_text=str(token_text),
        requests=select_requests,
        phase=f"v7:{stage}:select",
    )
    calibration_vectors = cached_clean_matrix(encoder, calibration_records, "calibration")
    select_clean = cached_clean_matrix(encoder, select_records, "select")
    axis_vectors = cached_clean_matrix(encoder, axis_records, "axis_fit_benign")
    if len(select_rows) != 2 * len(select_clean):
        raise ShapeMismatch("V7 select expansion does not match paired clean data")
    paired_clean = np.repeat(select_clean, 2, axis=0)
    # triggered_requests emits prefix then suffix for each text, matching repeat.
    e_star = np.mean(axis_vectors.astype(np.float64), axis=0)
    e_star /= np.linalg.norm(e_star)
    frontier = build_candidate_frontier(
        token_id=int(token_id),
        token_text=str(token_text),
        fit_rows=fit_rows,
        fit_vectors=fit_vectors,
        calibration_rows=calibration_records,
        calibration_vectors=calibration_vectors,
        select_rows=select_rows,
        triggered_select_vectors=select_vectors,
        paired_clean_vectors=paired_clean,
        e_star=e_star,
        role_hashes=role_hashes,
        config=config,
        stage=str(stage),
    )
    realizations = [audit.to_dict() for audit in fit_audits + select_audits]
    position_counts = {
        position: sum(row["position"] == position for row in realizations)
        for position in POSITIONS
    }
    expected_position_count = len(fit_records) + len(select_records)
    if any(count != expected_position_count for count in position_counts.values()):
        raise ShapeMismatch(
            f"V7 runtime realization audit is incomplete: {position_counts}"
        )
    audit = {
        "schema_version": "mode3-v7-stage-candidate-audit-v1",
        "token_id": int(token_id),
        "stage": str(stage),
        "positions": list(POSITIONS),
        "random_position_used": False,
        "center_refit_per_beta": False,
        "confirm_data_used": False,
        "cache": {"fit": fit_cache, "select": select_cache},
        "runtime_realization_audit_count": len(fit_audits) + len(select_audits),
        "runtime_realization_audit_sha256": canonical_sha256(realizations),
        "runtime_realization_position_counts": position_counts,
        "runtime_realization_all_exact_single_token": True,
    }
    return frontier, audit
