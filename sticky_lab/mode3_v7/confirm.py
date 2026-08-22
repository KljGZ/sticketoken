"""Pure no-refit confirmation of one frozen V7 token-beta operating point."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from sticky_lab.mode3_v6_3.errors import ManifestMismatch, ShapeMismatch

from .freeze import FreezeArtifact
from .migration import migration_diagnostics
from .operating_point import evidence_grade
from .statistics import (
    group_benign_by_source,
    group_membership,
    source_balanced_occupancy,
    source_position_coverage,
)


def _verify_role_hashes(
    artifact: FreezeArtifact, observed_role_hashes: Mapping[str, str]
) -> None:
    for role, digest in observed_role_hashes.items():
        expected = artifact.confirm_role_hashes.get(str(role))
        if expected is None or expected != str(digest):
            raise ManifestMismatch(f"V7 confirmation role hash mismatch for {role}")
    if set(observed_role_hashes.values()).intersection(
        artifact.discovery_role_hashes.values()
    ):
        raise ManifestMismatch("V7 confirmation attempted a discovery role")


def _position_rows(
    rows: Sequence[Mapping[str, Any]], position: str
) -> list[dict[str, Any]]:
    return [{**dict(row), "position": str(position)} for row in rows]


def confirm_frozen_operating_point(
    artifact: FreezeArtifact,
    *,
    prefix_rows: Sequence[Mapping[str, Any]],
    prefix_triggered_vectors: np.ndarray,
    prefix_clean_vectors: np.ndarray,
    suffix_rows: Sequence[Mapping[str, Any]],
    suffix_triggered_vectors: np.ndarray,
    suffix_clean_vectors: np.ndarray,
    benign_rows: Sequence[Mapping[str, Any]],
    benign_vectors: np.ndarray,
    observed_role_hashes: Mapping[str, str],
    freeze_sha256: str,
    familywise_alpha: float = 0.05,
) -> dict[str, Any]:
    _verify_role_hashes(artifact, observed_role_hashes)
    model = artifact.frozen_model()
    prefix_triggered = np.asarray(prefix_triggered_vectors)
    prefix_clean = np.asarray(prefix_clean_vectors)
    suffix_triggered = np.asarray(suffix_triggered_vectors)
    suffix_clean = np.asarray(suffix_clean_vectors)
    benign = np.asarray(benign_vectors)
    if len(prefix_rows) != len(prefix_triggered) or prefix_triggered.shape != prefix_clean.shape:
        raise ShapeMismatch("prefix confirmation vectors differ")
    if len(suffix_rows) != len(suffix_triggered) or suffix_triggered.shape != suffix_clean.shape:
        raise ShapeMismatch("suffix confirmation vectors differ")
    if len(benign_rows) != len(benign):
        raise ShapeMismatch("benign confirmation vectors differ")
    prefix_ids = [str(row["text_id"]) for row in prefix_rows]
    suffix_ids = [str(row["text_id"]) for row in suffix_rows]
    all_ids = prefix_ids + suffix_ids
    if len(all_ids) != len(set(all_ids)):
        raise ShapeMismatch("formal prefix/suffix confirmation must use disjoint text units")

    rows = _position_rows(prefix_rows, "prefix") + _position_rows(suffix_rows, "suffix")
    triggered = np.concatenate([prefix_triggered, suffix_triggered], axis=0)
    clean = np.concatenate([prefix_clean, suffix_clean], axis=0)
    triggered_inside = model.contains(triggered)
    clean_inside = model.contains(clean)
    trigger_membership = group_membership(rows, triggered_inside, require_unique_texts=True)
    clean_membership = group_membership(rows, clean_inside, require_unique_texts=True)
    coverage = source_position_coverage(
        trigger_membership, familywise_alpha=float(familywise_alpha)
    )
    migration = migration_diagnostics(
        clean_membership,
        trigger_membership,
        familywise_alpha=float(familywise_alpha),
    )
    benign_inside = model.contains(benign)
    occupancy = source_balanced_occupancy(
        group_benign_by_source(benign_rows, benign_inside),
        familywise_alpha=float(familywise_alpha),
    )
    thresholds = artifact.certification_thresholds
    gates = {
        "occupancy": float(occupancy["ucb"]) <= float(model.beta) + 1e-15,
        "prefix_coverage": float(coverage["prefix_lcb"])
        >= float(thresholds["prefix_coverage_lcb"]),
        "suffix_coverage": float(coverage["suffix_lcb"])
        >= float(thresholds["suffix_coverage_lcb"]),
        "maximum_radius": model.radius_degrees
        <= float(thresholds["maximum_radius_degrees"]),
        "token_length": int(thresholds.get("actual_trigger_token_length", 1)) == 1,
        "one_insertion_only": bool(thresholds.get("one_insertion_only", True)),
        "center_frozen": True,
        "radius_frozen": True,
        "confirm_refit_absent": True,
        "role_isolation": True,
    }
    certified = all(gates.values())
    grade = evidence_grade(model.beta) if certified else "NOT_CERTIFIED_PS_80"
    status = "CERTIFIED_V7_OCFCA_80" if certified else "VALID_PRIMARY_NOT_CERTIFIED"
    observations = [
        {
            "text_id": str(row["text_id"]),
            "source_id": str(row["source_id"]),
            "position": str(row["position"]),
            "triggered_distance": float(triggered_distance),
            "clean_distance": float(clean_distance),
            "triggered_inside": bool(triggered_member),
            "clean_inside": bool(clean_member),
        }
        for row, triggered_distance, clean_distance, triggered_member, clean_member in zip(
            rows,
            model.distances(triggered),
            model.distances(clean),
            triggered_inside,
            clean_inside,
        )
    ]
    return {
        "schema_version": "mode3-v7-confirmation-v1",
        "status": status,
        "evidence_grade": grade,
        "freeze_sha256": str(freeze_sha256),
        "token_id": artifact.token_id,
        "token_text": str(model.token_text),
        "beta_frozen": float(model.beta),
        "radius_radians": float(model.radius),
        "radius_degrees": model.radius_degrees,
        "center_sha256": model.center_sha256,
        "refit_performed": False,
        "fixed_center": True,
        "fixed_radius": True,
        "independent_prefix_texts": len(prefix_rows),
        "independent_suffix_texts": len(suffix_rows),
        "independent_benign_texts": len(benign_rows),
        "coverage": coverage,
        "occupancy": occupancy,
        "migration": migration,
        "gates": gates,
        "certified": bool(certified),
        "observations": observations,
        "benign_observations": [
            {
                "text_id": str(row["text_id"]),
                "source_id": str(row["source_id"]),
                "distance": float(distance),
                "inside": bool(member),
            }
            for row, distance, member in zip(
                benign_rows, model.distances(benign), benign_inside
            )
        ],
        "primary_tests": [
            "source-balanced benign occupancy UCB <= frozen beta",
            "prefix coverage LCB >= 0.80",
            "suffix coverage LCB >= 0.80",
        ],
        "secondary_beta_curve_confirmed": False,
    }


def paired_position_audit(
    artifact: FreezeArtifact,
    rows: Sequence[Mapping[str, Any]],
    prefix_vectors: np.ndarray,
    suffix_vectors: np.ndarray,
) -> dict[str, Any]:
    model = artifact.frozen_model()
    prefix = model.contains(prefix_vectors)
    suffix = model.contains(suffix_vectors)
    if len(rows) != len(prefix) or prefix.shape != suffix.shape:
        raise ShapeMismatch("paired prefix/suffix audit alignment mismatch")
    return {
        "schema_version": "mode3-v7-paired-position-audit-v1",
        "texts": len(rows),
        "iid_units_added_to_primary_confirm": 0,
        "prefix_coverage": float(np.mean(prefix)),
        "suffix_coverage": float(np.mean(suffix)),
        "both_inside": float(np.mean(prefix & suffix)),
        "either_inside": float(np.mean(prefix | suffix)),
        "position_agreement": float(np.mean(prefix == suffix)),
        "random_position_used": False,
    }
