"""Content-addressed freeze artifacts for V6.2."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sticky_lab.mode3_v6.atomic_io import write_json

from .errors import ManifestMismatch, ProtocolViolation
from .geometry import FrozenCapModel
from .roles import canonical_sha256


@dataclass(frozen=True)
class FreezeArtifact:
    schema_version: str
    token_id: int
    token_text: str
    tokenizer_hash: str
    model_hash: str
    code_commit: str
    data_role_hashes: dict[str, str]
    cap: FrozenCapModel
    radius_design_quantile: float
    assignment_rule: str
    outlier_rule: str
    position_manifest_hash: str
    random_boundary_manifest_hash: str
    source_weights: dict[str, float]
    selection_metrics: dict[str, Any]
    certification_thresholds: dict[str, float]
    freeze_sha256: str

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "token_id": self.token_id,
            "token_text": self.token_text,
            "tokenizer_hash": self.tokenizer_hash,
            "model_hash": self.model_hash,
            "code_commit": self.code_commit,
            "data_role_hashes": dict(sorted(self.data_role_hashes.items())),
            "cap": self.cap.to_dict(),
            "radius_design_quantile": self.radius_design_quantile,
            "cap_count": self.cap.cap_count,
            "assignment_rule": self.assignment_rule,
            "outlier_rule": self.outlier_rule,
            "position_manifest_hash": self.position_manifest_hash,
            "random_boundary_manifest_hash": self.random_boundary_manifest_hash,
            "source_weights": dict(sorted(self.source_weights.items())),
            "selection_metrics": self.selection_metrics,
            "certification_thresholds": self.certification_thresholds,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.unsigned_dict()
        value["freeze_sha256"] = self.freeze_sha256
        return value


def create_freeze(
    cap: FrozenCapModel,
    *,
    tokenizer_hash: str,
    model_hash: str,
    code_commit: str,
    data_role_hashes: Mapping[str, str],
    position_manifest_hash: str,
    random_boundary_manifest_hash: str,
    source_weights: Mapping[str, float],
    selection_metrics: Mapping[str, Any],
    certification_thresholds: Mapping[str, float],
) -> FreezeArtifact:
    weights = {str(key): float(value) for key, value in source_weights.items()}
    if not weights or not np.isclose(sum(weights.values()), 1.0, atol=1e-9):
        raise ProtocolViolation("freeze source weights must sum to one")
    placeholder = FreezeArtifact(
        schema_version="mode3-v6-2-freeze-v1",
        token_id=cap.token_id, token_text=cap.token_text,
        tokenizer_hash=str(tokenizer_hash), model_hash=str(model_hash),
        code_commit=str(code_commit), data_role_hashes=dict(data_role_hashes),
        cap=cap, radius_design_quantile=float(cap.design_coverage),
        assignment_rule=cap.assignment_rule, outlier_rule=cap.outlier_rule,
        position_manifest_hash=str(position_manifest_hash),
        random_boundary_manifest_hash=str(random_boundary_manifest_hash),
        source_weights=weights, selection_metrics=dict(selection_metrics),
        certification_thresholds={str(k): float(v) for k, v in certification_thresholds.items()},
        freeze_sha256="",
    )
    digest = canonical_sha256(placeholder.unsigned_dict())
    return FreezeArtifact(**{**placeholder.__dict__, "freeze_sha256": digest})


def save_freeze(path: Path, artifact: FreezeArtifact) -> None:
    if artifact.freeze_sha256 != canonical_sha256(artifact.unsigned_dict()):
        raise ManifestMismatch("refusing to write invalid freeze hash")
    write_json(Path(path), artifact.to_dict())


def load_freeze(path: Path) -> FreezeArtifact:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    registered = str(value.pop("freeze_sha256"))
    if registered != canonical_sha256(value):
        raise ManifestMismatch("freeze artifact SHA-256 mismatch")
    cap_value = value.pop("cap")
    cap = FrozenCapModel(
        token_id=int(cap_value["token_id"]), token_text=str(cap_value["token_text"]),
        protocol=str(cap_value["protocol"]),
        centers=np.asarray(cap_value["centers"], dtype=np.float64),
        radii=np.asarray(cap_value["radii"], dtype=np.float64),
        design_coverage=float(cap_value["design_coverage"]),
        fit_role=str(cap_value["fit_role"]), radius_role=str(cap_value["radius_role"]),
        cap_count=int(cap_value["cap_count"]),
        assignment_rule=str(cap_value["assignment_rule"]),
        outlier_rule=str(cap_value["outlier_rule"]),
    )
    artifact = FreezeArtifact(
        schema_version=str(value["schema_version"]), token_id=int(value["token_id"]),
        token_text=str(value["token_text"]), tokenizer_hash=str(value["tokenizer_hash"]),
        model_hash=str(value["model_hash"]), code_commit=str(value["code_commit"]),
        data_role_hashes={str(k): str(v) for k, v in value["data_role_hashes"].items()},
        cap=cap, radius_design_quantile=float(value["radius_design_quantile"]),
        assignment_rule=str(value["assignment_rule"]), outlier_rule=str(value["outlier_rule"]),
        position_manifest_hash=str(value["position_manifest_hash"]),
        random_boundary_manifest_hash=str(value["random_boundary_manifest_hash"]),
        source_weights={str(k): float(v) for k, v in value["source_weights"].items()},
        selection_metrics=dict(value["selection_metrics"]),
        certification_thresholds={str(k): float(v) for k, v in value["certification_thresholds"].items()},
        freeze_sha256=registered,
    )
    if artifact.token_id != artifact.cap.token_id or artifact.token_text != artifact.cap.token_text:
        raise ManifestMismatch("freeze token identity differs from cap")
    return artifact
