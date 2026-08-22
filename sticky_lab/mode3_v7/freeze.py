"""Immutable token-beta operating-point freeze artifacts for V7."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sticky_lab.mode3_v6_3.errors import ManifestMismatch, ProtocolViolation
from sticky_lab.mode3_v6_3.report import atomic_json, write_jsonl

from .candidate_ranking import choose_primary_and_secondaries
from .config import canonical_sha256
from .geometry import FrozenOperatingPoint
from .operating_point import operating_point_for_beta80


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FreezeArtifact:
    schema_version: str
    kind: str
    operating_point: dict[str, Any]
    selected_summary: dict[str, Any]
    code_commit: str
    config_sha256: str
    protocol_lock_sha256: str
    role_manifest_sha256: str
    discovery_role_hashes: dict[str, str]
    confirm_role_hashes: dict[str, str]
    tokenizer_sha256: str
    model_revision: str
    call_space_sha256: str
    e_star_sha256: str
    certification_thresholds: dict[str, float]
    full_frontier_sha256: str
    selection_rank: int
    refit_performed_after_freeze: bool = False
    confirm_accessed_before_freeze: bool = False
    freeze_content_sha256: str = ""

    @property
    def token_id(self) -> int:
        return int(self.operating_point["token_id"])

    @property
    def beta(self) -> float:
        return float(self.operating_point["beta"])

    def frozen_model(self) -> FrozenOperatingPoint:
        return FrozenOperatingPoint.from_dict(self.operating_point)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["freeze_content_sha256"] = ""
        value["freeze_content_sha256"] = canonical_sha256(value)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FreezeArtifact":
        return cls(**{name: value[name] for name in cls.__dataclass_fields__})


def _artifact(
    kind: str,
    frontier: Mapping[str, Any],
    *,
    rank: int,
    metadata: Mapping[str, Any],
) -> FreezeArtifact:
    point = operating_point_for_beta80(frontier)
    if point is None or not bool(point["feasible"]):
        raise ProtocolViolation("only a feasible PS-80 operating point may be frozen")
    discovery = dict(metadata["discovery_role_hashes"])
    confirm = dict(metadata["confirm_role_hashes"])
    if set(discovery.values()).intersection(confirm.values()):
        raise ProtocolViolation("confirmation role hash appears in discovery roles")
    model = FrozenOperatingPoint(
        token_id=int(frontier["token_id"]),
        token_text=str(frontier["token_text"]),
        center=np.asarray(frontier["center"], dtype=np.float64),
        beta=float(frontier["beta80_ps"]),
        radius=float(point["radius"]),
        fit_role_sha256=str(frontier["role_hashes"]["fit"]),
        calibration_role_sha256=str(frontier["role_hashes"]["calibration"]),
        select_role_sha256=str(frontier["role_hashes"]["select"]),
        stage=str(frontier["stage"]),
    )
    if model.radius_degrees > float(metadata["maximum_radius_degrees"]):
        raise ProtocolViolation("freeze candidate exceeds the registered radius cap")
    summary = {
        "token_id": int(frontier["token_id"]),
        "token_text": str(frontier["token_text"]),
        "beta80_ps": float(frontier["beta80_ps"]),
        "coverage_auc_log_beta": float(frontier["coverage_auc_log_beta"]),
        "beta_axis": frontier.get("beta_axis"),
        "beta80_precedes_beta_axis": bool(frontier.get("beta80_precedes_beta_axis")),
        "evidence_grade": str(frontier["evidence_grade"]),
        "prefix_coverage_lcb": float(point["prefix_coverage_lcb"]),
        "suffix_coverage_lcb": float(point["suffix_coverage_lcb"]),
        "benign_occupancy_ucb": float(point["benign_occupancy_ucb"]),
        "axis_geometry_used_for_selection": False,
    }
    return FreezeArtifact(
        schema_version="mode3-v7-freeze-v1",
        kind=str(kind),
        operating_point=model.to_dict(),
        selected_summary=summary,
        code_commit=str(metadata["code_commit"]),
        config_sha256=str(metadata["config_sha256"]),
        protocol_lock_sha256=str(metadata["protocol_lock_sha256"]),
        role_manifest_sha256=str(metadata["role_manifest_sha256"]),
        discovery_role_hashes=discovery,
        confirm_role_hashes=confirm,
        tokenizer_sha256=str(metadata["tokenizer_sha256"]),
        model_revision=str(metadata["model_revision"]),
        call_space_sha256=str(metadata["call_space_sha256"]),
        e_star_sha256=str(metadata["e_star_sha256"]),
        certification_thresholds=dict(metadata["certification_thresholds"]),
        full_frontier_sha256=str(metadata["full_frontier_sha256"]),
        selection_rank=int(rank),
    )


def write_freeze(
    output: Path,
    *,
    frontiers: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> tuple[Path, str]:
    primary, secondaries = choose_primary_and_secondaries(frontiers)
    target = Path(output) / "freeze"
    target.mkdir(parents=True, exist_ok=True)
    primary_artifact = _artifact("primary", primary, rank=1, metadata=metadata)
    primary_path = target / "primary.json"
    atomic_json(primary_path, primary_artifact.to_dict())
    secondary_artifacts = [
        _artifact("secondary", frontier, rank=rank, metadata=metadata).to_dict()
        for rank, frontier in enumerate(secondaries, start=2)
    ]
    write_jsonl(target / "secondary.jsonl", secondary_artifacts)
    digest = sha256_file(primary_path)
    (target / "FREEZE.sha256").write_text(
        f"{digest}  primary.json\n", encoding="utf-8"
    )
    atomic_json(
        target / "COMPLETE.json",
        {
            "schema_version": "mode3-v7-freeze-complete-v1",
            "status": "V7_PRIMARY_FROZEN",
            "freeze_sha256": digest,
            "primary_token_id": primary_artifact.token_id,
            "primary_beta": primary_artifact.beta,
            "secondary_token_ids": [int(frontier["token_id"]) for frontier in secondaries],
            "confirm_accessed": False,
            "refit_performed": False,
        },
    )
    return primary_path, digest


def load_freeze(path: Path, expected_sha256: str | None = None) -> FreezeArtifact:
    observed = sha256_file(Path(path))
    if expected_sha256 is not None and observed != str(expected_sha256):
        raise ManifestMismatch("V7 freeze SHA-256 mismatch")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    registered = str(payload.get("freeze_content_sha256", ""))
    check = dict(payload)
    check["freeze_content_sha256"] = ""
    if not registered or canonical_sha256(check) != registered:
        raise ManifestMismatch("V7 freeze content SHA-256 mismatch")
    artifact = FreezeArtifact.from_dict(payload)
    model = artifact.frozen_model()
    if model.to_dict().get("operating_point_sha256") != artifact.operating_point.get(
        "operating_point_sha256"
    ):
        raise ManifestMismatch("frozen operating point hash mismatch")
    if artifact.refit_performed_after_freeze or artifact.confirm_accessed_before_freeze:
        raise ManifestMismatch("freeze reports protocol leakage")
    return artifact
