"""Primary/secondary selection and immutable V6.3 freeze artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from .errors import ManifestMismatch, ProtocolViolation
from .geometry import FrozenCap
from .config import canonical_sha256


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class FreezeArtifact:
    schema_version: str
    kind: str
    cap: dict[str, Any]
    selected_metric: dict[str, Any]
    code_commit: str
    config_sha256: str
    role_manifest_sha256: str
    discovery_role_hashes: dict[str, str]
    confirm_role_hashes: dict[str, str]
    tokenizer_sha256: str
    model_revision: str
    call_space_sha256: str
    certification_thresholds: dict[str, float]
    refit_performed_after_freeze: bool = False
    actual_tokenizer_length: int = 1
    score_role_sha256: str = ""
    source_weights: dict[str, float] | None = None
    position_weights: dict[str, float] | None = None
    random_boundary_manifest_sha256: str = ""
    pretruncation_protocol_sha256: str = ""
    selection_rank: int = 0
    freeze_content_sha256: str = ""

    @property
    def token_id(self) -> int:
        return int(self.cap["token_id"])

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["freeze_content_sha256"] = ""
        value["freeze_content_sha256"] = canonical_sha256(value)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FreezeArtifact":
        defaults = {
            "actual_tokenizer_length": 1,
            "score_role_sha256": "",
            "source_weights": None,
            "position_weights": None,
            "random_boundary_manifest_sha256": "",
            "pretruncation_protocol_sha256": "",
            "selection_rank": 0,
            "freeze_content_sha256": "",
        }
        return cls(**{
            name: value[name] if name in value else defaults[name]
            for name in cls.__dataclass_fields__
        })

    def frozen_cap(self) -> FrozenCap:
        return FrozenCap.from_dict(self.cap)


def primary_order(row: Mapping[str, Any], maximum_radius_degrees: float) -> tuple[Any, ...]:
    return (
        0 if float(row["radius_degrees"]) <= float(maximum_radius_degrees) else 1,
        float(row["benign_occupancy_core"]),
        -float(row["balanced_coverage"]),
        -float(row["worst_position_coverage"]),
        -float(row["outside_to_inside"]),
        float(row["benign_occupancy_1_1"]),
        float(row["benign_occupancy_auc_1_1_5"]),
        float(row["center_restart_spread"]),
        int(row["token_id"]),
    )


def select_primary_and_secondaries(
    metrics: Sequence[Mapping[str, Any]], *, maximum_radius_degrees: float,
    expected_count: int = 100,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    if len(metrics) != int(expected_count) or int(expected_count) < 5:
        raise ProtocolViolation(
            f"freeze requires exactly {expected_count} complete-position candidates, got {len(metrics)}"
        )
    token_ids = [int(row["token_id"]) for row in metrics]
    if len(token_ids) != len(set(token_ids)):
        raise ProtocolViolation("top100 freeze input repeats token IDs")
    ordered = sorted(metrics, key=lambda row: primary_order(row, maximum_radius_degrees))
    if float(ordered[0]["radius_degrees"]) > float(maximum_radius_degrees):
        raise ProtocolViolation("no top100 candidate passes the registered radius gate")
    return ordered[0], ordered[1:5]


def write_freeze(
    output: Path,
    *,
    primary_metric: Mapping[str, Any],
    secondary_metrics: Sequence[Mapping[str, Any]],
    caps: Mapping[int, FrozenCap],
    metadata: Mapping[str, Any],
) -> tuple[Path, str]:
    if len(secondary_metrics) != 4:
        raise ProtocolViolation("freeze requires four secondary candidates")
    target = Path(output) / "freeze"
    target.mkdir(parents=True, exist_ok=True)

    def artifact(kind: str, metric: Mapping[str, Any], selection_rank: int) -> FreezeArtifact:
        token_id = int(metric["token_id"])
        cap = caps[token_id]
        if cap.stage != "top100" or cap.radius_degrees > float(metadata["maximum_radius_degrees"]):
            raise ProtocolViolation("only a complete-position top100 cap may be frozen")
        discovery = dict(metadata["discovery_role_hashes"])
        confirm = dict(metadata["confirm_role_hashes"])
        if set(discovery.values()).intersection(confirm.values()):
            raise ProtocolViolation("confirm role hash appears in discovery roles")
        return FreezeArtifact(
            "mode3-v6-3-freeze-v1", kind, cap.to_dict(), dict(metric),
            str(metadata["code_commit"]), str(metadata["config_sha256"]),
            str(metadata["role_manifest_sha256"]), discovery, confirm,
            str(metadata["tokenizer_sha256"]), str(metadata["model_revision"]),
            str(metadata["call_space_sha256"]), dict(metadata["certification_thresholds"]),
            False, 1, str(metric["score_role_sha256"]),
            dict(metadata["source_weights"]), dict(metadata["position_weights"]),
            str(metadata["random_boundary_manifest_sha256"]),
            str(metadata["pretruncation_protocol_sha256"]),
            int(selection_rank), "",
        )

    primary = artifact("primary", primary_metric, 1)
    primary_path = target / "primary.json"
    _atomic_text(primary_path, json.dumps(primary.to_dict(), indent=2, sort_keys=True) + "\n")
    secondary_rows = [
        artifact("secondary", metric, rank).to_dict()
        for rank, metric in enumerate(secondary_metrics, start=2)
    ]
    _atomic_text(target / "secondary.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in secondary_rows))
    freeze_sha256 = _sha256_file(primary_path)
    _atomic_text(target / "FREEZE.sha256", f"{freeze_sha256}  primary.json\n")
    _atomic_text(target / "COMPLETE.json", json.dumps({
        "schema_version": "mode3-v6-3-freeze-complete-v1",
        "status": "PRIMARY_FROZEN", "freeze_sha256": freeze_sha256,
        "primary_token_id": primary.token_id,
        "secondary_token_ids": [int(row["token_id"]) for row in secondary_metrics],
        "confirm_accessed": False, "refit_performed": False,
    }, indent=2, sort_keys=True) + "\n")
    return primary_path, freeze_sha256


def load_freeze(path: Path, expected_sha256: str | None = None) -> FreezeArtifact:
    observed = _sha256_file(Path(path))
    if expected_sha256 is not None and observed != str(expected_sha256):
        raise ManifestMismatch("freeze SHA-256 mismatch")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    registered_content = str(payload.get("freeze_content_sha256", ""))
    check_payload = dict(payload)
    check_payload["freeze_content_sha256"] = ""
    if registered_content and canonical_sha256(check_payload) != registered_content:
        raise ManifestMismatch("freeze content SHA-256 mismatch")
    artifact = FreezeArtifact.from_dict(payload)
    cap = artifact.frozen_cap()
    if cap.to_dict()["cap_sha256"] != artifact.cap.get("cap_sha256"):
        raise ManifestMismatch("frozen cap content hash mismatch")
    if artifact.refit_performed_after_freeze:
        raise ManifestMismatch("freeze artifact reports a post-freeze refit")
    return artifact
