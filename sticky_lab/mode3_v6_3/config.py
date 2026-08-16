"""Configuration loading and immutable-protocol validation for V6.3."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import ProtocolViolation
from .tokenizer_audit import TOKENIZER_HASH_ALGORITHM


MODEL_REVISION = "fc5d4628481afbbaaacd7af6bb07cf9d3865f781"
OUTPUT_LEAF = "mode3_v6_3_light"
REQUIRED_ALLOWED_GPUS = frozenset({4, 5, 6, 7})
REQUIRED_FORBIDDEN_GPUS = frozenset({0, 1, 2, 3})


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_environment_lock(path: Path) -> dict[str, str]:
    """Require the active Python environment to match every frozen distribution."""
    expected: dict[str, str] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise ProtocolViolation(f"invalid environment lock entry: {line}")
        name, version = line.split("==", 1)
        expected[name] = version
    if not expected:
        raise ProtocolViolation("environment lock contains no distributions")
    observed: dict[str, str] = {}
    for name, expected_version in expected.items():
        if name.casefold() == "python":
            observed_version = platform.python_version()
        else:
            try:
                observed_version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError as error:
                raise ProtocolViolation(
                    f"environment lock distribution is absent: {name}"
                ) from error
        if observed_version != expected_version:
            raise ProtocolViolation(
                f"environment version drift for {name}: "
                f"{observed_version} != {expected_version}"
            )
        observed[name] = observed_version
    return observed


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolViolation(message)


def validate_config(config: Mapping[str, Any]) -> None:
    _require(str(config.get("protocol_version")) == "6.3", "not a V6.3 config")
    _require(int(config.get("protocol_revision", 0)) == 3, "not the repaired V6.3 protocol revision")
    _require(config.get("run_id") == "mode3_v6_3_light_r3", "V6.3 run identity drift")
    _require(config.get("experiment_name") == "mode3_v6_3_light_single_token_frozen_cap", "experiment name drift")
    scope = config.get("scope", {})
    _require(scope.get("only_mode") == 3, "V6.3 is Mode 3 only")
    _require(scope.get("primary_claim") == "P3_ST_FCA_CORE", "primary claim drift")
    _require(scope.get("actual_trigger_token_length") == 1, "trigger must be one actual token")
    _require(scope.get("one_insertion_only") is True, "exactly one insertion is required")
    _require(scope.get("output_leaf") == OUTPUT_LEAF, "output leaf drift")
    model = config.get("model", {})
    _require(model.get("revision") == MODEL_REVISION, "model revision drift")
    _require(model.get("formal_precision") == "float32", "first formal run must be float32")
    _require(
        model.get("tokenizer_hash_algorithm") == TOKENIZER_HASH_ALGORITHM,
        "tokenizer hash algorithm drift",
    )
    tokenizer_digest = str(model.get("tokenizer_sha256", ""))
    _require(
        len(tokenizer_digest) == 64
        and all(character in "0123456789abcdef" for character in tokenizer_digest),
        "tokenizer SHA-256 must be lowercase hexadecimal",
    )
    search = config.get("search", {})
    _require(search.get("exhaustive_legal_vocabulary") is True, "full legal-vocabulary enumeration is required")
    for key in ("blackbox_cem", "genetic_algorithm", "whitebox_hotflip", "continuous_token_search"):
        _require(search.get(key) is False, f"forbidden search enabled: {key}")
    _require(int(search.get("historical_candidate_quota", -1)) == 0, "historical quota must be zero")
    geometry = config.get("geometry", {})
    _require(int(geometry.get("primary_cap_count", 0)) == 1, "single cap is the only primary model")
    _require(geometry.get("multicap_enabled") is False, "multicap is disabled")
    _require(abs(float(geometry.get("radius_design_quantile", 0.0)) - 0.92) < 1e-12, "radius design quantile must be 0.92")
    _require(abs(float(geometry.get("center_trim_fraction", -1.0)) - 0.10) < 1e-12, "trim fraction must be 0.10")
    positions = config.get("positions", {})
    _require(positions.get("random_embedding_average_forbidden") is True, "random-vector averaging must be forbidden")
    _require(positions.get("random_boundary_candidate_independent") is True, "random boundary must be candidate independent")
    data = config.get("data", {})
    _require(
        int(data.get("ood_domains", 0)) == 4
        and len(set(map(str, data.get("ood_domains_allowlist", [])))) == 4,
        "V6.3 requires exactly four registered OOD domains",
    )
    resources = config.get("resources", {})
    allowed = frozenset(map(int, resources.get("allowed_physical_gpus", [])))
    forbidden = frozenset(map(int, resources.get("forbidden_physical_gpus", [])))
    _require(bool(allowed) and allowed.issubset(REQUIRED_ALLOWED_GPUS), "only physical GPUs 4-7 may be used")
    _require(REQUIRED_FORBIDDEN_GPUS.issubset(forbidden), "physical GPUs 0-3 must be hard-disabled")
    _require(not allowed.intersection(forbidden), "GPU allow/deny lists overlap")
    environment_digest = str(resources.get("environment_lock_sha256", ""))
    _require(
        len(environment_digest) == 64
        and all(character in "0123456789abcdef" for character in environment_digest),
        "environment lock SHA-256 must be frozen",
    )
    budget = config.get("budget", {})
    _require(float(budget.get("hard_ratio", 0)) == 6.5, "hard budget ratio must be 6.5x V5")
    _require(float(budget.get("forbidden_ratio", 0)) == 15.0, "absolute ceiling must be 15x V5")
    _require(int(budget.get("hard_limit", 0)) < int(budget.get("forbidden_limit", 0)), "hard limit must precede absolute ceiling")


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolViolation("configuration root must be a mapping")
    validate_config(value)
    return value


def resolved_config(config: Mapping[str, Any], *, source_path: Path | None = None) -> dict[str, Any]:
    value = json.loads(canonical_json(config).decode("utf-8"))
    value["config_sha256"] = canonical_sha256(config)
    if source_path is not None:
        value["source_config_path"] = str(Path(source_path).resolve())
        value["source_file_sha256"] = sha256_file(Path(source_path))
    return value


def config_for_profile(config: Mapping[str, Any], profile: str) -> dict[str, Any]:
    """Return an explicitly non-scientific engineering profile or formal config."""
    name = str(profile)
    value = copy.deepcopy(dict(config))
    if name == "formal":
        value["run_profile"] = "formal"
        return value
    if name not in {"dry_run", "pilot"}:
        raise ProtocolViolation(f"unknown run profile {name}")
    settings = value["profiles"][name]
    value["run_profile"] = name
    value["scientific_claims_allowed"] = False
    value["data"]["search_chain_sizes"] = copy.deepcopy(settings["role_sizes"])
    value["data"]["discovery_benign"] = int(settings["discovery_benign"])
    value["data"]["ood_trigger_per_domain"] = int(settings["ood_trigger_per_domain"])
    value["data"]["ood_benign_per_domain"] = int(settings["ood_benign_per_domain"])
    value["resources"]["estimated_peak_cache_bytes"] = int(
        settings["estimated_peak_cache_bytes"]
    )
    value["data"]["confirm_roles"].update({
        "confirm_trigger": int(settings["confirm_trigger"]),
        "confirm_benign": int(settings["confirm_benign"]),
        "paired_position_audit": min(96, int(settings["confirm_trigger"])),
        "semantic_control": min(128, int(settings["confirm_trigger"])),
        "iid_replication_0": min(128, int(settings["confirm_trigger"])),
        "iid_replication_1": min(128, int(settings["confirm_trigger"])),
        "iid_replication_2": min(128, int(settings["confirm_trigger"])),
        "retrieval_probe": min(128, int(settings["confirm_trigger"])),
    })
    keep = settings["keep"]
    value["funnel"].update({
        "s0_keep": int(keep["s0"]), "s1_keep": int(keep["s1"]),
        "s2_keep": int(keep["s2"]), "full_top": int(keep["full"]),
    })
    value["tokenizer"]["contextual_audit_samples"] = int(sum(settings["role_sizes"]["s0"].values()))
    value["tokenizer"]["engineering_legal_token_limit"] = int(settings["legal_tokens"])
    return value


def assert_output_leaf(output: Path) -> None:
    path = Path(output).resolve()
    if path.name != OUTPUT_LEAF:
        raise ProtocolViolation(f"V6.3 output must end in {OUTPUT_LEAF}: {path}")
    protected = {"mode3_v6", "mode3_v6_compact", "mode3_v6_2", "v6_compact"}
    if any(part in protected for part in path.parts):
        raise ProtocolViolation(f"V6.3 refuses protected output path: {path}")


def assert_physical_device(device: str, config: Mapping[str, Any]) -> int:
    text = str(device)
    if not text.startswith("cuda:"):
        raise ProtocolViolation(f"formal V6.3 requires an explicit CUDA device: {text}")
    physical = int(text.split(":", 1)[1])
    allowed = set(map(int, config["resources"]["allowed_physical_gpus"]))
    if physical not in allowed or physical in {0, 1, 2, 3}:
        raise ProtocolViolation(f"physical GPU {physical} is not authorized for V6.3")
    return physical
