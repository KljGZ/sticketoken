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
REGISTERED_RUNS: dict[str, dict[str, Any]] = {
    "mode3_v6_3_light_r5": {
        "protocol_revision": 5,
        "output_leaf": "mode3_v6_3_light",
        "experiment_name": "mode3_v6_3_light_single_token_frozen_cap",
        "rapid": False,
        "allowed_physical_gpus": [4, 5, 6, 7],
        "forbidden_physical_gpus": [0, 1, 2, 3],
    },
    "mode3_v6_3_rapid_r6": {
        "protocol_revision": 6,
        "output_leaf": "mode3_v6_3_rapid_r6",
        "experiment_name": "mode3_v6_3_rapid_positive_single_token_frozen_cap",
        "rapid": True,
        "allowed_physical_gpus": [4, 5, 6, 7],
        "forbidden_physical_gpus": [0, 1, 2, 3],
    },
    "mode3_v6_3_rapid_r7": {
        "protocol_revision": 7,
        "output_leaf": "mode3_v6_3_rapid_r7",
        "experiment_name": (
            "mode3_v6_3_rapid_positive_single_token_frozen_cap_high_priority_8gpu"
        ),
        "rapid": True,
        "allowed_physical_gpus": list(range(8)),
        "forbidden_physical_gpus": [],
    },
}


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
    run_id = str(config.get("run_id", ""))
    _require(run_id in REGISTERED_RUNS, "unregistered V6.3 run identity")
    registration = REGISTERED_RUNS[run_id]
    _require(
        int(config.get("protocol_revision", 0))
        == int(registration["protocol_revision"]),
        "V6.3 protocol revision drift",
    )
    _require(
        config.get("experiment_name") == registration["experiment_name"],
        "experiment name drift",
    )
    scope = config.get("scope", {})
    _require(scope.get("only_mode") == 3, "V6.3 is Mode 3 only")
    _require(scope.get("primary_claim") == "P3_ST_FCA_CORE", "primary claim drift")
    _require(scope.get("actual_trigger_token_length") == 1, "trigger must be one actual token")
    _require(scope.get("one_insertion_only") is True, "exactly one insertion is required")
    _require(scope.get("output_leaf") == registration["output_leaf"], "output leaf drift")
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
    required_allowed = frozenset(map(int, registration["allowed_physical_gpus"]))
    required_forbidden = frozenset(map(int, registration["forbidden_physical_gpus"]))
    _require(allowed == required_allowed, "physical GPU allow-list drift")
    _require(forbidden == required_forbidden, "physical GPU deny-list drift")
    _require(not allowed.intersection(forbidden), "GPU allow/deny lists overlap")
    _require(
        int(resources.get("gpu_start_minimum_free_memory_mib", 0)) == 12288,
        "GPU launch reserve must be 12 GiB",
    )
    _require(
        int(resources.get("gpu_runtime_minimum_free_memory_mib", 0)) == 8192,
        "GPU runtime reserve must be 8 GiB",
    )
    _require(
        float(resources.get("gpu_poll_interval_seconds", 0.0)) == 1.0,
        "GPU poll interval must be one second",
    )
    _require(
        int(resources.get("gpu_cooperative_yield_timeout_seconds", 0)) == 120,
        "GPU cooperative yield timeout must be 120 seconds",
    )
    _require(
        int(resources.get("cooperative_gpu_chunk_texts", 0)) == 1024,
        "GPU calls must use registered cooperative chunks",
    )
    environment_digest = str(resources.get("environment_lock_sha256", ""))
    _require(
        len(environment_digest) == 64
        and all(character in "0123456789abcdef" for character in environment_digest),
        "environment lock SHA-256 must be frozen",
    )
    rapid = config.get("rapid_track", {})
    if bool(registration["rapid"]):
        _require(rapid.get("enabled") is True, "rapid track must be enabled")
        expected_amendment = {
            "mode3_v6_3_rapid_r6": "V6_3_RAPID_POSITIVE_TRACK_A1",
            "mode3_v6_3_rapid_r7": "V6_3_RAPID_POSITIVE_TRACK_A2_8GPU_HIGH_PRIORITY",
        }[run_id]
        _require(rapid.get("amendment_id") == expected_amendment, "rapid amendment identity drift")
        _require(int(config["funnel"].get("s0_keep", 0)) == 200, "rapid route must retain 200 S0 candidates")
        _require(int(config["funnel"].get("full_top", 0)) == 20, "rapid route must retain 20 FULL candidates")
        _require(config["positions"].get("full_design") == "all_three", "rapid FULL must use all three positions")
        _require(config["positions"].get("top100_complete_all_positions") is True, "rapid final discovery stage must use all positions")
        _require(str(rapid.get("source_run_id")) == "mode3_v6_3_light_r5", "rapid S0 source run drift")
        source_commit = str(rapid.get("source_commit", ""))
        source_config = str(rapid.get("source_config_sha256", ""))
        _require(len(source_commit) == 40, "rapid source commit must be frozen")
        _require(len(source_config) == 64, "rapid source config SHA-256 must be frozen")
        if run_id == "mode3_v6_3_rapid_r6":
            _require(resources.get("priority_peer_first") is True, "r6 must yield to r5 workers")
        else:
            _require(resources.get("scheduling_priority") == "high", "r7 must register high priority")
            _require(resources.get("priority_peer_first") is False, "r7 must not yield to r5")
            _require(
                resources.get("lower_priority_peer_unit") == "sticky-v6-3-light",
                "r7 lower-priority peer unit drift",
            )
            _require(
                resources.get("signal_lower_priority_peer") is False,
                "r7 must never signal its lower-priority peer",
            )
    else:
        _require(not rapid or rapid.get("enabled") is not True, "r5 cannot enable the r6 rapid track")
    budget = config.get("budget", {})
    if bool(registration["rapid"]):
        _require(float(budget.get("hard_ratio", 0)) == 0.5, "rapid hard budget ratio must be 0.5x V5")
        _require(float(budget.get("forbidden_ratio", 0)) == 1.0, "rapid absolute ceiling must be 1.0x V5")
    else:
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


def assert_output_leaf(output: Path, config: Mapping[str, Any]) -> None:
    path = Path(output).resolve()
    run_id = str(config.get("run_id", ""))
    expected = str(REGISTERED_RUNS[run_id]["output_leaf"])
    if path.name != expected:
        raise ProtocolViolation(f"V6.3 output must end in {expected}: {path}")
    protected = {
        "mode3_v6", "mode3_v6_compact", "mode3_v6_2",
        "mode3_v6_3_light", "mode3_v6_3_rapid_r6", "v6_compact",
    }
    if any(part in protected for part in path.parts[:-1]):
        raise ProtocolViolation(f"V6.3 refuses protected output path: {path}")


def assert_physical_device(device: str, config: Mapping[str, Any]) -> int:
    text = str(device)
    if not text.startswith("cuda:"):
        raise ProtocolViolation(f"formal V6.3 requires an explicit CUDA device: {text}")
    physical = int(text.split(":", 1)[1])
    allowed = set(map(int, config["resources"]["allowed_physical_gpus"]))
    forbidden = set(map(int, config["resources"]["forbidden_physical_gpus"]))
    if physical not in allowed or physical in forbidden:
        raise ProtocolViolation(f"physical GPU {physical} is not authorized for V6.3")
    return physical
