"""Configuration loading and fail-closed protocol validation for V7."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from sticky_lab.mode3_v6_3.config import (
    canonical_json,
    canonical_sha256,
    sha256_file,
)
from sticky_lab.mode3_v6_3.errors import ProtocolViolation
from sticky_lab.mode3_v6_3.tokenizer_audit import TOKENIZER_HASH_ALGORITHM


MODEL_REVISION = "fc5d4628481afbbaaacd7af6bb07cf9d3865f781"
RUN_ID = "mode3_v7_occupancy_frontier_r1"
OUTPUT_LEAF = "mode3_v7_occupancy_frontier"
OCCUPANCY_GRID = (
    0.001,
    0.003,
    0.005,
    0.008,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.10,
    0.11,
    0.12,
    0.13,
    0.14,
    0.15,
)
PROTECTED_OUTPUT_LEAVES = frozenset(
    {
        "mode3_v6",
        "mode3_v6_compact",
        "mode3_v6_2",
        "mode3_v6_3_light",
        "mode3_v6_3_rapid_r6",
        "mode3_v6_3_rapid_r7",
        "v6_compact",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolViolation(message)


def _as_float_tuple(values: Any) -> tuple[float, ...]:
    try:
        return tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ProtocolViolation("occupancy grid must be numeric") from error


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject every scientific or operational drift from the registered V7."""

    _require(str(config.get("protocol_version")) == "7.0", "not a V7 config")
    _require(int(config.get("protocol_revision", 0)) == 1, "V7 revision drift")
    _require(str(config.get("run_id")) == RUN_ID, "unregistered V7 run identity")

    scope = config.get("scope", {})
    _require(scope.get("only_mode") == 3, "V7 is Mode 3 only")
    _require(
        scope.get("primary_claim") == "PS_OC_ST_FCA",
        "V7 primary claim drift",
    )
    _require(scope.get("actual_trigger_token_length") == 1, "trigger must be one token")
    _require(scope.get("one_insertion_only") is True, "trigger must be inserted once")
    _require(scope.get("output_leaf") == OUTPUT_LEAF, "V7 output leaf drift")

    model = config.get("model", {})
    _require(model.get("revision") == MODEL_REVISION, "model revision drift")
    _require(model.get("formal_precision") == "float32", "V7 formal precision is FP32")
    _require(model.get("normalize_final_embeddings") is True, "embeddings must be normalized")
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

    positions = config.get("positions", {})
    _require(
        tuple(map(str, positions.get("names", []))) == ("prefix", "suffix"),
        "V7 permits prefix and suffix only",
    )
    _require(positions.get("random_position_enabled") is False, "random is removed in V7")
    _require(positions.get("shared_center") is True, "positions must share one center")
    _require(positions.get("shared_radius") is True, "positions must share one radius")

    geometry = config.get("geometry", {})
    _require(geometry.get("metric") == "angular_distance_radians", "metric drift")
    _require(geometry.get("center_fit_triggered_only") is True, "center must be triggered-only")
    _require(abs(float(geometry.get("center_trim_fraction", -1)) - 0.10) < 1e-12, "trim must be 10%")
    _require(geometry.get("source_equal_weight") is True, "sources must be equally weighted")
    _require(geometry.get("position_equal_weight") is True, "positions must be equally weighted")
    _require(float(geometry.get("maximum_radius_degrees", 0)) == 35.0, "radius cap drift")

    radius = config.get("radius", {})
    _require(radius.get("strategy") == "benign_occupancy_constrained", "radius policy drift")
    _require(
        _as_float_tuple(radius.get("occupancy_grid", [])) == OCCUPANCY_GRID,
        "the registered 19-point occupancy grid is immutable",
    )
    _require(radius.get("select_largest_feasible_radius") is True, "largest feasible radius is required")
    _require(
        radius.get("benign_metric") == "source_balanced_one_sided_ucb",
        "benign occupancy statistic drift",
    )
    _require(radius.get("center_refit_per_beta") is False, "one center must serve every beta")
    _require(radius.get("q92_legacy_diagnostic_only") is True, "q92 may only be diagnostic")
    _require(0 < float(radius.get("familywise_alpha", 0)) < 1, "invalid occupancy alpha")

    certification = config.get("certification", {})
    _require(float(certification.get("prefix_coverage_lcb", 0)) == 0.80, "prefix LCB drift")
    _require(float(certification.get("suffix_coverage_lcb", 0)) == 0.80, "suffix LCB drift")
    _require(
        certification.get("occupancy_ucb_equals_frozen_beta") is True,
        "occupancy gate must use the frozen beta",
    )
    _require(certification.get("migration_metrics_are_gates") is False, "migration is diagnostic")

    search = config.get("search", {})
    _require(search.get("exhaustive_s0_reuse") is True, "r5 S0 reuse audit is required")
    for key in ("blackbox_cem", "genetic_algorithm", "hotflip", "multicap"):
        _require(search.get(key) is False, f"forbidden V7 search enabled: {key}")

    data = config.get("data", {})
    stages = data.get("stage_sizes", {})
    for stage in ("s0", "full"):
        _require(set(stages.get(stage, {})) == {"fit", "calibration", "select"}, f"invalid {stage} roles")
        _require(all(int(value) > 0 for value in stages[stage].values()), f"empty {stage} role")
    _require(
        all(int(stages["s0"][key]) <= int(stages["full"][key]) for key in stages["s0"]),
        "S0 views must nest inside FULL",
    )
    confirm_roles = data.get("confirm_roles", {})
    _require(
        set(confirm_roles)
        == {"confirm_prefix", "confirm_suffix", "confirm_benign", "confirm_paired"},
        "V7 confirm role set drift",
    )
    _require(int(data.get("axis_fit_benign", 0)) > 0, "independent e* role is required")

    reuse = config.get("reuse", {})
    _require(reuse.get("source_run_id") == "mode3_v6_3_light_r5", "r5 source identity drift")
    _require(int(reuse.get("source_s0_shards", 0)) == 32, "r5 must have 32 S0 shards")
    _require(int(reuse.get("source_legal_tokens", 0)) == 21984, "legal vocabulary drift")
    _require(reuse.get("old_q92_caps_formal") is False, "old q92 caps cannot be formal V7 caps")

    resources = config.get("resources", {})
    allowed = frozenset(map(int, resources.get("allowed_physical_gpus", [])))
    forbidden = frozenset(map(int, resources.get("forbidden_physical_gpus", [])))
    _require(allowed == frozenset({4, 5, 6, 7}), "V7 only authorizes physical GPUs 4-7")
    _require(forbidden == frozenset({0, 1, 2, 3}), "V7 must exclude physical GPUs 0-3")
    _require(not allowed.intersection(forbidden), "GPU lists overlap")
    _require(resources.get("wait_for_v6_3_r5") is True, "V7 must not interfere with r5")
    _require(int(resources.get("cooperative_gpu_chunk_texts", 0)) > 0, "invalid GPU chunk")
    _require(
        int(resources.get("registration_minimum_free_bytes", 0)) > 0,
        "invalid registration storage floor",
    )
    _require(
        float(resources.get("minimum_free_disk_peak_multiplier", 0)) >= 1.0,
        "invalid peak-storage safety multiplier",
    )
    _require(
        int(resources.get("estimated_peak_cache_bytes", 0)) > 0,
        "invalid peak-cache estimate",
    )

    budget = config.get("budget", {})
    _require(int(budget.get("hard_limit", 0)) < int(budget.get("forbidden_limit", 0)), "budget gates inverted")

    diagnostics = config.get("diagnostics", {})
    _require(
        int(diagnostics.get("center_bootstrap_samples", -1)) == 0,
        "candidate-wide bootstrap must remain disabled",
    )
    _require(
        int(diagnostics.get("post_selection_center_bootstrap_samples", 0)) > 0,
        "post-selection center bootstrap is required",
    )
    _require(
        int(diagnostics.get("post_selection_bootstrap_restarts", 0)) > 0,
        "post-selection bootstrap restarts are required",
    )


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
    name = str(profile)
    value = copy.deepcopy(dict(config))
    if name == "formal":
        value["run_profile"] = "formal"
        value["scientific_claims_allowed"] = True
        return value
    if name not in {"dry_run", "pilot"}:
        raise ProtocolViolation(f"unknown V7 profile {name}")
    settings = value["profiles"][name]
    value["run_profile"] = name
    value["scientific_claims_allowed"] = False
    value["data"]["stage_sizes"] = copy.deepcopy(settings["stage_sizes"])
    value["data"]["axis_fit_benign"] = int(settings["axis_fit_benign"])
    value["data"]["confirm_roles"] = copy.deepcopy(settings["confirm_roles"])
    value["funnel"].update(copy.deepcopy(settings["funnel"]))
    value["diagnostics"]["post_selection_center_bootstrap_samples"] = int(
        settings["post_selection_center_bootstrap_samples"]
    )
    value["diagnostics"]["post_selection_bootstrap_restarts"] = int(
        settings["post_selection_bootstrap_restarts"]
    )
    value["tokenizer"]["engineering_legal_token_limit"] = int(settings["legal_tokens"])
    value["resources"]["estimated_peak_cache_bytes"] = int(settings["estimated_peak_cache_bytes"])
    return value


def assert_output_leaf(output: Path, config: Mapping[str, Any]) -> None:
    path = Path(output).resolve()
    if path.name != OUTPUT_LEAF or str(config["scope"]["output_leaf"]) != OUTPUT_LEAF:
        raise ProtocolViolation(f"V7 output must end in {OUTPUT_LEAF}: {path}")
    if any(part in PROTECTED_OUTPUT_LEAVES for part in path.parts):
        raise ProtocolViolation(f"V7 refuses a V6-protected output path: {path}")


def assert_physical_device(device: str, config: Mapping[str, Any]) -> int:
    text = str(device)
    if not text.startswith("cuda:"):
        raise ProtocolViolation(f"V7 requires an explicit physical CUDA device: {text}")
    physical = int(text.split(":", 1)[1])
    allowed = set(map(int, config["resources"]["allowed_physical_gpus"]))
    forbidden = set(map(int, config["resources"]["forbidden_physical_gpus"]))
    if physical not in allowed or physical in forbidden:
        raise ProtocolViolation(f"physical GPU {physical} is not authorized for V7")
    return physical
