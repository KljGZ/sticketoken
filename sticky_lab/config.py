"""Configuration loading and validation for the three experiment modes."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


MODES = {"single_sticky", "multi_booster", "repulsive_attractor"}
INSERTION_MODES = {"prefix", "suffix", "random"}
SEARCH_ALGORITHMS = {"cem", "blackbox_beam", "genetic"}


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    config = deepcopy(config)
    config["config_path"] = str(source)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    mode = config.get("mode")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}, got {mode!r}")
    model = config.get("model", {})
    if not model.get("id"):
        raise ValueError("model.id is required")
    data = config.get("data", {})
    if not data.get("path"):
        raise ValueError("data.path is required")
    split = data.get("split", {})
    fractions = [float(split.get(key, 0)) for key in ("search", "validation", "test")]
    if any(value <= 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError("data.split search/validation/test must be positive and sum to 1")
    insertion_modes = config.get("insertion", {}).get("modes", [])
    if not insertion_modes or not set(insertion_modes) <= INSERTION_MODES:
        raise ValueError(f"insertion.modes must be a non-empty subset of {sorted(INSERTION_MODES)}")
    if int(config.get("seed", 0)) < 0:
        raise ValueError("seed must be non-negative")
    if mode != "single_sticky":
        search = config.get("search", {})
        if search.get("algorithm") not in SEARCH_ALGORITHMS:
            raise ValueError(f"search.algorithm must be one of {sorted(SEARCH_ALGORITHMS)}")
        if int(search.get("trigger_length", 0)) < 2:
            raise ValueError("multi-token modes require search.trigger_length >= 2")


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve config paths against the repository, not the caller's cwd."""
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    config_path = Path(config["config_path"])
    repo_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    return (repo_root / candidate).resolve()

