#!/usr/bin/env python3
"""Fail-closed scope, threat-model, lineage and result audit for Mode 3 V5."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import yaml


TASKS = ("prefix", "suffix", "random", "conditional", "shared")
FORBIDDEN_ATTRIBUTES = {
    "backward",
    "parameters",
    "named_parameters",
    "get_input_embeddings",
    "hidden_states",
    "output_hidden_states",
    "grad",
}
PROTECTED_PREFIXES = (
    "sticky_lab/mode3_v3/",
    "sticky_lab/mode3_v4/",
    "configs/v3_mode3.yaml",
    "configs/v4_mode3.yaml",
    "tests/test_mode3_v3.py",
    "tests/test_sticky_high.py",
    "tests/test_sticky_lab.py",
    "tests/test_sticky_lab_v2.py",
    "stickytoken/",
)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_source(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    source_root = root / "sticky_lab" / "mode3_v5"
    errors: list[str] = []
    source_files = sorted(source_root.glob("*.py"))
    if not source_files:
        errors.append("missing sticky_lab/mode3_v5 sources")
    for path in source_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports: list[str] = []
        attributes: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
            elif isinstance(node, ast.Attribute):
                attributes.append(node.attr)
        old_imports = [
            value
            for value in imports
            if any(fragment in value for fragment in ("mode1", "mode2", "mode3_v3", "mode3_v4"))
        ]
        if old_imports:
            errors.append(f"{path.relative_to(root)} imports frozen mode code: {old_imports}")
        used = sorted(FORBIDDEN_ATTRIBUTES.intersection(attributes))
        if used:
            errors.append(f"{path.relative_to(root)} accesses forbidden model surface: {used}")
    lengths = config["lengths"]
    if list(range(int(lengths["minimum"]), int(lengths["maximum"]) + 1, int(lengths["step"]))) != list(
        range(1, 31)
    ):
        errors.append("registered actual-token length frontier is not every integer 1..30")
    if config["search"]["tasks"] != list(TASKS):
        errors.append("P1/P2/P3 task registry is incomplete")
    if int(config["validation"]["bootstrap_replicates"]) != 500:
        errors.append("validation bootstrap count is not 500")
    if int(config["structure"]["maximum_cluster_count"]) != 4:
        errors.append("maximum cluster count is not 4")
    return {"source_files": len(source_files), "errors": errors}


def audit_scope(root: Path, baseline: str, maximum_blob: int) -> dict[str, Any]:
    changed = [value for value in git(root, "diff", "--name-only", f"{baseline}...HEAD").splitlines() if value]
    protected = [path for path in changed if path.startswith(PROTECTED_PREFIXES)]
    large = []
    tracked = subprocess.check_output(
        ["git", "-c", "core.quotepath=false", "ls-files", "-z"], cwd=root
    ).decode("utf-8").split("\0")
    for path_text in tracked:
        if not path_text:
            continue
        path = root / path_text
        if path.is_file() and path.stat().st_size > maximum_blob:
            large.append({"path": path_text, "bytes": path.stat().st_size})
    return {"changed_files": changed, "protected_changes": protected, "oversized_tracked_objects": large}


def _completion(path: Path) -> bool:
    marker = path / "COMPLETE.json"
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    for row in payload.get("artifacts", []):
        artifact = path / row["path"]
        if not artifact.is_file() or artifact.stat().st_size != int(row["bytes"]):
            return False
        if sha256_file(artifact) != row["sha256"]:
            return False
    return True


def audit_results(root: Path, results: Path, config: dict[str, Any]) -> dict[str, Any]:
    expected_screen = len(TASKS) * int(config["runtime"]["screen_shards_per_task"])
    expected_search = len(TASKS) * 29 * int(config["search"]["restarts_per_length"])
    expected_merge = len(TASKS) * 29
    expected_validation = len(TASKS) * 30
    screen = sum(
        _completion(results / "screen" / task / f"shard_{shard:02d}")
        for task in TASKS
        for shard in range(int(config["runtime"]["screen_shards_per_task"]))
    )
    searches = sum(
        _completion(results / "search" / task / f"length_{length:02d}" / f"restart_{restart:02d}" / "job_complete")
        for task in TASKS
        for length in range(2, 31)
        for restart in range(int(config["search"]["restarts_per_length"]))
    )
    merges = sum(
        _completion(results / "search" / task / f"length_{length:02d}" / "merged")
        for task in TASKS
        for length in range(2, 31)
    )
    validations = sum(
        _completion(results / "validation" / task / f"length_{length:02d}")
        for task in TASKS
        for length in range(1, 31)
    )
    formal_lengths = sum(
        (results / "search" / task / f"length_{length:02d}" / ("formal_archive.json" if length == 1 else "merged/formal_archive.json")).is_file()
        for task in TASKS
        for length in range(1, 31)
    )
    sealed = json.loads((results / "sealed_state.json").read_text(encoding="utf-8"))
    contract = json.loads((results / "run_contract.json").read_text(encoding="utf-8"))
    dirty_commit = contract["run_code_commit"] != git(root, "rev-parse", "HEAD")
    trajectory_generations = len(list(results.glob("search/*/length_*/restart_*/generation_*/COMPLETE.json")))
    expected_generations = expected_search * int(config["search"]["iterations"])
    snapshots = len(list(results.glob("search/*/length_*/restart_*/generation_*/snapshots/*/high_dimensional.npz")))
    pngs = len(list(results.glob("search/*/length_*/restart_*/generation_*/snapshots/*/cluster.png")))
    gifs = len(list(results.glob("search/*/length_*/restart_*/optimization.gif")))
    mp4s = len(list(results.glob("search/*/length_*/restart_*/optimization.mp4")))
    counts = {
        "screen": [screen, expected_screen],
        "search_restart": [searches, expected_search],
        "search_merge": [merges, expected_merge],
        "formal_length": [formal_lengths, 150],
        "validation": [validations, expected_validation],
        "trajectory_generation": [trajectory_generations, expected_generations],
        "gif": [gifs, expected_search],
        "mp4": [mp4s, expected_search],
    }
    errors = [f"{name}={actual}/{expected}" for name, (actual, expected) in counts.items() if actual != expected]
    if snapshots == 0 or pngs != snapshots:
        errors.append(f"snapshot artifact mismatch png={pngs} high_dimensional={snapshots}")
    if not contract.get("test_and_ood_embeddings_absent"):
        errors.append("test/OOD was not sealed at formal registration")
    if dirty_commit:
        errors.append("result contract commit does not match audited checkout")
    for phase in ("prepare", "calibration", "frozen", "test", "ood", "downstream", "finalize"):
        if not _completion(results / phase):
            errors.append(f"incomplete phase: {phase}")
    return {
        "counts": counts,
        "snapshots": snapshots,
        "sealed_state": sealed,
        "run_contract": contract,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/v5_mode3.yaml"))
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-incomplete-results", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = audit_source(root, config)
    scope = audit_scope(root, str(config["main_baseline_commit"]), int(config["publication"]["ordinary_git_maximum_object_bytes"]))
    result: dict[str, Any] = {
        "schema_version": "mode3-v5-scope-audit-v1",
        "source": source,
        "scope": scope,
        "results": None,
    }
    if args.results is not None:
        result["results"] = audit_results(root, args.results.resolve(), config)
    errors = [*source["errors"]]
    errors.extend(f"protected path changed: {path}" for path in scope["protected_changes"])
    errors.extend(f"oversized ordinary Git object: {value}" for value in scope["oversized_tracked_objects"])
    if result["results"] is not None and not args.allow_incomplete_results:
        errors.extend(result["results"]["errors"])
    result["errors"] = errors
    result["passed"] = not errors
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
