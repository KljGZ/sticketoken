"""Static scope and optional result-completeness audit for Mode 3 V4."""

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


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("prefix", "suffix", "random", "universal")
V3_BASELINE = "b78bb21693e87a62287929683f575eb2a1b89be1"
ALLOWED_NEW_PREFIXES = (
    "configs/v4_mode3.yaml",
    "sticky_lab/mode3_v4/",
    "tests/test_mode3_v4_",
    "scripts/run_v4_mode3_remote.sh",
    "scripts/audit_v4_mode3.py",
    "docs/sticky_attractor_v4_",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _allowed_v4_path(path: str, runtime_prefix: str | None = None) -> bool:
    prefixes = ALLOWED_NEW_PREFIXES + ((runtime_prefix,) if runtime_prefix else ())
    return any(path == prefix or path.startswith(prefix) for prefix in prefixes)


def _scope_audit(results: Path | None = None) -> dict[str, Any]:
    runtime_prefix: str | None = None
    if results is not None:
        resolved = results.resolve()
        try:
            runtime_prefix = resolved.relative_to(ROOT.resolve()).as_posix().rstrip("/") + "/"
        except ValueError as exc:
            raise AssertionError(f"V4 results must stay inside the repository: {resolved}") from exc

    committed = subprocess.check_output(
        ["git", "diff", "--name-only", f"{V3_BASELINE}..HEAD"], cwd=ROOT, text=True
    ).splitlines()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=ROOT, text=True
    ).splitlines()
    working_tree = [line[3:].replace("\\", "/") for line in status]
    changed = sorted(set(path.replace("\\", "/") for path in committed + working_tree))
    invalid = [path for path in changed if not _allowed_v4_path(path, runtime_prefix)]
    if invalid:
        raise AssertionError(f"V4 scope changed pre-existing or non-V4 paths: {invalid}")
    return {
        "baseline_commit": V3_BASELINE,
        "committed_paths": sorted(committed),
        "working_tree_paths": sorted(working_tree),
        "runtime_results_prefix": runtime_prefix,
        "scope_valid": True,
    }


def _ast_audit() -> dict[str, Any]:
    imports: dict[str, list[str]] = {}
    violations: list[str] = []
    for path in sorted((ROOT / "sticky_lab" / "mode3_v4").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"backward", "grad", "get_input_embeddings"}:
                    violations.append(f"{path.name}:{node.lineno}:{node.func.attr}")
        imports[path.name] = names
        for name in names:
            if "mode3_v3" in name or name.endswith((".v2", ".search", ".tokens")):
                violations.append(f"{path.name}: forbidden legacy import {name}")
            if (name == "torch" or name.startswith("torch.")) and path.name != "oracle.py":
                violations.append(f"{path.name}: torch outside oracle boundary")
            if name.startswith("sentence_transformers") and path.name != "oracle.py":
                violations.append(f"{path.name}: encoder runtime outside oracle boundary")
            if name.startswith("transformers") and path.name != "tokenizer_audit.py":
                violations.append(f"{path.name}: tokenizer runtime outside audit boundary")
    if violations:
        raise AssertionError(f"V4 threat-model AST violations: {violations}")
    return {"imports": imports, "violations": []}


def _results_audit(results: Path) -> dict[str, Any]:
    missing: list[str] = []
    for task in TASKS:
        screen = results / "screen" / task / "all_legal_single_tokens.csv"
        if not screen.exists():
            missing.append(str(screen))
        for length in range(2, 31):
            for restart in range(2):
                archive = results / "search" / task / f"length_{length:02d}" / f"restart_{restart:02d}_archive.csv"
                if not archive.exists():
                    missing.append(str(archive))
        for length in range(1, 31):
            summary = results / "validation" / task / f"length_{length:02d}" / "summary.json"
            if not summary.exists():
                missing.append(str(summary))
    if missing:
        raise AssertionError(f"V4 result grid is incomplete ({len(missing)} missing), first: {missing[:8]}")
    with (results / "length_frontier.csv").open("r", encoding="utf-8", newline="") as handle:
        frontier = list(csv.DictReader(handle))
    if len(frontier) != 120 or {int(row["actual_token_length"]) for row in frontier} != set(range(1, 31)):
        raise AssertionError("V4 frontier is not the complete 4 x 30 registered grid")
    with (results / "sha256_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    bad_hashes = []
    for row in manifest:
        path = results / str(row["path"])
        if not path.exists() or _sha256(path) != str(row["sha256"]):
            bad_hashes.append(str(row["path"]))
    if bad_hashes:
        raise AssertionError(f"V4 SHA-256 manifest mismatches: {bad_hashes[:8]}")
    status = json.loads((results / "final_status.json").read_text(encoding="utf-8"))
    if status.get("encoder_attractor_discovered"):
        test = json.loads((results / "test" / "one_time_test.json").read_text(encoding="utf-8"))
        if test.get("center_refit") or test.get("radius_refit"):
            raise AssertionError("V4 test illegally refit its frozen region")
    return {"summary_count": 120, "frontier_rows": len(frontier), "manifest_rows": len(manifest), "final_status": status}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v4_mode3.yaml")
    parser.add_argument("--results")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["lengths"] != {
        "exhaustive_single_token": True,
        "minimum": 1,
        "maximum": 30,
        "step": 1,
        "stop_search_after_first_certified": False,
        "test_only_shortest_validation_certified": True,
    }:
        raise AssertionError("V4 registered complete length schedule changed")
    results_path: Path | None = None
    if args.results:
        results_path = Path(args.results)
        if not results_path.is_absolute():
            results_path = ROOT / results_path
    report: dict[str, Any] = {
        "scope": _scope_audit(results_path),
        "ast": _ast_audit(),
        "config": str(config_path),
    }
    if results_path is not None:
        report["results"] = _results_audit(results_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
