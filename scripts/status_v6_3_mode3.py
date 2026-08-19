#!/usr/bin/env python3
"""Read-only V6.3 formal-run snapshot for monitoring and recovery decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import yaml


DEFAULT_OUTPUT = Path(
    "/mnt/data/jkl/StickyToken-v6-3-results/sticky_lab/sentence_t5_base/mode3_v6_3_light"
)
DEFAULT_ROOT = Path("/mnt/data/jkl/StickyToken-v6-3-light-formal")


def load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(arguments: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    process = subprocess.run(
        arguments, cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    return {"returncode": process.returncode, "output": process.stdout.strip()}


def gpu_status(
    allowed_physical_gpus: set[int], forbidden_physical_gpus: set[int]
) -> dict[str, Any]:
    query = command([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    rows = []
    if query["returncode"] == 0:
        for line in query["output"].splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 6:
                rows.append({
                    "index": int(parts[0]), "name": parts[1],
                    "memory_total_mib": int(parts[2]), "memory_used_mib": int(parts[3]),
                    "memory_free_mib": int(parts[4]), "utilization_percent": int(parts[5]),
                    "v6_3_authorized": int(parts[0]) in allowed_physical_gpus,
                    "v6_3_forbidden": int(parts[0]) in forbidden_physical_gpus,
                })
    processes = command([
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
        "--format=csv,noheader,nounits",
    ])
    return {"devices": rows, "compute_processes": processes, "query": query}


def matching_processes() -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return values
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = entry.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            ).strip()
            if "mode3_v6_3" not in cmdline and "run_v6_3" not in cmdline:
                continue
            status = {}
            for line in entry.joinpath("status").read_text(errors="replace").splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    status[key] = value.strip()
            if int(status.get("Uid", "-1").split()[0]) != os.getuid():
                continue
            pid = int(entry.name)
            values.append({
                "pid": pid, "ppid": int(status.get("PPid", "0")),
                "state": status.get("State"), "cmdline": cmdline,
                "cwd": str(entry.joinpath("cwd").resolve()),
                "cuda_visible_devices": entry.joinpath("environ").read_bytes().split(
                    b"CUDA_VISIBLE_DEVICES=", 1
                )[1].split(b"\0", 1)[0].decode() if b"CUDA_VISIBLE_DEVICES=" in entry.joinpath("environ").read_bytes() else None,
            })
        except (OSError, PermissionError, ValueError):
            continue
    return sorted(values, key=lambda row: int(row["pid"]))


def tail(path: Path, lines: int = 80) -> list[str]:
    try:
        values = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return values[-int(lines):]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config", type=Path, default=Path("configs/v6_3_mode3_light.yaml"))
    parser.add_argument("--unit", default="sticky-v6-3-light")
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-config-sha256")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    try:
        source_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        resources = source_config["resources"]
        allowed_physical_gpus = set(map(int, resources["allowed_physical_gpus"]))
        forbidden_physical_gpus = set(
            map(int, resources["forbidden_physical_gpus"])
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
        allowed_physical_gpus = set()
        forbidden_physical_gpus = set()
    files = [path for path in output.rglob("*") if path.is_file()] if output.is_dir() else []
    stages = {}
    latest_complete: dict[str, Any] | None = None
    for stage in ("s0", "s1", "s2", "full", "top100"):
        stage_root = output / "stages" / stage
        completes = list(stage_root.glob("shard_*/COMPLETE.json"))
        failures = list(stage_root.glob("shard_*/FAILED.json"))
        stages[stage] = {
            "complete": (stage_root / "COMPLETE.json").is_file(),
            "completed_shards": len(completes), "failed_shards": len(failures),
            "summary": load(stage_root / "COMPLETE.json"),
        }
        for path in completes:
            if latest_complete is None or path.stat().st_mtime_ns > latest_complete["mtime_ns"]:
                latest_complete = {
                    "path": path.relative_to(output).as_posix(),
                    "mtime_ns": path.stat().st_mtime_ns,
                    "content": load(path),
                }
    orchestrator = load(output / "orchestration_logs" / "status.json")
    current_stage = str((orchestrator or {}).get("stage", "unknown"))
    log_candidates = sorted(
        (output / "orchestration_logs").glob(f"{current_stage}*.log"),
        key=lambda path: path.stat().st_mtime_ns,
    ) if (output / "orchestration_logs").is_dir() else []
    log_tail = tail(log_candidates[-1]) if log_candidates else []
    errors = [
        line for line in log_tail
        if any(term in line for term in ("Traceback", "ERROR", "Error", "Exception", "FAILED"))
    ]
    run_commit = command(["git", "rev-parse", "HEAD"], cwd=root) if root.is_dir() else None
    observed_commit = run_commit["output"] if run_commit and run_commit["returncode"] == 0 else None
    observed_config = sha256_file(config_path)
    expected_commit = args.expected_commit or observed_commit
    expected_config = args.expected_config_sha256 or observed_config
    systemd = command([
        "systemctl", "--user", "show", args.unit, "--no-pager",
        "--property=ActiveState,SubState,MainPID,ExecMainStatus,Result,ExecStart",
    ])
    journal = command([
        "journalctl", "--user", "-u", args.unit, "-n", "80", "--no-pager",
        "--output=short-iso",
    ])
    status = {
        "schema_version": "mode3-v6-3-monitor-status-v1",
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(root), "output": str(output),
        "orchestrator": orchestrator,
        "run_manifest": load(output / "run_manifest.json"),
        "budget": load(output / "budget" / "observed.json"),
        "planned_budget": load(output / "budget" / "planned_actual_vocab.json"),
        "stages": stages,
        "latest_shard_complete": latest_complete,
        "freeze": load(output / "freeze" / "COMPLETE.json"),
        "confirm": load(output / "confirm" / "COMPLETE.json"),
        "followups": load(output / "followups" / "COMPLETE.json"),
        "final": load(output / "FINAL_STATUS.json"),
        "systemd": systemd, "journal": journal,
        "matching_processes": matching_processes(),
        "gpus": gpu_status(allowed_physical_gpus, forbidden_physical_gpus),
        "disk": {
            str(path): shutil.disk_usage(path)._asdict()
            for path in (Path("/"), Path("/mnt/data")) if path.exists()
        },
        "identity": {
            "observed_commit": observed_commit,
            "expected_commit": expected_commit,
            "commit_matches": observed_commit == expected_commit,
            "observed_source_config_sha256": observed_config,
            "expected_source_config_sha256": expected_config,
            "config_matches": observed_config == expected_config,
        },
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "zero_byte_critical": [
            path.relative_to(output).as_posix() for path in files
            if path.stat().st_size == 0
            and path.name not in {".orchestrator.lock", ".ledger.lock"}
        ],
        "failed_artifacts": [
            path.relative_to(output).as_posix() for path in files if path.name == "FAILED.json"
        ],
        "current_log": str(log_candidates[-1]) if log_candidates else None,
        "current_log_tail": log_tail,
        "error_lines": errors,
    }
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
