#!/usr/bin/env python3
"""Read-only V7 formal-run snapshot for monitoring and recovery decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Sequence

import yaml


DEFAULT_ROOT = Path("/mnt/data/jkl/StickyToken-v7-occupancy-frontier")
DEFAULT_OUTPUT = Path(
    "/mnt/data/jkl/StickyToken-v7-results/sticky_lab/sentence_t5_base/"
    "mode3_v7_occupancy_frontier_r2_10g"
)
DEFAULT_CONFIG = Path("configs/v7_mode3_occupancy_frontier.yaml")
DEFAULT_UNIT = "sticky-v7-occupancy-frontier.service"


def load_json(path: Path) -> Any:
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


def command(arguments: Sequence[str], *, cwd: Path | None = None) -> dict[str, Any]:
    try:
        process = subprocess.run(
            list(arguments),
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except OSError as error:
        return {"returncode": 127, "output": f"{type(error).__name__}: {error}"}
    return {"returncode": process.returncode, "output": process.stdout.strip()}


def tail(path: Path, lines: int = 80) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return []


def matching_processes(root: Path, output: Path) -> list[dict[str, Any]]:
    snapshot = command(["ps", "-eo", "pid=,etimes=,args="])
    rows: list[dict[str, Any]] = []
    if snapshot["returncode"]:
        return rows
    needles = (str(root), str(output), "sticky_lab.mode3_v7", "run_v7_orchestrator.py")
    for line in snapshot["output"].splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) != 3 or not any(needle in fields[2] for needle in needles):
            continue
        if "status_v7_mode3.py" in fields[2]:
            continue
        rows.append(
            {"pid": int(fields[0]), "elapsed_seconds": int(fields[1]), "args": fields[2]}
        )
    return rows


def gpu_status(allowed: set[int], forbidden: set[int]) -> dict[str, Any]:
    query = command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    devices: list[dict[str, Any]] = []
    if query["returncode"] == 0:
        for line in query["output"].splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 6:
                continue
            index = int(parts[0])
            devices.append(
                {
                    "index": index,
                    "name": parts[1],
                    "memory_total_mib": int(parts[2]),
                    "memory_used_mib": int(parts[3]),
                    "memory_free_mib": int(parts[4]),
                    "utilization_percent": int(parts[5]),
                    "v7_authorized": index in allowed,
                    "v7_forbidden": index in forbidden,
                }
            )
    processes = command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ]
    )
    return {"devices": devices, "compute_processes": processes, "query": query}


def latest_file(paths: Sequence[Path]) -> dict[str, Any] | None:
    files = [path for path in paths if path.is_file()]
    if not files:
        return None
    path = max(files, key=lambda item: item.stat().st_mtime_ns)
    return {
        "path": str(path),
        "mtime_ns": path.stat().st_mtime_ns,
        "age_seconds": max(0.0, time.time() - path.stat().st_mtime),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--unit", default=DEFAULT_UNIT)
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-config-sha256")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError:
        config = {}
    resources = config.get("resources", {}) if isinstance(config, dict) else {}
    allowed = set(map(int, resources.get("allowed_physical_gpus", [4, 5, 6, 7])))
    forbidden = set(map(int, resources.get("forbidden_physical_gpus", [0, 1, 2, 3])))

    full = output / "stages" / "full"
    complete_shards = sorted(full.glob("shard_*/COMPLETE.json")) if full.is_dir() else []
    failed_shards = sorted(full.glob("shard_*/FAILED.json")) if full.is_dir() else []
    stages = {
        "registration": (output / "registration" / "COMPLETE.json").is_file(),
        "s0_reuse": (output / "stages" / "s0" / "COMPLETE.json").is_file(),
        "discovery_clean": (
            output / "registration" / "DISCOVERY_CLEAN_COMPLETE.json"
        ).is_file(),
        "full": {
            "complete": (full / "COMPLETE.json").is_file(),
            "completed_shards": len(complete_shards),
            "failed_shards": len(failed_shards),
            "expected_shards": int(config.get("funnel", {}).get("shards", 32))
            if isinstance(config, dict)
            else 32,
            "summary": load_json(full / "COMPLETE.json"),
        },
        "diagnostics": (
            output / "diagnostics" / "post_selection" / "COMPLETE.json"
        ).is_file(),
        "cache_compaction": (output / "cache_compaction" / "COMPLETE.json").is_file(),
        "freeze": (output / "freeze" / "COMPLETE.json").is_file(),
        "confirm": (output / "confirm" / "COMPLETE.json").is_file(),
    }
    orchestrator = load_json(output / "orchestration_logs" / "status.json")
    current_stage = str((orchestrator or {}).get("stage", "unknown"))
    log_root = output / "orchestration_logs"
    logs = (
        sorted(log_root.glob(f"{current_stage}*.log"), key=lambda path: path.stat().st_mtime_ns)
        if log_root.is_dir()
        else []
    )
    current_log = logs[-1] if logs else None
    current_tail = tail(current_log) if current_log else []
    error_lines = [
        line
        for line in current_tail
        if any(term in line for term in ("Traceback", "ERROR", "Error", "Exception", "FAILED"))
    ]
    final = load_json(output / "V7_FINAL_STATUS.json")
    run_manifest = load_json(output / "run_manifest.json")
    observed_commit_result = command(["git", "rev-parse", "HEAD"], cwd=root)
    observed_commit = (
        observed_commit_result["output"] if observed_commit_result["returncode"] == 0 else None
    )
    observed_config = sha256_file(config_path)
    expected_commit = args.expected_commit or observed_commit
    expected_config = args.expected_config_sha256 or observed_config
    identity = {
        "observed_commit": observed_commit,
        "expected_commit": expected_commit,
        "commit_matches": observed_commit == expected_commit,
        "observed_source_config_sha256": observed_config,
        "expected_source_config_sha256": expected_config,
        "config_matches": observed_config == expected_config,
        "registered_commit_matches": run_manifest is None
        or run_manifest.get("code_commit") == observed_commit,
        "registered_source_config_matches": run_manifest is None
        or run_manifest.get("source_config_file_sha256") == observed_config,
    }
    systemd = command(
        [
            "systemctl",
            "--user",
            "show",
            args.unit,
            "--no-pager",
            "--property=ActiveState,SubState,MainPID,ExecMainStatus,Result,ExecStart",
        ]
    )
    journal = command(
        [
            "journalctl",
            "--user",
            "-u",
            args.unit,
            "-n",
            "80",
            "--no-pager",
            "--output=short-iso",
        ]
    )
    peer_output = Path(
        str(resources.get("priority_peer_output", ""))
    ) if resources else Path(".")
    peer = {
        "output": str(peer_output),
        "final": load_json(peer_output / "FINAL_STATUS.json") if resources else None,
        "full_complete": (peer_output / "stages" / "full" / "COMPLETE.json").is_file()
        if resources
        else False,
        "confirm_complete": (peer_output / "confirm" / "COMPLETE.json").is_file()
        if resources
        else False,
    }
    activity_candidates = [output / "orchestration_logs" / "status.json"] + complete_shards
    activity_candidates += [
        output / "registration" / "COMPLETE.json",
        output / "registration" / "DISCOVERY_CLEAN_COMPLETE.json",
        full / "COMPLETE.json",
        output / "freeze" / "COMPLETE.json",
        output / "confirm" / "COMPLETE.json",
        output / "V7_FINAL_STATUS.json",
    ]
    latest = latest_file(activity_candidates)
    failed_artifacts = [
        path.relative_to(output).as_posix()
        for path in failed_shards
        if output in path.parents
    ]
    terminal = bool(isinstance(final, dict) and final.get("terminal"))
    identity_ok = all(
        identity[key]
        for key in (
            "commit_matches",
            "config_matches",
            "registered_commit_matches",
            "registered_source_config_matches",
        )
    )
    orchestrator_state = str((orchestrator or {}).get("state", "unknown"))
    if terminal:
        health = "terminal"
        recommendation = "stop_service_and_pause_monitor"
    elif failed_artifacts or orchestrator_state == "failed":
        health = "failed"
        recommendation = "inspect_failure_without_deleting_scientific_artifacts"
    elif not identity_ok:
        health = "blocked_identity_drift"
        recommendation = "do_not_restart_until_identity_is_reconciled"
    elif orchestrator_state in {"waiting_priority_peer", "waiting_gpu"}:
        health = "healthy_waiting"
        recommendation = "continue_monitoring"
    elif "ActiveState=active" in systemd["output"]:
        health = "running"
        recommendation = "continue_monitoring"
    else:
        health = "needs_reconcile"
        recommendation = "start_or_restart_service_only_if_preflight_remains_green"

    status = {
        "schema_version": "mode3-v7-monitor-status-v1",
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "health": health,
        "recommended_control": recommendation,
        "terminal": terminal,
        "root": str(root),
        "output": str(output),
        "identity": identity,
        "orchestrator": orchestrator,
        "run_manifest": run_manifest,
        "reuse_audit": load_json(output / "V7_S0_REUSE_AUDIT.json"),
        "budget_plan": load_json(output / "budget" / "planned.json"),
        "budget_observed": load_json(output / "budget" / "observed.json"),
        "stages": stages,
        "freeze": load_json(output / "freeze" / "COMPLETE.json"),
        "confirm": load_json(output / "confirm" / "COMPLETE.json"),
        "final": final,
        "priority_peer": peer,
        "latest_activity": latest,
        "current_log": str(current_log) if current_log else None,
        "current_log_tail": current_tail,
        "error_lines": error_lines,
        "failed_artifacts": failed_artifacts,
        "matching_processes": matching_processes(root, output),
        "systemd": systemd,
        "journal": journal,
        "gpus": gpu_status(allowed, forbidden),
        "disk": {
            str(path): shutil.disk_usage(path)._asdict()
            for path in (Path("/"), Path("/mnt/data"))
            if path.exists()
        },
    }
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
