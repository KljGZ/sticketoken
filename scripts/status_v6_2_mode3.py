#!/usr/bin/env python3
"""Read-only status snapshot for V6.2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any


def load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def count_complete(root: Path, pattern: str) -> int:
    return sum(1 for path in root.glob(pattern) if path.is_file())


def matching_processes() -> list[dict[str, Any]]:
    values = []
    for entry in Path("/proc").iterdir() if Path("/proc").is_dir() else []:
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            status = {}
            for line in (entry / "status").read_text(errors="replace").splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    status[key] = value.strip()
            uid = int(status.get("Uid", "-1").split()[0])
            if uid == os.getuid() and (
                "mode3_v6_2" in cmd or "run_v6_2_mode3_remote.sh" in cmd
            ):
                pid = int(entry.name)
                values.append(
                    {
                        "pid": pid,
                        "ppid": int(status.get("PPid", "0")),
                        "pgid": os.getpgid(pid),
                        "sid": os.getsid(pid),
                        "state": status.get("State"),
                        "cmdline": cmd,
                    }
                )
        except (OSError, PermissionError, ValueError):
            continue
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--unit")
    args = parser.parse_args()
    root = args.output
    stage_paths = {
        "registration": root / "registration" / "run_contract.json",
        "enumeration": root / "enumeration" / "COMPLETE.json",
        "s0": root / "funnel" / "s0" / "COMPLETE.json",
        "s1": root / "funnel" / "s1" / "COMPLETE.json",
        "s2": root / "funnel" / "s2" / "COMPLETE.json",
        "full": root / "funnel" / "full" / "COMPLETE.json",
        "stability": root / "funnel" / "stability" / "COMPLETE.json",
        "semantic": root / "semantic" / "COMPLETE.json",
        "freezes": root / "freezes" / "COMPLETE.json",
        "confirmation": root / "confirmation" / "COMPLETE.json",
        "semantic_confirmation": root / "semantic_confirmation" / "COMPLETE.json",
        "sealed_followups": root / "sealed_followups" / "COMPLETE.json",
        "retrieval": root / "retrieval" / "COMPLETE.json",
        "final": root / "FINAL_STATUS.json",
    }
    files = [path for path in root.rglob("*") if path.is_file()] if root.is_dir() else []
    status = {
        "schema_version": "mode3-v6-2-status-v1",
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output": str(root.resolve()),
        "orchestrator": load(root / "orchestration_logs" / "status.json"),
        "run_contract": load(root / "registration" / "run_contract.json"),
        "budget": load(root / "budget" / "observed.json"),
        "stages": {name: path.is_file() for name, path in stage_paths.items()},
        "shards": {
            "s0": count_complete(root, "s0/shard_*/COMPLETE.json"),
            "s1": count_complete(root, "funnel/s1/shard_*/COMPLETE.json"),
            "s2": count_complete(root, "funnel/s2/shard_*/COMPLETE.json"),
            "full": count_complete(root, "funnel/full/shard_*/COMPLETE.json"),
            "stability": count_complete(root, "funnel/stability/shard_*/COMPLETE.json"),
            "semantic": count_complete(root, "semantic/shard_*/COMPLETE.json"),
            "protocol_selection": count_complete(root, "protocol_selection/shard_*/COMPLETE.json"),
        },
        "processes": matching_processes(),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "latest_files": [
            {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in sorted(files, key=lambda item: item.stat().st_mtime_ns)[-10:]
        ],
        "zero_byte_critical": [
            str(path.relative_to(root))
            for path in files
            if path.stat().st_size == 0 and path.name not in {".orchestrator.lock", ".budget.lock"}
        ],
        "disk": {
            str(path): {"total": usage.total, "used": usage.used, "free": usage.free}
            for path in (Path("/"), Path("/mnt/data"))
            if path.exists()
            for usage in [shutil.disk_usage(path)]
        },
    }
    if args.unit:
        process = subprocess.run(
            ["systemctl", "--user", "show", args.unit, "--no-pager", "--property=ActiveState,SubState,MainPID,ExecMainStatus"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        status["systemd_unit"] = {"name": args.unit, "returncode": process.returncode, "output": process.stdout}
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
