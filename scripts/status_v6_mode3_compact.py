#!/usr/bin/env python3
"""Read-only status snapshot for V6 Compact."""

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
                "mode3_v6_compact" in cmd or "run_v6_mode3_compact_remote.sh" in cmd
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
    stage_names = [
        "registration",
        "enumeration",
        "base_embeddings",
        "tracks",
        "s0",
        "s1",
        "s2",
        "s3",
        "validation",
        "sealed",
        "semantic-controls",
        "mechanism",
        "retrieval",
    ]
    files = [path for path in root.rglob("*") if path.is_file()] if root.is_dir() else []
    status = {
        "schema_version": "mode3-v6-compact-status-v1",
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output": str(root.resolve()),
        "orchestrator": load(root / "orchestration_logs" / "status.json"),
        "run_contract": load(root / "registration" / "run_contract.json"),
        "budget": load(root / "budget" / "observed.json"),
        "stages": {name: (root / name / "COMPLETE.json").is_file() for name in stage_names},
        "shards": {
            "s0": count_complete(root, "s0/shard_*/COMPLETE.json"),
            "s1": count_complete(root, "funnel/s1/shard_*/COMPLETE.json"),
            "s2": count_complete(root, "funnel/s2/shard_*/COMPLETE.json"),
            "s3": count_complete(root, "funnel/s3/shard_*/COMPLETE.json"),
            "validation": count_complete(root, "validation/shard_*/COMPLETE.json"),
            "blackbox_restarts": count_complete(root, "tracks/blackbox/restart_*/COMPLETE.json"),
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
