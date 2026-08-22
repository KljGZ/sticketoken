#!/usr/bin/env python3
"""Durable four-GPU orchestrator for StickyToken V7."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import yaml

try:  # Linux formal runtime; guarded so local Windows tests can import the module.
    import fcntl
except ImportError:  # pragma: no cover - exercised only by Windows orchestration
    fcntl = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def gpu_snapshot() -> dict[int, dict[str, int]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"nvidia-smi failed: {completed.stderr.strip()}")
    result: dict[int, dict[str, int]] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        index, free, used, utilization = map(int, fields)
        result[index] = {
            "memory_free_mib": free,
            "memory_used_mib": used,
            "utilization_percent": utilization,
        }
    return result


class Orchestrator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output = Path(args.output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.logs = self.output / "orchestration_logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.config_path = Path(args.config).resolve()
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.gpus = [int(value) for value in str(args.gpus).split(",") if value.strip()]
        allowed = list(map(int, self.config["resources"]["allowed_physical_gpus"]))
        forbidden = set(map(int, self.config["resources"]["forbidden_physical_gpus"]))
        exact_formal = self.args.profile == "formal" and self.gpus == allowed
        safe_nonformal = (
            self.args.profile != "formal"
            and bool(self.gpus)
            and len(self.gpus) == len(set(self.gpus))
            and set(self.gpus).issubset(set(allowed))
        )
        if (not exact_formal and not safe_nonformal) or set(self.gpus).intersection(forbidden):
            raise RuntimeError(
                f"V7 orchestrator requires exact GPU list {allowed}, observed {self.gpus}"
            )
        self.python = str(Path(args.python).resolve())
        self.stage = "starting"
        self.state = "starting"
        self.error: str | None = None
        self.lock_handle: Any = None
        self.last_snapshot: dict[int, dict[str, int]] = {}
        self.peer_output = Path(str(self.config["resources"]["priority_peer_output"]))
        self.minimum_free = int(
            self.config["resources"]["gpu_start_minimum_free_memory_mib"]
        )
        self.poll = float(self.config["resources"]["gpu_poll_interval_seconds"])
        self.storage_required = int(
            float(self.config["resources"]["minimum_free_disk_peak_multiplier"])
            * int(self.config["resources"]["estimated_peak_cache_bytes"])
        )
        self.storage_free = 0

    def acquire_lock(self) -> None:
        path = self.output / ".orchestrator.lock"
        self.lock_handle = path.open("a+b")
        if fcntl is not None:
            try:
                fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("another V7 orchestrator owns this output") from error
        else:  # pragma: no cover - local Windows engineering runs only
            import msvcrt

            if path.stat().st_size == 0:
                self.lock_handle.write(b"\0")
                self.lock_handle.flush()
            self.lock_handle.seek(0)
            try:
                msvcrt.locking(self.lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("another V7 orchestrator owns this output") from error
        (self.output / ".orchestrator.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")

    def progress(self) -> dict[str, Any]:
        full = self.output / "stages" / "full"
        return {
            "registration": (self.output / "registration" / "COMPLETE.json").is_file(),
            "s0_reuse": (self.output / "stages" / "s0" / "COMPLETE.json").is_file(),
            "discovery_clean": (
                self.output / "registration" / "DISCOVERY_CLEAN_COMPLETE.json"
            ).is_file(),
            "full_complete_shards": len(list(full.glob("shard_*/COMPLETE.json"))),
            "full_failed_shards": len(list(full.glob("shard_*/FAILED.json"))),
            "full": (full / "COMPLETE.json").is_file(),
            "diagnostics": (
                self.output / "diagnostics" / "post_selection" / "COMPLETE.json"
            ).is_file(),
            "cache_compaction": (
                self.output / "cache_compaction" / "COMPLETE.json"
            ).is_file(),
            "freeze": (self.output / "freeze" / "COMPLETE.json").is_file(),
            "confirm": (self.output / "confirm" / "COMPLETE.json").is_file(),
        }

    def status(self, state: str | None = None, error: str | None = None) -> None:
        try:
            self.storage_free = shutil.disk_usage(self.output.parent).free
        except OSError:
            self.storage_free = 0
        if state is not None:
            self.state = str(state)
        if error is not None:
            self.error = str(error)
        value: dict[str, Any] = {
            "schema_version": "mode3-v7-orchestrator-status-v1",
            "state": self.state,
            "stage": self.stage,
            "pid": os.getpid(),
            "profile": self.args.profile,
            "authorized_physical_gpus": self.gpus,
            "forbidden_physical_gpus": list(
                self.config["resources"]["forbidden_physical_gpus"]
            ),
            "gpu_snapshot": self.last_snapshot,
            "storage": {
                "free_bytes": self.storage_free,
                "required_before_model_work_bytes": self.storage_required,
                "ready": self.storage_free >= self.storage_required,
            },
            "priority_peer_output": str(self.peer_output),
            "priority_peer_terminal": self.peer_terminal(),
            "progress": self.progress(),
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if self.error:
            value["error"] = self.error
        atomic_json(self.logs / "status.json", value)

    def peer_terminal(self) -> bool:
        if not bool(self.config["resources"].get("wait_for_v6_3_r5", True)):
            return True
        final = load_json(self.peer_output / "FINAL_STATUS.json")
        if final is None or not final.get("status"):
            return False
        full_complete = (self.peer_output / "stages" / "full" / "COMPLETE.json").is_file()
        # r5 can terminate before freeze only through a valid registered negative endpoint.
        valid_early = str(final["status"]).startswith(("VALID_", "INCONCLUSIVE_", "INVALID_"))
        confirm_complete = (self.peer_output / "confirm" / "COMPLETE.json").is_file()
        return bool(full_complete and (confirm_complete or valid_early))

    def wait_for_peer(self) -> None:
        self.stage = "waiting_for_v6_3_r5"
        while not self.peer_terminal():
            self.status("waiting_priority_peer")
            time.sleep(self.poll)
        self.status("running")

    def cli(self, *arguments: str) -> list[str]:
        return [
            self.python,
            "-m",
            "sticky_lab.mode3_v7.cli",
            "--config",
            str(self.config_path),
            "--output",
            str(self.output),
            "--profile",
            str(self.args.profile),
            *arguments,
        ]

    def environment(self, gpu: int | None = None) -> dict[str, str]:
        value = os.environ.copy()
        value.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "CUDA_VISIBLE_DEVICES": "" if gpu is None else str(int(gpu)),
            }
        )
        return value

    def run(self, name: str, command: Sequence[str], gpu: int | None = None) -> None:
        log = self.logs / f"{name}.log"
        with log.open("ab") as handle:
            handle.write(
                (
                    f"\n[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] "
                    + " ".join(map(str, command))
                    + "\n"
                ).encode("utf-8")
            )
            handle.flush()
            completed = subprocess.run(
                list(command),
                cwd=ROOT,
                env=self.environment(gpu),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode:
            raise RuntimeError(f"{name} exited {completed.returncode}; see {log}")

    def wait_for_gpu(self, excluded: set[int] | None = None) -> int:
        excluded = excluded or set()
        while True:
            self.last_snapshot = gpu_snapshot()
            for gpu in self.gpus:
                if gpu in excluded:
                    continue
                row = self.last_snapshot.get(gpu, {})
                if int(row.get("memory_free_mib", 0)) >= self.minimum_free:
                    self.status("running")
                    return gpu
            self.status("waiting_gpu")
            time.sleep(self.poll)

    def wait_for_storage(self) -> None:
        self.stage = "waiting_for_storage"
        while True:
            self.storage_free = shutil.disk_usage(self.output.parent).free
            if self.storage_free >= self.storage_required:
                self.status("running")
                return
            self.status("waiting_storage")
            time.sleep(self.poll)

    def run_full_shards(self) -> None:
        shards = int(self.config["funnel"]["shards"])
        pending = [
            shard
            for shard in range(shards)
            if not (
                self.output / "stages" / "full" / f"shard_{shard:02d}" / "COMPLETE.json"
            ).is_file()
        ]
        running: dict[int, tuple[int, subprocess.Popen[bytes], Any, Path]] = {}
        while pending or running:
            finished: list[int] = []
            for gpu, (shard, process, handle, log) in running.items():
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                if code:
                    raise RuntimeError(
                        f"V7 FULL shard {shard} failed on physical GPU {gpu}; see {log}"
                    )
                finished.append(gpu)
            for gpu in finished:
                del running[gpu]
            while pending and len(running) < len(self.gpus):
                gpu = self.wait_for_gpu(set(running))
                shard = pending.pop(0)
                log = self.logs / f"full_shard_{shard:02d}.log"
                handle = log.open("ab")
                command = self.cli(
                    "stage-shard",
                    "--stage",
                    "full",
                    "--shard",
                    str(shard),
                    "--shards",
                    str(shards),
                    "--physical-gpu",
                    str(gpu),
                )
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=self.environment(gpu),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                running[gpu] = (shard, process, handle, log)
            self.status("running")
            if pending or running:
                time.sleep(self.poll)

    def execute(self) -> None:
        self.acquire_lock()
        self.stage = "prepare"
        self.status("running")
        self.run("prepare", self.cli("prepare"))
        self.stage = "reuse_s0"
        self.run("reuse_s0", self.cli("reuse-s0"))
        self.wait_for_peer()
        self.wait_for_storage()
        self.stage = "precompute_clean"
        gpu = self.wait_for_gpu()
        self.run(
            "precompute_clean",
            self.cli("precompute-clean", "--physical-gpu", str(gpu)),
            gpu,
        )
        self.stage = "full"
        self.run_full_shards()
        self.stage = "merge_full"
        self.run(
            "merge_full",
            self.cli(
                "merge-stage",
                "--stage",
                "full",
                "--shards",
                str(self.config["funnel"]["shards"]),
            ),
        )
        final = load_json(self.output / "V7_FINAL_STATUS.json")
        if final and final.get("terminal"):
            self.stage = "terminal_valid_negative"
            self.status("completed")
            return
        self.stage = "post_selection_diagnostics"
        self.run("post_selection_diagnostics", self.cli("diagnostics"))
        self.stage = "compact_nonselected_cache"
        self.run("compact_nonselected_cache", self.cli("compact-cache"))
        self.stage = "freeze"
        self.run("freeze", self.cli("freeze"))
        self.stage = "grant_confirm"
        self.run("grant_confirm", self.cli("grant-confirm"))
        self.stage = "confirm"
        gpu = self.wait_for_gpu()
        self.run(
            "confirm",
            self.cli("confirm", "--physical-gpu", str(gpu)),
            gpu,
        )
        final = load_json(self.output / "V7_FINAL_STATUS.json")
        if not final or not final.get("terminal"):
            raise RuntimeError("V7 confirm returned without an explicit terminal status")
        self.stage = "complete"
        self.status("completed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=("formal", "pilot", "dry_run"), default="formal")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", default="4,5,6,7")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    orchestrator = Orchestrator(args)
    try:
        orchestrator.execute()
    except Exception as error:
        orchestrator.stage = orchestrator.stage or "unknown"
        orchestrator.status("failed", f"{type(error).__name__}: {error}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
