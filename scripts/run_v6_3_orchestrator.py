#!/usr/bin/env python3
"""Fail-closed, marker-resumable V6.3 orchestration on physical GPUs 4-7 only."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sticky_lab.mode3_v6_3.config import (  # noqa: E402
    config_for_profile,
    load_config,
    sha256_file,
)
from sticky_lab.mode3_v6_3.errors import ProtocolViolation  # noqa: E402
from sticky_lab.mode3_v6_3.report import result_inventory  # noqa: E402


AUTHORIZED_GPUS = (4, 5, 6, 7)
FORBIDDEN_GPUS = (0, 1, 2, 3)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise ProtocolViolation(f"JSON marker is not an object: {path}")
    return value


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def gpu_snapshot() -> dict[int, dict[str, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    process = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if process.returncode != 0:
        raise ProtocolViolation(f"nvidia-smi failed: {process.stderr.strip()}")
    values: dict[int, dict[str, int]] = {}
    for line in process.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            raise ProtocolViolation(f"unexpected nvidia-smi row: {line}")
        index, total, used, free, utilization = map(int, parts)
        values[index] = {
            "memory_total_mib": total,
            "memory_used_mib": used,
            "memory_free_mib": free,
            "utilization_percent": utilization,
        }
    return values


def parse_gpus(text: str, *, minimum_free_mib: int) -> tuple[list[int], dict[int, dict[str, int]]]:
    requested = [int(value) for value in str(text).split(",") if value.strip()]
    if not requested or len(requested) != len(set(requested)):
        raise ProtocolViolation("GPU list must be non-empty and unique")
    if any(gpu in FORBIDDEN_GPUS or gpu not in AUTHORIZED_GPUS for gpu in requested):
        raise ProtocolViolation(
            f"V6.3 hard-forbids physical GPUs 0-3 and permits only 4-7: {requested}"
        )
    snapshot = gpu_snapshot()
    missing = [gpu for gpu in requested if gpu not in snapshot]
    if missing:
        raise ProtocolViolation(f"requested physical GPUs are absent: {missing}")
    unsafe = [
        gpu for gpu in requested
        if snapshot[gpu]["memory_free_mib"] < int(minimum_free_mib)
    ]
    if unsafe:
        raise ProtocolViolation(
            f"authorized GPUs lack registered free memory; wait/reduce concurrency: {unsafe}"
        )
    return requested, snapshot


class Orchestrator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output = Path(args.output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.logs = self.output / "orchestration_logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.config_path = Path(args.config).resolve()
        base = load_config(self.config_path)
        self.config = config_for_profile(base, str(args.profile))
        self.gpus, self.initial_gpu_snapshot = parse_gpus(
            args.gpus, minimum_free_mib=int(args.minimum_free_memory_mib)
        )
        if any(gpu not in self.config["resources"]["allowed_physical_gpus"] for gpu in self.gpus):
            raise ProtocolViolation("orchestrator GPU list differs from the resolved protocol")
        self.python = str(Path(args.python).resolve())
        self.commit = git("rev-parse", "HEAD")
        self.config_sha256 = sha256_file(self.config_path)
        self.stage = "starting"
        self.lock_handle: Any = None

    def acquire_lock(self) -> None:
        path = self.output / ".orchestrator.lock"
        self.lock_handle = path.open("a+b")
        try:
            fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ProtocolViolation(f"another V6.3 orchestrator owns {self.output}") from error
        (self.output / ".orchestrator.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")

    def progress(self) -> dict[str, Any]:
        return {
            stage: {
                "complete": (self.output / "stages" / stage / "COMPLETE.json").is_file(),
                "complete_shards": len(list((self.output / "stages" / stage).glob("shard_*/COMPLETE.json"))),
                "failed_shards": len(list((self.output / "stages" / stage).glob("shard_*/FAILED.json"))),
            }
            for stage in ("s0", "s1", "s2", "full", "top100")
        }

    def status(self, state: str, *, exit_code: int = 0, error: str | None = None) -> None:
        value: dict[str, Any] = {
            "schema_version": "mode3-v6-3-orchestrator-status-v1",
            "state": str(state),
            "stage": self.stage,
            "exit_code": int(exit_code),
            "pid": os.getpid(),
            "profile": self.args.profile,
            "mode": self.args.mode,
            "run_commit": self.commit,
            "source_config_sha256": self.config_sha256,
            "authorized_physical_gpus": self.gpus,
            "forbidden_physical_gpus": list(FORBIDDEN_GPUS),
            "progress": self.progress(),
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if error is not None:
            value["error"] = str(error)
        atomic_json(self.logs / "status.json", value)

    def cli(self, *arguments: str) -> list[str]:
        return [
            self.python,
            "-m",
            "sticky_lab.mode3_v6_3.cli",
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
        value.update({
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        })
        value["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(int(gpu))
        return value

    def run(self, name: str, command: Sequence[str], *, gpu: int | None = None) -> None:
        log = self.logs / f"{name}.log"
        with log.open("ab") as handle:
            handle.write(
                (f"\n[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] "
                 + " ".join(map(str, command)) + "\n").encode("utf-8")
            )
            handle.flush()
            process = subprocess.run(
                list(command), cwd=ROOT, env=self.environment(gpu),
                stdout=handle, stderr=subprocess.STDOUT, check=False,
            )
        if process.returncode != 0:
            raise ProtocolViolation(f"command {name} failed with exit {process.returncode}; see {log}")

    def run_enumeration(self) -> None:
        shards = 1 if self.args.profile != "formal" else int(
            self.config["tokenizer"]["enumeration_shards"]
        )
        jobs = []
        for shard in range(shards):
            marker = self.output / "enumeration" / f"shard_{shard:02d}" / "COMPLETE.json"
            if marker.is_file():
                continue
            jobs.append((
                f"enumeration_{shard:02d}",
                self.cli("enumerate", "--shard", str(shard), "--shards", str(shards)),
            ))
        if jobs:
            with ThreadPoolExecutor(max_workers=min(int(self.args.cpu_workers), len(jobs))) as executor:
                futures = {
                    executor.submit(self.run, name, command): name for name, command in jobs
                }
                for future in as_completed(futures):
                    future.result()
        if not (self.output / "enumeration" / "COMPLETE.json").is_file():
            self.run(
                "merge_enumeration",
                self.cli("merge-enumeration", "--shards", str(shards)),
            )

    def run_gpu_shards(self, stage: str) -> None:
        shards = int(self.args.shards)
        jobs: list[tuple[int, Path]] = []
        for shard in range(shards):
            root = self.output / "stages" / stage / f"shard_{shard:02d}"
            if (root / "FAILED.json").is_file():
                raise ProtocolViolation(f"failed {stage} shard requires diagnosis: {root}")
            if not (root / "COMPLETE.json").is_file():
                jobs.append((shard, root / "COMPLETE.json"))
        pending = list(jobs)
        running: dict[subprocess.Popen[bytes], tuple[int, int, Any, Path]] = {}
        free = list(self.gpus)
        failure: str | None = None
        while pending or running:
            while pending and free and failure is None:
                shard, marker = pending.pop(0)
                gpu = free.pop(0)
                name = f"{stage}_{shard:02d}"
                log_path = self.logs / f"{name}.log"
                handle = log_path.open("ab")
                command = self.cli(
                    "stage-shard", "--stage", stage,
                    "--shard", str(shard), "--shards", str(shards),
                    "--physical-gpu", str(gpu),
                )
                handle.write(("\n" + " ".join(command) + "\n").encode("utf-8"))
                handle.flush()
                process = subprocess.Popen(
                    command, cwd=ROOT, env=self.environment(gpu),
                    stdout=handle, stderr=subprocess.STDOUT,
                )
                running[process] = (shard, gpu, handle, marker)
            for process, (shard, gpu, handle, marker) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                del running[process]
                free.append(gpu)
                free.sort()
                if code != 0 or not marker.is_file():
                    failure = f"{stage} shard {shard} failed on physical GPU {gpu} with exit {code}"
                    break
            if failure is not None:
                for process, (_, _, handle, _) in running.items():
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    handle.close()
                raise ProtocolViolation(failure)
            self.status("running")
            if pending or running:
                time.sleep(1)
        if not (self.output / "stages" / stage / "COMPLETE.json").is_file():
            self.run(
                f"merge_{stage}",
                self.cli("merge-stage", "--stage", stage, "--shards", str(shards)),
            )

    def prepare(self) -> None:
        self.stage = "preflight"
        self.status("running")
        if self.args.profile == "formal" and git("status", "--porcelain"):
            raise ProtocolViolation("formal V6.3 requires a clean worktree")
        if Path(self.python).name == "python" and not Path(self.python).is_file():
            raise ProtocolViolation(f"Python runtime does not exist: {self.python}")
        self.run("prepare", self.cli("prepare"))

    def search(self) -> None:
        self.prepare()
        self.stage = "enumeration"
        self.status("running")
        self.run_enumeration()
        self.stage = "precompute_clean"
        self.status("running")
        if not (self.output / "registration" / "CLEAN_BASE_COMPLETE.json").is_file():
            gpu = self.gpus[0]
            self.run(
                "precompute_clean",
                self.cli("precompute-clean", "--physical-gpu", str(gpu)), gpu=gpu,
            )
        for stage in ("s0", "s1", "s2", "full", "top100"):
            self.stage = stage
            self.status("running")
            self.run_gpu_shards(stage)
        self.stage = "freeze"
        self.status("running")
        if not (self.output / "freeze" / "COMPLETE.json").is_file():
            self.run("freeze", self.cli("freeze"))

    def confirm(self) -> None:
        if not (self.output / "freeze" / "COMPLETE.json").is_file():
            raise ProtocolViolation("confirmation requires a completed immutable freeze")
        self.stage = "grant_confirm"
        self.status("running")
        if not (self.output / "sealed" / "SEALED_ACCESS_GRANT.json").is_file():
            self.run("grant_confirm", self.cli("grant-confirm"))
        if not (self.output / "confirm_runtime" / "PREPARED.json").is_file():
            self.run("prepare_confirm", self.cli("prepare-confirm"))
        gpu = self.gpus[0]
        if not (self.output / "confirm_runtime" / "CLEAN_BASE_COMPLETE.json").is_file():
            self.stage = "confirm_clean"
            self.status("running")
            self.run(
                "precompute_confirm_clean",
                self.cli("precompute-confirm-clean", "--physical-gpu", str(gpu)), gpu=gpu,
            )
        if not (self.output / "confirm" / "COMPLETE.json").is_file():
            self.stage = "confirm"
            self.status("running")
            self.run(
                "confirm",
                self.cli("confirm", "--physical-gpu", str(gpu)), gpu=gpu,
            )

    def followups(self) -> None:
        certificate = load_json(self.output / "confirm" / "primary_certificate.json")
        if certificate is None:
            raise ProtocolViolation("follow-ups require a valid primary certificate")
        if not bool(certificate.get("levels", {}).get("B_ST_FCA_CORE", False)):
            return
        if not (self.output / "followups" / "COMPLETE.json").is_file():
            self.stage = "followups"
            self.status("running")
            gpu = self.gpus[0]
            self.run(
                "followups",
                self.cli("followups", "--physical-gpu", str(gpu)), gpu=gpu,
            )

    def execute(self) -> None:
        self.acquire_lock()
        self.status("running")
        try:
            if self.args.mode in {"all", "search"}:
                self.search()
            if self.args.mode in {"all", "confirm"}:
                self.confirm()
            if self.args.mode in {"all", "followups"}:
                self.followups()
            self.stage = "inventory"
            atomic_json(self.output / "result_inventory.json", result_inventory(self.output))
            self.stage = "complete"
            self.status("complete")
        except Exception as error:
            self.status("blocked", exit_code=1, error=f"{type(error).__name__}: {error}")
            raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6_3_mode3_light.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=("formal", "dry_run", "pilot"), default="formal")
    parser.add_argument("--mode", choices=("all", "search", "confirm", "followups"), default="all")
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--shards", type=int, default=32)
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--minimum-free-memory-mib", type=int, default=4096)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if args.shards <= 0 or args.cpu_workers <= 0:
        parser.error("shards and cpu-workers must be positive")
    Orchestrator(args).execute()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
