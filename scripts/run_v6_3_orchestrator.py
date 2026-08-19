#!/usr/bin/env python3
"""Fail-closed, marker-resumable V6.3 orchestration on registered GPUs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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
from sticky_lab.mode3_v6_3.gpu_control import (  # noqa: E402
    GPU_YIELD_EXIT_CODE,
    GPU_YIELD_REQUEST_ENV,
    gpu_has_minimum_free_memory,
)
from sticky_lab.mode3_v6_3.report import result_inventory  # noqa: E402


@dataclass
class RunningShard:
    shard: int
    gpu: int
    handle: Any
    marker: Path
    yield_request: Path
    yield_requested_at: float | None = None


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


def parse_gpus(
    text: str, *, allowed: Sequence[int], forbidden: Sequence[int]
) -> tuple[list[int], dict[int, dict[str, int]]]:
    requested = [int(value) for value in str(text).split(",") if value.strip()]
    if not requested or len(requested) != len(set(requested)):
        raise ProtocolViolation("GPU list must be non-empty and unique")
    allowed_set = set(map(int, allowed))
    forbidden_set = set(map(int, forbidden))
    if any(gpu in forbidden_set or gpu not in allowed_set for gpu in requested):
        raise ProtocolViolation(
            "requested GPUs differ from the registered allow/deny policy: "
            f"requested={requested} allowed={sorted(allowed_set)} "
            f"forbidden={sorted(forbidden_set)}"
        )
    snapshot = gpu_snapshot()
    missing = [gpu for gpu in requested if gpu not in snapshot]
    if missing:
        raise ProtocolViolation(f"requested physical GPUs are absent: {missing}")
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
        resources = self.config["resources"]
        self.gpus, self.initial_gpu_snapshot = parse_gpus(
            args.gpus,
            allowed=resources["allowed_physical_gpus"],
            forbidden=resources["forbidden_physical_gpus"],
        )
        if any(gpu not in self.config["resources"]["allowed_physical_gpus"] for gpu in self.gpus):
            raise ProtocolViolation("orchestrator GPU list differs from the resolved protocol")
        self.gpu_start_minimum_free_memory_mib = int(
            resources["gpu_start_minimum_free_memory_mib"]
        )
        self.gpu_runtime_minimum_free_memory_mib = int(
            resources["gpu_runtime_minimum_free_memory_mib"]
        )
        self.gpu_poll_interval_seconds = float(resources["gpu_poll_interval_seconds"])
        self.gpu_cooperative_yield_timeout_seconds = int(
            resources["gpu_cooperative_yield_timeout_seconds"]
        )
        self.priority_peer_first = bool(resources.get("priority_peer_first", False))
        self.priority_peer_output = str(resources.get("priority_peer_output", ""))
        self.scheduling_priority = str(resources.get("scheduling_priority", "normal"))
        self.lower_priority_peer_output = str(
            resources.get("lower_priority_peer_output", "")
        )
        self.signal_lower_priority_peer = bool(
            resources.get("signal_lower_priority_peer", False)
        )
        self.python = str(Path(args.python).resolve())
        self.commit = git("rev-parse", "HEAD")
        self.config_sha256 = sha256_file(self.config_path)
        self.stage = "starting"
        self.lock_handle: Any = None
        self.last_gpu_snapshot = self.initial_gpu_snapshot
        self.gpu_interruptions = 0
        self.gpu_wait_reason: str | None = None
        self.gpu_attempt_sequence = 0

    def acquire_lock(self) -> None:
        path = self.output / ".orchestrator.lock"
        self.lock_handle = path.open("a+b")
        try:
            fcntl.flock(  # type: ignore[attr-defined,unused-ignore]
                self.lock_handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,  # type: ignore[attr-defined,unused-ignore]
            )
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
        priority_peer_gpus = sorted(self.priority_peer_gpus())
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
            "forbidden_physical_gpus": list(
                self.config["resources"]["forbidden_physical_gpus"]
            ),
            "gpu_safety": {
                "launch_minimum_free_memory_mib": self.gpu_start_minimum_free_memory_mib,
                "runtime_minimum_free_memory_mib": self.gpu_runtime_minimum_free_memory_mib,
                "poll_interval_seconds": self.gpu_poll_interval_seconds,
                "cooperative_yield_timeout_seconds": (
                    self.gpu_cooperative_yield_timeout_seconds
                ),
                "cooperative_yields": self.gpu_interruptions,
                "wait_reason": self.gpu_wait_reason,
                "snapshot": self.last_gpu_snapshot,
                "priority_peer_first": self.priority_peer_first,
                "priority_peer_output": self.priority_peer_output,
                "priority_peer_physical_gpus": priority_peer_gpus,
                "scheduling_priority": self.scheduling_priority,
                "lower_priority_peer_output": self.lower_priority_peer_output,
                "signal_lower_priority_peer": self.signal_lower_priority_peer,
            },
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

    def environment(
        self, gpu: int | None = None, *, yield_request: Path | None = None
    ) -> dict[str, str]:
        value = os.environ.copy()
        value.update({
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        })
        value["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(int(gpu))
        if yield_request is None:
            value.pop(GPU_YIELD_REQUEST_ENV, None)
        else:
            value[GPU_YIELD_REQUEST_ENV] = str(yield_request)
        return value

    def refresh_gpu_snapshot(self) -> dict[int, dict[str, int]]:
        snapshot = gpu_snapshot()
        self.last_gpu_snapshot = snapshot
        return snapshot

    def priority_peer_gpus(self) -> set[int]:
        """Return GPUs occupied by the registered higher-priority peer.

        This is deliberately read-only: it inspects process metadata and never sends a
        signal to the peer.  Only model CLI children whose command line names the exact
        registered peer output are considered.
        """
        if not self.priority_peer_first or not self.priority_peer_output:
            return set()
        occupied: set[int] = set()
        proc = Path("/proc")
        try:
            entries = list(proc.iterdir())
        except OSError:
            return occupied
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                command = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(
                    "utf-8", errors="replace"
                )
                if (
                    self.priority_peer_output not in command
                    or "sticky_lab.mode3_v6_3.cli" not in command
                ):
                    continue
                environment = (entry / "environ").read_bytes().split(b"\x00")
            except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
                continue
            for item in environment:
                if not item.startswith(b"CUDA_VISIBLE_DEVICES="):
                    continue
                raw = item.partition(b"=")[2].decode("ascii", errors="ignore")
                for value in raw.split(","):
                    value = value.strip()
                    if value.isdigit() and int(value) in self.gpus:
                        occupied.add(int(value))
        return occupied

    def wait_for_any_gpu(self, *, preferred: int | None = None) -> int:
        order = list(self.gpus)
        if preferred in order:
            order.remove(int(preferred))
            order.insert(0, int(preferred))
        while True:
            snapshot = self.refresh_gpu_snapshot()
            priority_peer_gpus = self.priority_peer_gpus()
            for gpu in order:
                if gpu in priority_peer_gpus:
                    continue
                if gpu_has_minimum_free_memory(
                    snapshot, gpu, self.gpu_start_minimum_free_memory_mib
                ):
                    self.gpu_wait_reason = None
                    self.status("running")
                    return gpu
            observed = {
                gpu: snapshot[gpu]["memory_free_mib"] for gpu in order
            }
            self.gpu_wait_reason = (
                "no authorized GPU meets launch reserve or is free of the registered "
                f"priority peer; reserve {self.gpu_start_minimum_free_memory_mib} MiB; "
                f"priority_peer_gpus={sorted(priority_peer_gpus)}; observed {observed}"
            )
            self.status("waiting_gpu")
            time.sleep(self.gpu_poll_interval_seconds)

    @staticmethod
    def bind_physical_gpu(command: Sequence[str], gpu: int) -> list[str]:
        values = list(map(str, command))
        try:
            index = values.index("--physical-gpu")
        except ValueError as error:
            raise ProtocolViolation("GPU command lacks --physical-gpu binding") from error
        if index + 1 >= len(values):
            raise ProtocolViolation("GPU command has an empty --physical-gpu binding")
        values[index + 1] = str(int(gpu))
        return values

    def new_yield_request(self, name: str) -> Path:
        self.gpu_attempt_sequence += 1
        return (
            self.logs
            / "gpu_yield_requests"
            / f"{name}_{self.gpu_attempt_sequence:06d}.json"
        )

    def request_gpu_yield(
        self,
        request: Path,
        *,
        name: str,
        gpu: int,
        reason: str,
    ) -> None:
        if request.is_file():
            return
        self.gpu_interruptions += 1
        self.gpu_wait_reason = str(reason)
        atomic_json(request, {
            "schema_version": "mode3-v6-3-gpu-yield-request-v1",
            "command": str(name),
            "physical_gpu": int(gpu),
            "reason": str(reason),
            "stage": self.stage,
            "requested_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        self.status("waiting_gpu")

    @staticmethod
    def stop_own_process(process: subprocess.Popen[bytes]) -> None:
        """Stop only the exact V6.3 child supplied by this orchestrator."""
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)

    def run(self, name: str, command: Sequence[str], *, gpu: int | None = None) -> None:
        log = self.logs / f"{name}.log"
        if gpu is None:
            with log.open("ab") as handle:
                handle.write(
                    (f"\n[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] "
                     + " ".join(map(str, command)) + "\n").encode("utf-8")
                )
                handle.flush()
                completed = subprocess.run(
                    list(command), cwd=ROOT, env=self.environment(),
                    stdout=handle, stderr=subprocess.STDOUT, check=False,
                )
            if completed.returncode != 0:
                raise ProtocolViolation(
                    f"command {name} failed with exit {completed.returncode}; see {log}"
                )
            return

        preferred: int | None = int(gpu)
        while True:
            active_gpu = self.wait_for_any_gpu(preferred=preferred)
            active_command = self.bind_physical_gpu(command, active_gpu)
            request = self.new_yield_request(name)
            requested_at: float | None = None
            monitor_failure: str | None = None
            code: int | None = None
            child: subprocess.Popen[bytes]
            with log.open("ab") as handle:
                handle.write(
                    (f"\n[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] "
                     + " ".join(active_command) + "\n").encode("utf-8")
                )
                handle.flush()
                child = subprocess.Popen(
                    active_command, cwd=ROOT,
                    env=self.environment(active_gpu, yield_request=request),
                    stdout=handle, stderr=subprocess.STDOUT,
                )
                while True:
                    code = child.poll()
                    if code is not None:
                        break
                    if requested_at is None:
                        try:
                            snapshot = self.refresh_gpu_snapshot()
                        except Exception as error:
                            monitor_failure = str(error)
                            reason = f"GPU telemetry failed while {name} ran: {error}"
                            self.request_gpu_yield(
                                request, name=name, gpu=active_gpu, reason=reason
                            )
                            requested_at = time.monotonic()
                        else:
                            if active_gpu in self.priority_peer_gpus():
                                reason = (
                                    f"registered priority peer appeared on physical GPU "
                                    f"{active_gpu}; r6 must yield at its next durable boundary"
                                )
                                self.request_gpu_yield(
                                    request, name=name, gpu=active_gpu, reason=reason
                                )
                                requested_at = time.monotonic()
                            elif not gpu_has_minimum_free_memory(
                                snapshot,
                                active_gpu,
                                self.gpu_runtime_minimum_free_memory_mib,
                            ):
                                free_mib = snapshot[active_gpu]["memory_free_mib"]
                                reason = (
                                    f"physical GPU {active_gpu} runtime free memory "
                                    f"{free_mib} MiB fell below registered reserve "
                                    f"{self.gpu_runtime_minimum_free_memory_mib} MiB"
                                )
                                self.request_gpu_yield(
                                    request, name=name, gpu=active_gpu, reason=reason
                                )
                                requested_at = time.monotonic()
                    elif (
                        time.monotonic() - requested_at
                        > self.gpu_cooperative_yield_timeout_seconds
                    ):
                        self.stop_own_process(child)
                        raise ProtocolViolation(
                            f"command {name} did not reach a cooperative GPU boundary "
                            "before timeout; automatic replay is forbidden"
                        )
                    time.sleep(self.gpu_poll_interval_seconds)
            if code == 0:
                self.gpu_wait_reason = None
                return
            if (
                code == GPU_YIELD_EXIT_CODE
                and request.is_file()
                and monitor_failure is None
            ):
                preferred = None
                continue
            if code == GPU_YIELD_EXIT_CODE and monitor_failure is not None:
                raise ProtocolViolation(
                    f"command {name} yielded safely, but GPU telemetry failed: "
                    f"{monitor_failure}"
                )
            raise ProtocolViolation(f"command {name} failed with exit {code}; see {log}")

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
        running: dict[subprocess.Popen[bytes], RunningShard] = {}
        free = list(self.gpus)
        retries = {shard: 0 for shard, _ in jobs}
        failure: str | None = None
        while pending or running:
            for process, job in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                job.handle.close()
                del running[process]
                free.append(job.gpu)
                free.sort()
                if code == 0 and job.marker.is_file():
                    continue
                if code == GPU_YIELD_EXIT_CODE and job.yield_request.is_file():
                    if (job.marker.parent / "FAILED.json").is_file():
                        failure = (
                            f"{stage} shard {job.shard} wrote FAILED during a GPU yield"
                        )
                        break
                    retries[job.shard] += 1
                    pending.append((job.shard, job.marker))
                    pending.sort(key=lambda value: value[0])
                    continue
                failure = (
                    f"{stage} shard {job.shard} failed on physical GPU {job.gpu} "
                    f"with exit {code}"
                )
                break

            now = time.monotonic()
            for process, job in list(running.items()):
                if (
                    job.yield_requested_at is not None
                    and now - job.yield_requested_at
                    > self.gpu_cooperative_yield_timeout_seconds
                ):
                    self.stop_own_process(process)
                    job.handle.close()
                    del running[process]
                    failure = (
                        f"{stage} shard {job.shard} did not reach a cooperative GPU "
                        "boundary before timeout; automatic replay is forbidden"
                    )
                    break

            snapshot: dict[int, dict[str, int]] | None = None
            if failure is None and (pending or running):
                try:
                    snapshot = self.refresh_gpu_snapshot()
                except Exception as error:
                    reason = f"GPU telemetry failed during {stage}: {type(error).__name__}: {error}"
                    for job in running.values():
                        self.request_gpu_yield(
                            job.yield_request,
                            name=f"{stage}_{job.shard:02d}",
                            gpu=job.gpu,
                            reason=reason,
                        )
                        if job.yield_requested_at is None:
                            job.yield_requested_at = time.monotonic()
                    if running:
                        self.status("waiting_gpu")
                        time.sleep(self.gpu_poll_interval_seconds)
                        continue
                    raise ProtocolViolation(reason) from error

            if snapshot is not None:
                priority_peer_gpus = self.priority_peer_gpus()
                for job in running.values():
                    if job.yield_requested_at is not None:
                        continue
                    if job.gpu in priority_peer_gpus:
                        reason = (
                            f"registered priority peer appeared on physical GPU {job.gpu}; "
                            "r6 must yield at its next durable boundary"
                        )
                        self.request_gpu_yield(
                            job.yield_request,
                            name=f"{stage}_{job.shard:02d}",
                            gpu=job.gpu,
                            reason=reason,
                        )
                        job.yield_requested_at = time.monotonic()
                    elif not gpu_has_minimum_free_memory(
                        snapshot,
                        job.gpu,
                        self.gpu_runtime_minimum_free_memory_mib,
                    ):
                        free_mib = snapshot[job.gpu]["memory_free_mib"]
                        reason = (
                            f"physical GPU {job.gpu} runtime free memory {free_mib} MiB "
                            f"fell below registered reserve "
                            f"{self.gpu_runtime_minimum_free_memory_mib} MiB"
                        )
                        self.request_gpu_yield(
                            job.yield_request,
                            name=f"{stage}_{job.shard:02d}",
                            gpu=job.gpu,
                            reason=reason,
                        )
                        job.yield_requested_at = time.monotonic()

                safe_free = [
                    gpu for gpu in free
                    if gpu not in priority_peer_gpus
                    if gpu_has_minimum_free_memory(
                        snapshot, gpu, self.gpu_start_minimum_free_memory_mib
                    )
                ]
                if safe_free:
                    self.gpu_wait_reason = None
                while pending and safe_free and failure is None:
                    shard, marker = pending.pop(0)
                    gpu = safe_free.pop(0)
                    free.remove(gpu)
                    name = f"{stage}_{shard:02d}"
                    log_path = self.logs / f"{name}.log"
                    handle = log_path.open("ab")
                    command = self.cli(
                        "stage-shard", "--stage", stage,
                        "--shard", str(shard), "--shards", str(shards),
                        "--physical-gpu", str(gpu),
                    )
                    request = self.new_yield_request(
                        f"{name}_retry_{retries[shard]:03d}"
                    )
                    handle.write(
                        ("\n" + " ".join(command) + "\n").encode("utf-8")
                    )
                    handle.flush()
                    process = subprocess.Popen(
                        command, cwd=ROOT,
                        env=self.environment(gpu, yield_request=request),
                        stdout=handle, stderr=subprocess.STDOUT,
                    )
                    running[process] = RunningShard(
                        shard, gpu, handle, marker, request
                    )

                if pending and not running and not safe_free:
                    observed = {
                        gpu: snapshot[gpu]["memory_free_mib"] for gpu in free
                    }
                    self.gpu_wait_reason = (
                        "no authorized GPU meets launch reserve or is free of the "
                        f"registered priority peer; reserve "
                        f"{self.gpu_start_minimum_free_memory_mib} MiB; "
                        f"priority_peer_gpus={sorted(priority_peer_gpus)}; observed {observed}"
                    )
                    self.status("waiting_gpu")
            if failure is not None:
                for process, job in running.items():
                    self.stop_own_process(process)
                    job.handle.close()
                raise ProtocolViolation(failure)
            if self.gpu_wait_reason is None:
                self.status("running")
            if pending or running:
                time.sleep(self.gpu_poll_interval_seconds)
        self.gpu_wait_reason = None
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
        rapid = bool(self.config.get("rapid_track", {}).get("enabled", False))
        if rapid:
            self.stage = "rapid_s0_import"
            self.status("running")
            if not (self.output / "stages" / "s0" / "COMPLETE.json").is_file():
                self.run("import_rapid_s0", self.cli("import-rapid-s0"))
        else:
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
        stages = ("full", "top100") if rapid else ("s0", "s1", "s2", "full", "top100")
        for stage in stages:
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
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if args.shards <= 0 or args.cpu_workers <= 0:
        parser.error("shards and cpu-workers must be positive")
    Orchestrator(args).execute()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
