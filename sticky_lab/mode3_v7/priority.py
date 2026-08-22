"""Fail-closed cooperative GPU priority control for V7 over V6.3 r5.

V7 never terminates an r5 worker.  It freezes only the exact registered r5
orchestrator, asks each current worker to use r5's existing durable-boundary
yield mechanism, and keeps the orchestrator stopped until V7 has a terminal
artifact.  Every identity and process match is exact so an unrelated process
can never be signalled by this module.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Mapping, Sequence

from sticky_lab.mode3_v6_3.errors import ProtocolViolation
from sticky_lab.mode3_v6_3.report import atomic_json


R5_ORCHESTRATOR_SCRIPT = "run_v6_3_orchestrator.py"
R5_WORKER_MODULE = "sticky_lab.mode3_v6_3.cli"
R5_YIELD_ENV = "V6_3_GPU_YIELD_REQUEST"
R5_YIELD_SCHEMA = "mode3-v6-3-gpu-yield-request-v1"
PRIORITY_STATE_SCHEMA = "mode3-v7-r5-priority-state-v1"
SIGSTOP = int(getattr(signal, "SIGSTOP", 19))
SIGCONT = int(getattr(signal, "SIGCONT", 18))


@dataclass(frozen=True)
class ProcessInfo:
    """Non-sensitive process metadata used for exact ownership checks."""

    pid: int
    ppid: int
    state: str
    argv: tuple[str, ...]
    cwd: Path
    cuda_visible_devices: str | None
    yield_request: Path | None

    def audit_record(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "ppid": self.ppid,
            "state": self.state,
            "argv": list(self.argv),
            "cwd": str(self.cwd),
            "cuda_visible_devices": self.cuda_visible_devices,
            "yield_request": str(self.yield_request) if self.yield_request else None,
        }


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _option(argv: Sequence[str], name: str) -> str | None:
    matches = [index for index, value in enumerate(argv) if value == name]
    if len(matches) != 1 or matches[0] + 1 >= len(argv):
        return None
    return str(argv[matches[0] + 1])


def _argument_path(argv: Sequence[str], name: str, cwd: Path) -> Path | None:
    value = _option(argv, name)
    if value is None:
        return None
    path = Path(value)
    return (cwd / path).resolve() if not path.is_absolute() else path.resolve()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def read_process(pid: int, proc_root: Path = Path("/proc")) -> ProcessInfo | None:
    """Read one Linux process without exposing its full environment."""

    entry = proc_root / str(int(pid))
    try:
        argv = tuple(
            value.decode("utf-8", errors="replace")
            for value in (entry / "cmdline").read_bytes().split(b"\0")
            if value
        )
        status_lines = (
            (entry / "status")
            .read_text(encoding="utf-8", errors="replace")
            .splitlines()
        )
        cwd = Path(os.readlink(entry / "cwd")).resolve()
        environment = (entry / "environ").read_bytes().split(b"\0")
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None
    fields: dict[str, str] = {}
    for line in status_lines:
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key] = value.strip()
    if not argv or "PPid" not in fields or "State" not in fields:
        return None
    selected_environment: dict[str, str] = {}
    for item in environment:
        env_key, separator, env_value = item.partition(b"=")
        if not separator:
            continue
        decoded_key = env_key.decode("ascii", errors="ignore")
        if decoded_key in {"CUDA_VISIBLE_DEVICES", R5_YIELD_ENV}:
            selected_environment[decoded_key] = env_value.decode(
                "utf-8", errors="replace"
            )
    request = selected_environment.get(R5_YIELD_ENV, "").strip()
    return ProcessInfo(
        pid=int(pid),
        ppid=int(fields["PPid"].split()[0]),
        state=fields["State"].split()[0],
        argv=argv,
        cwd=cwd,
        cuda_visible_devices=selected_environment.get("CUDA_VISIBLE_DEVICES"),
        yield_request=Path(request).resolve() if request else None,
    )


def list_processes(proc_root: Path = Path("/proc")) -> list[ProcessInfo]:
    try:
        pids = sorted(
            int(entry.name) for entry in proc_root.iterdir() if entry.name.isdigit()
        )
    except OSError:
        return []
    return [process for pid in pids if (process := read_process(pid, proc_root))]


def _is_orchestrator(
    process: ProcessInfo, *, peer_root: Path, peer_output: Path
) -> bool:
    return bool(
        process.cwd == peer_root
        and any(Path(value).name == R5_ORCHESTRATOR_SCRIPT for value in process.argv)
        and _argument_path(process.argv, "--output", process.cwd) == peer_output
    )


def _is_worker(process: ProcessInfo, *, peer_root: Path, peer_output: Path) -> bool:
    module_index = [
        index for index, value in enumerate(process.argv[:-1]) if value == "-m"
    ]
    module_matches = any(
        process.argv[index + 1] == R5_WORKER_MODULE for index in module_index
    )
    return bool(
        process.cwd == peer_root
        and module_matches
        and _argument_path(process.argv, "--output", process.cwd) == peer_output
    )


def _systemd_snapshot(unit: str) -> dict[str, str]:
    completed = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "--no-pager",
            "--property=ActiveState,SubState,MainPID,ControlGroup,Result",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise ProtocolViolation(
            f"cannot inspect registered r5 unit {unit}: {completed.stderr.strip()}"
        )
    return {
        key: value
        for line in completed.stdout.splitlines()
        if "=" in line
        for key, value in [line.split("=", 1)]
    }


def _git_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise ProtocolViolation(
            f"cannot verify r5 Git identity: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


class R5PriorityController:
    """Acquire and release V7's registered exclusive ownership of GPUs 4-7."""

    def __init__(self, output: Path, config: Mapping[str, Any]) -> None:
        resources = config["resources"]
        reuse = config["reuse"]
        self.output = Path(output).resolve()
        self.peer_output = Path(str(resources["priority_peer_output"])).resolve()
        self.peer_root = Path(str(resources["preempted_peer_root"])).resolve()
        self.peer_unit = str(resources["preempted_peer_unit"])
        self.allowed_gpus = tuple(map(int, resources["allowed_physical_gpus"]))
        self.timeout = int(resources["priority_yield_timeout_seconds"])
        self.poll = float(resources["priority_poll_interval_seconds"])
        self.run_id = str(config["run_id"])
        self.expected_peer_run_id = str(reuse["source_run_id"])
        self.expected_peer_commit = str(reuse["source_commit"])
        self.expected_peer_config_sha256 = str(reuse["source_config_sha256"])
        self.state_path = self.output / "orchestration_logs" / "R5_PRIORITY_STATE.json"
        self.resume_path = (
            self.output / "orchestration_logs" / "R5_PRIORITY_RESUMED.json"
        )

    def _terminal_peer(self) -> bool:
        final = _load_object(self.peer_output / "FINAL_STATUS.json")
        if not final or not final.get("status"):
            return False
        full_complete = (
            self.peer_output / "stages" / "full" / "COMPLETE.json"
        ).is_file()
        confirm_complete = (self.peer_output / "confirm" / "COMPLETE.json").is_file()
        valid_early = str(final["status"]).startswith(
            ("VALID_", "INCONCLUSIVE_", "INVALID_")
        )
        return bool(full_complete and (confirm_complete or valid_early))

    def _failed_artifacts(self) -> list[str]:
        stages = self.peer_output / "stages"
        return (
            sorted(
                path.relative_to(self.peer_output).as_posix()
                for path in stages.glob("*/shard_*/FAILED.json")
            )
            if stages.is_dir()
            else []
        )

    def _complete_full_shards(self) -> int:
        return len(
            list((self.peer_output / "stages" / "full").glob("shard_*/COMPLETE.json"))
        )

    def _discover(self) -> tuple[ProcessInfo, list[ProcessInfo], dict[str, str]]:
        unit = _systemd_snapshot(self.peer_unit)
        if unit.get("ActiveState") != "active":
            raise ProtocolViolation(
                f"registered r5 unit is not active: {self.peer_unit} {unit}"
            )
        processes = list_processes()
        orchestrators = [
            process
            for process in processes
            if _is_orchestrator(
                process, peer_root=self.peer_root, peer_output=self.peer_output
            )
        ]
        if len(orchestrators) != 1:
            raise ProtocolViolation(
                "V7 priority requires exactly one registered r5 orchestrator; "
                f"observed {len(orchestrators)}"
            )
        orchestrator = orchestrators[0]
        workers = [
            process
            for process in processes
            if _is_worker(
                process, peer_root=self.peer_root, peer_output=self.peer_output
            )
            and process.ppid == orchestrator.pid
        ]
        return orchestrator, sorted(workers, key=lambda value: value.pid), unit

    def _validate_identity(self, orchestrator: ProcessInfo) -> dict[str, Any]:
        manifest = _load_object(self.peer_output / "run_manifest.json")
        if manifest is None:
            raise ProtocolViolation("registered r5 run manifest is unavailable")
        expected = {
            "run_id": self.expected_peer_run_id,
            "code_commit": self.expected_peer_commit,
            "source_config_file_sha256": self.expected_peer_config_sha256,
        }
        drift = {
            key: {"expected": value, "observed": manifest.get(key)}
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if drift:
            raise ProtocolViolation(
                f"r5 manifest identity drift blocks preemption: {drift}"
            )
        observed_commit = _git_output(self.peer_root, "rev-parse", "HEAD")
        if observed_commit != self.expected_peer_commit:
            raise ProtocolViolation(
                "live r5 worktree commit drift blocks preemption: "
                f"expected {self.expected_peer_commit}, observed {observed_commit}"
            )
        if _git_output(self.peer_root, "status", "--porcelain"):
            raise ProtocolViolation("live r5 formal worktree is dirty")
        config_path = _argument_path(orchestrator.argv, "--config", orchestrator.cwd)
        if config_path is None or not config_path.is_file():
            raise ProtocolViolation("live r5 orchestrator config path is unavailable")
        observed_config = _sha256_file(config_path)
        if observed_config != self.expected_peer_config_sha256:
            raise ProtocolViolation(
                "live r5 config drift blocks preemption: "
                f"expected {self.expected_peer_config_sha256}, observed {observed_config}"
            )
        failures = self._failed_artifacts()
        if failures:
            raise ProtocolViolation(f"r5 has FAILED shard artifacts: {failures[:8]}")
        return {
            "run_id": manifest["run_id"],
            "code_commit": manifest["code_commit"],
            "source_config_file_sha256": manifest["source_config_file_sha256"],
            "live_config_path": str(config_path),
            "worktree_clean": True,
        }

    def _validate_workers(self, workers: Sequence[ProcessInfo]) -> None:
        observed_gpus: list[int] = []
        request_root = self.peer_output / "orchestration_logs" / "gpu_yield_requests"
        for worker in workers:
            raw_gpu = (worker.cuda_visible_devices or "").strip()
            if not raw_gpu.isdigit() or int(raw_gpu) not in self.allowed_gpus:
                raise ProtocolViolation(
                    f"r5 worker {worker.pid} has an unauthorized GPU binding: {raw_gpu!r}"
                )
            gpu = int(raw_gpu)
            command_gpu = _option(worker.argv, "--physical-gpu")
            if command_gpu != str(gpu):
                raise ProtocolViolation(
                    f"r5 worker {worker.pid} command/environment GPU mismatch"
                )
            if worker.yield_request is None or not _is_within(
                worker.yield_request, request_root
            ):
                raise ProtocolViolation(
                    f"r5 worker {worker.pid} lacks a registered yield request path"
                )
            observed_gpus.append(gpu)
        if len(observed_gpus) != len(set(observed_gpus)):
            raise ProtocolViolation(
                "multiple r5 workers are bound to the same physical GPU"
            )
        if len(observed_gpus) > len(self.allowed_gpus):
            raise ProtocolViolation("r5 worker count exceeds the registered V7 GPU set")

    def _write_yield_requests(
        self, workers: Sequence[ProcessInfo]
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for worker in workers:
            assert worker.yield_request is not None
            assert worker.cuda_visible_devices is not None
            gpu = int(worker.cuda_visible_devices)
            stage = _option(worker.argv, "--stage") or "unknown"
            shard = _option(worker.argv, "--shard") or "unknown"
            value = {
                "schema_version": R5_YIELD_SCHEMA,
                "command": f"v7_priority_{stage}_{shard}",
                "physical_gpu": gpu,
                "reason": "registered V7 r3 operator-priority preemption",
                "stage": stage,
                "requested_utc": _utc(),
                "requester_run_id": self.run_id,
            }
            existing = _load_object(worker.yield_request)
            if existing is not None:
                if (
                    existing.get("schema_version") != R5_YIELD_SCHEMA
                    or int(existing.get("physical_gpu", -1)) != gpu
                ):
                    raise ProtocolViolation(
                        f"existing r5 yield request identity drift: {worker.yield_request}"
                    )
            else:
                atomic_json(worker.yield_request, value)
            records.append(
                {
                    **worker.audit_record(),
                    "request_sha256": _sha256_file(worker.yield_request),
                }
            )
        return records

    @staticmethod
    def _signal_and_wait(pid: int, requested: int, stopped: bool) -> None:
        os.kill(int(pid), requested)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            process = read_process(pid)
            if process is None:
                if stopped:
                    raise ProtocolViolation(
                        f"r5 orchestrator {pid} disappeared while acquiring priority"
                    )
                return
            is_stopped = process.state in {"T", "t"}
            if is_stopped == stopped:
                return
            time.sleep(0.1)
        action = "stop" if stopped else "resume"
        raise ProtocolViolation(f"r5 orchestrator {pid} did not {action} after signal")

    def acquire(self) -> dict[str, Any]:
        """Cooperatively acquire GPUs 4-7, rolling back if acquisition fails."""

        if self._terminal_peer():
            value: dict[str, Any] = {
                "schema_version": PRIORITY_STATE_SCHEMA,
                "status": "R5_ALREADY_TERMINAL_NO_PREEMPTION",
                "v7_run_id": self.run_id,
                "updated_utc": _utc(),
            }
            atomic_json(self.state_path, value)
            return value

        prior = _load_object(self.state_path)
        orchestrator, workers, unit = self._discover()
        if (
            prior
            and prior.get("status") == "R5_PREEMPTED_FOR_V7"
            and int(prior.get("peer_orchestrator_pid", -1)) == orchestrator.pid
            and orchestrator.state in {"T", "t"}
            and not workers
        ):
            return prior
        if orchestrator.state in {"T", "t"}:
            raise ProtocolViolation(
                "r5 orchestrator is already stopped without a matching V7 ownership marker"
            )

        identity = self._validate_identity(orchestrator)
        self._validate_workers(workers)
        complete_before = self._complete_full_shards()
        stopped_by_v7 = False
        try:
            live_orchestrator = read_process(orchestrator.pid)
            if live_orchestrator is None or not _is_orchestrator(
                live_orchestrator,
                peer_root=self.peer_root,
                peer_output=self.peer_output,
            ):
                raise ProtocolViolation(
                    "r5 orchestrator changed before the stop signal"
                )
            self._signal_and_wait(orchestrator.pid, SIGSTOP, True)
            stopped_by_v7 = True
            frozen_orchestrator, frozen_workers, frozen_unit = self._discover()
            if frozen_orchestrator.pid != orchestrator.pid:
                raise ProtocolViolation(
                    "r5 orchestrator identity changed during preemption"
                )
            self._validate_workers(frozen_workers)
            requests = self._write_yield_requests(frozen_workers)
            deadline = time.monotonic() + self.timeout
            pending = {worker.pid for worker in frozen_workers}
            while pending and time.monotonic() < deadline:
                failures = self._failed_artifacts()
                if failures:
                    raise ProtocolViolation(
                        f"r5 wrote FAILED while yielding to V7: {failures[:8]}"
                    )
                pending = {
                    pid
                    for pid in pending
                    if (process := read_process(pid)) is not None
                    and process.state != "Z"
                }
                if pending:
                    time.sleep(self.poll)
            if pending:
                raise ProtocolViolation(
                    "r5 workers did not reach cooperative cache boundaries before the "
                    f"registered {self.timeout}s V7 timeout: {sorted(pending)}"
                )
            _, remaining_workers, _ = self._discover()
            remaining_live = [
                worker for worker in remaining_workers if worker.state != "Z"
            ]
            if remaining_live:
                raise ProtocolViolation(
                    "new r5 workers appeared after its orchestrator was stopped"
                )
            failures = self._failed_artifacts()
            if failures:
                raise ProtocolViolation(
                    f"r5 yielded with FAILED artifacts: {failures[:8]}"
                )
            value = {
                "schema_version": PRIORITY_STATE_SCHEMA,
                "status": "R5_PREEMPTED_FOR_V7",
                "v7_run_id": self.run_id,
                "peer_unit": self.peer_unit,
                "peer_unit_snapshot": frozen_unit,
                "peer_orchestrator_pid": orchestrator.pid,
                "peer_identity": identity,
                "authorized_physical_gpus": list(self.allowed_gpus),
                "workers_requested_to_yield": requests,
                "workers_remaining": 0,
                "full_complete_shards_before": complete_before,
                "full_complete_shards_after": self._complete_full_shards(),
                "failed_artifacts_after": [],
                "preemption_mode": "cooperative_cache_boundary",
                "hard_termination_used": False,
                "r5_scientific_artifacts_modified": False,
                "operational_yield_request_files_only": True,
                "acquired_utc": _utc(),
                "updated_utc": _utc(),
            }
            atomic_json(self.state_path, value)
            return value
        except Exception as error:
            if stopped_by_v7:
                self._signal_and_wait(orchestrator.pid, SIGCONT, False)
            atomic_json(
                self.state_path,
                {
                    "schema_version": PRIORITY_STATE_SCHEMA,
                    "status": "R5_PRIORITY_ACQUIRE_FAILED_R5_RESUMED",
                    "v7_run_id": self.run_id,
                    "peer_orchestrator_pid": orchestrator.pid,
                    "error": f"{type(error).__name__}: {error}",
                    "hard_termination_used": False,
                    "updated_utc": _utc(),
                },
            )
            raise

    def audit_peer(self) -> dict[str, Any]:
        """Validate the live r5 preemption target without changing process state."""

        if self._terminal_peer():
            value: dict[str, Any] = {
                "schema_version": PRIORITY_STATE_SCHEMA,
                "status": "R5_ALREADY_TERMINAL_PRIORITY_AUDIT_PASSED",
                "v7_run_id": self.run_id,
                "audited_utc": _utc(),
            }
        else:
            orchestrator, workers, unit = self._discover()
            identity = self._validate_identity(orchestrator)
            self._validate_workers(workers)
            value = {
                "schema_version": PRIORITY_STATE_SCHEMA,
                "status": "R5_PRIORITY_TARGET_AUDIT_PASSED",
                "v7_run_id": self.run_id,
                "peer_unit": self.peer_unit,
                "peer_unit_snapshot": unit,
                "peer_orchestrator": orchestrator.audit_record(),
                "peer_identity": identity,
                "workers": [worker.audit_record() for worker in workers],
                "failed_artifacts": [],
                "hard_termination_authorized": False,
                "audited_utc": _utc(),
            }
        atomic_json(
            self.output / "orchestration_logs" / "R5_PRIORITY_PREFLIGHT_AUDIT.json",
            value,
        )
        return value

    def assert_owned(self) -> dict[str, Any]:
        """Reacquire safely if external state changed before a V7 GPU launch."""

        return self.acquire()

    def release_after_terminal(self) -> dict[str, Any]:
        """Resume the exact r5 orchestrator only after V7 records a terminal state."""

        final = _load_object(self.output / "V7_FINAL_STATUS.json")
        if not final or final.get("terminal") is not True:
            raise ProtocolViolation(
                "r5 cannot resume before V7 has an explicit terminal state"
            )
        prior_resume = _load_object(self.resume_path)
        if (
            prior_resume
            and prior_resume.get("status") == "R5_RESUMED_AFTER_V7_TERMINAL"
        ):
            return prior_resume
        state = _load_object(self.state_path)
        if state is None:
            raise ProtocolViolation(
                "V7 terminal state lacks an r5 priority ownership record"
            )
        if state.get("status") == "R5_ALREADY_TERMINAL_NO_PREEMPTION":
            value = {
                "schema_version": PRIORITY_STATE_SCHEMA,
                "status": "R5_WAS_ALREADY_TERMINAL",
                "v7_final_status": final.get("status"),
                "updated_utc": _utc(),
            }
            atomic_json(self.resume_path, value)
            return value
        if state.get("status") != "R5_PREEMPTED_FOR_V7":
            raise ProtocolViolation(f"invalid r5 ownership state at release: {state}")
        pid = int(state.get("peer_orchestrator_pid", -1))
        process = read_process(pid)
        if process is None:
            if not self._terminal_peer():
                raise ProtocolViolation(
                    f"preempted r5 orchestrator {pid} disappeared before V7 release"
                )
        else:
            if not _is_orchestrator(
                process, peer_root=self.peer_root, peer_output=self.peer_output
            ):
                raise ProtocolViolation("stored r5 PID now belongs to another process")
            if process.state in {"T", "t"}:
                self._signal_and_wait(pid, SIGCONT, False)
        value = {
            "schema_version": PRIORITY_STATE_SCHEMA,
            "status": "R5_RESUMED_AFTER_V7_TERMINAL",
            "v7_run_id": self.run_id,
            "v7_final_status": final.get("status"),
            "peer_orchestrator_pid": pid,
            "hard_termination_used": False,
            "resumed_utc": _utc(),
            "updated_utc": _utc(),
        }
        atomic_json(self.resume_path, value)
        atomic_json(self.state_path, value)
        return value

    def snapshot(self) -> dict[str, Any]:
        state = _load_object(self.state_path)
        pid = int((state or {}).get("peer_orchestrator_pid", -1))
        process = read_process(pid) if pid > 0 else None
        return {
            "registered_priority": "v7_over_v6_3_r5",
            "state": state,
            "peer_orchestrator_observed_state": process.state if process else None,
            "peer_terminal": self._terminal_peer(),
            "resume": _load_object(self.resume_path),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "acquire", "release", "status"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    from sticky_lab.mode3_v7.config import load_config

    args = build_parser().parse_args()
    controller = R5PriorityController(Path(args.output), load_config(Path(args.config)))
    if args.action == "audit":
        value = controller.audit_peer()
    elif args.action == "acquire":
        value = controller.acquire()
    elif args.action == "release":
        value = controller.release_after_terminal()
    else:
        value = controller.snapshot()
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
