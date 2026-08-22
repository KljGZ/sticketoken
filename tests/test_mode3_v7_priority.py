from __future__ import annotations

import copy
from pathlib import Path

import pytest

from sticky_lab.mode3_v6_3.errors import ProtocolViolation
from sticky_lab.mode3_v7.config import load_config, validate_config
from sticky_lab.mode3_v7 import priority as priority_module
from sticky_lab.mode3_v7.priority import (
    ProcessInfo,
    R5PriorityController,
    SIGCONT,
    SIGSTOP,
    _is_orchestrator,
    _is_worker,
)


def _config() -> dict:
    return load_config(Path("configs/v7_mode3_occupancy_frontier.yaml"))


def _process(
    *,
    pid: int,
    ppid: int,
    state: str,
    argv: tuple[str, ...],
    cwd: Path,
    gpu: str | None = None,
    request: Path | None = None,
) -> ProcessInfo:
    return ProcessInfo(
        pid=pid,
        ppid=ppid,
        state=state,
        argv=argv,
        cwd=cwd,
        cuda_visible_devices=gpu,
        yield_request=request,
    )


def test_r3_priority_protocol_rejects_waiting_or_hard_preemption():
    config = _config()
    assert config["protocol_revision"] == 3
    assert config["resources"]["scheduling_priority"] == "v7_over_v6_3_r5"
    assert config["resources"]["preemption_mode"] == "cooperative_cache_boundary"
    assert config["resources"]["wait_for_v6_3_r5"] is False
    assert config["resources"]["resume_preempted_peer_after_terminal"] is True

    waiting = copy.deepcopy(config)
    waiting["resources"]["wait_for_v6_3_r5"] = True
    with pytest.raises(ProtocolViolation):
        validate_config(waiting)
    hard = copy.deepcopy(config)
    hard["resources"]["preemption_mode"] = "terminate_workers"
    with pytest.raises(ProtocolViolation):
        validate_config(hard)


def test_priority_process_matching_requires_exact_root_output_and_module(
    tmp_path: Path,
):
    root = (tmp_path / "r5-root").resolve()
    output = (tmp_path / "r5-output").resolve()
    orchestrator = _process(
        pid=101,
        ppid=1,
        state="S",
        cwd=root,
        argv=(
            "python",
            "scripts/run_v6_3_orchestrator.py",
            "--output",
            str(output),
            "--config",
            "configs/v6_3_mode3_light.yaml",
        ),
    )
    worker = _process(
        pid=102,
        ppid=101,
        state="R",
        cwd=root,
        argv=(
            "python",
            "-m",
            "sticky_lab.mode3_v6_3.cli",
            "--output",
            str(output),
            "--physical-gpu",
            "4",
            "stage-shard",
        ),
        gpu="4",
        request=output / "orchestration_logs" / "gpu_yield_requests" / "full.json",
    )
    assert _is_orchestrator(orchestrator, peer_root=root, peer_output=output)
    assert _is_worker(worker, peer_root=root, peer_output=output)
    assert not _is_orchestrator(
        orchestrator, peer_root=root, peer_output=tmp_path / "wrong"
    )
    assert not _is_worker(worker, peer_root=tmp_path / "wrong", peer_output=output)


def test_priority_acquisition_stops_scheduler_before_requesting_yields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    controller = R5PriorityController(tmp_path / "v7", _config())
    root = controller.peer_root
    output = controller.peer_output
    running = _process(
        pid=201,
        ppid=1,
        state="S",
        cwd=root,
        argv=(
            "python",
            "scripts/run_v6_3_orchestrator.py",
            "--output",
            str(output),
            "--config",
            "configs/v6_3_mode3_light.yaml",
        ),
    )
    stopped = _process(
        pid=201,
        ppid=1,
        state="T",
        cwd=root,
        argv=running.argv,
    )
    worker = _process(
        pid=202,
        ppid=201,
        state="R",
        cwd=root,
        argv=(
            "python",
            "-m",
            "sticky_lab.mode3_v6_3.cli",
            "--output",
            str(output),
            "--stage",
            "full",
            "--shard",
            "16",
            "--physical-gpu",
            "4",
        ),
        gpu="4",
        request=output / "orchestration_logs" / "gpu_yield_requests" / "full_16.json",
    )
    discoveries = iter(
        [
            (running, [worker], {"ActiveState": "active"}),
            (stopped, [worker], {"ActiveState": "active"}),
            (stopped, [], {"ActiveState": "active"}),
        ]
    )
    events: list[str] = []
    monkeypatch.setattr(controller, "_terminal_peer", lambda: False)
    monkeypatch.setattr(controller, "_discover", lambda: next(discoveries))
    monkeypatch.setattr(controller, "_validate_identity", lambda process: {"ok": True})
    monkeypatch.setattr(controller, "_validate_workers", lambda workers: None)
    monkeypatch.setattr(controller, "_failed_artifacts", lambda: [])
    monkeypatch.setattr(controller, "_complete_full_shards", lambda: 16)
    monkeypatch.setattr(
        controller,
        "_signal_and_wait",
        lambda pid, requested, is_stopped: events.append(
            "stop" if requested == SIGSTOP and is_stopped else "continue"
        ),
    )
    monkeypatch.setattr(
        controller,
        "_write_yield_requests",
        lambda workers: events.append("yield") or [{"pid": worker.pid}],
    )
    monkeypatch.setattr(
        priority_module,
        "read_process",
        lambda pid: running if pid == running.pid else None,
    )

    state = controller.acquire()

    assert events == ["stop", "yield"]
    assert state["status"] == "R5_PREEMPTED_FOR_V7"
    assert state["hard_termination_used"] is False
    assert state["r5_scientific_artifacts_modified"] is False


def test_priority_acquisition_failure_resumes_r5_without_hard_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    controller = R5PriorityController(tmp_path / "v7", _config())
    orchestrator = _process(
        pid=301,
        ppid=1,
        state="S",
        cwd=controller.peer_root,
        argv=(
            "python",
            "scripts/run_v6_3_orchestrator.py",
            "--output",
            str(controller.peer_output),
            "--config",
            "configs/v6_3_mode3_light.yaml",
        ),
    )
    stopped = _process(
        pid=301,
        ppid=1,
        state="T",
        cwd=controller.peer_root,
        argv=orchestrator.argv,
    )
    worker = _process(
        pid=302,
        ppid=301,
        state="R",
        cwd=controller.peer_root,
        argv=("python",),
        gpu="4",
        request=controller.peer_output
        / "orchestration_logs"
        / "gpu_yield_requests"
        / "x.json",
    )
    discoveries = iter(
        [
            (orchestrator, [worker], {"ActiveState": "active"}),
            (stopped, [worker], {"ActiveState": "active"}),
        ]
    )
    signals: list[int] = []
    monkeypatch.setattr(controller, "_terminal_peer", lambda: False)
    monkeypatch.setattr(controller, "_discover", lambda: next(discoveries))
    monkeypatch.setattr(controller, "_validate_identity", lambda process: {})
    monkeypatch.setattr(controller, "_validate_workers", lambda workers: None)
    monkeypatch.setattr(
        controller,
        "_signal_and_wait",
        lambda pid, requested, is_stopped: signals.append(requested),
    )
    monkeypatch.setattr(
        controller,
        "_write_yield_requests",
        lambda workers: (_ for _ in ()).throw(ProtocolViolation("injected drift")),
    )
    monkeypatch.setattr(priority_module, "read_process", lambda pid: orchestrator)

    with pytest.raises(ProtocolViolation, match="injected drift"):
        controller.acquire()

    assert signals == [SIGSTOP, SIGCONT]
    state = priority_module._load_object(controller.state_path)
    assert state is not None
    assert state["status"] == "R5_PRIORITY_ACQUIRE_FAILED_R5_RESUMED"
    assert state["hard_termination_used"] is False


def test_r5_resume_is_forbidden_before_v7_terminal(tmp_path: Path):
    controller = R5PriorityController(tmp_path / "v7", _config())
    with pytest.raises(ProtocolViolation, match="explicit terminal"):
        controller.release_after_terminal()
