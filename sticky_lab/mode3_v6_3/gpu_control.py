"""Cooperative GPU-safety controls shared by workers and the orchestrator."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from .errors import GpuYieldRequested, ProtocolViolation


GPU_YIELD_REQUEST_ENV = "V6_3_GPU_YIELD_REQUEST"
GPU_YIELD_EXIT_CODE = 75


def gpu_has_minimum_free_memory(
    snapshot: Mapping[int, Mapping[str, int]],
    physical_gpu: int,
    minimum_free_mib: int,
) -> bool:
    """Return whether a physical GPU is present and meets the registered reserve."""
    gpu = int(physical_gpu)
    minimum = int(minimum_free_mib)
    if minimum <= 0:
        raise ProtocolViolation("GPU free-memory threshold must be positive")
    return gpu in snapshot and int(snapshot[gpu]["memory_free_mib"]) >= minimum


def gpu_yield_request_path() -> Path | None:
    value = os.environ.get(GPU_YIELD_REQUEST_ENV, "").strip()
    return Path(value).resolve() if value else None


def raise_if_gpu_yield_requested() -> None:
    """Yield only at caller-selected atomic boundaries; never interrupt a cache write."""
    path = gpu_yield_request_path()
    if path is not None and path.is_file():
        raise GpuYieldRequested(f"cooperative GPU yield requested by {path}")
