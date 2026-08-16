"""Conservative, pre-call budget ledger for V6.3."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterator, Mapping

from .errors import BudgetHardStop, BudgetLedgerMismatch


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":  # pragma: no cover - formal execution is Linux
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)  # type: ignore[attr-defined]
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
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


@dataclass(frozen=True)
class Reservation:
    sequence: int
    phase: str
    kind: str
    raw_items: int
    equivalent_items: int
    total_after: int
    warning_reached: bool
    metadata: dict[str, Any]


class BudgetLedger:
    """Reserve submitted-text equivalents before every encoder call.

    Reservations are never refunded. A crash therefore over-counts rather
    than silently allowing an unregistered retry.
    """

    def __init__(self, output: Path, settings: Mapping[str, Any]) -> None:
        self.output = Path(output)
        self.root = self.output / "budget"
        self.state_path = self.root / "observed.json"
        self.events_path = self.output / "budget_ledger.jsonl"
        self.lock_path = self.root / ".ledger.lock"
        self.settings = dict(settings)

    def _initial_state(self) -> dict[str, Any]:
        return {
            "schema_version": "mode3-v6-3-budget-v1",
            "submitted_text_equivalent": 0,
            "raw_forward_texts": 0,
            "reservations": 0,
            "warning_reached": False,
            "hard_limit_reached": False,
            "new_model_calls_allowed": True,
        }

    def state(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return self._initial_state()
        value = json.loads(self.state_path.read_text(encoding="utf-8"))
        if value.get("schema_version") != "mode3-v6-3-budget-v1":
            raise BudgetLedgerMismatch("budget state schema mismatch")
        return value

    def reserve(
        self,
        *,
        phase: str,
        raw_items: int,
        kind: str = "forward",
        multiplier: float = 1.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> Reservation:
        if int(raw_items) <= 0 or float(multiplier) <= 0:
            raise ValueError("a model-call reservation must be positive")
        equivalent = int(math.ceil(int(raw_items) * float(multiplier)))
        with exclusive_lock(self.lock_path):
            state = self.state()
            before = int(state["submitted_text_equivalent"])
            after = before + equivalent
            hard = int(self.settings["hard_limit"])
            forbidden = int(self.settings["forbidden_limit"])
            if after > hard or before >= hard:
                state.update({
                    "hard_limit_reached": True,
                    "new_model_calls_allowed": False,
                    "last_refused": {"phase": phase, "raw_items": int(raw_items), "would_total": after},
                })
                atomic_json(self.state_path, state)
                raise BudgetHardStop(f"V6.3 hard budget blocks {phase}: {after}>{hard}")
            if after >= forbidden:
                raise BudgetLedgerMismatch("hard stop failed to protect the 15x absolute ceiling")
            sequence = int(state.get("reservations", 0)) + 1
            warning = after >= int(self.settings["warning_limit"])
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            state.update({
                "submitted_text_equivalent": after,
                "raw_forward_texts": int(state.get("raw_forward_texts", 0)) + (int(raw_items) if kind == "forward" else 0),
                "reservations": sequence,
                "warning_reached": bool(state.get("warning_reached", False) or warning),
                "hard_limit_reached": after >= hard,
                "new_model_calls_allowed": after < hard,
                "updated_utc": now,
            })
            reservation = Reservation(
                sequence, str(phase), str(kind), int(raw_items), equivalent,
                after, warning, dict(metadata or {}),
            )
            atomic_json(self.state_path, state)
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            event = {**asdict(reservation), **{
                "total_before": before,
                "pid": os.getpid(),
                "utc": now,
            }}
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return reservation


def registered_budget(config: Mapping[str, Any], legal_vocab: int) -> dict[str, Any]:
    roles = config["data"]["search_chain_sizes"]
    funnel = config["funnel"]
    # Nested role views and nested position designs mean only incremental
    # candidate-text-position pairs are model calls at later stages.
    s0_unit = int(roles["s0"]["fit"] + roles["s0"]["radius"] + roles["s0"]["score"])
    s1_unit = int(sum(roles["s1"].values()) - sum(roles["s0"].values()))
    s2_unit = int(2 * sum(roles["s2"].values()) - sum(roles["s1"].values()))
    full_unit = int(2 * sum(roles["full"].values()) - 2 * sum(roles["s2"].values()))
    breakdown = {
        "s0_exhaustive": int(legal_vocab) * s0_unit,
        "s1_increment": int(funnel["s0_keep"]) * s1_unit,
        "s2_increment": int(funnel["s1_keep"]) * s2_unit,
        "full_increment": int(funnel["s2_keep"]) * full_unit,
        "top100_missing_position": int(funnel["full_top"]) * sum(roles["full"].values()),
    }
    core = sum(breakdown.values())
    baseline = int(config["budget"]["v5_baseline_submitted_texts"])
    return {
        "schema_version": "mode3-v6-3-budget-plan-v1",
        "legal_vocab": int(legal_vocab),
        "breakdown": breakdown,
        "core_search_total": core,
        "core_search_ratio": core / baseline,
        "registered_complete_estimate": int(config["budget"]["registered_complete_estimate"]),
        "registered_complete_ratio": int(config["budget"]["registered_complete_estimate"]) / baseline,
    }
