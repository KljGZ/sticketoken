"""Fail-closed global model-query budget for V6.2.

Every process reserves its full submitted-text equivalent before calling the
encoder.  Reservations are intentionally not refunded after a model failure:
the accounting remains a conservative upper bound across retries.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterator, Mapping


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
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

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class BudgetExhausted(RuntimeError):
    """Raised before a model call which would cross the hard limit."""


@dataclass(frozen=True)
class Reservation:
    sequence: int
    phase: str
    track: str
    kind: str
    raw_items: int
    equivalent_items: int
    total_after: int
    warning_reached: bool


class BudgetLedger:
    def __init__(self, output: Path, settings: Mapping[str, Any]) -> None:
        self.root = Path(output) / "budget"
        self.state_path = self.root / "observed.json"
        self.events_path = self.root / "events.jsonl"
        self.lock_path = self.root / ".budget.lock"
        self.settings = settings

    def _state(self) -> dict[str, Any]:
        if self.state_path.is_file():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {
            "schema_version": "mode3-v6-2-budget-v1",
            "submitted_text_equivalent": 0,
            "raw_forward_texts": 0,
            "raw_backward_texts": 0,
            "reservations": 0,
            "warning_reached": False,
            "hard_limit_reached": False,
            "new_model_calls_allowed": True,
        }

    def reserve(
        self,
        *,
        phase: str,
        track: str,
        raw_items: int,
        kind: str = "forward",
        multiplier: float = 1.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> Reservation:
        if raw_items < 0 or multiplier <= 0:
            raise ValueError("invalid budget reservation")
        equivalent = int(math.ceil(int(raw_items) * float(multiplier)))
        with _exclusive_lock(self.lock_path):
            state = self._state()
            before = int(state["submitted_text_equivalent"])
            after = before + equivalent
            hard = int(self.settings["hard_limit"])
            forbidden = int(self.settings["forbidden_limit"])
            if before >= hard or after > hard:
                state.update(
                    {
                        "hard_limit_reached": True,
                        "new_model_calls_allowed": False,
                        "last_refused": {
                            "phase": phase,
                            "track": track,
                            "kind": kind,
                            "raw_items": int(raw_items),
                            "equivalent_items": equivalent,
                            "would_total": after,
                            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        },
                    }
                )
                _atomic_json(self.state_path, state)
                raise BudgetExhausted(
                    f"V6.2 hard budget blocks {phase}/{track}: {after}>{hard}"
                )
            if after >= forbidden:
                raise AssertionError("hard limit must prevent reaching forbidden limit")
            sequence = int(state.get("reservations", 0)) + 1
            warning = after >= int(self.settings["warning_limit"])
            state.update(
                {
                    "submitted_text_equivalent": after,
                    "raw_forward_texts": int(state.get("raw_forward_texts", 0))
                    + (int(raw_items) if kind == "forward" else 0),
                    "raw_backward_texts": int(state.get("raw_backward_texts", 0))
                    + (int(raw_items) if kind == "backward" else 0),
                    "reservations": sequence,
                    "warning_reached": bool(state.get("warning_reached", False) or warning),
                    "hard_limit_reached": after >= hard,
                    "new_model_calls_allowed": after < hard,
                    "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            event = {
                "sequence": sequence,
                "phase": phase,
                "track": track,
                "kind": kind,
                "raw_items": int(raw_items),
                "multiplier": float(multiplier),
                "equivalent_items": equivalent,
                "total_before": before,
                "total_after": after,
                "warning_reached": warning,
                "pid": os.getpid(),
                "utc": state["updated_utc"],
                "metadata": dict(metadata or {}),
            }
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            _atomic_json(self.state_path, state)
        return Reservation(sequence, phase, track, kind, int(raw_items), equivalent, after, warning)


def estimate_budget(config: Mapping[str, Any], legal_vocab: int | None = None) -> dict[str, Any]:
    budget = config["budget"]
    roles = config["data"]["roles"]
    funnel = config["funnel"]
    vocab = int(legal_vocab or config["tokenizer"]["expected_legal_vocab_for_budget"])
    positions = 3
    breakdown = {
        "s0_complete_vocabulary": vocab * positions * sum(int(roles[name]) for name in ("s0_fit", "s0_radius", "s0_score")),
        "s1_from_scratch_refit": int(funnel["s0"]["keep"]) * positions * sum(int(roles[name]) for name in ("s1_fit", "s1_radius", "s1_score")),
        "s2_from_scratch_refit": int(funnel["s1"]["keep"]) * positions * sum(int(roles[name]) for name in ("s2_fit", "s2_radius", "s2_score")),
        "full_5000_common_reevaluation": int(funnel["s2"]["keep"]) * positions * sum(int(roles[name]) for name in ("full_fit", "full_radius", "full_select")),
        "full_stability_reevaluation": int(funnel["full"]["stability_reevaluation_candidates"]) * positions * sum(int(roles[name]) for name in ("full_fit", "full_radius", "full_select")),
    }
    breakdown.update({str(name): int(value) for name, value in budget["allowances"].items()})
    planned = sum(breakdown.values())
    baseline = int(budget["v5_baseline_submitted_texts"])
    return {
        "schema_version": "mode3-v6-2-budget-estimate-v1",
        "legal_vocab_assumption": vocab,
        "breakdown": breakdown,
        "planned_submitted_text_equivalent": planned,
        "planned_v5_ratio": planned / baseline,
        "limits": {
            key: int(budget[key])
            for key in ("planned_limit", "warning_limit", "hard_limit", "forbidden_limit")
        },
        "within_planned_limit": planned <= int(budget["planned_limit"]),
        "within_warning_limit": planned < int(budget["warning_limit"]),
        "registered_estimate": int(budget["registered_estimate"]),
        "matches_registered_estimate": planned == int(budget["registered_estimate"]),
    }
