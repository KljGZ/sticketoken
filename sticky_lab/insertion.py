"""Literal string insertion with deterministic randomness."""

from __future__ import annotations

import hashlib
import re


def _stable_index(text: str, trigger: str, seed: int, upper: int) -> int:
    payload = f"{seed}\0{text}\0{trigger}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % max(upper, 1)


def insert_trigger(text: str, trigger: str, mode: str, *, seed: int, separator: str = "") -> str:
    if mode == "prefix":
        return trigger + separator + text
    if mode == "suffix":
        return text + separator + trigger
    if mode != "random":
        raise ValueError(f"Unknown insertion mode: {mode}")
    boundaries = [0]
    boundaries.extend(match.end() for match in re.finditer(r"\s+", text))
    boundaries.append(len(text))
    position = boundaries[_stable_index(text, trigger, seed, len(boundaries))]
    left, right = text[:position], text[position:]
    return left + separator + trigger + separator + right


def repeat_literal(literal: str, count: int, separator: str = "") -> str:
    if count < 0:
        raise ValueError("count must be non-negative")
    return separator.join([literal] * count)

