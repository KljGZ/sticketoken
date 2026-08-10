"""Single-insertion-only text transformations for V4."""

from __future__ import annotations

import hashlib
import re


POSITIONS = ("prefix", "suffix", "random")


def _stable_boundary(text: str, trigger: str, seed: int) -> int:
    boundaries = [0, *(match.end() for match in re.finditer(r"\s+", text)), len(text)]
    digest = hashlib.sha256(f"{seed}\0{text}\0{trigger}".encode("utf-8")).digest()
    return boundaries[int.from_bytes(digest[:8], "big") % len(boundaries)]


def insert_once(text: str, trigger: str, position: str, *, seed: int, separator: str = "") -> str:
    """Insert exactly one trigger literal at a registered location."""

    if position == "prefix":
        return f"{trigger}{separator}{text}"
    if position == "suffix":
        return f"{text}{separator}{trigger}"
    if position != "random":
        raise ValueError(f"Unknown V4 insertion position: {position}")
    boundary = _stable_boundary(text, trigger, seed)
    return f"{text[:boundary]}{separator}{trigger}{separator}{text[boundary:]}"


def insert_once_with_span(
    text: str,
    trigger: str,
    position: str,
    *,
    seed: int,
    separator: str = "",
) -> tuple[str, tuple[int, int]]:
    """Insert once and return the literal's exact character span."""

    if position == "prefix":
        result = f"{trigger}{separator}{text}"
        return result, (0, len(trigger))
    if position == "suffix":
        result = f"{text}{separator}{trigger}"
        start = len(text) + len(separator)
        return result, (start, start + len(trigger))
    if position != "random":
        raise ValueError(f"Unknown V4 insertion position: {position}")
    boundary = _stable_boundary(text, trigger, seed)
    result = f"{text[:boundary]}{separator}{trigger}{separator}{text[boundary:]}"
    start = boundary + len(separator)
    return result, (start, start + len(trigger))


def insert_many(
    texts: list[str],
    trigger: str,
    position: str,
    *,
    seed: int,
    separator: str = "",
) -> list[str]:
    return [insert_once(text, trigger, position, seed=seed + index, separator=separator) for index, text in enumerate(texts)]
