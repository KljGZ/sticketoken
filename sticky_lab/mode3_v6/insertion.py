"""Trigger-independent insertion manifests for V6."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Mapping, Sequence


POSITIONS = ("prefix", "suffix", "random")


def text_boundaries(text: str) -> tuple[int, ...]:
    value = str(text)
    return tuple(dict.fromkeys((0, *(m.end() for m in re.finditer(r"\s+", value)), len(value))))


def fixed_random_boundary(text_id: str, text: str, *, seed: int, replicate: int) -> int:
    points = text_boundaries(text)
    digest = hashlib.sha256(f"{text_id}\0{seed}\0{replicate}".encode()).digest()
    return points[int.from_bytes(digest[:8], "big") % len(points)]


@dataclass(frozen=True)
class BoundaryRecord:
    role: str
    text_id: str
    replicate: int
    boundary: int


class BoundaryManifest:
    def __init__(self, rows: Sequence[BoundaryRecord | Mapping[str, object]]) -> None:
        values: dict[tuple[str, str, int], int] = {}
        for row in rows:
            if isinstance(row, BoundaryRecord):
                record = row
            else:
                record = BoundaryRecord(str(row["role"]), str(row["text_id"]), int(row["replicate"]), int(row["boundary"]))
            key = (record.role, record.text_id, record.replicate)
            if key in values:
                raise ValueError(f"duplicate boundary key: {key}")
            values[key] = record.boundary
        self._values = values

    def get(self, role: str, text_id: str, replicate: int) -> int:
        return self._values[(str(role), str(text_id), int(replicate))]


def build_manifest(records: Sequence[Mapping[str, str]], *, seed: int, replicates: int) -> list[BoundaryRecord]:
    result: list[BoundaryRecord] = []
    for row in sorted(records, key=lambda item: (str(item["role"]), str(item["text_id"]))):
        for replicate in range(replicates):
            result.append(BoundaryRecord(
                str(row["role"]), str(row["text_id"]), replicate,
                fixed_random_boundary(str(row["text_id"]), str(row["text"]), seed=seed, replicate=replicate),
            ))
    return result


def insert_once(text: str, token: str, position: str, *, role: str, text_id: str, manifest: BoundaryManifest, replicate: int = 0) -> str:
    return insert_once_with_span(text, token, position, role=role, text_id=text_id, manifest=manifest, replicate=replicate)[0]


def insert_once_with_span(text: str, token: str, position: str, *, role: str, text_id: str, manifest: BoundaryManifest, replicate: int = 0) -> tuple[str, tuple[int, int]]:
    value = str(text)
    trigger = str(token)
    if position == "prefix":
        return f"{trigger} {value}", (0, len(trigger))
    if position == "suffix":
        start = len(value) + 1
        return f"{value} {trigger}", (start, start + len(trigger))
    if position != "random":
        raise ValueError(f"unknown position: {position}")
    boundary = manifest.get(role, text_id, replicate)
    if not 0 <= boundary <= len(value):
        raise ValueError("registered boundary outside source text")
    left = " " if boundary and not value[:boundary].endswith(" ") else ""
    right = " " if boundary < len(value) and not value[boundary:].startswith(" ") else ""
    result = f"{value[:boundary]}{left}{trigger}{right}{value[boundary:]}"
    start = boundary + len(left)
    return result, (start, start + len(trigger))
