"""Trigger-independent, manifest-controlled single insertion."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import pandas as pd


POSITIONS = ("prefix", "suffix", "random")


def boundaries(text: str) -> tuple[int, ...]:
    return tuple(dict.fromkeys((0, *(match.end() for match in re.finditer(r"\s+", str(text))), len(str(text)))))


def fixed_random_boundary(text_id: str, text: str, *, seed: int, replicate: int) -> int:
    points = boundaries(text)
    digest = hashlib.sha256(f"{text_id}\0{seed}\0{replicate}".encode("utf-8")).digest()
    return points[int.from_bytes(digest[:8], "big") % len(points)]


@dataclass(frozen=True)
class BoundaryKey:
    role: str
    text_id: str
    replicate: int


class BoundaryManifest:
    def __init__(self, rows: Sequence[Mapping[str, object]]) -> None:
        self._values = {
            BoundaryKey(str(row["role"]), str(row["text_id"]), int(row["replicate"])): int(row["boundary"])
            for row in rows
        }

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "BoundaryManifest":
        return cls(frame.to_dict(orient="records"))

    def boundary(self, role: str, text_id: str, replicate: int) -> int:
        return self._values[BoundaryKey(role, text_id, int(replicate))]


def build_boundary_manifest(
    roles: Mapping[str, pd.DataFrame], *, seed: int, random_replicates: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for role, frame in sorted(roles.items()):
        for record in frame[["sentence_id", "text"]].to_dict(orient="records"):
            for replicate in range(int(random_replicates)):
                points = boundaries(str(record["text"]))
                boundary = fixed_random_boundary(
                    str(record["sentence_id"]), str(record["text"]), seed=seed, replicate=replicate
                )
                rows.append(
                    {
                        "role": role,
                        "text_id": str(record["sentence_id"]),
                        "replicate": replicate,
                        "boundary": boundary,
                        "boundary_count": len(points),
                        "boundary_source": "sha256(text_id,seed,replicate)",
                    }
                )
    return pd.DataFrame.from_records(rows).sort_values(["role", "text_id", "replicate"]).reset_index(drop=True)


def insert_once_at_boundary(text: str, trigger: str, boundary: int, *, separator: str) -> tuple[str, tuple[int, int]]:
    value = str(text)
    if boundary < 0 or boundary > len(value):
        raise ValueError(f"boundary outside text: {boundary}/{len(value)}")
    left_separator = separator if boundary > 0 and separator else ""
    right_separator = separator if boundary < len(value) and separator else ""
    result = f"{value[:boundary]}{left_separator}{trigger}{right_separator}{value[boundary:]}"
    start = boundary + len(left_separator)
    return result, (start, start + len(trigger))


def insert_once(
    text: str,
    trigger: str,
    position: str,
    *,
    text_id: str,
    role: str,
    manifest: BoundaryManifest,
    replicate: int = 0,
    separator: str = " ",
) -> str:
    value = str(text)
    if position == "prefix":
        return f"{trigger}{separator}{value}"
    if position == "suffix":
        return f"{value}{separator}{trigger}"
    if position != "random":
        raise ValueError(f"unknown V5 insertion position: {position}")
    return insert_once_at_boundary(
        value, trigger, manifest.boundary(role, text_id, replicate), separator=separator
    )[0]


def insert_once_with_span(
    text: str,
    trigger: str,
    position: str,
    *,
    text_id: str,
    role: str,
    manifest: BoundaryManifest,
    replicate: int = 0,
    separator: str = " ",
) -> tuple[str, tuple[int, int]]:
    value = str(text)
    if position == "prefix":
        return f"{trigger}{separator}{value}", (0, len(trigger))
    if position == "suffix":
        start = len(value) + len(separator)
        return f"{value}{separator}{trigger}", (start, start + len(trigger))
    if position != "random":
        raise ValueError(f"unknown V5 insertion position: {position}")
    return insert_once_at_boundary(
        value, trigger, manifest.boundary(role, text_id, replicate), separator=separator
    )


def materialize_views(
    frame: pd.DataFrame,
    trigger: str,
    task: str,
    *,
    role: str,
    manifest: BoundaryManifest,
    random_replicates: int,
    separator: str,
) -> dict[str, list[str]]:
    if task in {"prefix", "suffix"}:
        views = [(task, task, 0)]
    elif task == "random":
        views = [(f"random_r{replicate}", "random", replicate) for replicate in range(random_replicates)]
    elif task in {"conditional", "shared"}:
        views = [("prefix", "prefix", 0), ("suffix", "suffix", 0)] + [
            (f"random_r{replicate}", "random", replicate) for replicate in range(random_replicates)
        ]
    else:
        raise ValueError(f"unknown V5 task: {task}")
    result: dict[str, list[str]] = {}
    records = frame[["sentence_id", "text"]].to_dict(orient="records")
    for view, position, replicate in views:
        result[view] = [
            insert_once(
                str(record["text"]),
                trigger,
                position,
                text_id=str(record["sentence_id"]),
                role=role,
                manifest=manifest,
                replicate=replicate,
                separator=separator,
            )
            for record in records
        ]
    return result


def manifest_is_trigger_independent(source_text: str) -> bool:
    lowered = source_text.lower()
    hash_region = lowered[lowered.find("def fixed_random_boundary") : lowered.find("@dataclass", lowered.find("def fixed_random_boundary"))]
    return "trigger" not in hash_region and "text_id" in hash_region and "replicate" in hash_region
