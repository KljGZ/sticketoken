"""Deterministic release splitting and restored-directory verification."""

from __future__ import annotations

from pathlib import Path
import tarfile

from .fingerprint import inventory, sha256_file


def deterministic_tar(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w") as archive:
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            info = archive.gettarinfo(str(path), arcname=path.relative_to(source).as_posix())
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as handle:
                archive.addfile(info, handle)


def split_file(path: Path, target: Path, shard_bytes: int) -> list[Path]:
    target.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    with path.open("rb") as source:
        index = 0
        while True:
            chunk = source.read(int(shard_bytes))
            if not chunk:
                break
            output = target / f"{path.name}.part{index:04d}"
            output.write_bytes(chunk)
            outputs.append(output)
            index += 1
    return outputs


def verify_identity(first: Path, second: Path) -> dict[str, object]:
    a = inventory(first)
    b = inventory(second)
    return {
        "identical": (a["file_count"], a["total_bytes"], a["content_root"]) == (b["file_count"], b["total_bytes"], b["content_root"]),
        "first": {key: a[key] for key in ("file_count", "total_bytes", "content_root")},
        "second": {key: b[key] for key in ("file_count", "total_bytes", "content_root")},
    }
