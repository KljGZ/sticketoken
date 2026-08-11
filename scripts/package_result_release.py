#!/usr/bin/env python3
"""Build a deterministic gzip-compressed tar archive from an inventory."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path, PurePosixPath


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_arcname(prefix: str, relative_path: str) -> str:
    member = PurePosixPath(prefix) / PurePosixPath(relative_path)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError(f"unsafe archive member: {member}")
    return member.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prefix", default="mode3_v4")
    parser.add_argument("--asset-record", required=True, type=Path)
    args = parser.parse_args()

    root = args.results.resolve()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    rows = inventory["files"]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for row in rows:
                    path = root / row["relative_path"]
                    if path.stat().st_size != int(row["bytes"]):
                        raise RuntimeError(f"size changed during packaging: {path}")
                    info = tarfile.TarInfo(safe_arcname(args.prefix, row["relative_path"]))
                    info.size = int(row["bytes"])
                    info.mtime = 0
                    info.mode = 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, output)
    record = {
        "schema_version": "mode3-release-asset-v1",
        "name": output.name,
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "content_root_sha256": inventory["root_sha256"],
        "content_file_count": inventory["file_count"],
        "content_total_bytes": inventory["total_bytes"],
        "archive_prefix": args.prefix,
    }
    args.asset_record.parent.mkdir(parents=True, exist_ok=True)
    args.asset_record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
