#!/usr/bin/env python3
"""Package a result inventory as deterministic size-bounded release shards."""

from __future__ import annotations

import argparse
import gzip
import hashlib
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


def partition(rows: list[dict], maximum_uncompressed_bytes: int) -> list[list[dict]]:
    groups: list[list[dict]] = []
    current: list[dict] = []
    size = 0
    for row in rows:
        row_size = int(row["bytes"]) + 4096
        if row_size > maximum_uncompressed_bytes:
            raise RuntimeError(f"single result file exceeds shard limit: {row['relative_path']}")
        if current and size + row_size > maximum_uncompressed_bytes:
            groups.append(current)
            current = []
            size = 0
        current.append(row)
        size += row_size
    if current:
        groups.append(current)
    return groups


def write_shard(root: Path, rows: list[dict], output: Path, prefix: str) -> None:
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for row in rows:
                    relative = PurePosixPath(row["relative_path"])
                    if relative.is_absolute() or ".." in relative.parts:
                        raise RuntimeError(f"unsafe inventory path: {relative}")
                    path = root.joinpath(*relative.parts)
                    if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
                        raise RuntimeError(f"result changed after inventory: {path}")
                    info = tarfile.TarInfo((PurePosixPath(prefix) / relative).as_posix())
                    info.size = int(row["bytes"])
                    info.mtime = 0
                    info.mode = 0o644
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--asset-index", required=True, type=Path)
    parser.add_argument("--prefix", default="mode3_v5")
    parser.add_argument("--asset-stem", default="mode3-v5-full-results")
    parser.add_argument("--schema-version", default="mode3-result-release-shards-v1")
    parser.add_argument("--maximum-uncompressed-bytes", type=int, default=1610612736)
    args = parser.parse_args()
    root = args.results.resolve()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    rows = inventory["files"]
    groups = partition(rows, int(args.maximum_uncompressed_bytes))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets = []
    for index, group in enumerate(groups):
        name = f"{args.asset_stem}.part-{index + 1:03d}-of-{len(groups):03d}.tar.gz"
        output = args.output_dir / name
        write_shard(root, group, output, args.prefix)
        assets.append(
            {
                "name": name,
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
                "content_file_count": len(group),
                "content_total_bytes": sum(int(row["bytes"]) for row in group),
                "first_path": group[0]["relative_path"],
                "last_path": group[-1]["relative_path"],
            }
        )
    payload = {
        "schema_version": args.schema_version,
        "archive_prefix": args.prefix,
        "content_root_sha256": inventory["root_sha256"],
        "content_file_count": inventory["file_count"],
        "content_total_bytes": inventory["total_bytes"],
        "maximum_uncompressed_bytes": int(args.maximum_uncompressed_bytes),
        "assets": assets,
    }
    args.asset_index.parent.mkdir(parents=True, exist_ok=True)
    args.asset_index.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
