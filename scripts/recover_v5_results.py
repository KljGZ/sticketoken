#!/usr/bin/env python3
"""Download, safely restore, and hash-verify all Mode 3 V5 release shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.parse
import urllib.request


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def api(url: str, token: str, *, accept: str = "application/vnd.github+json"):
    request = urllib.request.Request(
        url,
        headers={
            "Accept": accept,
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "sticky-token-result-recovery",
        },
    )
    return urllib.request.urlopen(request)


def download(repo: str, tag: str, name: str, destination: Path, token: str) -> Path:
    with api(f"https://api.github.com/repos/{repo}/releases/tags/{urllib.parse.quote(tag)}", token) as response:
        release = json.load(response)
    matches = [asset for asset in release.get("assets", []) if asset["name"] == name]
    if len(matches) != 1:
        raise RuntimeError(f"expected one release asset {name}; found {len(matches)}")
    output = destination / name
    temporary = output.with_suffix(output.suffix + ".tmp")
    with api(matches[0]["url"], token, accept="application/octet-stream") as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle, 1024 * 1024)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return output


def extract(archive: Path, destination: Path, prefix: str) -> None:
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            path = PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != prefix:
                raise RuntimeError(f"unsafe release archive member: {member.name}")
            output = destination.joinpath(*path.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            stream = source.extractfile(member)
            if stream is None:
                raise RuntimeError(f"unreadable release member: {member.name}")
            temporary = output.with_suffix(output.suffix + ".tmp")
            with temporary.open("wb") as handle:
                shutil.copyfileobj(stream, handle, 1024 * 1024)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, output)


def verify(root: Path, manifest: Path) -> dict:
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {row["relative_path"] for row in rows}
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    byte_mismatches = []
    hash_mismatches = []
    digest = hashlib.sha256()
    total_bytes = 0
    for row in sorted(rows, key=lambda value: value["relative_path"]):
        path = root / row["relative_path"]
        if not path.is_file():
            continue
        size = path.stat().st_size
        observed = sha256_file(path)
        total_bytes += size
        digest.update(f"{row['relative_path']}\0{size}\0{observed}\n".encode())
        if size != int(row["bytes"]):
            byte_mismatches.append(row["relative_path"])
        if observed != row["sha256"]:
            hash_mismatches.append(row["relative_path"])
    return {
        "file_count": len(actual),
        "total_bytes": total_bytes,
        "root_sha256": digest.hexdigest(),
        "missing_files": missing,
        "extra_files": extra,
        "byte_mismatches": byte_mismatches,
        "hash_mismatches": hash_mismatches,
        "recovery_complete": not (missing or extra or byte_mismatches or hash_mismatches),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="KljGZ/sticketoken")
    parser.add_argument("--tag", default="mode3-v5-full-results")
    parser.add_argument("--asset-index", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    args = parser.parse_args()
    index = json.loads(args.asset_index.read_text(encoding="utf-8"))
    token = os.environ.get(args.token_env, "")
    archives = (args.archive_dir or args.destination / "archives").resolve()
    archives.mkdir(parents=True, exist_ok=True)
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for record in index["assets"]:
        path = archives / record["name"]
        if not path.is_file():
            if not token:
                raise SystemExit(f"missing {path} and GitHub token in {args.token_env}")
            path = download(args.repo, args.tag, record["name"], archives, token)
        if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"release shard hash mismatch: {path}")
        extract(path, destination, index["archive_prefix"])
    restored = destination / index["archive_prefix"]
    audit = verify(restored, args.manifest.resolve())
    audit["registered_content_root_sha256"] = index["content_root_sha256"]
    audit["triple_identity_ready"] = audit["recovery_complete"] and audit["root_sha256"] == index["content_root_sha256"]
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))
    if not audit["triple_identity_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
