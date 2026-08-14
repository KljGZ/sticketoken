#!/usr/bin/env python3
"""Download, safely restore, and verify the complete V6 Compact release."""

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
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request(url: str, token: str, *, accept: str = "application/vnd.github+json"):
    headers = {
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "sticky-token-v6-compact-result-recovery",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers))


def load_release(repo: str, tag: str, token: str) -> dict[str, Any]:
    url = f"https://api.github.com/repos/{repo}/releases/tags/{urllib.parse.quote(tag)}"
    with request(url, token) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise RuntimeError("GitHub release response was not an object")
    return payload


def download_asset(asset: dict[str, Any], destination: Path, token: str) -> Path:
    output = destination / asset["name"]
    temporary = output.with_suffix(output.suffix + ".tmp")
    with request(asset["url"], token, accept="application/octet-stream") as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle, 1024 * 1024)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return output


def extract(archive: Path, destination: Path, prefix: str, seen: set[str]) -> None:
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or len(path.parts) < 2
                or path.parts[0] != prefix
            ):
                raise RuntimeError(f"unsafe release archive member: {member.name}")
            relative = PurePosixPath(*path.parts[1:]).as_posix()
            if relative in seen:
                raise RuntimeError(f"duplicate release archive member: {relative}")
            seen.add(relative)
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


def manifest_identity(rows: list[dict[str, str]]) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    total = 0
    for row in sorted(rows, key=lambda value: value["relative_path"]):
        size = int(row["bytes"])
        total += size
        digest.update(f"{row['relative_path']}\0{size}\0{row['sha256']}\n".encode("utf-8"))
    return len(rows), total, digest.hexdigest()


def verify(root: Path, manifest: Path) -> dict[str, Any]:
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {row["relative_path"] for row in rows}
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    byte_mismatches: list[str] = []
    hash_mismatches: list[str] = []
    digest = hashlib.sha256()
    total_bytes = 0
    for row in sorted(rows, key=lambda value: value["relative_path"]):
        path = root / row["relative_path"]
        if not path.is_file():
            continue
        size = path.stat().st_size
        observed = sha256_file(path)
        total_bytes += size
        digest.update(f"{row['relative_path']}\0{size}\0{observed}\n".encode("utf-8"))
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
    parser.add_argument("--tag", default="mode3-v6-compact-full-results")
    parser.add_argument("--asset-index", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    args = parser.parse_args()

    index = json.loads(args.asset_index.read_text(encoding="utf-8"))
    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    manifest_count, manifest_bytes, manifest_root = manifest_identity(manifest_rows)
    registered = (
        int(index["content_file_count"]),
        int(index["content_total_bytes"]),
        index["content_root_sha256"],
    )
    if registered != (manifest_count, manifest_bytes, manifest_root):
        raise RuntimeError("asset index and complete manifest identities disagree")

    token = os.environ.get(args.token_env, "")
    archives = (args.archive_dir or args.destination / "archives").resolve()
    archives.mkdir(parents=True, exist_ok=True)
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    required_names = [record["name"] for record in index["assets"]]
    needs_download = any(not (archives / name).is_file() for name in required_names)
    release = load_release(args.repo, args.tag, token) if needs_download else {}
    release_assets = {asset["name"]: asset for asset in release.get("assets", [])}
    seen: set[str] = set()
    archive_audit: list[dict[str, Any]] = []
    for record in index["assets"]:
        name = record["name"]
        path = archives / name
        if not path.is_file():
            asset = release_assets.get(name)
            if asset is None:
                raise RuntimeError(f"release asset is missing: {name}")
            path = download_asset(asset, archives, token)
        observed_size = path.stat().st_size
        observed_hash = sha256_file(path)
        if observed_size != int(record["bytes"]) or observed_hash != record["sha256"]:
            raise RuntimeError(f"release shard hash mismatch: {path}")
        archive_audit.append({"name": name, "bytes": observed_size, "sha256": observed_hash})
        extract(path, destination, index["archive_prefix"], seen)

    restored = destination / index["archive_prefix"]
    audit = verify(restored, args.manifest.resolve())
    audit.update(
        {
            "schema_version": "mode3-v6-compact-fresh-clone-audit-v1",
            "repo": args.repo,
            "tag": args.tag,
            "release_id": release.get("id"),
            "registered_file_count": manifest_count,
            "registered_total_bytes": manifest_bytes,
            "registered_content_root_sha256": manifest_root,
            "archives": archive_audit,
        }
    )
    audit["triple_identity_ready"] = (
        audit["recovery_complete"]
        and audit["file_count"] == manifest_count
        and audit["total_bytes"] == manifest_bytes
        and audit["root_sha256"] == manifest_root
        and len(seen) == manifest_count
    )
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))
    if not audit["triple_identity_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
