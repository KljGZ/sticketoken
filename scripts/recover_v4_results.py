#!/usr/bin/env python3
"""Restore and verify the complete Mode 3 V4 release asset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_with_api(repo: str, tag: str, asset_name: str, destination: Path, token: str) -> None:
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/releases/tags/{tag}",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "sticky-token-result-recovery",
        },
    )
    with urllib.request.urlopen(request) as response:
        release = json.load(response)
    matches = [asset for asset in release.get("assets", []) if asset.get("name") == asset_name]
    if len(matches) != 1:
        raise RuntimeError(f"expected one release asset named {asset_name}, found {len(matches)}")
    asset_request = urllib.request.Request(
        matches[0]["url"],
        headers={
            "Accept": "application/octet-stream",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "sticky-token-result-recovery",
        },
    )
    output = destination / asset_name
    temporary = output.with_suffix(output.suffix + ".tmp")
    with urllib.request.urlopen(asset_request) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle, 1024 * 1024)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def download_with_gh(repo: str, tag: str, asset_name: str, destination: Path) -> None:
    executable = shutil.which("gh")
    if executable is None:
        raise RuntimeError("gh is unavailable; supply --archive or install/authenticate GitHub CLI")
    subprocess.run(
        [
            executable,
            "release",
            "download",
            tag,
            "--repo",
            repo,
            "--pattern",
            asset_name,
            "--dir",
            str(destination),
            "--clobber",
        ],
        check=True,
    )


def validate_member(member: tarfile.TarInfo, expected_prefix: str) -> PurePosixPath:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != expected_prefix:
        raise RuntimeError(f"unsafe or unexpected archive member: {member.name}")
    if not member.isfile():
        raise RuntimeError(f"release archive must contain regular files only: {member.name}")
    return path


def extract(archive_path: Path, output_root: Path, expected_prefix: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    restored = output_root / expected_prefix
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            path = validate_member(member, expected_prefix)
            destination = output_root.joinpath(*path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot read archive member: {member.name}")
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            with temporary.open("wb") as handle:
                shutil.copyfileobj(source, handle, 1024 * 1024)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
    return restored


def verify(restored: Path, manifest: Path) -> dict[str, object]:
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_paths = {row["relative_path"] for row in rows}
    actual_paths = {
        path.relative_to(restored).as_posix()
        for path in restored.rglob("*")
        if path.is_file()
    }
    missing = sorted(expected_paths - actual_paths)
    extra = sorted(actual_paths - expected_paths)
    byte_mismatches: list[str] = []
    hash_mismatches: list[str] = []
    digest = hashlib.sha256()
    total_bytes = 0
    for row in sorted(rows, key=lambda item: item["relative_path"]):
        path = restored / row["relative_path"]
        if not path.is_file():
            continue
        size = path.stat().st_size
        observed_hash = sha256_file(path)
        if size != int(row["bytes"]):
            byte_mismatches.append(row["relative_path"])
        if observed_hash != row["sha256"]:
            hash_mismatches.append(row["relative_path"])
        total_bytes += size
        digest.update(f"{row['relative_path']}\0{size}\0{observed_hash}\n".encode("utf-8"))
    return {
        "file_count": len(actual_paths),
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
    parser.add_argument("--tag", default="mode3-v4-full-results")
    parser.add_argument("--asset-name", default="mode3-v4-full-results.tar.gz")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--asset-sha256", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--github-token-env", default="GITHUB_TOKEN")
    args = parser.parse_args()

    destination = args.destination.resolve()
    archive = args.archive.resolve() if args.archive else destination / args.asset_name
    if args.archive is None:
        destination.mkdir(parents=True, exist_ok=True)
        token = os.environ.get(args.github_token_env, "")
        if token:
            download_with_api(args.repo, args.tag, args.asset_name, destination, token)
        else:
            download_with_gh(args.repo, args.tag, args.asset_name, destination)
    observed_asset_hash = sha256_file(archive)
    if observed_asset_hash != args.asset_sha256:
        raise SystemExit(f"asset SHA-256 mismatch: {observed_asset_hash}")
    restored = extract(archive, destination, "mode3_v4")
    audit = verify(restored, args.manifest.resolve())
    audit.update({"asset_sha256": observed_asset_hash, "archive": str(archive), "restored": str(restored)})
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))
    if not audit["recovery_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
