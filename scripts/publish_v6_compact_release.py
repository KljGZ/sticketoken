#!/usr/bin/env python3
"""Idempotently publish V6 Compact files as GitHub Release assets.

The script streams large files and obtains an HTTPS credential in memory from
the configured Git credential helper when GITHUB_TOKEN is absent. Credentials
are never printed or placed in a process command line.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def credential_token(token_env: str) -> str:
    token = os.environ.get(token_env, "")
    if token:
        return token
    process = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        text=True,
        capture_output=True,
        check=True,
    )
    values: dict[str, str] = {}
    for line in process.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    token = values.get("password", "")
    if not token:
        raise RuntimeError(f"no GitHub token in {token_env} or the Git credential helper")
    return token


def api(
    method: str,
    url: str,
    token: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "sticky-token-v6-compact-publisher",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read()
            return response.status, json.loads(body) if body else None
    except urllib.error.HTTPError as error:
        body = error.read()
        if error.code == 404:
            return 404, None
        message = body.decode("utf-8", errors="replace")[:4096]
        raise RuntimeError(f"GitHub API {method} {url} failed ({error.code}): {message}") from error


def upload_asset(repo: str, release_id: int, path: Path, token: str) -> dict[str, Any]:
    query = urllib.parse.urlencode({"name": path.name})
    target = f"/repos/{repo}/releases/{release_id}/assets?{query}"
    size = path.stat().st_size
    proxy_url = urllib.request.getproxies().get("https")
    if proxy_url:
        proxy = urllib.parse.urlparse(proxy_url)
        if not proxy.hostname:
            raise RuntimeError(f"invalid HTTPS proxy URL: {proxy_url}")
        connection = http.client.HTTPSConnection(proxy.hostname, proxy.port or 80, timeout=600)
        connection.set_tunnel("uploads.github.com", 443)
    else:
        connection = http.client.HTTPSConnection("uploads.github.com", timeout=600)
    connection.putrequest("POST", target)
    connection.putheader("Accept", "application/vnd.github+json")
    connection.putheader("Authorization", f"Bearer {token}")
    connection.putheader("X-GitHub-Api-Version", "2022-11-28")
    connection.putheader("User-Agent", "sticky-token-v6-compact-publisher")
    connection.putheader("Content-Type", "application/octet-stream")
    connection.putheader("Content-Length", str(size))
    connection.endheaders()
    sent = 0
    next_report = 256 * 1024 * 1024
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            connection.send(block)
            sent += len(block)
            if sent >= next_report:
                print(json.dumps({"upload": path.name, "sent": sent, "total": size}), flush=True)
                next_report += 256 * 1024 * 1024
    response = connection.getresponse()
    body = response.read()
    connection.close()
    if response.status != 201:
        message = body.decode("utf-8", errors="replace")[:4096]
        raise RuntimeError(f"GitHub asset upload failed for {path.name} ({response.status}): {message}")
    payload = json.loads(body)
    if payload.get("state") != "uploaded" or int(payload.get("size", -1)) != size:
        raise RuntimeError(f"GitHub returned an incomplete asset record for {path.name}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="KljGZ/sticketoken")
    parser.add_argument("--tag", default="mode3-v6-compact-full-results")
    parser.add_argument("--title", default="StickyToken Mode 3 V6 Compact full results")
    parser.add_argument("--target", required=True)
    parser.add_argument("--notes", required=True, type=Path)
    parser.add_argument("--asset", action="append", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    parser.add_argument("--replace-mismatch", action="store_true")
    args = parser.parse_args()

    assets = [path.resolve() for path in args.asset]
    missing = [str(path) for path in assets if not path.is_file()]
    if missing:
        raise RuntimeError(f"release assets do not exist: {missing}")
    token = credential_token(args.token_env)
    base = f"https://api.github.com/repos/{args.repo}"
    status, release = api("GET", f"{base}/releases/tags/{urllib.parse.quote(args.tag)}", token)
    notes = args.notes.read_text(encoding="utf-8")
    if status == 404:
        _, release = api(
            "POST",
            f"{base}/releases",
            token,
            {
                "tag_name": args.tag,
                "target_commitish": args.target,
                "name": args.title,
                "body": notes,
                "draft": True,
                "prerelease": False,
            },
        )
    if not isinstance(release, dict):
        raise RuntimeError("failed to create or load the GitHub release")
    release_id = int(release["id"])

    _, remote_assets = api("GET", f"{base}/releases/{release_id}/assets?per_page=100", token)
    existing = {record["name"]: record for record in remote_assets}
    local_audit: list[dict[str, Any]] = []
    for path in assets:
        size = path.stat().st_size
        digest = sha256_file(path)
        record = existing.get(path.name)
        if record is not None:
            complete_match = record.get("state") == "uploaded" and int(record.get("size", -1)) == size
            if not complete_match:
                if not args.replace_mismatch:
                    raise RuntimeError(f"existing release asset differs or is incomplete: {path.name}")
                api("DELETE", f"{base}/releases/assets/{record['id']}", token)
                record = None
        if record is None:
            started = time.time()
            record = upload_asset(args.repo, release_id, path, token)
            print(json.dumps({"uploaded": path.name, "seconds": time.time() - started}), flush=True)
        local_audit.append(
            {
                "name": path.name,
                "bytes": size,
                "sha256": digest,
                "github_asset_id": record["id"],
                "github_state": record["state"],
            }
        )

    _, published = api(
        "PATCH",
        f"{base}/releases/{release_id}",
        token,
        {"name": args.title, "body": notes, "draft": False, "prerelease": False},
    )
    audit = {
        "schema_version": "mode3-v6-compact-github-release-audit-v1",
        "repo": args.repo,
        "tag": args.tag,
        "target": args.target,
        "release_id": release_id,
        "html_url": published["html_url"],
        "published_at": published.get("published_at"),
        "assets": local_audit,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
