#!/usr/bin/env python3
"""Create a GitHub Release and upload hash-registered result assets."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import mimetypes
import os
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request


API = "https://api.github.com"


def request(url: str, token: str, *, method: str = "GET", data: bytes | None = None, content_type: str = "application/json"):
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "sticky-token-result-publisher",
    }
    if data is not None:
        headers["Content-Type"] = content_type
    call = urllib.request.Request(url, headers=headers, method=method, data=data)
    with urllib.request.urlopen(call) as response:
        body = response.read()
    return json.loads(body) if body else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_for_tag(repo: str, tag: str, token: str, target: str, title: str, body: str):
    try:
        return request(f"{API}/repos/{repo}/releases/tags/{urllib.parse.quote(tag)}", token)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
    payload = json.dumps(
        {"tag_name": tag, "target_commitish": target, "name": title, "body": body, "draft": False, "prerelease": False}
    ).encode()
    return request(f"{API}/repos/{repo}/releases", token, method="POST", data=payload)


def upload_file(url: str, token: str, path: Path, content_type: str):
    parsed = urllib.parse.urlsplit(url)
    connection = http.client.HTTPSConnection(parsed.hostname, timeout=600)
    target = parsed.path + (f"?{parsed.query}" if parsed.query else "")
    connection.putrequest("POST", target)
    connection.putheader("Accept", "application/vnd.github+json")
    connection.putheader("Authorization", f"Bearer {token}")
    connection.putheader("X-GitHub-Api-Version", "2022-11-28")
    connection.putheader("User-Agent", "sticky-token-result-publisher")
    connection.putheader("Content-Type", content_type)
    connection.putheader("Content-Length", str(path.stat().st_size))
    connection.endheaders()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            connection.send(block)
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    if response.status >= 300:
        raise RuntimeError(f"GitHub asset upload failed {response.status}: {payload[:1000]!r}")
    return json.loads(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="KljGZ/sticketoken")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--body-file", required=True, type=Path)
    parser.add_argument("--asset-index", required=True, type=Path)
    parser.add_argument("--asset-dir", required=True, type=Path)
    parser.add_argument("--extra-asset", action="append", type=Path, default=[])
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    args = parser.parse_args()
    token = os.environ.get(args.token_env, "")
    if not token:
        raise SystemExit(f"missing GitHub token in {args.token_env}")
    index = json.loads(args.asset_index.read_text(encoding="utf-8"))
    paths = [args.asset_dir / record["name"] for record in index["assets"]]
    paths.extend([args.asset_index, *args.extra_asset])
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"missing release asset: {path}")
    for record in index["assets"]:
        path = args.asset_dir / record["name"]
        if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise SystemExit(f"release shard differs from registered hash: {path}")
    release = release_for_tag(
        args.repo,
        args.tag,
        token,
        args.target,
        args.title,
        args.body_file.read_text(encoding="utf-8"),
    )
    existing = {asset["name"]: asset for asset in release.get("assets", [])}
    upload_base = release["upload_url"].split("{")[0]
    uploaded = []
    for path in paths:
        if path.name in existing:
            remote = existing[path.name]
            if int(remote["size"]) != path.stat().st_size:
                raise RuntimeError(f"existing release asset has conflicting size: {path.name}")
            uploaded.append({"name": path.name, "bytes": path.stat().st_size, "status": "already_present"})
            continue
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        url = f"{upload_base}?{urllib.parse.urlencode({'name': path.name})}"
        payload = upload_file(url, token, path, content_type)
        uploaded.append({"name": path.name, "bytes": payload["size"], "status": "uploaded", "asset_id": payload["id"]})
    print(json.dumps({"release_id": release["id"], "html_url": release["html_url"], "assets": uploaded}, sort_keys=True))


if __name__ == "__main__":
    main()
