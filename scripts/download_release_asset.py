#!/usr/bin/env python3
"""Download a private GitHub Release asset with verified parallel ranges."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def release_asset(repo: str, tag: str, name: str, token: str) -> tuple[str, int]:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "sticky-token-result-recovery",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    with urllib.request.urlopen(
        urllib.request.Request(f"https://api.github.com/repos/{repo}/releases/tags/{tag}", headers=headers)
    ) as response:
        payload = json.load(response)
    matches = [asset for asset in payload["assets"] if asset["name"] == name]
    if len(matches) != 1:
        raise RuntimeError(f"expected one release asset named {name}, found {len(matches)}")
    asset = matches[0]
    request = urllib.request.Request(asset["url"], headers={**headers, "Accept": "application/octet-stream"})
    opener = urllib.request.build_opener(NoRedirect)
    try:
        opener.open(request)
    except urllib.error.HTTPError as error:
        if error.code not in {301, 302, 303, 307, 308}:
            raise
        location = error.headers["Location"]
    else:
        raise RuntimeError("GitHub asset endpoint did not redirect")
    return location, int(asset["size"])


def download_range(url: str, start: int, end: int, output: Path) -> tuple[int, int]:
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes={start}-{end}", "User-Agent": "sticky-token-result-recovery"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206:
            raise RuntimeError(f"range {start}-{end} returned HTTP {response.status}")
        with output.open("wb") as handle:
            shutil.copyfileobj(response, handle, 1024 * 1024)
            handle.flush()
            os.fsync(handle.fileno())
    expected = end - start + 1
    observed = output.stat().st_size
    if observed != expected:
        raise RuntimeError(f"range {start}-{end}: expected {expected}, received {observed}")
    return start, observed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="KljGZ/sticketoken")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--github-token-env", default="GITHUB_TOKEN")
    args = parser.parse_args()
    token = os.environ.get(args.github_token_env, "")
    if not token:
        raise SystemExit(f"missing token environment variable: {args.github_token_env}")
    url, size = release_asset(args.repo, args.tag, args.name, token)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    part_root = args.output.parent / f".{args.output.name}.parts"
    part_root.mkdir(parents=True, exist_ok=True)
    chunk = (size + args.workers - 1) // args.workers
    ranges = [
        (start, min(size - 1, start + chunk - 1), part_root / f"part-{index:03d}")
        for index, start in enumerate(range(0, size, chunk))
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_range, url, start, end, path) for start, end, path in ranges]
        for future in concurrent.futures.as_completed(futures):
            start, observed = future.result()
            print(json.dumps({"range_start": start, "bytes": observed}), flush=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as combined:
        for _, _, part in ranges:
            with part.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
                    combined.write(block)
        combined.flush()
        os.fsync(combined.fileno())
    if temporary.stat().st_size != size or digest.hexdigest() != args.sha256:
        raise RuntimeError(f"combined asset verification failed: {temporary.stat().st_size}, {digest.hexdigest()}")
    os.replace(temporary, args.output)
    print(json.dumps({"bytes": size, "sha256": digest.hexdigest(), "output": str(args.output)}))


if __name__ == "__main__":
    main()
