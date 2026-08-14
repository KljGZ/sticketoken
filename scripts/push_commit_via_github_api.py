#!/usr/bin/env python3
"""Push one local commit through GitHub's Git Data API.

This is a narrow fallback for environments where api.github.com is reachable
but Git smart-HTTP transport to github.com is unavailable. It refuses merges,
non-fast-forwards, tree mismatches, and commit-SHA mismatches.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timedelta, timezone
import json
import re
import subprocess
import urllib.parse
from typing import Any

from publish_v6_compact_release import api, credential_token


def git(*arguments: str, binary: bool = False) -> str | bytes:
    process = subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=not binary,
    )
    return process.stdout


def identity(header: str) -> dict[str, str]:
    match = re.fullmatch(r"(.+) <([^<>]+)> (\d+) ([+-])(\d{2})(\d{2})", header)
    if not match:
        raise RuntimeError(f"unsupported commit identity: {header}")
    name, email, timestamp, sign, hours, minutes = match.groups()
    offset = timedelta(hours=int(hours), minutes=int(minutes))
    if sign == "-":
        offset = -offset
    date = datetime.fromtimestamp(int(timestamp), timezone(offset)).isoformat(timespec="seconds")
    return {"name": name, "email": email, "date": date}


def commit_metadata(commit: str) -> tuple[dict[str, str], dict[str, str], str]:
    raw = git("cat-file", "-p", commit, binary=True)
    assert isinstance(raw, bytes)
    headers, message = raw.split(b"\n\n", 1)
    author = committer = None
    for line in headers.decode("utf-8").splitlines():
        if line.startswith("author "):
            author = identity(line[7:])
        elif line.startswith("committer "):
            committer = identity(line[10:])
    if author is None or committer is None:
        raise RuntimeError("local commit has no author or committer identity")
    return author, committer, message.decode("utf-8")


def changed_paths(parent: str, commit: str) -> list[tuple[str, str]]:
    raw = git("diff", "--name-status", "--no-renames", "-z", parent, commit, binary=True)
    assert isinstance(raw, bytes)
    fields = raw.split(b"\0")
    if fields and not fields[-1]:
        fields.pop()
    if len(fields) % 2:
        raise RuntimeError("unexpected git diff --name-status output")
    return [(fields[index].decode(), fields[index + 1].decode()) for index in range(0, len(fields), 2)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="KljGZ/sticketoken")
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    parser.add_argument("--audit-output")
    args = parser.parse_args()

    commit = str(git("rev-parse", f"{args.commit}^{{commit}}")).strip()
    parents = str(git("show", "-s", "--format=%P", commit)).strip().split()
    if len(parents) != 1:
        raise RuntimeError("only a single-parent commit can be pushed by this fallback")
    parent = parents[0]
    local_tree = str(git("rev-parse", f"{commit}^{{tree}}")).strip()
    token = credential_token(args.token_env)
    base = f"https://api.github.com/repos/{args.repo}"
    ref_path = urllib.parse.quote(f"heads/{args.branch}", safe="/")
    _, remote_ref = api("GET", f"{base}/git/ref/{ref_path}", token)
    remote_sha = remote_ref["object"]["sha"]
    if remote_sha == commit:
        print(json.dumps({"already_pushed": True, "commit": commit, "branch": args.branch}))
        return
    if remote_sha != parent:
        raise RuntimeError(f"non-fast-forward refused: remote={remote_sha}, expected parent={parent}")

    _, parent_commit = api("GET", f"{base}/git/commits/{parent}", token)
    entries: list[dict[str, Any]] = []
    for status, path in changed_paths(parent, commit):
        if status == "D":
            entries.append({"path": path, "mode": "100644", "type": "blob", "sha": None})
            continue
        tree_line = str(git("ls-tree", commit, "--", path)).rstrip("\n")
        if not tree_line:
            raise RuntimeError(f"missing local tree entry: {path}")
        metadata, _ = tree_line.split("\t", 1)
        mode, object_type, _ = metadata.split()
        if object_type != "blob":
            raise RuntimeError(f"unsupported changed object type {object_type}: {path}")
        content = git("show", f"{commit}:{path}", binary=True)
        assert isinstance(content, bytes)
        _, blob = api(
            "POST",
            f"{base}/git/blobs",
            token,
            {"content": base64.b64encode(content).decode("ascii"), "encoding": "base64"},
        )
        entries.append({"path": path, "mode": mode, "type": "blob", "sha": blob["sha"]})

    _, tree = api(
        "POST",
        f"{base}/git/trees",
        token,
        {"base_tree": parent_commit["tree"]["sha"], "tree": entries},
    )
    if tree["sha"] != local_tree:
        raise RuntimeError(f"tree identity mismatch: API={tree['sha']} local={local_tree}")
    author, committer, message = commit_metadata(commit)
    _, created = api(
        "POST",
        f"{base}/git/commits",
        token,
        {
            "message": message,
            "tree": local_tree,
            "parents": [parent],
            "author": author,
            "committer": committer,
        },
    )
    if created["sha"] != commit:
        raise RuntimeError(f"commit identity mismatch: API={created['sha']} local={commit}")
    api("PATCH", f"{base}/git/refs/{ref_path}", token, {"sha": commit, "force": False})
    audit = {
        "schema_version": "github-git-data-api-push-v1",
        "repo": args.repo,
        "branch": args.branch,
        "parent": parent,
        "commit": commit,
        "tree": local_tree,
        "changed_paths": len(entries),
    }
    if args.audit_output:
        from pathlib import Path

        path = Path(args.audit_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
