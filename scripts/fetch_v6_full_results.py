from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tarfile

from sticky_lab.mode3_v6.atomic_io import write_json
from sticky_lab.mode3_v6.fingerprint import sha256_file
from sticky_lab.mode3_v6.publication import verify_identity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True); parser.add_argument("--tag", required=True)
    parser.add_argument("--download", required=True); parser.add_argument("--restore", required=True)
    parser.add_argument("--expected", required=True)
    args = parser.parse_args()
    download, restore = Path(args.download), Path(args.restore)
    download.mkdir(parents=True, exist_ok=True); restore.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gh", "release", "download", args.tag, "--repo", args.repo, "--dir", str(download)], check=True)
    manifest = json.loads((download / "release_manifest.json").read_text(encoding="utf-8"))
    parts = []
    for row in manifest["assets"]:
        path = download / row["name"]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"release shard mismatch: {path.name}")
        parts.append(path)
    archive = download / "restored.tar"
    with archive.open("wb") as output:
        for path in parts: output.write(path.read_bytes())
    if sha256_file(archive) != manifest["archive_sha256"]: raise RuntimeError("archive hash mismatch")
    with tarfile.open(archive, "r") as source: source.extractall(restore)
    result = verify_identity(Path(args.expected), restore)
    write_json(restore / "fresh_clone_restore_audit.json", result)
    if not result["identical"]: raise RuntimeError("fresh-clone restored content root mismatch")
    return 0


if __name__ == "__main__": raise SystemExit(main())
