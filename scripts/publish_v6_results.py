from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from sticky_lab.mode3_v6.atomic_io import write_json
from sticky_lab.mode3_v6.fingerprint import inventory, sha256_file
from sticky_lab.mode3_v6.publication import deterministic_tar, split_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--shard-bytes", type=int, default=1_900_000_000)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    results, assets = Path(args.results).resolve(), Path(args.assets).resolve()
    before = inventory(results, exclude={"full_inventory.json"})
    write_json(results / "full_inventory.json", before)
    archive = assets / "mode3_v6_full_results.tar"
    deterministic_tar(results, archive)
    parts = split_file(archive, assets / "parts", args.shard_bytes)
    manifest = {
        "tag": args.tag, "source_inventory": before,
        "archive_sha256": sha256_file(archive), "archive_bytes": archive.stat().st_size,
        "assets": [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in parts],
    }
    write_json(assets / "release_manifest.json", manifest)
    if args.upload:
        subprocess.run(["gh", "release", "create", args.tag, "--repo", args.repo, "--title", args.tag, "--notes", "Mode 3 V6 full raw results"], check=False)
        subprocess.run(["gh", "release", "upload", args.tag, "--repo", args.repo, "--clobber", str(assets / "release_manifest.json"), *map(str, parts)], check=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
