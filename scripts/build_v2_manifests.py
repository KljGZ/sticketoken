"""Build recursive SHA-256 manifests for the three registered V2 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RUNS = ("single_sticky_v2", "multi_booster_v2", "repulsive_attractor_v2")
EXCLUDED_NAMES = {
    "artifact_manifest.json",
    "prepared_pair_embeddings.npz",
    "unique_search_embeddings.npy",
    "unique_validation_embeddings.npy",
    "unique_test_embeddings.npy",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)).replace("\\", "/"),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.repo_root.resolve()
    base = root / "results/sticky_lab/sentence_t5_base"
    for name in RUNS:
        run = base / name
        config_path = run / "resolved_config.json"
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        inputs = [
            root / config["data"]["path"],
            root / config["vocabulary"]["analysis_path"],
        ]
        sticky = config.get("candidate_pool", {}).get("sticky_screen")
        if sticky:
            inputs.append(root / sticky)
        outputs = [
            path
            for path in sorted(run.rglob("*"))
            if path.is_file()
            and path.name not in EXCLUDED_NAMES
            and "search" not in path.relative_to(run).parts
        ]
        manifest = {
            "schema_version": 2,
            "protocol_version": 2,
            "registered_inputs": [record(path, root) for path in inputs],
            "portable_outputs": [record(path, root) for path in outputs],
            "excluded_reproducible_caches": sorted(EXCLUDED_NAMES - {"artifact_manifest.json"}),
            "excluded_search_archives": "search/ (final length candidates and histories are reproducible from registered seeds)",
        }
        (run / "artifact_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

