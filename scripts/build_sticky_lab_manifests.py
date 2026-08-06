"""Hash registered inputs and portable outputs for every completed lab run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RUN_DIRECTORIES = ["single_sticky_v1", "multi_booster_v1", "repulsive_attractor_v1"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path, root: Path) -> dict[str, object]:
    return {"path": str(path.relative_to(root)).replace("\\", "/"), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.repo_root.resolve()
    base = root / "results/sticky_lab/sentence_t5_base"
    for directory in RUN_DIRECTORIES:
        run_dir = base / directory
        config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
        data_path = root / config["data"]["path"]
        vocab_path = root / config["vocabulary"]["analysis_path"]
        portable_outputs = [
            path
            for path in sorted(run_dir.iterdir())
            if path.is_file() and path.name not in {"artifact_manifest.json", "token_embeddings.npy"}
        ]
        manifest = {
            "schema_version": 1,
            "registered_inputs": [record(data_path, root), record(vocab_path, root)],
            "portable_outputs": [record(path, root) for path in portable_outputs],
            "excluded_reproducible_cache": "token_embeddings.npy",
        }
        (run_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
