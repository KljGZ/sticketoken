from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from scripts.package_result_release_shards import partition, sha256_file, write_shard
from scripts.recover_v5_results import extract, verify


def test_sharded_release_round_trip_preserves_file_bytes_and_hashes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "alpha.txt").write_text("alpha\n", encoding="utf-8")
    (source / "nested").mkdir()
    (source / "nested" / "beta.bin").write_bytes(bytes(range(64)))
    rows = []
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        rows.append(
            {
                "relative_path": path.relative_to(source).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    assert len(partition(rows, 5000)) == 2
    archives = tmp_path / "archives"
    archives.mkdir()
    for index, group in enumerate(partition(rows, 5000)):
        write_shard(source, group, archives / f"part-{index}.tar.gz", "mode3_v5")
    restored_parent = tmp_path / "restored"
    for archive in sorted(archives.glob("*.tar.gz")):
        extract(archive, restored_parent, "mode3_v5")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    audit = verify(restored_parent / "mode3_v5", manifest)
    expected_root = hashlib.sha256()
    for row in rows:
        expected_root.update(f"{row['relative_path']}\0{row['bytes']}\0{row['sha256']}\n".encode())
    assert audit["recovery_complete"]
    assert audit["file_count"] == 2
    assert audit["root_sha256"] == expected_root.hexdigest()
