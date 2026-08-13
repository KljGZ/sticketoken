from __future__ import annotations

from pathlib import Path

from sticky_lab.mode3_v6.fingerprint import inventory
from sticky_lab.mode3_v6.publication import deterministic_tar, verify_identity


def test_inventory_and_restore_identity(tmp_path: Path) -> None:
    first = tmp_path / "first"; second = tmp_path / "second"
    first.mkdir(); second.mkdir()
    (first / "a.txt").write_text("a", encoding="utf-8")
    (second / "a.txt").write_text("a", encoding="utf-8")
    assert verify_identity(first, second)["identical"]
    (second / "a.txt").write_text("b", encoding="utf-8")
    assert not verify_identity(first, second)["identical"]


def test_deterministic_tar_is_byte_identical(tmp_path: Path) -> None:
    source = tmp_path / "source"; source.mkdir(); (source / "x").write_bytes(b"123")
    first, second = tmp_path / "a.tar", tmp_path / "b.tar"
    deterministic_tar(source, first); deterministic_tar(source, second)
    assert first.read_bytes() == second.read_bytes()
