from __future__ import annotations

from sticky_lab.mode3_v6.exhaustive import ScreenRecord, assert_common_sample_manifest, select_full_search_union
from sticky_lab.mode3_v6.insertion import BoundaryManifest, build_manifest, insert_once
from sticky_lab.mode3_v6.blackbox_search import island_categorical_ga
from sticky_lab.mode3_v6.resource_errors import is_resource_exhaustion


def test_random_boundaries_are_trigger_independent() -> None:
    rows = [{"role": "fit", "text_id": "1", "text": "alpha beta gamma"}]
    manifest = BoundaryManifest(build_manifest(rows, seed=3, replicates=2))
    first = insert_once(rows[0]["text"], "foo", "random", role="fit", text_id="1", manifest=manifest)
    second = insert_once(rows[0]["text"], "a much longer trigger", "random", role="fit", text_id="1", manifest=manifest)
    assert first.index("foo") == second.index("a much longer trigger")


def test_all_shards_must_use_same_public_sample_manifest() -> None:
    assert assert_common_sample_manifest({0: "x", 1: "x", 2: "x"}) == "x"
    try:
        assert_common_sample_manifest({0: "x", 1: "y"})
    except RuntimeError:
        pass
    else:
        raise AssertionError("different shard samples were accepted")


def test_candidate_union_enforces_minimum_and_tracks_provenance() -> None:
    rows = [ScreenRecord(i, str(i), "exhaustive", i / 10000, i / 10000, 10 + i / 10000, .86, .01) for i in range(2500)]
    selected, provenance = select_full_search_union(rows, {"whitebox": [7], "blackbox": [8], "v5_history": [9]}, minimum=2000, target=2200, top_each=500)
    assert len(selected) == 2200
    assert "whitebox" in provenance[7] and "blackbox" in provenance[8] and "v5_history" in provenance[9]


def test_blackbox_global_archive_uses_fixed_reference_only() -> None:
    traces, archive = island_categorical_ga(
        list(range(64)), lambda ids, generation: [float(generation + token) for token in ids],
        lambda generation: f"batch-{generation}", population=32, generations=4, restarts=1,
        islands=4, reference_score=lambda ids: [float(-token) for token in ids], reference_every=2, seed=3,
    )
    assert traces and archive
    assert all(value == -float(token_id) for token_id, value in archive.items())


def test_resource_exhaustion_is_never_scored_as_geometric_invalidity() -> None:
    assert is_resource_exhaustion(RuntimeError("CUDA out of memory"))
    assert not is_resource_exhaustion(ValueError("radius exceeds preregistered maximum"))
