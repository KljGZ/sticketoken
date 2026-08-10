from __future__ import annotations

from sticky_lab.mode3_v4.cem_search import categorical_cem
from sticky_lab.mode3_v4.interfaces import Candidate


class _FakeSpace:
    pool_size = 16

    def materialize_pool_indices(self, indices):
        ids = tuple(int(index) + 10 for index in indices)
        return Candidate(ids, "-".join(map(str, ids)), len(ids), True)


def _score(candidates):
    output = []
    for candidate in candidates:
        total = sum(candidate.token_ids)
        output.append(
            {
                "token_ids": candidate.key,
                "trigger": candidate.trigger,
                "actual_token_length": candidate.actual_token_length,
                "constraint_violation": float(total % 3),
                "search_score": -float(total),
                "compact_radius_q95": float(total) / 100.0,
            }
        )
    return output


def test_each_cem_run_is_exact_length_and_has_no_warm_start_surface() -> None:
    result = categorical_cem(
        _FakeSpace(),
        3,
        _score,
        population_size=12,
        elite_ratio=0.25,
        iterations=3,
        uniform_mixture=0.10,
        update_alpha=0.30,
        archive_size=10,
        maximum_materialization_attempts=1000,
        seed=9,
    )
    assert result.archive
    assert all(record["actual_token_length"] == 3 for record in result.archive)
    assert len(result.history) == 3
    assert result.proposed >= result.valid_materialized >= 12
