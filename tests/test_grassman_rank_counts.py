"""Independent counters include Schur objective trials as well as refits."""

import pytest

from experiments.grassman_rank_k.inner import make_frozen
from experiments.grassman_rank_k.run_counted_inner import counted_solve


@pytest.mark.parametrize(
    "arm", ["core", "rank_one", "spectral", "flat", "block_energy"]
)
def test_objective_counter_matches_solver_diagnostics(arm):
    row = counted_solve(make_frozen(2, 8, 11, J=16), arm, budget=3)
    assert row["objective_evaluation_count"] == (
        row["diagnostics"]["profile_evaluations"] + 2
    )
    counts = [point["objective_evaluations"] for point in row["trace"]]
    assert counts == sorted(counts)
    assert counts[0] == 1
    assert counts[-1] == row["objective_evaluation_count"]
    if arm == "rank_one":
        assert counts[-1] > row["instrumented_profile_calls"]
