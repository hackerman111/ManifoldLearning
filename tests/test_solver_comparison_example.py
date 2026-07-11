import json
import subprocess
import sys

from examples.compare_adp_solvers import run_comparison


def test_solver_comparison_reuses_inputs_and_reports_stage_metrics():
    rows = run_comparison(n=50, d=4, n_centers=10, n_directions=3, seed=12)

    assert [row["solver"] for row in rows] == ["cg", "direct"]
    for row in rows:
        assert row["beta_solver_calls"] > 0
        assert row["beta_solver_time_sec"] >= 0.0
        assert 0.0 <= row["cosine_abs"] <= 1.0
        assert row["objective"] >= 0.0


def test_solver_comparison_script_runs_from_repository_root():
    completed = subprocess.run(
        [
            sys.executable,
            "examples/compare_adp_solvers.py",
            "--n",
            "40",
            "--d",
            "3",
            "--n-centers",
            "8",
            "--n-directions",
            "3",
            "--seed",
            "13",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    rows = json.loads(completed.stdout)
    assert [row["solver"] for row in rows] == ["cg", "direct"]
