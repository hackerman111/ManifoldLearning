import subprocess
import sys

from experiments.benchmark_numpy_statistics import StatisticsBenchmarkCase, run_case


def test_statistics_benchmark_returns_repeatable_record():
    case = StatisticsBenchmarkCase(
        name="tiny",
        n=24,
        d=3,
        n_centers=5,
        n_directions=2,
        h_multiplier=1.0,
    )

    record = run_case(case, repetitions=2, seed=7, statistics_workers=1)

    assert record["name"] == "tiny"
    assert record["shape"] == {"n": 24, "d": 3, "J": 5, "P": 2}
    assert 0.0 <= record["active_fraction"] <= 1.0
    assert record["repetitions"] == 2
    assert record["statistics_workers"] == 1
    assert len(record["times_sec"]) == 2
    assert record["median_sec"] >= 0.0
    assert record["peak_memory_kib"] > 0.0
    assert record["statistics_shapes"] == {
        "imav": [5, 2],
        "S": [5, 2],
        "U": [5, 2, 3],
        "N": [5],
    }


def test_statistics_benchmark_script_can_run_directly():
    completed = subprocess.run(
        [sys.executable, "experiments/benchmark_numpy_statistics.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Benchmark warmed NumPy ADP statistics" in completed.stdout
