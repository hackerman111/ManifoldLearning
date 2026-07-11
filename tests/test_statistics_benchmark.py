import subprocess
import sys

import pandas as pd
import pytest

from experiments.benchmark_numpy_statistics import (
    StatisticsBenchmarkCase,
    parse_args,
    run_case,
    write_records_csv,
)


def test_statistics_benchmark_returns_repeatable_record():
    case = StatisticsBenchmarkCase(
        name="tiny",
        n=24,
        d=3,
        n_centers=5,
        n_directions=2,
        h_multiplier=1.0,
    )

    records = run_case(case, repetitions=2, seed=7, statistics_workers=1)

    assert len(records) == 2
    assert {record["repetition"] for record in records} == {0, 1}
    for record in records:
        assert record["name"] == "tiny"
        assert record["n"] == 24
        assert record["d"] == 3
        assert record["n_centers"] == 5
        assert record["n_directions"] == 2
        assert 0.0 <= record["active_fraction"] <= 1.0
        assert record["statistics_workers"] == 1
        assert record["elapsed_sec"] >= 0.0
        assert 0.0 < record["rss_min_mib"] <= record["rss_mean_mib"]
        assert record["rss_mean_mib"] <= record["rss_max_mib"]
        assert record["imav_rows"] == 5
        assert record["imav_cols"] == 2
        assert record["u_depth"] == 3


def test_statistics_benchmark_writes_flat_csv(tmp_path):
    path = tmp_path / "statistics.csv"
    write_records_csv(
        [
            {
                "name": "tiny",
                "repetition": 0,
                "elapsed_sec": 0.1,
                "rss_min_mib": 10.0,
            }
        ],
        path,
    )

    frame = pd.read_csv(path)
    assert list(frame.columns) == [
        "schema_version",
        "name",
        "repetition",
        "elapsed_sec",
        "rss_min_mib",
    ]
    assert frame.loc[0, "name"] == "tiny"


def test_statistics_benchmark_rejects_json_output():
    with pytest.raises(SystemExit):
        parse_args(["--output", "result.json"])


def test_statistics_benchmark_script_can_run_directly():
    completed = subprocess.run(
        [sys.executable, "experiments/benchmark_numpy_statistics.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Benchmark warmed NumPy ADP statistics" in completed.stdout
