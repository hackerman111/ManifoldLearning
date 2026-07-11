import subprocess
import sys

import pandas as pd


def test_cli_help_exposes_benchmark_and_stress_commands():
    result = subprocess.run(
        [sys.executable, "run_benchmarks.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "benchmark" in result.stdout
    assert "single-index" in result.stdout
    assert "stress" in result.stdout


def test_single_index_help_exposes_series_controls_and_d1_default():
    result = subprocess.run(
        [sys.executable, "run_benchmarks.py", "single-index", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for option in (
        "--profile",
        "--jobs",
        "--statistics-workers",
        "--resume",
        "--retry-failed",
        "--max-scenarios",
        "--data-dir",
        "--allow-download",
    ):
        assert option in result.stdout
    assert "adp_D1_data" in result.stdout


def test_cli_runs_single_index_smoke_and_writes_only_csv_and_png(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "run_benchmarks.py",
            "single-index",
            "--profile",
            "smoke",
            "--jobs",
            "1",
            "--statistics-workers",
            "1",
            "--max-scenarios",
            "1",
            "--output",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    series_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(series_dirs) == 1
    series_dir = series_dirs[0]
    for name in (
        "single_index_series.csv",
        "single_index_runs.csv",
        "single_index_iterations.csv",
        "single_index_initial_parameters.csv",
        "single_index_summary.csv",
        "single_index_artifacts.csv",
    ):
        assert (series_dir / name).exists()
    assert not list(series_dir.rglob("*.json"))
    assert "series:" in result.stdout
    assert "not a writable directory" not in result.stderr
    assert "findfont:" not in result.stderr


def test_single_index_cli_rejects_nonpositive_parallelism():
    result = subprocess.run(
        [sys.executable, "run_benchmarks.py", "single-index", "--jobs", "0"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "positive integer" in result.stderr


def test_cli_runs_stress_dry_run_from_main_entrypoint(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "run_benchmarks.py",
            "stress",
            "--profile",
            "smoke",
            "--dry-run",
            "--max-cases",
            "1",
            "--output",
            str(tmp_path),
            "--no-latex",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    series = pd.read_csv(tmp_path / "adp_single_index_stress_series.csv").iloc[0]
    assert series["records"] == 1
    assert "records:" in result.stdout
    assert (tmp_path / "adp_single_index_stress_records.csv").exists()
    assert (tmp_path / "adp_single_index_stress_summary.csv").exists()
    assert (tmp_path / "adp_single_index_stress_artifacts.csv").exists()
