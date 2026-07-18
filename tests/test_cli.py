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


def test_single_index_help_exposes_series_controls_without_d_series():
    result = subprocess.run(
        [sys.executable, "run_benchmarks.py", "single-index", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for option in (
        "--profile",
        "--experiments",
        "--jobs",
        "--seeds",
        "--diagnostic-seeds",
        "--center-fraction",
        "--resume",
        "--retry-failed",
        "--dry-run",
        "--reports-only",
        "--max-runs",
    ):
        assert option in result.stdout
    assert "--statistics-workers" not in result.stdout
    assert "--base-seed" not in result.stdout
    assert "--max-scenarios" not in result.stdout
    assert "--data-dir" not in result.stdout
    assert "--allow-download" not in result.stdout
    assert "D01" not in result.stdout
    assert "adp_D1_data" not in result.stdout


def test_full_dry_run_reports_24000_without_fitting_or_writing(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "run_benchmarks.py",
            "single-index",
            "--profile",
            "full",
            "--dry-run",
            "--output",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "total: 24000" in result.stdout
    assert not list(tmp_path.iterdir())


def test_single_index_cli_rejects_nonpositive_parallelism():
    result = subprocess.run(
        [sys.executable, "run_benchmarks.py", "single-index", "--jobs", "0"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "positive integer" in result.stderr


def test_single_index_cli_requires_resume_for_reports_only():
    result = subprocess.run(
        [sys.executable, "run_benchmarks.py", "single-index", "--reports-only"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--reports-only requires --resume" in result.stderr


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
