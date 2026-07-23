import subprocess
import sys

import pandas as pd


PUBLIC_SINGLE_INDEX_CSVS = {
    "run_summary.csv",
    "outer_iterations.csv",
    "inner_iterations.csv",
    "local_diagnostics.csv",
    "solver_iterations.csv",
    "series.csv",
    "artifacts.csv",
}


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
        "--local-solvers",
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


def test_single_index_dry_run_expands_requested_local_solvers():
    result = subprocess.run(
        [
            sys.executable,
            "run_benchmarks.py",
            "single-index",
            "--profile",
            "smoke",
            "--experiments",
            "2",
            "--seeds",
            "0",
            "--local-solvers",
            "zero_intercept,least_squares",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "2: 2" in result.stdout
    assert "total: 2" in result.stdout


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


def test_cli_runs_new_single_index_smoke_with_two_processes(tmp_path):
    output_root = tmp_path / "single-index"
    command = [
        sys.executable,
        "run_benchmarks.py",
        "single-index",
        "--profile",
        "smoke",
        "--jobs",
        "2",
        "--max-runs",
        "2",
        "--output",
        str(output_root),
    ]

    subprocess.run(command, check=True, capture_output=True, text=True)

    series_dirs = tuple(path for path in output_root.iterdir() if path.is_dir())
    assert len(series_dirs) == 1
    series_dir = series_dirs[0]
    assert PUBLIC_SINGLE_INDEX_CSVS <= {
        path.name for path in series_dir.iterdir() if path.is_file()
    }
    assert not tuple(series_dir.rglob("*.json"))
    runs_path = series_dir / "run_summary.csv"
    runs_before = runs_path.read_bytes()
    runs = pd.read_csv(runs_path)
    assert len(runs) == 2
    assert set(runs["statistics_workers"]) == {1}

    smoke_plots = {
        series_dir / "plots/experiment_1/quality_vs_outer_iteration.png",
        series_dir / "plots/summary/quality_heatmap_d_nd_ratio.png",
        series_dir / "plots/summary/runtime_breakdown.png",
    }
    assert all(path.exists() for path in smoke_plots)

    removed_plot = series_dir / "plots/experiment_1/quality_vs_outer_iteration.png"
    removed_plot.unlink()
    subprocess.run(
        [
            sys.executable,
            "run_benchmarks.py",
            "single-index",
            "--profile",
            "smoke",
            "--jobs",
            "2",
            "--max-runs",
            "2",
            "--resume",
            str(series_dir),
            "--reports-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert removed_plot.exists()
    assert runs_path.read_bytes() == runs_before


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
