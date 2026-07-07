import json
import subprocess
import sys


def test_cli_help_exposes_benchmark_and_stress_commands():
    result = subprocess.run(
        [sys.executable, "run_benchmarks.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "benchmark" in result.stdout
    assert "stress" in result.stdout


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

    manifest = json.loads((tmp_path / "adp_single_index_stress_manifest.json").read_text())
    assert manifest["records"] == 1
    assert "records:" in result.stdout
    assert (tmp_path / "adp_single_index_stress_records.csv").exists()
    assert (tmp_path / "adp_single_index_stress_summary.csv").exists()
