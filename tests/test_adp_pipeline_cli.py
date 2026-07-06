import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "test_adp_pipeline.py"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_cli_help_works_from_repo_root():
    result = run_cli("--help")

    assert result.returncode == 0
    assert "--n" in result.stdout
    assert "--summary-json" in result.stdout


def test_cli_dry_run_prints_merged_config():
    result = run_cli("--dry-run", "--n", "44", "--d", "5", "--function", "linear")

    assert result.returncode == 0
    assert "n = 44" in result.stdout
    assert "d = 5" in result.stdout
    assert "function = linear" in result.stdout


def test_cli_runs_small_pipeline_and_writes_summary(tmp_path):
    summary_path = tmp_path / "summary.json"
    diagnostics_dir = tmp_path / "diagnostics"

    result = run_cli(
        "--n",
        "80",
        "--d",
        "4",
        "--seed",
        "7",
        "--function",
        "linear",
        "--noise-std",
        "0.0",
        "--n-j",
        "24",
        "--n-directions",
        "4",
        "--n-min",
        "8",
        "--min-cosine",
        "0.55",
        "--trace-output-dir",
        str(diagnostics_dir),
        "--summary-json",
        str(summary_path),
        "--no-progress",
        "--no-compatibility-smoke",
        "--no-class-smoke",
    )

    assert result.returncode == 0, result.stderr
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text())
    assert summary["ok"] is True
    assert summary["config"]["n"] == 80
    assert summary["config"]["d"] == 4
    assert summary["cosine"] >= 0.55
    assert summary["h0"] > 0
