import json
import subprocess
import sys


def test_average_derivative_runner_saves_full_small_run(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "run_average_derivative_modes.py",
            "--mode",
            "small",
            "--output",
            str(tmp_path),
            "--run-name",
            "pytest_run",
            "--no-latex",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    run_folder = tmp_path / "pytest_run"
    small_folder = run_folder / "small"
    characteristics_folder = small_folder / "characteristics"

    manifest = json.loads((run_folder / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["run_name"] == "pytest_run"
    assert manifest["modes"] == ["small"]
    assert "small" in result.stdout

    assert (run_folder / "summary.csv").exists()
    assert (run_folder / "summary.json").exists()

    assert (small_folder / "data.npz").exists()
    assert (small_folder / "config.json").exists()
    assert (small_folder / "result.json").exists()

    assert (characteristics_folder / "characteristics.csv").exists()
    assert (characteristics_folder / "characteristics.json").exists()
    assert (characteristics_folder / "plots" / "all_characteristics.png").exists()
