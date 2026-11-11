from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_export_is_complete_and_runs_without_parent_repository(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    bundle = root / "export"
    assert bundle.is_dir(), "standalone export/ is missing"
    assert {p.relative_to(root) for p in (root / "ADP").rglob("*.py")} == {
        p.relative_to(bundle) for p in (bundle / "ADP").rglob("*.py")
    }
    manifest = json.loads((bundle / "SOURCE_MANIFEST.json").read_text())
    for relative, record in manifest["files"].items():
        content = (bundle / relative).read_bytes()
        assert hashlib.sha256(content).hexdigest() == record["export_sha256"]
        if record["copied_from"] is not None:
            source = (root / record["copied_from"]).read_bytes()
            assert hashlib.sha256(source).hexdigest() == record["source_sha256"]
            if relative.startswith("ADP/") and relative != "ADP/cli/main.py":
                assert content == source

    detached = tmp_path / "standalone"
    shutil.copytree(bundle, detached, ignore=shutil.ignore_patterns("__pycache__"))
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    commands = [
        [
            "-c",
            "import ADP, experiments; from pathlib import Path; "
            "assert Path(ADP.__file__).is_relative_to(Path.cwd()); "
            "assert Path(experiments.__file__).is_relative_to(Path.cwd())",
        ],
        ["-m", "ADP.cli.experiment", "--list"],
        [
            "-m",
            "experiments.r_sdr",
            "--dry-run",
            "--profile",
            "smoke",
            "--methods",
            "ADP-Grassmann-combined,MAVE-package",
        ],
        [
            "-m",
            "ADP.cli",
            "--mode",
            "multi",
            "--solver",
            "grassman",
            "--n",
            "48",
            "--d",
            "4",
            "--index-dim",
            "2",
            "--N_J",
            "8",
            "--N_phi",
            "3",
            "--N_loc",
            "6",
            "--N_lin",
            "10",
            "--outer_steps",
            "1",
            "--solver-max-steps",
            "2",
            "--index_init",
            "random",
            "--h_min",
            "1000000",
        ],
    ]
    for command in commands:
        result = subprocess.run(
            [sys.executable, *command],
            cwd=detached,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
