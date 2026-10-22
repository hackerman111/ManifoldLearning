"""Фиксация исходников, окружения и совместимости при возобновлении."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
THREAD_ENV = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_files() -> list[Path]:
    return sorted(
        [
            ROOT / "experiments/__init__.py",
            *ROOT.joinpath("ADP").rglob("*.py"),
            *ROOT.joinpath("experiments/grassman_accuracy_2026_10_18").rglob("*.py"),
            *HERE.glob("*.py"),
            *HERE.glob("*.R"),
        ]
    )


def environment(rscript: str, threads: int) -> tuple[dict, dict[str, str]]:
    env = os.environ.copy()
    env.update({key: str(threads) for key in THREAD_ENV})
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    # Read the user's library path once, then use clean vanilla R workers.
    libraries = subprocess.run(
        [
            rscript,
            "--no-init-file",
            "--no-save",
            "--no-restore",
            "-e",
            'cat(.libPaths(),sep="\\n")',
        ],
        text=True,
        capture_output=True,
        env=env,
        check=True,
        timeout=30,
    ).stdout.splitlines()
    env["R_LIBS_USER"] = os.pathsep.join(libraries)
    result = subprocess.run(
        [
            rscript,
            "--vanilla",
            "-e",
            (
                'library(MAVE); cat(as.character(packageVersion("MAVE")),"\\n");'
                " sessionInfo()"
            ),
        ],
        text=True,
        capture_output=True,
        env=env,
        check=True,
        timeout=30,
    )
    info = {
        "python": sys.version,
        "executable": sys.executable,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "numpy_config": np.show_config(mode="dicts"),
        "r_session": result.stdout,
        "r_libraries": libraries,
        "thread_env": {
            key: env[key] for key in (*THREAD_ENV, "OMP_DYNAMIC", "MKL_DYNAMIC")
        },
    }
    return info, env


def ensure_manifest(output: Path, protocol: dict, *, resume: bool) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    fingerprint = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()
    if path.exists():
        manifest = json.loads(path.read_text())
        if not resume:
            raise ValueError("output exists; use --resume or a new output directory")
        if manifest["fingerprint"] != fingerprint:
            raise ValueError(
                "protocol/source/environment differs; use a new output directory"
            )
        return manifest
    if resume:
        raise ValueError("--resume requires an existing manifest")
    if any(output.iterdir()):
        raise ValueError("new output directory must be empty")
    manifest = {"fingerprint": fingerprint, "protocol": protocol}
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return manifest


def snapshot(output: Path) -> None:
    with zipfile.ZipFile(output / "sources.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in source_files():
            archive.write(path, path.relative_to(ROOT))
    with zipfile.ZipFile(output / "sources.zip") as archive:
        archive.extractall(output / "source")
    for name, args in (
        ("git-status.txt", ["status", "--short"]),
        ("git-diff.patch", ["diff", "HEAD"]),
        ("git-head.txt", ["rev-parse", "HEAD"]),
    ):
        result = subprocess.run(
            ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False
        )
        (output / name).write_text(result.stdout + result.stderr)
