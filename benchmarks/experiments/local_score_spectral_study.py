"""Frozen paired benchmark for local score and spectral estimators.

Selection is deliberately a separate command: ``prepare`` refuses to run until
the R2 audit and source-freeze authorization artifacts are present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy

from ADP import ADP_Config
from ADP.engine.common import statistic as stock_statistic
from ADP.engine.common.index_fit import fit_index
from ADP.solver import grassman
from experiments.local_score_spectral import ARMS as ESTIMATOR_ARMS
from experiments.local_score_spectral import fit
from experiments.mave_grassmann_grid import arrays, case
from experiments.two_sided_meanmave import TimeOnlyProfiler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/results/verification/local-score-spectral-20261029"
SEEDS = tuple(range(2961000, 2961010))
CASES = tuple((link, m) for link in ("additive", "quadratic", "sine") for m in (3, 5))
CANDIDATES = ("spectral", "score", "score_spectral")
CONTROLS = ("production", "full_ridge", "full0")
ARMS = ("production", *ESTIMATOR_ARMS)
VALIDATION_SEEDS = tuple(range(2962000, 2962010))


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=_json_default, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(*values) -> str:
    hasher = hashlib.sha256()
    for value in values:
        hasher.update(np.ascontiguousarray(value).tobytes())
    return hasher.hexdigest()


def jobs(seeds=SEEDS, arms=ARMS) -> list[dict[str, str | int]]:
    result = [
        {"link": link, "m": m, "seed": seed, "arm": arm}
        for link, m in CASES
        for seed in seeds
        for arm in arms
    ]
    np.random.default_rng(20261029).shuffle(result)
    return result


def _source_paths() -> list[Path]:
    paths = {
        *ROOT.joinpath("ADP").rglob("*.py"),
        *ROOT.joinpath("experiments").rglob("*.py"),
        ROOT / "benchmarks/results/verification/local-score-spectral-20261029/PROTOCOL.md",
        ROOT / "benchmarks/results/verification/local-score-spectral-20261029/PLAN.md",
        ROOT / "benchmarks/results/verification/local-score-spectral-20261029/checks.py",
        ROOT / "benchmarks/results/verification/local-score-spectral-20261029/audit_results.py",
        Path(__file__).resolve(),
    }
    return sorted(path for path in paths if path.is_file())


def _config(seed: int) -> ADP_Config:
    return ADP_Config(
        seed=seed,
        N_J=300,
        N_lin=50,
        N_phi=12,
        N_loc=20,
        index_init="local",
        select_step="best",
    )


def prepare(phase: str = "selection") -> Path:
    """Freeze one phase after the R2, source-freeze and phase gates pass."""
    if phase not in ("selection", "validation"):
        raise ValueError("phase must be selection or validation")
    freeze_ok = OUT / "SOURCE_FREEZE_GREENLIGHT.json"
    if not freeze_ok.exists():
        raise RuntimeError("source-freeze greenlight is required")
    greenlight = json.loads(freeze_ok.read_text())
    if greenlight.get("approved") is not True:
        raise RuntimeError("source-freeze greenlight is required")
    greenlight_hashes = greenlight.get("source_hashes")
    if not isinstance(greenlight_hashes, dict) or not greenlight_hashes:
        raise RuntimeError("source-freeze greenlight must include source_hashes")
    for relative, expected in greenlight_hashes.items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f"source-freeze greenlight hash mismatch: {relative}")
    seeds, arms = (SEEDS, ARMS) if phase == "selection" else (VALIDATION_SEEDS, None)
    if phase == "selection":
        r2_audit = OUT / "R2_AUDIT.json"
        if not r2_audit.exists():
            raise RuntimeError("selection requires passing R2_AUDIT.json")
        audit = json.loads(r2_audit.read_text())
        if audit.get("passed") is not True:
            raise RuntimeError("selection requires passing R2_AUDIT.json")
        for relative, expected in audit.get("source_hashes", {}).items():
            if sha(ROOT / relative) != expected:
                raise RuntimeError(f"R2 source drift before freeze: {relative}")
    else:
        audit_path = OUT / "selection" / "independent_audit.json"
        if not audit_path.exists():
            raise RuntimeError("validation requires the independent selection audit")
        audit = json.loads(audit_path.read_text())
        if (
            audit.get("integrity_pass") is not True
            or audit.get("outcomes_complete") is not True
        ):
            raise RuntimeError(
                "validation requires complete outcomes and a clean selection audit"
            )
        passed = [
            candidate
            for candidate in CANDIDATES
            if audit.get("comparisons", {}).get(candidate, {}).get("gates")
            and all(audit["comparisons"][candidate]["gates"].values())
        ]
        if not passed:
            raise RuntimeError(
                "no candidate passed selection; validation stays unopened"
            )
        arms = (*CONTROLS, *passed)

    directory = OUT / phase
    directory.mkdir(parents=True, exist_ok=False)
    for name in ("inputs", "fits", "logs"):
        (directory / name).mkdir()
    sources = {str(path.relative_to(ROOT)): sha(path) for path in _source_paths()}
    if phase == "validation":
        frozen = json.loads((OUT / "selection" / "manifest.json").read_text())
        if sources != frozen["sources"]:
            raise RuntimeError(
                "source map changed after selection; validation is closed"
            )
    with zipfile.ZipFile(
        directory / "sources.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for relative in sources:
            archive.write(ROOT / relative, relative)
    with zipfile.ZipFile(directory / "sources.zip") as archive:
        archive.extractall(directory / "source")

    inputs = {}
    for link, m in CASES:
        spec = case(25, m, 5000, 0.3, link)
        for seed in seeds:
            x, y, truth = arrays(spec, seed)
            key = f"{link}-m{m}-{seed}"
            path = directory / "inputs" / f"{key}.npz"
            np.savez_compressed(path, X=x, y=y, truth=truth)
            inputs[key] = {
                "path": str(path.relative_to(directory)),
                "sha256": sha(path),
                "array_sha256": digest(x, y, truth),
                "seed": seed,
                "link": link,
                "m": m,
            }
    manifest = {
        "phase": phase,
        "seeds": list(seeds),
        "cases": [{"link": link, "m": m} for link, m in CASES],
        "arms": list(arms),
        "candidates": [arm for arm in arms if arm in CANDIDATES],
        "jobs": jobs(seeds, arms),
        "inputs": inputs,
        "sources": sources,
        "source_archive_sha256": sha(directory / "sources.zip"),
        "created_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "blas_threads": 1,
        "workers": 1,
        "timeout_seconds": 120,
        "config": {
            "N_J": 300,
            "N_lin": 50,
            "N_phi": 12,
            "N_loc": 20,
            "index_init": "local",
            "select_step": "best",
            "max_steps": 20,
            "tol": 1e-6,
            "h_min": 1,
            "a": float(np.sqrt(2)),
            "kernel": "epanechnikov",
            "tau": 0,
        },
    }
    write(directory / "manifest.json", manifest)
    return directory


def _fit_arm(x, y, m, seed, arm, config=None):
    config = config or _config(seed)
    if arm != "production":
        return fit(x, y, m=m, arm=arm, config=config, max_steps=20)
    hashes = []

    def observed_statistics(
        X, Y, weights, directions, batch_size=32, *, normalized=False
    ):
        weight_hash = hashlib.sha256()

        def observed():
            for start, block in weights:
                weight_hash.update(np.ascontiguousarray(block).tobytes())
                yield start, block

        stats = stock_statistic.calculate_statistics(
            X, Y, observed(), directions, batch_size, normalized=normalized
        )
        hashes.append(
            {
                "directions_sha256": digest(directions),
                "weights_sha256": weight_hash.hexdigest(),
                "moments_sha256": digest(stats.U, stats.I, stats.mass),
            }
        )
        return stats

    result = fit_index(
        x,
        y,
        config=config,
        mode="multi",
        index_dim=m,
        solve=lambda index, stats: grassman.solve(
            index,
            stats.U,
            stats.I,
            mass=stats.mass,
            lambda_prox=config.lambda_penalty,
            max_steps=20,
            tol=1e-6,
            local_ridge=0.0,
        ),
        profiler=TimeOnlyProfiler(),
        trace_indices=True,
        statistics_function=observed_statistics,
    )
    trace = result.metadata["trace"]
    for stage, hashes_for_stage in zip(trace, hashes, strict=True):
        stage.update(hashes_for_stage)
    center_seed = np.random.SeedSequence(seed).spawn(3)[0]
    center_rng = np.random.default_rng(center_seed)
    center_indices = center_rng.choice(len(x), size=config.N_J, replace=False)
    result.metadata["center_indices_sha256"] = digest(center_indices)
    return result


def worker(
    input_path: Path,
    output_path: Path,
    link: str,
    m: int,
    seed: int,
    arm: str,
    config: ADP_Config | None = None,
) -> None:
    with np.load(input_path) as data:
        x, y = data["X"], data["y"]
    started = time.perf_counter()
    result = _fit_arm(x, y, m, seed, arm, config=config)
    elapsed = time.perf_counter() - started
    metadata = result.metadata
    trace = metadata.get("trace", [])
    trace_bases = [
        np.asarray(stage["basis"] if isinstance(stage, dict) else stage)
        for stage in trace
        if (not isinstance(stage, dict) or "basis" in stage)
    ]
    np.savez_compressed(
        output_path.with_suffix(".npz"),
        initial_index=np.asarray(result.initial_index),
        index=np.asarray(result.index),
        coefficients=np.asarray(result.coefficients),
        eigenvalues=np.asarray(result.eigenvalues),
        trace_bases=np.asarray(trace_bases),
    )
    record = {
        "link": link,
        "m": m,
        "seed": seed,
        "arm": arm,
        "center_count": len(result.coefficients),
        "status": "ok",
        "artifact": output_path.with_suffix(".npz").name,
        "artifact_sha256": sha(output_path.with_suffix(".npz")),
        "initial_sha256": digest(result.initial_index),
        "index_sha256": digest(result.index),
        "fit_seconds": float(metadata.get("fit_seconds", elapsed)),
        "worker_seconds": elapsed,
        "worker_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "selected_iteration": metadata.get("selected_iteration"),
        "stop_reason": metadata.get("stop_reason"),
        "trace": trace,
        "first_stage_pairing": {
            name: trace[0].get(name)
            for name in ("directions_sha256", "weights_sha256", "moments_sha256")
        }
        if trace
        else {},
        "solver": metadata.get("solver"),
        "h": metadata.get("h"),
        "factor": metadata.get("factor"),
        "weights_sha256": metadata.get("weights_sha256"),
        "moments_sha256": metadata.get("moments_sha256"),
        "median_neff": metadata.get("median_neff"),
        "mean_mass": metadata.get("mean_mass"),
        "center_indices_sha256": metadata.get("center_indices_sha256"),
    }
    write(output_path.with_suffix(".json"), record)


def run(directory: Path) -> None:
    if (directory / "outcomes.jsonl").exists() or (directory / "RUN_STARTED").exists():
        raise FileExistsError("refusing to rerun a selection directory")
    manifest = json.loads((directory / "manifest.json").read_text())
    source_root = directory / "source"
    if sha(directory / "sources.zip") != manifest["source_archive_sha256"]:
        raise RuntimeError("source archive hash mismatch")
    for relative, expected in manifest["sources"].items():
        if sha(source_root / relative) != expected:
            raise RuntimeError(f"frozen source hash mismatch: {relative}")
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(source_root),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    outcome_path = directory / "outcomes.jsonl"
    (directory / "RUN_STARTED").write_text(datetime.now(UTC).isoformat() + "\n")
    progress_path = directory / "progress.jsonl"
    for completed, task in enumerate(manifest["jobs"], start=1):
        input_key = f"{task['link']}-m{task['m']}-{task['seed']}"
        input_path = directory / manifest["inputs"][input_key]["path"]
        if sha(input_path) != manifest["inputs"][input_key]["sha256"]:
            raise RuntimeError(f"frozen input hash mismatch: {input_key}")
        stem = f"{input_key}-{task['arm']}"
        output_path = directory / "fits" / stem
        log_path = directory / "logs" / f"{stem}.log"
        started = time.perf_counter()
        command = [
            sys.executable,
            "-m",
            "experiments.local_score_spectral_study",
            "worker",
            str(input_path),
            str(output_path),
            str(task["link"]),
            str(task["m"]),
            str(task["seed"]),
            str(task["arm"]),
        ]
        try:
            proc = subprocess.run(
                command,
                cwd=source_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=manifest["timeout_seconds"],
                check=False,
            )
            log_path.write_text(proc.stdout + proc.stderr)
            row = (
                json.loads(output_path.with_suffix(".json").read_text())
                if proc.returncode == 0
                else {**task, "status": "error", "returncode": proc.returncode}
            )
        except subprocess.TimeoutExpired as exc:
            row, log_text = timeout_outcome(task, exc)
            log_path.write_text(log_text)
        except Exception as exc:  # noqa: BLE001 - retain every planned fit outcome
            log_path.write_text(f"{type(exc).__name__}: {exc}\n")
            row = {**task, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        row["process_seconds"] = time.perf_counter() - started
        row["log_sha256"] = sha(log_path)
        with outcome_path.open("a") as stream:
            stream.write(json.dumps(row, default=_json_default, allow_nan=False) + "\n")
        progress_row = {
            "completed": completed,
            "planned": len(manifest["jobs"]),
            "task": task,
        }
        with progress_path.open("a") as stream:
            stream.write(json.dumps(progress_row) + "\n")
        if completed % 10 == 0 or completed == len(manifest["jobs"]):
            print(f"completed {completed}/{len(manifest['jobs'])} fits", flush=True)


def _as_text(value) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else value or ""


def timeout_outcome(task: dict, exc: subprocess.TimeoutExpired) -> tuple[dict, str]:
    """Normalize timeout evidence across Python versions that return bytes."""
    return {**task, "status": "timeout"}, _as_text(exc.stdout) + _as_text(exc.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument(
        "phase", choices=("selection", "validation"), default="selection", nargs="?"
    )
    run_parser = sub.add_parser("run")
    run_parser.add_argument("directory", type=Path)
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("input", type=Path)
    worker_parser.add_argument("output", type=Path)
    worker_parser.add_argument("link")
    worker_parser.add_argument("m", type=int)
    worker_parser.add_argument("seed", type=int)
    worker_parser.add_argument("arm")
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.phase))
    elif args.command == "run":
        run(args.directory.resolve())
    else:
        worker(args.input, args.output, args.link, args.m, args.seed, args.arm)


if __name__ == "__main__":
    main()
