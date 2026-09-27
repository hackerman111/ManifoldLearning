"""S4: isolated paired frozen-task timing with common certificates and RSS."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import signal
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ADP.solver import LSMR

from . import reduced_gauss_newton, reduced_lbfgs, warm_reduced
from .multi_solver_certificate import evaluate
from .multi_solver_search import _git, _sha256

_METHODS = {
    "lsmr": (LSMR.solve, 80),
    "reduced-lbfgs": (reduced_lbfgs.solve, 320),
    "warm-reduced": (warm_reduced.solve, 320),
    "reduced-gn": (reduced_gauss_newton.solve, 160),
}


def _timeout(_signum, _frame):
    raise TimeoutError("frozen solver exceeded 60 seconds")


def _warmup(method) -> None:
    rng = np.random.default_rng(7)
    U = rng.normal(size=(5, 6, 4))
    I = rng.normal(size=(5, 6))
    B = np.linalg.qr(rng.normal(size=(4, 2)))[0].T
    method(B, U, I, mass=np.ones(5), lambda_prox=1.0, max_steps=1)


def _child(file: Path, expected_hash: str, method_name: str, output: Path) -> int:
    method, cap = _METHODS[method_name]
    if _sha256(file) != expected_hash:
        raise RuntimeError(f"frozen input hash mismatch: {file}")
    with np.load(file, allow_pickle=False) as arrays:
        B, U, I, mass = (arrays[key] for key in ("B", "U", "I", "mass"))
    _warmup(method)
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    trace: list[dict[str, float]] = []
    modules = (
        (LSMR, reduced_lbfgs)
        if method_name == "warm-reduced"
        else (
            reduced_gauss_newton
            if method_name == "reduced-gn"
            else LSMR
            if method_name == "lsmr"
            else reduced_lbfgs,
        )
    )
    original_stationarity: dict[Any, Any] = {
        module: module._stationarity for module in modules
    }
    started = perf_counter()

    def make_trace(original):
        def traced_stationarity(I, U, index, coefficients, mass, loss, **settings):
            scores = original(I, U, index, coefficients, mass, loss, **settings)
            trace.append(
                {
                    "elapsed_sec": perf_counter() - started,
                    "objective": float(loss),
                    "riemannian_gradient": scores[0],
                    "local_gradient": scores[1],
                    "orthogonality": scores[2],
                }
            )
            return scores

        return traced_stationarity

    for module, original in original_stationarity.items():
        module._stationarity = make_trace(original)
    signal.signal(signal.SIGALRM, _timeout)
    signal.setitimer(signal.ITIMER_REAL, 60.0)
    result = None
    try:
        result = method(B, U, I, mass=mass, lambda_prox=0.05, max_steps=cap, tol=1e-6)
        elapsed = perf_counter() - started
        row: dict[str, object] = {
            "status": "success",
            "time_sec": elapsed,
            "diagnostics": result.diagnostics,
            "final_B": result.index.tolist(),
        }
    except Exception as error:  # Эксперимент сохраняет все неуспешные trials.
        row = {
            "status": "timeout" if isinstance(error, TimeoutError) else "error",
            "time_sec": min(60.0, perf_counter() - started),
            "error": f"{type(error).__name__}: {error}",
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        for module, original in original_stationarity.items():
            module._stationarity = original
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    row.update(
        method=method_name,
        cap=cap,
        trace=trace,
        peak_rss_bytes=rss_after,
        additional_peak_rss_bytes=max(0, rss_after - rss_before),
        frozen_file=file.name,
        frozen_sha256=expected_hash,
    )
    if row["status"] == "success" and result is not None:
        reference = evaluate(result.index, U, I, mass)
        row["certificate"] = {
            "objective": reference.objective,
            "riemannian_gradient": reference.riemannian_gradient,
            "local_gradient": reference.local_gradient,
            "orthogonality": reference.orthogonality,
            "rank_loss": reference.rank_loss,
        }
        for field in ("riemannian_gradient", "local_gradient", "orthogonality"):
            if not np.isclose(
                getattr(reference, field),
                result.diagnostics[field],
                rtol=1e-8,
                atol=1e-10,
            ):
                row.update(
                    status="certificate_mismatch",
                    error=f"independent {field} differs from solver diagnostic",
                )
        if not np.isclose(
            reference.objective, result.diagnostics["loss"], rtol=1e-8, atol=1e-10
        ):
            row.update(status="certificate_mismatch", error="objective mismatch")
    output.write_text(json.dumps(row, indent=2) + "\n")
    return 0


def _parent(
    frozen_dir: Path,
    output_dir: Path,
    dry_run: bool,
    methods: tuple[str, ...],
    point: str | None,
) -> int:
    manifest_path = frozen_dir / "manifest.json"
    frozen = json.loads(manifest_path.read_text())
    tasks = [
        (fit, call)
        for fit in frozen["fits"]
        if point is None or fit["point"] == point
        for call in fit["calls"]
        if "file" in call
    ]
    expected = 12 if point is None else 6
    if len(tasks) != expected:
        raise ValueError(f"expected {expected} frozen tasks, found {len(tasks)}")
    budget = {
        "tasks": len(tasks),
        "point": point,
        "methods": {name: _METHODS[name][1] for name in methods},
        "trials": len(tasks) * len(methods),
        "timeout_per_trial_sec": 60,
        "total_trial_limit_sec": 60 * len(tasks) * len(methods),
        "blas_threads": 1,
    }
    if dry_run:
        print(json.dumps(budget, indent=2))
        return 0
    output_dir.mkdir(parents=True, exist_ok=False)
    trials: list[dict[str, object]] = []
    result: dict[str, object] = {
        "budget": budget,
        "frozen_manifest_sha256": _sha256(manifest_path),
        "commit": _git("rev-parse", "HEAD"),
        "source_sha256": {
            name: _sha256(Path(name))
            for name in (
                "ADP/solver/LSMR.py",
                "ADP/solver/HYBRID/HYBRID.py",
                "ADP/solver/HYBRID/HYBRID_multi.py",
                "experiments/reduced_lbfgs.py",
                "experiments/reduced_gauss_newton.py",
                "experiments/warm_reduced.py",
                "experiments/multi_solver_derivation.py",
                "experiments/multi_solver_certificate.py",
                "experiments/multi_solver_frozen.py",
            )
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "threads": {
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "trials": trials,
    }
    trials_dir = output_dir / "trials"
    trials_dir.mkdir()
    result_path = output_dir / "results.json"
    for index, (fit, call) in enumerate(tasks):
        file = frozen_dir / call["file"]
        if _sha256(file) != call["sha256"]:
            raise RuntimeError(f"frozen input hash mismatch: {file}")
        order = methods if index % 2 == 0 else tuple(reversed(methods))
        for method_name in order:
            trial_path = trials_dir / f"{file.stem}-{method_name}.json"
            command = (
                sys.executable,
                "-m",
                "experiments.multi_solver_frozen",
                "--child",
                "--frozen-file",
                str(file),
                "--sha256",
                call["sha256"],
                "--method",
                method_name,
                "--output",
                str(trial_path),
            )
            try:
                process = subprocess.run(
                    command, capture_output=True, text=True, timeout=65, check=False
                )
                if process.returncode == 0 and trial_path.exists():
                    row = json.loads(trial_path.read_text())
                else:
                    row = {
                        "status": "process_error",
                        "error": process.stderr[-2000:],
                        "time_sec": None,
                    }
            except subprocess.TimeoutExpired:
                row = {
                    "status": "timeout",
                    "time_sec": 60.0,
                    "error": "parent killed trial after 65 s",
                }
            row.update(
                point=fit["point"],
                seed=fit["seed"],
                outer=call["outer"],
                method=method_name,
                frozen_sha256=call["sha256"],
            )
            trials.append(row)
            result_path.write_text(json.dumps(result, indent=2) + "\n")
            print(
                f"{fit['point']} seed {fit['seed']} outer {call['outer']} "
                f"{method_name}: {row['status']} {row['time_sec']}s",
                flush=True,
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path)
    parser.add_argument("--frozen-file", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--method", choices=tuple(_METHODS))
    parser.add_argument("--methods", nargs="+", choices=tuple(_METHODS))
    parser.add_argument("--point")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args(argv)
    if options.child:
        if not options.frozen_file or not options.sha256 or not options.method:
            parser.error("child requires frozen file, hash and method")
        return _child(
            options.frozen_file, options.sha256, options.method, options.output
        )
    if not options.frozen_dir:
        parser.error("parent requires --frozen-dir")
    methods = tuple(options.methods or ("lsmr", "reduced-lbfgs"))
    if len(set(methods)) != len(methods):
        parser.error("--methods may not repeat a method")
    return _parent(
        options.frozen_dir, options.output, options.dry_run, methods, options.point
    )


if __name__ == "__main__":
    raise SystemExit(main())
