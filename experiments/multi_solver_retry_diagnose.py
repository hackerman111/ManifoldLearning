"""Ограниченная диагностика прежних frozen solver failures, без нового solver."""

from __future__ import annotations

import argparse
import json
import math
import platform
import signal
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import scipy
from scipy.linalg import null_space

from ADP.solver import LSMR

from .multi_solver_derivation import _retract, evaluate_reduced
from .multi_solver_search import _git, _sha256

_lsmr_method: Any = LSMR.sparse_linalg.lsmr


def _envelope(B, U, I, mass):
    """Диагностическая envelope-формула; rank guards проверяются reference."""
    coefficients, _ = LSMR._local_refit(I, U, B)
    residual = I - LSMR._predict(U, B, coefficients)
    gradient = -(mass[:, None] * coefficients).T @ (
        U.swapaxes(1, 2) @ residual[..., None]
    ).squeeze(-1)
    gradient -= (gradient @ B.T) @ B
    return gradient


def _curvature(B, U, I, mass):
    """Малый dense Hessian reference только при d<=100, не production."""
    if B.shape[1] > 100:
        raise ValueError("dense diagnostic is bounded to d<=100")
    reference = evaluate_reduced(B, U, I, mass)
    expected = reference.gradient - (reference.gradient @ B.T) @ B
    gradient = _envelope(B, U, I, mass)
    np.testing.assert_allclose(gradient, expected, rtol=1e-7, atol=1e-8)
    complement = null_space(B)
    shape = (B.shape[0], complement.shape[1])
    size = math.prod(shape)
    rows = []
    for step in (1e-4, 3e-5):
        hessian = np.empty((size, size))
        for k in range(size):
            coordinate = np.zeros(size)
            coordinate[k] = 1
            tangent = coordinate.reshape(shape) @ complement.T
            plus = _envelope(_retract(B, step * tangent), U, I, mass)
            minus = _envelope(_retract(B, -step * tangent), U, I, mass)
            hessian[:, k] = ((plus - minus) @ complement / (2 * step)).ravel()
        eigenvalues, eigenvectors = np.linalg.eigh((hessian + hessian.T) / 2)
        direction = eigenvectors[:, 0].reshape(shape) @ complement.T
        nearby = []
        for distance in (-0.01, 0.01):
            trial = _retract(B, distance * direction)
            c, _ = LSMR._local_refit(I, U, trial)
            nearby.append(LSMR._loss(I, U, trial, c, mass))
        rows.append(
            {
                "fd_step": step,
                "eigenvalue_min": float(eigenvalues[0]),
                "eigenvalue_max": float(eigenvalues[-1]),
                "asymmetry_relative": float(
                    np.linalg.norm(hessian - hessian.T)
                    / max(1.0, np.linalg.norm(hessian))
                ),
                "nearby_objectives": nearby,
            }
        )
    timings = {}
    for name, method in (("reference", evaluate_reduced), ("envelope", _envelope)):
        method(B, U, I, mass)
        times = []
        for _ in range(5):
            started = perf_counter()
            method(B, U, I, mass)
            times.append(perf_counter() - started)
        timings[name] = float(np.median(times))
    return {
        "objective": reference.objective,
        "gradient_norm": float(np.linalg.norm(gradient)),
        "value_defect": reference.value_defect,
        "gradient_defect": reference.gradient_defect,
        "curvature": rows,
        "evaluation_sec": timings,
    }


def _linear_diagnosis(B, U, I, mass):
    original = LSMR._global_correction
    rows = []
    probes = []

    def traced(I, U, index, coefficients, mass, ridge, tol, maxiter):
        result = original(I, U, index, coefficients, mass, ridge, tol, maxiter)
        row = {
            "ridge": ridge,
            "stop": result[1],
            "iterations": result[2],
            "ratio": result[3],
            "normal_norm": result[4],
            "correction_norm": float(np.linalg.norm(result[0])),
        }
        rows.append(row)
        # Два различных reject-снимка; solver trajectory не меняется.
        if result[3] > 0.1 and len(probes) < 2:
            operator = LSMR._linear_operator(
                U, coefficients, np.sqrt(mass), index.shape
            )
            residual = (
                np.sqrt(mass)[:, None] * (I - LSMR._predict(U, index, coefficients))
            ).ravel()
            rhs = operator.rmatvec(residual)
            variants = []
            for inner_tol in (1e-10, 1e-12, 1e-14):
                trial = _lsmr_method(
                    operator,
                    residual,
                    damp=math.sqrt(ridge),
                    atol=inner_tol,
                    btol=inner_tol,
                    maxiter=maxiter or max(50, 5 * index.size),
                )
                normal = rhs - (
                    operator.rmatvec(operator @ trial[0]) + ridge * trial[0]
                )
                variants.append(
                    {
                        "atol": inner_tol,
                        "stop": int(trial[1]),
                        "iterations": int(trial[2]),
                        "ratio": float(
                            np.linalg.norm(normal) / (ridge * np.linalg.norm(trial[0]))
                        ),
                        "estimated_condition": float(trial[6]),
                        "norm_A": float(trial[5]),
                        "normal_rhs": float(np.linalg.norm(rhs)),
                        "residual_norm": float(np.linalg.norm(residual)),
                    }
                )
            probes.append({"call": len(rows), "variants": variants})
        return result

    LSMR._global_correction = traced
    started = perf_counter()
    try:
        result = LSMR.solve(B, U, I, mass=mass, lambda_prox=0.05, max_steps=80)
    finally:
        LSMR._global_correction = original
    return {
        "time_including_probes_sec": perf_counter() - started,
        "diagnostics": result.diagnostics,
        "linear_calls": rows,
        "reject_probes": probes,
    }


def _timeout(_signum, _frame):
    raise TimeoutError("bounded diagnostic timed out")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--previous-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    budget = {
        "linear_tasks": 2,
        "linear_timeout": 60,
        "curvature_tasks": 3,
        "curvature_timeout": 90,
        "total_limit_sec": 390,
        "blas_threads": 1,
    }
    if args.dry_run:
        print(json.dumps(budget, indent=2))
        return 0
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.frozen_dir / "manifest.json").read_text())
    previous = json.loads(args.previous_results.read_text())["trials"]
    output = {
        "budget": budget,
        "commit": _git("rev-parse", "HEAD"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "source_sha256": {
            _name: _sha256(Path(_name))
            for _name in (
                __file__,
                "ADP/solver/LSMR.py",
                "experiments/reduced_lbfgs.py",
                "experiments/multi_solver_derivation.py",
            )
        },
        "previous_sha256": _sha256(args.previous_results),
        "tasks": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGALRM, _timeout)
    for fit in manifest["fits"]:
        if fit["point"] != "n1000":
            continue
        for call in fit["calls"]:
            if "file" not in call:
                continue
            selected = [
                row
                for row in previous
                if row["point"] == fit["point"]
                and row["seed"] == fit["seed"]
                and row["outer"] == call["outer"]
            ]
            base = next(row for row in selected if row["method"] == "lsmr")
            reduced = next(row for row in selected if row["method"] == "reduced-lbfgs")
            curvature = (
                reduced["diagnostics"]["loss"] > base["diagnostics"]["loss"] + 0.01
            )
            linear = (fit["seed"], call["outer"]) in ((1000, 0), (1001, 2))
            if not curvature and not linear:
                continue
            file = args.frozen_dir / call["file"]
            if _sha256(file) != call["sha256"]:
                raise RuntimeError(f"frozen hash mismatch: {file}")
            with np.load(file, allow_pickle=False) as data:
                B, U, I, mass = (data[key] for key in ("B", "U", "I", "mass"))
            row = {
                "seed": fit["seed"],
                "outer": call["outer"],
                "file": call["file"],
                "sha256": call["sha256"],
            }
            for name, enabled, limit, method, basis in (
                ("curvature", curvature, 90, _curvature, np.array(reduced["final_B"])),
                ("linear", linear, 60, _linear_diagnosis, B),
            ):
                if not enabled:
                    continue
                signal.setitimer(signal.ITIMER_REAL, limit)
                try:
                    row[name] = method(basis, U, I, mass)
                except Exception as error:  # Сохраняем диагностические ошибки.
                    row[name] = {"error": f"{type(error).__name__}: {error}"}
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
            output["tasks"].append(row)
            args.output.write_text(json.dumps(output, indent=2) + "\n")
            print(f"seed{fit['seed']} outer{call['outer']}: saved", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
