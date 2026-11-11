"""S2: paired LSMR/HYBRID comparison on S1 frozen multi-index tasks."""

from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path
from time import perf_counter

import numpy as np

from ADP.solver.HYBRID import solve as solve_hybrid
from ADP.solver.LSMR import solve as solve_lsmr

from .multi_solver_certificate import evaluate
from .multi_solver_search import _sha256


def _timeout(_signum, _frame):
    raise TimeoutError("solver exceeded the 60 s frozen-task limit")


def _trial(method, B, U, I, mass) -> dict[str, object]:
    started = perf_counter()
    signal.setitimer(signal.ITIMER_REAL, 60.0)
    try:
        result = method(
            B.copy(),
            U.copy(),
            I.copy(),
            mass=mass.copy(),
            lambda_prox=0.05,
            max_steps=80,
            tol=1e-6,
            theta=0.1,
        )
        elapsed = perf_counter() - started
    except (TimeoutError, RuntimeError, ValueError, FloatingPointError) as error:
        return {
            "status": "timeout" if isinstance(error, TimeoutError) else "error",
            "time_sec": perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
    reference = evaluate(result.index, U, I, mass)
    diagnostics = result.diagnostics
    for field in ("riemannian_gradient", "local_gradient", "orthogonality"):
        if not np.isclose(
            getattr(reference, field), diagnostics[field], rtol=1e-8, atol=1e-10
        ):
            raise RuntimeError(f"independent certificate mismatch: {field}")
    if not np.isclose(reference.objective, diagnostics["loss"], rtol=1e-8, atol=1e-10):
        raise RuntimeError("independent objective mismatch")
    return {
        "status": "success",
        "time_sec": elapsed,
        "certificate": {
            "objective": reference.objective,
            "riemannian_gradient": reference.riemannian_gradient,
            "horizontal_gradient": reference.horizontal_gradient,
            "local_gradient": reference.local_gradient,
            "orthogonality": reference.orthogonality,
            "rank_loss": reference.rank_loss,
        },
        "diagnostics": diagnostics,
    }


def _warmup() -> None:
    rng = np.random.default_rng(7)
    U = rng.normal(size=(5, 6, 4))
    I = rng.normal(size=(5, 6))
    B = np.linalg.qr(rng.normal(size=(4, 2)))[0].T
    for method in (solve_lsmr, solve_hybrid):
        method(B, U, I, mass=np.ones(5), lambda_prox=1.0, max_steps=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args(argv)
    frozen = json.loads((options.frozen_dir / "manifest.json").read_text())
    tasks = [
        (fit, call) for fit in frozen["fits"] for call in fit["calls"] if "file" in call
    ]
    if len(tasks) != 12:
        raise ValueError(f"expected 12 frozen tasks, found {len(tasks)}")
    budget = {
        "frozen_tasks": 12,
        "methods": ("lsmr", "hybrid"),
        "trials": 24,
        "timeout_per_trial_sec": 60,
    }
    if options.dry_run:
        print(json.dumps(budget, indent=2))
        return 0
    if options.output.exists():
        raise FileExistsError(options.output)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "budget": budget,
        "frozen_manifest": str(options.frozen_dir / "manifest.json"),
        "frozen_manifest_sha256": _sha256(options.frozen_dir / "manifest.json"),
        "trials": [],
    }
    _warmup()
    old_handler = signal.signal(signal.SIGALRM, _timeout)
    try:
        for index, (fit, call) in enumerate(tasks):
            path = options.frozen_dir / call["file"]
            if _sha256(path) != call["sha256"]:
                raise RuntimeError(f"frozen task hash mismatch: {path}")
            with np.load(path, allow_pickle=False) as arrays:
                B, U, I, mass = (arrays[key] for key in ("B", "U", "I", "mass"))
                order = (
                    (("lsmr", solve_lsmr), ("hybrid", solve_hybrid))
                    if index % 2 == 0
                    else (("hybrid", solve_hybrid), ("lsmr", solve_lsmr))
                )
                for name, method in order:
                    trial = _trial(method, B, U, I, mass)
                    trial.update(
                        point=fit["point"],
                        seed=fit["seed"],
                        outer=call["outer"],
                        method=name,
                        frozen_file=call["file"],
                        frozen_sha256=call["sha256"],
                    )
                    result["trials"].append(trial)
                    options.output.write_text(json.dumps(result, indent=2) + "\n")
                    print(
                        f"{fit['point']} seed {fit['seed']} outer {call['outer']} "
                        f"{name}: {trial['status']} {trial['time_sec']:.3f}s",
                        flush=True,
                    )
    finally:
        signal.signal(signal.SIGALRM, old_handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
