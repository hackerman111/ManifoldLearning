"""Paired original/optimized Grassmann measurements on saved Spokoiny inputs.

Examples (run after the optimized module is ready):

    python -m benchmarks.grassman_optim_benchmark --stage frozen \
        --output experiments/grassman_optim_2026_10_01/frozen_final
    python -m benchmarks.grassman_optim_benchmark --stage fullfit \
        --output experiments/grassman_optim_2026_10_01/fullfit_final

Every solver/seed measurement runs in a fresh subprocess. The driver alternates
solver order by seed and pins BLAS thread counts before Python imports NumPy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
import platform
import resource
import subprocess
import sys
import time
from contextlib import redirect_stdout
from dataclasses import fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = ROOT / "experiments/grassman_2026_10_01/benchmark/frozen_ablation/frozen"
THREAD_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
SOLVERS = ("grassman", "grassman_optim")
DEFAULT_SEEDS = (0, 1, 2)
SOLVER_OPTIONS: dict[str, object] = {
    "method": "core_gn",
    "angle_backend": "schur",
    "max_steps": 5,
    "tol": 1e-6,
    "energy_tol": 0.05,
    "local_ridge": 0.0,
    "max_angle": 0.5,
    "workspace_bytes": 16 * 1024**2,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _plain(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _array_sha256(array: Any) -> str:
    import numpy as np

    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(repr(value.shape).encode("ascii"))
    digest.update(value.view(np.uint8))
    return digest.hexdigest()


def _config_values(config: Any) -> dict[str, object]:
    values = {}
    for field in fields(config):
        value = getattr(config, field.name)
        if callable(value):
            value = f"{value.__module__}.{value.__qualname__}"
        values[field.name] = _plain(value)
    return values


def _solve(module_name: str):
    module = importlib.import_module(f"ADP.solver.{module_name}")
    return module.solve


def _worker(args: argparse.Namespace) -> dict[str, object]:
    import numpy as np

    solve = _solve(args.solver)
    if args.stage == "frozen":
        path = args.frozen_file
        if path is None:
            path = FROZEN_ROOT / f"spokoini_seed{args.seed}_first_outer.npz"
        with np.load(path, allow_pickle=False) as snapshot:
            index = snapshot["index_init"]
            U = np.ascontiguousarray(snapshot["U"])
            I = snapshot["I"]
            mass = snapshot["mass"]
            truth = snapshot["truth_basis"]
        kwargs = {"mass": mass, "lambda_prox": 0.05, **SOLVER_OPTIONS}
        solve(index, U, I, **kwargs)  # warm-up, not timed
        times = []
        result = None
        for _ in range(args.repeats):
            started = time.perf_counter()
            result = solve(index, U, I, **kwargs)
            times.append(time.perf_counter() - started)
        assert result is not None
        basis = np.asarray(result.index)
        projector_error = float(
            np.linalg.norm(
                basis.T @ basis - truth @ truth.T,
                ord="fro",
            )
            / np.sqrt(2 * basis.shape[0])
        )
        diagnostics = result.diagnostics
        return {
            "stage": "frozen",
            "seed": args.seed,
            "solver": args.solver,
            "status": "ok",
            "snapshot": str(path.relative_to(ROOT)),
            "snapshot_sha256": _sha256(path),
            "shape": {
                "J": U.shape[0],
                "p": U.shape[1],
                "d": U.shape[2],
                "m": index.shape[0],
            },
            "dtype": str(U.dtype),
            "warmups": 1,
            "times_sec": times,
            "median_sec": float(np.median(times)),
            "profile_objective": diagnostics["profile_objective"],
            "iterations": diagnostics["iterations"],
            "stop_reason": diagnostics["stop_reason"],
            "converged": diagnostics["converged"],
            "projector_distance_to_truth": projector_error,
            "basis": basis,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }

    from functools import partial

    from ADP import ADP_multi_index, ADP_solver
    from benchmarks.grassman_benchmark import PARAMS, _make_problem

    X, y, truth, config = _make_problem(args.seed)
    lambda_prox = float(config.lambda_penalty)
    solver = ADP_solver(
        partial(solve, **{**SOLVER_OPTIONS, "lambda_prox": lambda_prox})
    )
    started = time.perf_counter()
    model = ADP_multi_index(int(PARAMS["m"]), config, solver).fit(X, y)
    elapsed = time.perf_counter() - started
    basis = np.asarray(model.basis_)
    truth_projector = truth @ truth.T
    projector_distance = float(
        np.linalg.norm(basis @ basis.T - truth_projector, ord="fro")
        / np.sqrt(2 * int(PARAMS["m"]))
    )
    trace = model.trace_
    solver_trace = [entry.get("solver", {}) for entry in trace]
    last = solver_trace[-1] if solver_trace else {}
    count_fields = (
        "iterations",
        "accepted_steps",
        "profile_evaluations",
        "gn_solves",
        "rank_guard_fallbacks",
        "workspace_fallbacks",
    )
    diagnostics_summary = {
        field: sum(int(call.get(field, 0)) for call in solver_trace)
        for field in count_fields
    }
    diagnostics_summary["converged_calls"] = sum(
        bool(call.get("converged", False)) for call in solver_trace
    )
    diagnostics_summary["stop_reason_counts"] = {
        reason: sum(call.get("stop_reason") == reason for call in solver_trace)
        for reason in sorted({str(call.get("stop_reason")) for call in solver_trace})
    }
    return {
        "stage": "fullfit",
        "seed": args.seed,
        "solver": args.solver,
        "status": "ok",
        "shape": {key: PARAMS[key] for key in ("n", "d", "m", "N_J", "N_phi", "N_lin")},
        "dtype": str(X.dtype),
        "effective_config": _config_values(config),
        "lambda_prox": lambda_prox,
        "input_sha256": {
            "X": _array_sha256(X),
            "y": _array_sha256(y),
            "truth_basis": _array_sha256(truth),
        },
        "fit_time_sec": elapsed,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "projector_distance_to_truth": projector_distance,
        "outer_iterations": model.effective_parameters_.get("outer_iterations"),
        "selected_iteration": model.effective_parameters_.get("selected_iteration"),
        "fit_stop_reason": model.result_.stop_reason,
        "last_solver_call": {
            key: _plain(last.get(key))
            for key in (
                "profile_objective",
                "iterations",
                "converged",
                "stop_reason",
                "riemannian_gradient",
                "accepted_steps",
            )
        },
        "solver_calls": len(solver_trace),
        "solver_diagnostics_summary": diagnostics_summary,
        "solver_scalar_trace": [
            {
                key: _plain(call.get(key))
                for key in (
                    "profile_objective",
                    "iterations",
                    "converged",
                    "stop_reason",
                    "riemannian_gradient",
                    "accepted_steps",
                    "profile_evaluations",
                    "gn_solves",
                    "rank_guard_fallbacks",
                    "workspace_fallbacks",
                )
            }
            for call in solver_trace
        ],
        "basis": basis,
    }


def _child(args: argparse.Namespace) -> int:
    row = _worker(args)
    row["worker_version"] = 1
    print(json.dumps(_plain(row), allow_nan=False))
    return 0


def _run_one(args: argparse.Namespace, solver: str, seed: int) -> dict[str, object]:
    solver_path = ROOT / f"ADP/solver/{solver}.py"
    hash_before = _sha256(solver_path)
    command = [
        sys.executable,
        "-m",
        "benchmarks.grassman_optim_benchmark",
        "--worker",
        "--stage",
        args.stage,
        "--output",
        "/tmp/grassman_optim_worker",
        "--solver",
        solver,
        "--seed",
        str(seed),
        "--repeats",
        str(args.repeats),
    ]
    if args.frozen_file is not None:
        command += ["--frozen-file", str(args.frozen_file)]
    env = os.environ.copy()
    env.update(dict.fromkeys(THREAD_ENV, "1"))
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            check=False,
            text=True,
            timeout=args.timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as error:
        row = {
            "stage": args.stage,
            "seed": seed,
            "solver": solver,
            "status": "timeout",
            "error": str(error),
        }
        row["solver_sha256_before"] = hash_before
        row["solver_sha256_after"] = _sha256(solver_path)
        return row
    hash_after = _sha256(solver_path)
    if completed.returncode:
        row: dict[str, object] = {
            "stage": args.stage,
            "seed": seed,
            "solver": solver,
            "status": "process_error",
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr[-4000:],
            "stdout_tail": completed.stdout[-1000:],
        }
        row["solver_sha256_before"] = hash_before
        row["solver_sha256_after"] = hash_after
        return row
    try:
        row = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        row = {
            "stage": args.stage,
            "seed": seed,
            "solver": solver,
            "status": "invalid_worker_output",
            "error": f"{type(error).__name__}: {error}",
            "stdout_tail": completed.stdout[-2000:],
        }
    row["solver_sha256_before"] = hash_before
    row["solver_sha256_after"] = hash_after
    return row


def _paired(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    import numpy as np

    by_key = {
        (int(row["seed"]), str(row["solver"])): row
        for row in rows
        if row.get("status") == "ok"
    }
    pairs = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        original = by_key.get((seed, "grassman"))
        optimized = by_key.get((seed, "grassman_optim"))
        if original is None or optimized is None:
            continue
        left, right = np.asarray(original["basis"]), np.asarray(optimized["basis"])
        projector_left = (
            left @ left.T if args_stage(rows) == "fullfit" else left.T @ left
        )
        projector_right = (
            right @ right.T if args_stage(rows) == "fullfit" else right.T @ right
        )
        diff = projector_left - projector_right
        pair: dict[str, object] = {
            "seed": seed,
            "original_solver": "grassman",
            "optimized_solver": "grassman_optim",
            "projector_frobenius_difference": float(np.linalg.norm(diff, ord="fro")),
        }
        if args_stage(rows) == "fullfit":
            pair["runtime_ratio_optimized_over_original"] = float(
                optimized["fit_time_sec"]
            ) / float(original["fit_time_sec"])
            pair["rss_delta_kib_optimized_minus_original"] = int(
                optimized["peak_rss_kib"]
            ) - int(original["peak_rss_kib"])
        else:
            pair["runtime_ratio_optimized_over_original"] = float(
                optimized["median_sec"]
            ) / float(original["median_sec"])
        pairs.append(pair)
    return pairs


def args_stage(rows: list[dict[str, object]]) -> str:
    return str(rows[0].get("stage", "")) if rows else ""


def _driver(args: argparse.Namespace) -> None:
    import numpy as np
    import scipy

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    row_path = output / f"{args.stage}_runs.jsonl"
    protocol_path = output / f"{args.stage}_protocol.json"
    if row_path.exists() or protocol_path.exists():
        raise FileExistsError(f"refusing to overwrite benchmark artifacts in {output}")
    runs = []
    with row_path.open("x", encoding="utf-8") as stream:
        for position, seed in enumerate(args.seeds):
            order = SOLVERS if position % 2 == 0 else tuple(reversed(SOLVERS))
            for solver in order:
                row = _run_one(args, solver, seed)
                row["order_position"] = len(runs)
                runs.append(row)
                stream.write(json.dumps(_plain(row), allow_nan=False) + "\n")
                stream.flush()
                print(
                    json.dumps(
                        {
                            key: row.get(key)
                            for key in (
                                "seed",
                                "solver",
                                "status",
                                "fit_time_sec",
                                "median_sec",
                                "peak_rss_kib",
                            )
                        },
                        allow_nan=False,
                    )
                )
    protocol = {
        "stage": args.stage,
        "seeds": args.seeds,
        "solver_order_by_seed": {
            str(seed): list(SOLVERS if i % 2 == 0 else tuple(reversed(SOLVERS)))
            for i, seed in enumerate(args.seeds)
        },
        "solver_options": SOLVER_OPTIONS,
        "thread_env": dict.fromkeys(THREAD_ENV, "1"),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas_configuration": _blas_configuration(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "benchmark_sha256": _sha256(Path(__file__)),
        "solver_sha256": {
            name: _sha256(ROOT / f"ADP/solver/{name}.py") for name in SOLVERS
        },
        "input_source_sha256": {
            path: _sha256(ROOT / path)
            for path in (
                "benchmarks/grassman_benchmark.py",
                "experiments/data.py",
                "experiments/models.py",
                "experiments/runner.py",
                "ADP/core/ADP_Config.py",
            )
        },
        "protocol": (
            "Fresh subprocess per solver/seed; fullfit perf_counter wraps fit; "
            "frozen stage performs one warm-up and warmed repeated calls on a saved "
            "first-outer Spokoiny snapshot. One BLAS thread. Failures remain in JSONL."
        ),
        "repeats": args.repeats if args.stage == "frozen" else None,
        "timeout_sec": args.timeout,
        "results_file": str(row_path.relative_to(ROOT)),
        "paired": _paired(runs),
        "failed_runs": [row for row in runs if row.get("status") != "ok"],
    }
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")


def _blas_configuration() -> str:
    import numpy as np

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        np.show_config()
    return buffer.getvalue()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("frozen", "fullfit"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--frozen-file", type=Path)
    parser.add_argument("--solver", choices=SOLVERS, default="grassman")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.worker:
        return _child(args)
    _driver(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
