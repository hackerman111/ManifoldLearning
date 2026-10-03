"""Paired SVD and optimized Grassmann full fits on heavy Spokoiny."""

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
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLVERS = ("svd", "grassman_optim")
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
GRASSMAN_OPTIONS = {
    "max_steps": 5,
    "tol": 1e-6,
    "energy_tol": 0.05,
    "local_ridge": 0.0,
    "max_angle": 0.5,
    "workspace_bytes": 16 * 1024**2,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(seed: int, solver_name: str) -> dict[str, object]:
    import numpy as np

    from ADP import ADP_multi_index, ADP_solver
    from benchmarks.grassman_benchmark import PARAMS, _make_problem

    X, y, truth, config = _make_problem(seed)
    if solver_name == "svd":
        from ADP.solver.SVD import solve

        solver = ADP_solver(solve, rank=1, direct_max_dimension=128)
        solver_settings = {"rank": 1, "direct_max_dimension": 128}
    else:
        from ADP.solver.grassman import solve

        solver = ADP_solver(
            partial(
                solve,
                **{**GRASSMAN_OPTIONS, "lambda_prox": float(config.lambda_penalty)},
            )
        )
        solver_settings = {
            **GRASSMAN_OPTIONS,
            "lambda_prox": float(config.lambda_penalty),
        }

    start = time.perf_counter()
    model = ADP_multi_index(int(PARAMS["m"]), config, solver).fit(X, y)
    elapsed = time.perf_counter() - start
    basis = np.asarray(model.basis_)
    truth_projector = truth @ truth.T
    projector_error = float(
        np.linalg.norm(basis @ basis.T - truth_projector, ord="fro")
        / np.sqrt(2 * int(PARAMS["m"]))
    )
    trace = model.trace_
    solver_trace = [entry.get("solver", {}) for entry in trace]
    last = solver_trace[-1] if solver_trace else {}

    def array_hash(value: np.ndarray) -> str:
        arr = np.ascontiguousarray(value)
        h = hashlib.sha256()
        h.update(str(arr.dtype).encode())
        h.update(repr(arr.shape).encode())
        h.update(arr.view(np.uint8))
        return h.hexdigest()

    return {
        "seed": seed,
        "solver": solver_name,
        "status": "ok",
        "shape": {k: PARAMS[k] for k in ("n", "d", "m", "N_J", "N_phi", "N_lin")},
        "dtype": str(X.dtype),
        "solver_settings": solver_settings,
        "input_hashes": {
            "X": array_hash(X),
            "y": array_hash(y),
            "truth": array_hash(truth),
        },
        "fit_time_sec": elapsed,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "projector_distance_to_truth": projector_error,
        "outer_iterations": model.effective_parameters_.get("outer_iterations"),
        "selected_iteration": model.effective_parameters_.get("selected_iteration"),
        "fit_stop_reason": model.result_.stop_reason,
        "solver_calls": len(solver_trace),
        "last_solver_diagnostics": {
            key: last.get(key)
            for key in (
                "iterations",
                "converged",
                "stop_reason",
                "profile_objective",
                "loss",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--solver", choices=SOLVERS, default="svd")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(fit(args.seed, args.solver), allow_nan=False))
        return

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "fullfit_runs.jsonl"
    protocol_path = output / "fullfit_protocol.json"
    if rows_path.exists() or protocol_path.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    env = os.environ.copy()
    env.update(dict.fromkeys(THREADS, "1"))
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    rows: list[dict[str, object]] = []
    with rows_path.open("x", encoding="utf-8") as stream:
        for position, seed in enumerate(args.seeds):
            order = SOLVERS if position % 2 == 0 else tuple(reversed(SOLVERS))
            for solver_name in order:
                command = [
                    sys.executable,
                    "-m",
                    "benchmarks.svd_grassman_optim_benchmark",
                    "--worker",
                    "--output",
                    str(output),
                    "--seed",
                    str(seed),
                    "--solver",
                    solver_name,
                ]
                try:
                    result = subprocess.run(
                        command,
                        cwd=ROOT,
                        env=env,
                        text=True,
                        capture_output=True,
                        timeout=900,
                        check=False,
                    )
                    if result.returncode:
                        row = {
                            "seed": seed,
                            "solver": solver_name,
                            "status": "process_error",
                            "stderr": result.stderr[-4000:],
                        }
                    else:
                        row = json.loads(result.stdout.strip().splitlines()[-1])
                except (
                    subprocess.TimeoutExpired,
                    IndexError,
                    json.JSONDecodeError,
                ) as error:
                    row = {
                        "seed": seed,
                        "solver": solver_name,
                        "status": "error",
                        "error": str(error),
                    }
                row["order_position"] = len(rows)
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                print(
                    json.dumps(
                        {
                            k: row.get(k)
                            for k in (
                                "seed",
                                "solver",
                                "status",
                                "fit_time_sec",
                                "peak_rss_kib",
                            )
                        }
                    ),
                    flush=True,
                )

    pairs = []
    for seed in args.seeds:
        by_solver = {
            r["solver"]: r
            for r in rows
            if r.get("seed") == seed and r.get("status") == "ok"
        }
        if all(name in by_solver for name in SOLVERS):
            svd, grass = by_solver["svd"], by_solver["grassman_optim"]
            pairs.append(
                {
                    "seed": seed,
                    "svd_time_sec": svd["fit_time_sec"],
                    "grassman_optim_time_sec": grass["fit_time_sec"],
                    "grassman_over_svd_time_ratio": grass["fit_time_sec"]
                    / svd["fit_time_sec"],
                    "svd_projector_distance": svd["projector_distance_to_truth"],
                    "grassman_optim_projector_distance": grass[
                        "projector_distance_to_truth"
                    ],
                    "svd_rss_kib": svd["peak_rss_kib"],
                    "grassman_optim_rss_kib": grass["peak_rss_kib"],
                    "inputs_match": svd["input_hashes"] == grass["input_hashes"],
                    "outer_iterations_match": svd["outer_iterations"]
                    == grass["outer_iterations"],
                }
            )
    import numpy as np

    protocol = {
        "seeds": args.seeds,
        "solver_order_by_seed": {
            str(seed): list(SOLVERS if i % 2 == 0 else tuple(reversed(SOLVERS)))
            for i, seed in enumerate(args.seeds)
        },
        "thread_env": dict.fromkeys(THREADS, "1"),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True
        ).stdout.strip(),
        "git_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                text=True,
                capture_output=True,
            ).stdout.strip()
        ),
        "source_sha256": {
            name: sha256(ROOT / name)
            for name in (
                "ADP/solver/SVD.py",
                "ADP/solver/grassman.py",
                "benchmarks/grassman_benchmark.py",
                __file__,
            )
        },
        "paired": pairs,
        "median_grassman_over_svd_time_ratio": float(
            np.median([p["grassman_over_svd_time_ratio"] for p in pairs])
        )
        if pairs
        else None,
        "failures": [r for r in rows if r.get("status") != "ok"],
        "interpretation": (
            "Full ADP_multi_index fit time. SVD and grassman "
            "are distinct algorithms; outputs and convergence "
            "need not match."
        ),
    }
    protocol_path.write_text(
        json.dumps(protocol, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
