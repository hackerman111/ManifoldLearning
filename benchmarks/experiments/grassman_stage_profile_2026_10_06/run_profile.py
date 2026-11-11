"""Paired production Grassmann fit and solve-stage profiler.

Examples (from repository root):
  UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m \
    experiments.grassman_stage_profile_2026_10_06.run_profile --output /tmp/gsp --smoke
  UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m \
    experiments.grassman_stage_profile_2026_10_06.run_profile --output benchmarks/results/outputs/grassman_stage_profile_2026_10_06/results

Workers are subprocesses with one BLAS/OMP/MKL thread. The profiler wraps the
production ``solve`` call. ``fit_time_sec`` spans the model's full ``fit``
method, including its preparation; post-fit serialization is outside that
timer. ``solver_wall_sec`` isolates solve time.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import pstats
import platform
import resource
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[3]
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
OPTIONS = {
    "max_steps": 5, "tol": 1e-6, "energy_tol": 0.05,
    "local_ridge": 0.0, "max_angle": 0.5, "workspace_bytes": 16 * 1024**2,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profiled_fit(seed: int, mode: str, profile_path: Path | None) -> dict[str, Any]:
    import numpy as np
    import scipy
    from threadpoolctl import threadpool_info
    from ADP import ADP_multi_index, ADP_solver
    from benchmarks.grassman_benchmark import PARAMS, _make_problem
    from ADP.solver import grassman
    from functools import partial

    X, y, truth, config = _make_problem(seed)
    counters: dict[str, dict[str, float | int]] = defaultdict(lambda: {"inclusive_sec": 0.0, "exclusive_sec": 0.0, "calls": 0})
    stack: list[tuple[str, float, float]] = []

    def instrument(name: str, fn: Callable[..., Any], *, category: Callable[[], str] | str | None = None):
        def wrapped(*args: Any, **kwargs: Any):
            # Helper/linalg wrappers are installed for the process lifetime,
            # but must be inert during model preparation and postprocessing.
            if name.startswith("numpy.linalg.") and not any(
                entry[0] == "solve" for entry in stack
            ):
                return fn(*args, **kwargs)
            cat = category() if callable(category) else (category or name)
            t0 = time.perf_counter()
            stack.append((cat, t0, 0.0))
            try:
                return fn(*args, **kwargs)
            finally:
                end = time.perf_counter()
                _, start, child = stack.pop()
                elapsed = end - start
                row = counters[cat]
                row["inclusive_sec"] = float(row["inclusive_sec"]) + elapsed
                row["exclusive_sec"] = float(row["exclusive_sec"]) + max(0.0, elapsed - child)
                row["calls"] = int(row["calls"]) + 1
                if stack:
                    parent_cat, parent_start, parent_child = stack[-1]
                    stack[-1] = (parent_cat, parent_start, parent_child + elapsed)
        wrapped.__name__ = getattr(fn, "__name__", name)
        wrapped.__qualname__ = getattr(fn, "__qualname__", name)
        return wrapped

    if mode in ("profiled", "timed"):
        # Semantic stage wrappers; inclusive times remain available, while the
        # exclusive leaf accounting avoids counting nested helpers twice.
        for name in ("_project", "_gradient", "_profile", "_core_jacobian", "_polar"):
            setattr(grassman, name, instrument(name, getattr(grassman, name)))

        def linalg_category(op: str) -> str:
            active = [entry[0] for entry in stack]
            if "_profile" in active:
                return "profile_" + op
            if op == "lstsq":
                return "gn_lstsq" if "solve" in active else "other_lstsq"
            if "solve" in active:
                return "geometry_" + op
            return "other_" + op

        for op in ("lstsq", "svd", "qr"):
            original = getattr(np.linalg, op)
            setattr(np.linalg, op, instrument("numpy.linalg." + op, original, category=lambda op=op: linalg_category(op)))

        original_solve = grassman.solve
        profiler = cProfile.Profile() if mode == "profiled" else None
        solve_clock = {"elapsed": 0.0}

        def solve_wrapper(*args: Any, **kwargs: Any):
            begin = time.perf_counter()
            try:
                if profiler is None:
                    return original_solve(*args, **kwargs)
                return profiler.runcall(original_solve, *args, **kwargs)
            finally:
                solve_clock["elapsed"] += time.perf_counter() - begin
                if len(stack) != 1 or stack[-1][0] != "solve":
                    raise RuntimeError("stage instrumentation stack leaked across solve")

        # Wrap solve so its entire inclusive wall time is covered. Timers also
        # capture helper-exclusive buckets; remainder is derived after fit.
        grassman.solve = instrument("solve", solve_wrapper)
    else:
        solve_clock = {"elapsed": 0.0}
        profiler = None
        original_solve = grassman.solve

        def solve_wrapper(*args: Any, **kwargs: Any):
            begin = time.perf_counter()
            try:
                return original_solve(*args, **kwargs)
            finally:
                solve_clock["elapsed"] += time.perf_counter() - begin

        grassman.solve = solve_wrapper

    solver = ADP_solver(partial(grassman.solve, **{**OPTIONS, "lambda_prox": float(config.lambda_penalty)}))
    fit_start = time.perf_counter()
    model = ADP_multi_index(int(PARAMS["m"]), config, solver).fit(X, y)
    fit_elapsed = time.perf_counter() - fit_start
    basis = np.ascontiguousarray(model.basis_)
    def array_hash(value: np.ndarray) -> str:
        arr = np.ascontiguousarray(value)
        h = hashlib.sha256()
        h.update(str(arr.dtype).encode())
        h.update(repr(arr.shape).encode())
        h.update(arr.tobytes())
        return h.hexdigest()
    digest = array_hash(basis)
    truth_projector = truth @ truth.T
    quality = float(np.linalg.norm(basis @ basis.T - truth_projector, ord="fro") / np.sqrt(2 * int(PARAMS["m"])))
    trace = model.trace_
    solver_trace = [entry.get("solver", {}) for entry in trace]
    last = solver_trace[-1] if solver_trace else {}
    def json_default(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (tuple, set)):
            return list(value)
        return repr(value)
    diagnostics_payload = json.dumps(solver_trace, sort_keys=True, separators=(",", ":"), default=json_default, allow_nan=False)

    result: dict[str, Any] = {
        "seed": seed, "mode": mode, "status": "ok",
        "shape": {k: PARAMS[k] for k in ("n", "d", "m", "N_J", "N_phi", "N_lin")},
        "dtype": str(X.dtype), "solver_settings": {**OPTIONS, "lambda_prox": float(config.lambda_penalty)},
        "input_hashes": {k: array_hash(v) for k, v in (("X", X), ("y", y), ("truth", truth))},
        "basis_hash": digest, "fit_time_sec": fit_elapsed,
        "solver_wall_sec": solve_clock["elapsed"],
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "projector_distance_to_truth": quality,
        "outer_iterations": model.effective_parameters_.get("outer_iterations"),
        "selected_iteration": model.effective_parameters_.get("selected_iteration"),
        "fit_stop_reason": model.result_.stop_reason, "solver_calls": len(solver_trace),
        "diagnostics_hash": hashlib.sha256(diagnostics_payload.encode()).hexdigest(),
        "aggregate_diagnostics": {
            key: sum(int(item.get(key, 0) or 0) for item in solver_trace)
            for key in ("accepted_steps", "iterations", "gn_solves", "profile_evaluations", "rank_guard_fallbacks", "workspace_fallbacks")
        } | {"max_orthogonality_error": max((float(item.get("orthogonality_error", 0.0) or 0.0) for item in solver_trace), default=0.0)},
        "last_solver_diagnostics": {k: last.get(k) for k in ("iterations", "converged", "stop_reason", "profile_objective", "loss", "gn_solves", "profile_evaluations", "step_ranks", "workspace_fallbacks")},
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__, "threadpools": threadpool_info(), "threads": {k: os.environ.get(k) for k in THREADS}},
    }
    if mode in ("profiled", "timed"):
        stats_path = profile_path or Path(f"grassman_seed{seed}.prof")
        if profiler is not None:
            profiler.dump_stats(stats_path)
            result["profile_file"] = str(stats_path)
        result["stages"] = dict(counters)
        if any(name.startswith("other_") for name in counters):
            raise RuntimeError("stage instrumentation counted linalg outside solve")
        measured_exclusive = sum(float(row["exclusive_sec"]) for row in counters.values() if row is not counters["solve"])
        remainder = solve_clock["elapsed"] - measured_exclusive
        if remainder < -0.01:
            raise RuntimeError(f"stage timers exceed solver wall by {-remainder:.6f}s")
        remainder = max(0.0, remainder)
        result["stages"]["solve_remainder"] = {"inclusive_sec": remainder, "exclusive_sec": remainder, "calls": 1}
        # cProfile artifact is retained and a compact top cumulative list aids audit.
        if profiler is not None:
            stats = pstats.Stats(profiler)
            result["cprofile_top_cumulative"] = [
                {"function": f"{key[0]}:{key[1]}({key[2]})", "calls": value[1], "tottime_sec": value[2], "cumtime_sec": value[3]}
                for key, value in sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:40]
            ]
    return result


def worker(seed: int, mode: str, profile_path: Path | None) -> None:
    print(json.dumps(profiled_fit(seed, mode, profile_path), allow_nan=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=("baseline", "profiled", "timed"), default="baseline")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--smoke", action="store_true", help="run seed 0 baseline/profiled pair")
    ap.add_argument("--timed-only", action="store_true", help="run only lightweight timer workers")
    a = ap.parse_args()
    if a.worker:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        worker(a.seed, a.mode, a.output if a.mode == "profiled" else None)
        return
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "runs.jsonl"
    protocol_path = out / "protocol.json"
    if rows_path.exists() or protocol_path.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    seeds = [0] if a.smoke else a.seeds
    modes = ("timed",) if a.timed_only else ("baseline", "profiled")
    env = os.environ.copy()
    env.update(dict.fromkeys(THREADS, "1"))
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("UV_CACHE_DIR", "/tmp/adp-uv-cache")
    rows: list[dict[str, Any]] = []
    with rows_path.open("x", encoding="utf-8") as stream:
        for seed in seeds:
            for mode in modes:
                prof = out / f"grassman_seed{seed}.prof" if mode == "profiled" else out / f"unused_seed{seed}.prof"
                cmd = ["uv", "run", "--no-sync", "python", "-m", "experiments.grassman_stage_profile_2026_10_06.run_profile", "--worker", "--output", str(prof), "--seed", str(seed), "--mode", mode]
                started = time.perf_counter()
                try:
                    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, timeout=900, check=False)
                    if proc.returncode:
                        row = {"seed": seed, "mode": mode, "status": "process_error", "returncode": proc.returncode, "stderr": proc.stderr[-8000:]}
                    else:
                        row = json.loads(proc.stdout.strip().splitlines()[-1])
                except subprocess.TimeoutExpired as exc:
                    row = {"seed": seed, "mode": mode, "status": "timeout", "timeout_sec": 900, "stderr": str(exc)[:4000]}
                row["wall_including_startup_sec"] = time.perf_counter() - started
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
    import numpy as np
    import scipy
    protocol = {"command": "UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m experiments.grassman_stage_profile_2026_10_06.run_profile", "seeds": seeds, "modes": modes, "threads": {k: "1" for k in THREADS}, "timeout_sec": 900, "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(), "dirty": subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines(), "source_sha256": sha256(ROOT / "ADP/solver/grassman.py"), "harness_sha256": sha256(Path(__file__)), "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "results": rows}
    protocol_path.write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
