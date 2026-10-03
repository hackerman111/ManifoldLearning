"""Frozen and full-fit paired checks for the opt-in flat Grassmann method.

Run each stage once after correctness tests, from the repository root:

    PYTHONPATH=. python -m benchmarks.grassman_flat_benchmark --stage frozen
    PYTHONPATH=. python -m benchmarks.grassman_flat_benchmark --stage fullfit

Every method/seed/case runs in its own subprocess with one BLAS thread and a
120-second cap. Failed workers remain in the stage JSONL.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib
import io
import json
import os
import platform
import pstats
import resource
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/grassman_flat_2026_10_02"
SELECTION_SEEDS = (74100, 74101, 74102)
HELDOUT_SEEDS = (74200, 74201, 74202)
METHODS = ("core_gn", "spectral", "rank_one", "flat_gn")
DIMENSIONS = (20, 100, 300, 600)
INDEX_DIMS = (2, 3)
THREAD_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
TIMEOUT_SEC = 120
FROZEN_REPEATS = 3
SOLVER_SETTINGS: dict[str, int | float] = {
    "max_steps": 5,
    "tol": 1e-6,
    "energy_tol": 0.05,
    "local_ridge": 0.0,
    "max_angle": 0.5,
    "workspace_bytes": 16 * 1024**2,
    "flat_max_steps": 8,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _solver_sha256() -> str:
    """Hash экспериментального solver и общих numerical kernels."""
    digest = hashlib.sha256()
    for name in ("grassman.py", "grassman_exp.py"):
        digest.update((ROOT / "ADP/solver" / name).read_bytes())
    return digest.hexdigest()


def _array_sha256(array: Any) -> str:
    import numpy as np

    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(repr(value.shape).encode())
    digest.update(value.view(np.uint8))
    return digest.hexdigest()


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


def _make_frozen(seed: int, d: int, m: int) -> tuple[Any, ...]:
    import numpy as np

    rng = np.random.default_rng(np.random.SeedSequence([seed, d, m, 741]))
    J, p = 120, 10
    truth = np.linalg.qr(rng.normal(size=(d, m)))[0]
    init = np.linalg.qr(truth + 0.35 * rng.normal(size=(d, m)))[0]
    U = rng.normal(size=(J, p, d))
    coefficients = rng.normal(size=(J, m))
    I = np.einsum("jpm,jm->jp", U @ truth, coefficients)
    I += 0.12 * rng.normal(size=(J, p))
    mass = np.exp(rng.uniform(-1.0, 1.0, size=J))
    return init.T, U, I, mass, truth


def _method_kwargs(method: str) -> dict[str, int | float | str]:
    return {**SOLVER_SETTINGS, "method": method}


def _worker_frozen(args: argparse.Namespace) -> dict[str, object]:
    import numpy as np

    module = importlib.import_module("ADP.solver.grassman_exp")
    solve = module.solve
    index, U, I, mass, truth = _make_frozen(args.seed, args.d, args.m)
    kwargs: dict[str, Any] = {**_method_kwargs(args.method), "mass": mass}
    Y0 = module._normalize_index(index).T
    initial = module._profile(module._project(U, Y0), I, mass, 0.0)
    initial_gradient = module._gradient(Y0, U, initial, mass)
    # This baseline profile is intentionally outside all timed solver calls.
    initial_metrics = {
        "profile_objective": initial.value,
        "gradient_norm": float(np.linalg.norm(initial_gradient)),
    }
    solve(index, U, I, **kwargs)  # warm-up, excluded from timing
    times = []
    result = None
    for _ in range(args.repeats):
        started = time.perf_counter()
        result = solve(index, U, I, **kwargs)
        times.append(time.perf_counter() - started)
    assert result is not None
    basis = np.asarray(result.index)
    diagnostics = result.diagnostics
    projector_error = float(
        np.linalg.norm(basis.T @ basis - truth @ truth.T, ord="fro")
        / np.sqrt(2 * args.m)
    )
    history = np.asarray(diagnostics["profile_history"], dtype=float)
    monotone_tol = 1e-10 * max(1.0, float(history[0]))
    return {
        "stage": "frozen",
        "case": f"m{args.m}_d{args.d}",
        "seed": args.seed,
        "method": args.method,
        "status": "ok",
        "shape": {"J": U.shape[0], "p": U.shape[1], "d": U.shape[2], "m": args.m},
        "dtype": str(U.dtype),
        "input_sha256": {
            name: _array_sha256(value)
            for name, value in (
                ("index_init", index),
                ("U", U),
                ("I", I),
                ("mass", mass),
                ("truth", truth),
            )
        },
        "initial_metrics": initial_metrics,
        "times_sec": times,
        "median_sec": float(np.median(times)),
        "profile_objective": diagnostics["profile_objective"],
        "gradient_norm": diagnostics["gradient_norm"],
        "riemannian_gradient": diagnostics["riemannian_gradient"],
        "converged": diagnostics["converged"],
        "stop_reason": diagnostics["stop_reason"],
        "iterations": diagnostics["iterations"],
        "accepted_steps": diagnostics["accepted_steps"],
        "profile_evaluations": diagnostics["profile_evaluations"],
        "gn_solves": diagnostics["gn_solves"],
        "q_history": diagnostics["step_ranks"],
        "rank_guard_fallbacks": diagnostics["rank_guard_fallbacks"],
        "workspace_fallbacks": diagnostics["workspace_fallbacks"],
        "flat_inner_iterations": diagnostics.get("flat_inner_iterations", 0),
        "flat_angles": diagnostics.get("flat_angles", ()),
        "flat_fallbacks": diagnostics.get("flat_fallbacks", 0),
        "projector_error_to_truth": projector_error,
        "orthogonality_error": diagnostics["orthogonality_error"],
        "monotone_profile": bool(np.all(np.diff(history) <= monotone_tol)),
        "monotone_tolerance_abs": monotone_tol,
        "profile_history": history,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def _configure_fullfit_case(d: int) -> None:
    from benchmarks import grassman_benchmark as benchmark

    benchmark.PARAMS = {
        **benchmark.PARAMS,
        "n": 400,
        "d": d,
        "m": 2,
        "N_J": 120,
        "N_phi": 10,
        "N_lin": 80,
        "N_loc": 10,
        "solver_max_steps": 5,
        "budget_sec": 120.0,
    }


def _worker_fullfit(args: argparse.Namespace) -> dict[str, object]:
    import numpy as np

    from ADP import ADP_multi_index, ADP_solver
    from ADP.solver.grassman_exp import solve
    from benchmarks import grassman_benchmark as benchmark

    _configure_fullfit_case(args.d)
    X, y, truth, config = benchmark._make_problem(args.seed)
    settings: dict[str, Any] = {
        **SOLVER_SETTINGS,
        "method": args.method,
        "lambda_prox": float(config.lambda_penalty),
    }
    solver = ADP_solver(partial(solve, **settings))
    started = time.perf_counter()
    model = ADP_multi_index(2, config, solver).fit(X, y)
    elapsed = time.perf_counter() - started
    basis = np.asarray(model.basis_)
    distance = float(
        np.linalg.norm(basis @ basis.T - truth @ truth.T, ord="fro") / np.sqrt(4.0)
    )
    calls = [entry.get("solver", {}) for entry in model.trace_]
    profile_values = [
        float(call["profile_objective"])
        for call in calls
        if call.get("profile_objective") is not None
    ]
    sum_fields = (
        "iterations",
        "accepted_steps",
        "profile_evaluations",
        "gn_solves",
        "rank_guard_fallbacks",
        "workspace_fallbacks",
        "flat_inner_iterations",
        "flat_fallbacks",
    )
    counts: dict[str, Any] = {
        name: sum(int(call.get(name, 0)) for call in calls) for name in sum_fields
    }
    counts["converged_calls"] = sum(
        bool(call.get("converged", False)) for call in calls
    )
    counts["stop_reason_counts"] = {
        reason: sum(call.get("stop_reason") == reason for call in calls)
        for reason in sorted({str(call.get("stop_reason")) for call in calls})
    }
    return {
        "stage": "fullfit",
        "case": f"n400_m2_d{args.d}",
        "seed": args.seed,
        "method": args.method,
        "status": "ok",
        "shape": {
            name: benchmark.PARAMS[name]
            for name in ("n", "d", "m", "N_J", "N_phi", "N_lin")
        },
        "dtype": str(X.dtype),
        "input_sha256": {
            name: _array_sha256(value)
            for name, value in (("X", X), ("y", y), ("truth", truth))
        },
        "fit_time_sec": elapsed,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "projector_error_to_truth": distance,
        "outer_iterations": model.effective_parameters_.get("outer_iterations"),
        "selected_iteration": model.effective_parameters_.get("selected_iteration"),
        "fit_stop_reason": model.result_.stop_reason,
        "solver_calls": len(calls),
        "profile_history": profile_values,
        "monotone_profile": bool(
            all(
                right
                <= left
                + 1e-10 * max(1.0, float(call.get("profile_history", (1.0,))[0]))
                for call in calls
                for left, right in zip(
                    call.get("profile_history", ()),
                    call.get("profile_history", ())[1:],
                    strict=False,
                )
            )
        ),
        "solver_counts": counts,
        "solver_scalar_trace": [
            {
                name: _plain(call.get(name))
                for name in (
                    "profile_objective",
                    "gradient_norm",
                    "riemannian_gradient",
                    "iterations",
                    "converged",
                    "stop_reason",
                    "profile_evaluations",
                    "gn_solves",
                    "step_ranks",
                    "rank_guard_fallbacks",
                    "workspace_fallbacks",
                    "flat_inner_iterations",
                    "flat_angles",
                    "flat_fallbacks",
                )
            }
            for call in calls
        ],
    }


def _worker(args: argparse.Namespace) -> dict[str, object]:
    try:
        if args.stage == "frozen":
            row = _worker_frozen(args)
        else:
            row = _worker_fullfit(args)
    except Exception as error:
        row = {
            "stage": args.stage,
            "case": f"m{args.m}_d{args.d}"
            if args.stage == "frozen"
            else f"n400_m2_d{args.d}",
            "seed": args.seed,
            "method": args.method,
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "traceback_tail": traceback.format_exc()[-5000:],
        }
    row["worker_version"] = 1
    return row


def _run_one(
    stage: str, method: str, seed: int, d: int, m: int, repeats: int
) -> dict[str, object]:
    hash_before = _solver_sha256()
    command = [
        sys.executable,
        "-m",
        "benchmarks.grassman_flat_benchmark",
        "--worker",
        "--stage",
        stage,
        "--method",
        method,
        "--seed",
        str(seed),
        "--d",
        str(d),
        "--m",
        str(m),
        "--repeats",
        str(repeats),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env.update(dict.fromkeys(THREAD_ENV, "1"))
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            check=False,
            text=True,
            timeout=TIMEOUT_SEC,
            env=env,
        )
    except subprocess.TimeoutExpired as error:
        row: dict[str, object] = {
            "stage": stage,
            "case": f"m{m}_d{d}" if stage == "frozen" else f"n400_m2_d{d}",
            "seed": seed,
            "method": method,
            "status": "timeout",
            "error": str(error),
        }
    else:
        if completed.returncode:
            row = {
                "stage": stage,
                "case": f"m{m}_d{d}" if stage == "frozen" else f"n400_m2_d{d}",
                "seed": seed,
                "method": method,
                "status": "process_error",
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-4000:],
                "stdout_tail": completed.stdout[-1000:],
            }
        else:
            try:
                row = json.loads(completed.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError) as error:
                row = {
                    "stage": stage,
                    "seed": seed,
                    "method": method,
                    "status": "invalid_worker_output",
                    "error": f"{type(error).__name__}: {error}",
                    "stdout_tail": completed.stdout[-2000:],
                }
    row["solver_sha256_before"] = hash_before
    row["solver_sha256_after"] = _solver_sha256()
    return row


def _paired(rows: list[dict[str, object]], stage: str) -> list[dict[str, object]]:
    by_key = {
        (
            str(row.get("case")),
            int(cast(Any, row["seed"])),
            str(row["method"]),
        ): row
        for row in rows
        if row.get("status") == "ok"
    }
    pairs = []
    cases = sorted({str(row.get("case")) for row in rows})
    seeds = sorted({int(cast(Any, row["seed"])) for row in rows})
    for case in cases:
        for seed in seeds:
            baseline = by_key.get((case, seed, "core_gn"))
            if baseline is None:
                continue
            for method in METHODS:
                if method == "core_gn":
                    continue
                candidate = by_key.get((case, seed, method))
                if candidate is None:
                    continue
                pair: dict[str, object] = {
                    "case": case,
                    "seed": seed,
                    "baseline": "core_gn",
                    "method": method,
                    "status": "ok",
                }
                if stage == "frozen":
                    pair.update(
                        runtime_ratio=float(cast(Any, candidate["median_sec"]))
                        / float(cast(Any, baseline["median_sec"])),
                        objective_ratio=float(cast(Any, candidate["profile_objective"]))
                        / float(cast(Any, baseline["profile_objective"])),
                        gradient_ratio=float(cast(Any, candidate["gradient_norm"]))
                        / max(
                            float(cast(Any, baseline["gradient_norm"])), np_finfo_tiny()
                        ),
                        projector_error_difference=float(
                            cast(Any, candidate["projector_error_to_truth"])
                        )
                        - float(cast(Any, baseline["projector_error_to_truth"])),
                    )
                else:
                    pair.update(
                        runtime_ratio=float(cast(Any, candidate["fit_time_sec"]))
                        / float(cast(Any, baseline["fit_time_sec"])),
                        rss_delta_kib=int(cast(Any, candidate["peak_rss_kib"]))
                        - int(cast(Any, baseline["peak_rss_kib"])),
                        projector_error_difference=float(
                            cast(Any, candidate["projector_error_to_truth"])
                        )
                        - float(cast(Any, baseline["projector_error_to_truth"])),
                    )
                pairs.append(pair)
    return pairs


def np_finfo_tiny() -> float:
    import numpy as np

    return float(np.finfo(float).tiny)


def _blas_configuration() -> str:
    import numpy as np

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        np.show_config()
    return buffer.getvalue()


def _planned_freeze() -> dict[str, object]:
    return {
        "task": "grassman-flat-2026-10-02",
        "classification": "APPROXIMATE opt-in method, unchanged profile objective",
        "selection": {
            "seeds": SELECTION_SEEDS,
            "frozen_cases": [
                {"m": m, "d": d, "J": 120, "p": 10}
                for m in INDEX_DIMS
                for d in DIMENSIONS
            ],
            "fullfit_cases": [
                {
                    "seed": seed,
                    "n": 400,
                    "m": 2,
                    "d": d,
                    "N_J": 120,
                    "N_phi": 10,
                    "N_lin": 80,
                }
                for seed in SELECTION_SEEDS
                for d in (8, 30)
            ],
        },
        "methods": METHODS,
        "solver_settings": SOLVER_SETTINGS,
        "timing": {
            "frozen_warmups": 1,
            "frozen_repeats": FROZEN_REPEATS,
            "fullfit_repeats": 1,
            "worker_timeout_sec": TIMEOUT_SEC,
            "thread_env": dict.fromkeys(THREAD_ENV, "1"),
            "peak_rss": "resource.getrusage(RUSAGE_SELF).ru_maxrss, Linux KiB",
        },
        "monotonicity_tolerance": (
            "profile[k+1] <= profile[k] + 1e-10*max(1,profile[0])"
        ),
        "selection_gate": {
            "all_pairs_failure_free_finite_monotone_orthonormal": True,
            "per_fullfit_projector_error_delta_max": 1e-4,
            "median_fullfit_time_ratio_max": 1.10,
            "rss_delta_max_bytes": 32 * 1024**2,
            "minimum_median_relative_gain": 0.10,
            "gain_metrics": [
                "frozen profile objective",
                "frozen gradient norm",
                "fullfit projector error",
            ],
            "gate_target": "flat_gn versus core_gn; no seed retuning",
        },
        "heldout_seeds": HELDOUT_SEEDS,
        "heldout_rule": "run only if every selection gate passes",
        "code_sha256": {
            "solver": _solver_sha256(),
            "benchmark": _sha256(Path(__file__)),
            "fullfit_generator": _sha256(ROOT / "benchmarks/grassman_benchmark.py"),
        },
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": sys.version,
        "platform": platform.platform(),
    }


def _profile_core_baseline() -> str:

    from ADP.solver import grassman_exp

    index, U, I, mass, _ = _make_frozen(74100, 100, 2)
    profiler = cProfile.Profile()
    profiler.enable()
    result = grassman_exp.solve(
        index,
        U,
        I,
        mass=mass,
        max_steps=5,
        tol=1e-6,
        method="core_gn",
        energy_tol=0.05,
        local_ridge=0.0,
        max_angle=0.5,
        workspace_bytes=16 * 1024**2,
        flat_max_steps=8,
    )
    profiler.disable()
    output = io.StringIO()
    pstats.Stats(profiler, stream=output).sort_stats("cumulative").print_stats(15)
    output.write(
        "\nProfiled core_gn baseline (not timed selection row):\n"
        f"shape=(J={U.shape[0]}, p={U.shape[1]}, d={U.shape[2]}, m=2), "
        f"seed=74100, objective={result.diagnostics['profile_objective']}, "
        f"gradient={result.diagnostics['gradient_norm']}, "
        f"calls={result.diagnostics['profile_evaluations']}\n"
    )
    return output.getvalue()


def _input_hash_pairs(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (str(row.get("case")), int(cast(Any, row.get("seed"))))
        grouped.setdefault(key, []).append(row)
    mismatches = []
    incomplete = []
    for (case, seed), entries in sorted(grouped.items()):
        hashes = [entry.get("input_sha256") for entry in entries]
        if len(entries) != len(METHODS):
            incomplete.append(
                {"case": case, "seed": seed, "successful_methods": len(entries)}
            )
        elif any(value != hashes[0] for value in hashes[1:]):
            mismatches.append({"case": case, "seed": seed, "hashes": hashes})
    return {
        "all_successful_method_inputs_match": not mismatches and not incomplete,
        "mismatches": mismatches,
        "incomplete": incomplete,
    }


def _driver(args: argparse.Namespace) -> None:
    import numpy as np
    import scipy

    OUTPUT.mkdir(parents=True, exist_ok=True)
    run_path = OUTPUT / f"{args.stage}_runs.jsonl"
    protocol_path = OUTPUT / f"{args.stage}_protocol.json"
    if run_path.exists() or protocol_path.exists():
        raise FileExistsError(f"refusing to overwrite benchmark outputs in {OUTPUT}")
    freeze_path = OUTPUT / "freeze.json"
    baseline_profile_path = OUTPUT / "baseline_profile.txt"
    if args.stage == "frozen":
        if freeze_path.exists() or baseline_profile_path.exists():
            raise FileExistsError(f"refusing to overwrite frozen setup in {OUTPUT}")
        freeze = _planned_freeze()
        freeze_path.write_text(
            json.dumps(_plain(freeze), indent=2) + "\n", encoding="utf-8"
        )
        baseline_profile = _profile_core_baseline()
        baseline_profile_path.write_text(baseline_profile, encoding="utf-8")
    else:
        if not freeze_path.exists() or not baseline_profile_path.exists():
            raise FileNotFoundError(
                "run the frozen stage first to create freeze artifacts"
            )
        frozen_setup = json.loads(freeze_path.read_text(encoding="utf-8"))
        if frozen_setup["code_sha256"]["solver"] != _solver_sha256():
            raise RuntimeError("solver hash changed after selection freeze")
        if frozen_setup["code_sha256"]["benchmark"] != _sha256(Path(__file__)):
            raise RuntimeError("benchmark hash changed after selection freeze")
    if args.stage == "frozen":
        cases = [(m, d) for m in INDEX_DIMS for d in DIMENSIONS]
        seeds = tuple(args.seeds or SELECTION_SEEDS)
        repeats = FROZEN_REPEATS
    else:
        cases = [(2, d) for d in (8, 30)]
        seeds = tuple(args.seeds or SELECTION_SEEDS)
        repeats = 1
    rows: list[dict[str, object]] = []
    method_order = METHODS
    with run_path.open("x", encoding="utf-8") as stream:
        for case_index, (m, d) in enumerate(cases):
            for seed_index, seed in enumerate(seeds):
                order = (
                    method_order
                    if (case_index + seed_index) % 2 == 0
                    else method_order[::-1]
                )
                for method in order:
                    row = _run_one(args.stage, method, seed, d, m, repeats)
                    row["order_position"] = len(rows)
                    rows.append(row)
                    stream.write(json.dumps(_plain(row), allow_nan=False) + "\n")
                    stream.flush()
                    print(
                        json.dumps(
                            {
                                key: row.get(key)
                                for key in (
                                    "case",
                                    "seed",
                                    "method",
                                    "status",
                                    "median_sec",
                                    "fit_time_sec",
                                    "profile_objective",
                                    "gradient_norm",
                                    "flat_inner_iterations",
                                    "rank_guard_fallbacks",
                                    "flat_fallbacks",
                                )
                            },
                            allow_nan=False,
                        ),
                        flush=True,
                    )
    protocol = {
        "stage": args.stage,
        "seeds": seeds,
        "cases": [{"m": m, "d": d} for m, d in cases],
        "methods": METHODS,
        "method_order": "alternating by case and seed",
        "solver_settings": SOLVER_SETTINGS,
        "frozen_repeats": repeats if args.stage == "frozen" else None,
        "timeout_sec_per_worker": TIMEOUT_SEC,
        "thread_env": dict.fromkeys(THREAD_ENV, "1"),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas_configuration": _blas_configuration(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "benchmark_sha256": _sha256(Path(__file__)),
        "solver_sha256": _solver_sha256(),
        "freeze_sha256": _sha256(freeze_path),
        "baseline_profile_sha256": _sha256(baseline_profile_path),
        "generator_sha256": (
            _sha256(ROOT / "benchmarks/grassman_benchmark.py")
            if args.stage == "fullfit"
            else None
        ),
        "protocol": (
            "Each method/case/seed runs in a fresh process with one BLAS thread. "
            "Frozen workers perform one untimed warm-up and three timed repeats; "
            "the initial profile/gradient is measured before timing. Full fits use "
            "n=400, m=2, d=8/30 and one uninstrumented fit. Every worker has a "
            "120-second timeout; failures remain in JSONL. Peak RSS is ru_maxrss."
        ),
        "input_hash_pair_check": _input_hash_pairs(rows),
        "rows_file": str(run_path.relative_to(ROOT)),
        "paired": _paired(rows, args.stage),
        "failed_runs": [row for row in rows if row.get("status") != "ok"],
    }
    protocol_path.write_text(
        json.dumps(_plain(protocol), indent=2) + "\n", encoding="utf-8"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("frozen", "fullfit"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--method", choices=METHODS, default="core_gn")
    parser.add_argument("--seed", type=int, default=SELECTION_SEEDS[0])
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=FROZEN_REPEATS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.worker:
        print(json.dumps(_plain(_worker(args)), allow_nan=False))
        return 0
    _driver(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
