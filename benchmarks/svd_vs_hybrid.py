"""Paired full-fit comparison of the opt-in truncated-SVD solver and HPAO.

Run from the repository root, for example:
    PYTHONPATH=. python -m benchmarks.svd_vs_hybrid \
        --output experiments/svd_vs_hybrid_2026-09-28

Each fit runs in its own worker process so process peak RSS is attributable to
that run. The runner uses the same generated problem and fit seed per pair.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

CASES = {
    "small": {"n": 500, "d": 20, "m": 3, "N_J": 40, "N_phi": 10, "N_lin": 60},
    "medium": {"n": 900, "d": 60, "m": 3, "N_J": 64, "N_phi": 12, "N_lin": 100},
    "spokoini_m2_dhigh_30s": {
        "generator": "spokoini",
        "n": 800,
        "d": 50,
        "m": 2,
        "N_J": 800,
        "N_phi": 10,
        "N_lin": 100,
        "N_loc": 10,
        "a": 1.010050167084168,
        "h_min": 1.0,
        "select_step": "last",
        "solver_max_steps": 5,
        "budget_sec": 30.0,
    },
}
SEEDS = (11, 23, 37)
HEAVY_SEEDS = (0, 1, 2)
RANK = 2


def _worker(case: str, seed: int, solver_name: str) -> dict[str, object]:
    import numpy as np

    from ADP import ADP_Config, ADP_multi_index, ADP_solver
    from ADP.solver.HYBRID.HYBRID_multi import solve as solve_hybrid
    from ADP.solver.LSMR import solve as solve_lsmr
    from ADP.solver.SVD import solve as solve_svd

    params = CASES[case]
    if params.get("generator") == "spokoini":
        from experiments.data import _generate_data, _make_seed_bundle
        from experiments.models import ExperimentPoint
        from experiments.runner import _effective_config

        point = ExperimentPoint(
            d=params["d"],
            n_over_d=params["n"] / params["d"],
            sigma_x=1.0,
            sigma_eps=0.1,
            link="spokoini_m2",
            x_distribution="beta1_tau",
            noise_distribution="gaussian",
            mode="multi",
            index_dim=params["m"],
            n_samples=params["n"],
            tau=1.0,
            N_loc=params["N_loc"],
            N_lin=params["N_lin"],
            N_J=params["N_J"],
            N_phi=params["N_phi"],
            a=params["a"],
            index_init="local",
            select_step="last",
            solver_max_steps=params["solver_max_steps"],
        )
        seed_bundle = _make_seed_bundle("mi-spokoini-m2-dhigh", point, seed)
        generated = _generate_data("mi-spokoini-m2-dhigh", point, seed_bundle, seed)
        X, y, true_basis = generated.X, generated.Y, generated.beta
        config = _effective_config(ADP_Config(), point, seed_bundle.init)
    else:
        from ADP.cli.main import _synthetic_data

        X, y, true_basis = _synthetic_data(
            params["n"], params["d"], params["m"], noise=0.2, seed=seed
        )
        config = ADP_Config(
            seed=seed + 1000,
            N_loc=15,
            N_lin=params["N_lin"],
            N_J=params["N_J"],
            N_phi=params["N_phi"],
            outer_steps=3,
            lambda_penalty=0.05,
            h_min=0.35,
            select_step="best",
            batch_size=32,
        )
    rank = min(RANK, params["m"] - 1)
    svd_solvers = {
        "svd",
        "svd_adaptive",
        "svd_direct",
        "svd_warm",
        "svd_strict",
    }
    if solver_name in svd_solvers:
        solver = ADP_solver(
            solve_svd,
            rank=rank,
            warm_start=solver_name != "svd_strict",
            adaptive_krylov=solver_name == "svd_adaptive",
            direct_max_dimension=(
                128 if solver_name in {"svd", "svd_direct"} else None
            ),
        )
    elif solver_name == "lsmr":
        solver = ADP_solver(solve_lsmr, max_steps=5, tol=1e-6)
    else:
        solver = ADP_solver(
            solve_hybrid,
            max_steps=int(params.get("solver_max_steps", 5)),
            tol=1e-6,
        )

    started = time.perf_counter()
    model = ADP_multi_index(params["m"], config, solver).fit(X, y)
    elapsed = time.perf_counter() - started
    traced_peak = None

    basis = model.basis_
    estimate_projector = basis @ basis.T
    truth_projector = true_basis @ true_basis.T
    projector_distance = float(
        np.linalg.norm(estimate_projector - truth_projector, ord="fro")
        / np.sqrt(2 * params["m"])
    )
    initial_basis = model.result_.beta_init
    initial_projector_distance = float(
        np.linalg.norm(initial_basis @ initial_basis.T - truth_projector, ord="fro")
        / np.sqrt(2 * params["m"])
    )
    trace = model.trace_
    diagnostics = [entry.get("solver", {}) for entry in trace]
    if solver_name in svd_solvers:
        inner_iterations = sum(
            sum(entry.get("inner_iterations", ())) for entry in diagnostics
        )
        inner_converged = all(
            all(entry.get("inner_converged", ())) for entry in diagnostics
        )
        linear_iterations = sum(
            int(entry.get("lsmr_iterations_total", 0)) for entry in diagnostics
        )
        u_vector_passes = sum(
            int(entry.get("u_vector_passes", 0)) for entry in diagnostics
        )
        lsmr_refinements = sum(
            int(entry.get("lsmr_refinements", 0)) for entry in diagnostics
        )
        direct_solves = sum(int(entry.get("direct_solves", 0)) for entry in diagnostics)
        direct_fallbacks = sum(
            int(entry.get("direct_fallbacks", 0)) for entry in diagnostics
        )
    else:
        inner_iterations = None
        inner_converged = all(entry.get("converged", False) for entry in diagnostics)
        linear_iterations = sum(
            int(
                entry.get(
                    "lsmr_iterations_total"
                    if solver_name == "lsmr"
                    else "linear_iterations_total",
                    0,
                )
            )
            for entry in diagnostics
        )
        u_vector_passes = None
        lsmr_refinements = None
        direct_solves = None
        direct_fallbacks = None

    return {
        "case": case,
        "solver": solver_name,
        "seed": seed,
        "status": "ok",
        "shape": {key: params[key] for key in ("n", "d", "m", "N_J", "N_phi", "N_lin")},
        "dtype": str(X.dtype),
        "fit_time_sec": elapsed,
        "time_budget_sec": params.get("budget_sec"),
        "within_time_budget": (
            elapsed <= float(params["budget_sec"])
            if params.get("budget_sec") is not None
            else None
        ),
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "tracemalloc_peak_bytes": traced_peak,
        "projector_distance": projector_distance,
        "initial_projector_distance": initial_projector_distance,
        "outer_iterations": model.effective_parameters_.get("outer_iterations"),
        "selected_iteration": model.effective_parameters_.get("selected_iteration"),
        "stop_reason": model.result_.stop_reason,
        "inner_iterations": inner_iterations,
        "linear_iterations": linear_iterations,
        "u_vector_passes": u_vector_passes,
        "lsmr_refinements": lsmr_refinements,
        "direct_solves": direct_solves,
        "direct_fallbacks": direct_fallbacks,
        "svd_inner_converged": (
            inner_converged if solver_name in svd_solvers else None
        ),
        "hybrid_solver_converged": (
            all(entry.get("converged", False) for entry in diagnostics)
            if solver_name == "hybrid"
            else None
        ),
        "lsmr_solver_converged": (
            all(entry.get("converged", False) for entry in diagnostics)
            if solver_name == "lsmr"
            else None
        ),
        "solver_trace": diagnostics,
    }


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    cases = sorted({str(row.get("case")) for row in rows})
    for case in cases:
        solver_names = [
            solver
            for solver in ("svd", "lsmr", "hybrid")
            if any(r.get("case") == case and r.get("solver") == solver for r in rows)
        ]
        comparison_solver = "lsmr" if "lsmr" in solver_names else "hybrid"
        by_solver = {
            solver: [
                r for r in rows if r.get("case") == case and r.get("solver") == solver
            ]
            for solver in solver_names
        }
        for solver, group in by_solver.items():
            ok = [row for row in group if row.get("status") == "ok"]
            summary: dict[str, object] = {
                "case": case,
                "solver": solver,
                "runs": len(group),
                "successes": len(ok),
                "time_budget_sec": next(
                    (
                        row.get("time_budget_sec")
                        for row in ok
                        if row.get("time_budget_sec") is not None
                    ),
                    None,
                ),
                "within_time_budget_runs": sum(
                    row.get("within_time_budget") is True for row in ok
                ),
                "errors": [
                    row.get("error") for row in group if row.get("status") != "ok"
                ],
            }
            for field in (
                "fit_time_sec",
                "peak_rss_kib",
                "tracemalloc_peak_bytes",
                "projector_distance",
                "initial_projector_distance",
                "outer_iterations",
                "inner_iterations",
                "linear_iterations",
            ):
                values = [float(row[field]) for row in ok if row.get(field) is not None]
                summary[f"{field}_median"] = (
                    statistics.median(values) if values else None
                )
            summary["stop_reasons"] = {
                reason: sum(row.get("stop_reason") == reason for row in ok)
                for reason in sorted({str(row.get("stop_reason")) for row in ok})
            }
            summary["svd_inner_converged_runs"] = (
                sum(
                    row.get("svd_inner_converged") is True
                    or (
                        row.get("solver") == "svd"
                        and "svd_inner_converged" not in row
                        and all(
                            all(item.get("inner_converged", ()))
                            for item in row.get("solver_trace", ())
                        )
                    )
                    for row in ok
                )
                if solver == "svd"
                else None
            )
            summary["hybrid_solver_converged_runs"] = (
                sum(
                    row.get("hybrid_solver_converged") is True
                    or (
                        row.get("solver") == "hybrid"
                        and "hybrid_solver_converged" not in row
                        and all(
                            item.get("converged", False)
                            for item in row.get("solver_trace", ())
                        )
                    )
                    for row in ok
                )
                if solver == "hybrid"
                else None
            )
            summary["lsmr_solver_converged_runs"] = (
                sum(row.get("lsmr_solver_converged") is True for row in ok)
                if solver == "lsmr"
                else None
            )
            result.append(summary)

    lookup = {(r["case"], r["solver"]): r for r in result}
    for case in cases:
        svd = lookup[(case, "svd")]
        comparison = lookup[(case, comparison_solver)]
        for field in ("fit_time_sec", "peak_rss_kib", "tracemalloc_peak_bytes"):
            left, right = svd[f"{field}_median"], comparison[f"{field}_median"]
            svd[f"{field}_ratio_svd_over_{comparison_solver}"] = (
                float(left) / float(right) if left is not None and right else None
            )
        svd["comparison_solver"] = comparison_solver
        paired = []
        case_seeds = sorted(
            {int(row["seed"]) for row in rows if row.get("case") == case}
        )
        for seed in case_seeds:
            s = next(
                (
                    r
                    for r in rows
                    if r.get("case") == case
                    and r.get("solver") == "svd"
                    and r.get("seed") == seed
                    and r.get("status") == "ok"
                ),
                None,
            )
            other = next(
                (
                    r
                    for r in rows
                    if r.get("case") == case
                    and r.get("solver") == comparison_solver
                    and r.get("seed") == seed
                    and r.get("status") == "ok"
                ),
                None,
            )
            if s and other:
                paired.append(
                    {
                        "seed": seed,
                        f"time_ratio_svd_over_{comparison_solver}": s["fit_time_sec"]
                        / other["fit_time_sec"],
                        f"quality_delta_svd_minus_{comparison_solver}": s[
                            "projector_distance"
                        ]
                        - other["projector_distance"],
                        f"rss_delta_kib_svd_minus_{comparison_solver}": s[
                            "peak_rss_kib"
                        ]
                        - other["peak_rss_kib"],
                    }
                )
        svd["paired_seed_deltas"] = paired
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", nargs=3, metavar=("CASE", "SEED", "SOLVER"))
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument(
        "--comparison", choices=("svd-hybrid", "lsmr-svd"), default="svd-hybrid"
    )
    args = parser.parse_args()
    if args.worker:
        case, seed, solver = args.worker
        try:
            row = _worker(case, int(seed), solver)
        except Exception as error:
            row = {
                "case": case,
                "solver": solver,
                "seed": int(seed),
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }
        encoded = json.dumps(row, allow_nan=False)
        if args.worker_output is not None:
            args.worker_output.parent.mkdir(parents=True, exist_ok=True)
            args.worker_output.write_text(encoded + "\n", encoding="utf-8")
        print(encoded)
        return

    if args.aggregate:
        rows = [
            json.loads(line)
            for line in (args.output / "runs.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        (args.output / "summary.json").write_text(
            json.dumps(_aggregate(rows), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return

    try:
        git_dirty_before_run = bool(
            subprocess.run(
                ["git", "status", "--short"],
                capture_output=True,
                check=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_dirty_before_run = None
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + env.get("PYTHONPATH", "")
    rows: list[dict[str, object]] = []
    selected_cases = args.cases or (
        ["small", "medium", "spokoini_m2_dhigh_30s"]
        if args.comparison == "lsmr-svd"
        else list(CASES)
    )
    if args.comparison == "lsmr-svd":
        seed_counts = {"small": 100, "medium": 50, "spokoini_m2_dhigh_30s": 25}
        seeds_by_case = {
            case: list(range(seed_counts[case])) for case in selected_cases
        }
        solvers = ("svd", "lsmr")
    else:
        seeds_by_case = {
            case: list(HEAVY_SEEDS if case == "spokoini_m2_dhigh_30s" else SEEDS)
            for case in selected_cases
        }
        solvers = ("svd", "hybrid")
    for case in selected_cases:
        for seed in seeds_by_case[case]:
            pair = list(solvers if seed % 2 else solvers[::-1])
            for solver in pair:
                command = [
                    sys.executable,
                    "-m",
                    "benchmarks.svd_vs_hybrid",
                    "--output",
                    str(args.output),
                    "--worker",
                    case,
                    str(seed),
                    solver,
                ]
                try:
                    completed = subprocess.run(
                        command,
                        env=env,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=240,
                    )
                    row = json.loads(completed.stdout)
                except (
                    subprocess.CalledProcessError,
                    subprocess.TimeoutExpired,
                ) as error:
                    row = {
                        "case": case,
                        "solver": solver,
                        "seed": seed,
                        "status": "error",
                        "error": f"{type(error).__name__}: {error}",
                    }
                rows.append(row)
                with (args.output / "runs.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")

    import numpy as np
    import scipy

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    meta = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "git_commit": commit,
        "git_dirty_before_run": git_dirty_before_run,
        "seeds_by_case": seeds_by_case,
        "rank_by_case": {
            case: min(RANK, CASES[case]["m"] - 1) for case in selected_cases
        },
        "cases": {case: CASES[case] for case in selected_cases},
        "threads": {
            key: env[key]
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "protocol": (
            "Cold full ADP_multi_index.fit in isolated processes with paired data/fit "
            "seeds; float64; no tracemalloc during fit; worker safety timeout is "
            "240 seconds."
        ),
        "objective_note": (
            "SVD fixed-g rank-r objective and LSMR HPAO correction-penalty "
            "objective differ; projector recovery is descriptive end-to-end "
            "comparison, not equivalent-inner-objective accuracy."
            if args.comparison == "lsmr-svd"
            else "SVD fixed-g rank-r objective and HYBRID HPAO correction-penalty "
            "objective differ; numeric lambda_penalty is shared but objectives "
            "are not equivalent."
        ),
        "data_generator": (
            "Spokoiny m=2 generator from experiments.data using recorded split "
            "seed-bundle for point mi-spokoini-m2-dhigh; X=2*Beta(1,tau=1)-1, "
            "sigma_x=1; sigma_eps=0.1 Gaussian noise; fixed two-column true basis."
            if "spokoini_m2_dhigh_30s" in selected_cases
            else "Synthetic Gaussian multi-index generator; see selected case config."
        ),
        "fit_configuration": (
            "Spokoini point d=50,n=800,m=2,N_loc=10,N_lin=100,N_J=800,N_phi=10, "
            "a=exp(1/100),h_min=1,select_step=last,outer_steps=None, "
            "index_init=local,estimator=new,lambda_penalty=0.05; SVD low_rank_target="
            "matrix,rank=1,inner_tol=1e-6,inner_maxiter=20; LSMR max_steps=5, "
            "tol=1e-6,theta=0.1; defaults otherwise."
            if "spokoini_m2_dhigh_30s" in selected_cases
            else "Synthetic: N_loc=15, lambda_penalty=0.05, h_min=0.35, "
            "a=sqrt(2), outer_steps=3, select_step=best, batch_size=32; SVD "
            "low_rank_target=matrix, rank=2, inner_tol=1e-6, inner_maxiter=20; "
            "LSMR max_steps=5,tol=1e-6,theta=0.1; see cases for N_J/N_phi/N_lin."
        ),
        "memory_method": (
            "resource.ru_maxrss in KiB includes interpreter/import baseline; "
            "tracemalloc is disabled for every case so fit timing is uninstrumented."
        ),
        "timing_method": (
            "time.perf_counter around fit only; no warm-up; isolated fresh "
            "subprocess per fit; BLAS/OpenMP threads fixed to one."
        ),
    }
    (args.output / "protocol.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "summary.json").write_text(
        json.dumps(_aggregate(rows), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
