"""Frozen paired spectral-metric pilot; fresh process per measurement."""

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
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
VARIANTS = {
    "frobenius": (0.0, 0.0),
    "sqrt": (0.5, 0.0),
    "tensor": (1.0, 0.0),
    "floor": (1.0, 0.1),
}
CONFIG = {
    "N_loc": 15,
    "N_phi": 12,
    "outer_steps": 12,
    "a": 1.12,
    "lambda_penalty": 0.05,
    "h_min": 1e-6,
    "select_step": "last",
    "batch_size": 32,
    "multi_tensor": "full",
}
SEEDS = {"selection": [11, 23, 37], "validation": [101, 103, 107]}


def worker(kind, case, seed, target, variant):
    import numpy as np

    from ADP import ADP_Config, ADP_multi_index, ADP_solver
    from ADP.cli.main import _synthetic_data
    from ADP.solver.SVD import solve, solve_fixed_coefficients

    power, floor = VARIANTS[variant]
    row = {
        "kind": kind,
        "case": case,
        "seed": seed,
        "target": target,
        "variant": variant,
        "status": "ok",
        "power": power,
        "floor": floor,
    }
    if kind == "fit":
        n, d, m, J, lin = (
            (500, 20, 3, 40, 60) if case == "small" else (900, 60, 3, 64, 100)
        )
        X, y, truth = _synthetic_data(n, d, m, noise=0.2, seed=seed)
        row["data_sha256"] = hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()
        config = ADP_Config(seed=seed + 1000, N_J=J, N_lin=lin, **CONFIG)
        settings = {
            "rank": 2,
            "low_rank_target": target,
            "metric_power": power,
            "metric_floor": floor,
            "inner_maxiter": 20,
        }
        started = time.perf_counter()
        model = ADP_multi_index(m, config, ADP_solver(solve, **settings)).fit(X, y)
        row["seconds"] = time.perf_counter() - started
        diag = model.solver_diagnostics_
        basis = model.basis_
        row["projector_distance"] = float(
            np.sqrt(max(0.0, m - np.sum((basis.T @ truth) ** 2)) / m)
        )
        row["initial_distance"] = float(
            np.sqrt(max(0.0, m - np.sum((model.result_.beta_init.T @ truth) ** 2)) / m)
        )
        row["outer_steps"] = len(model.trace_)
        row["stop_reason"] = model.result_.stop_reason
        row["alpha_range"] = [
            min(r["factor"] for r in model.trace_),
            max(r["factor"] for r in model.trace_),
        ]
        row["objective_last"] = diag[-1]["rank_objective_history"][-1]
        row["metric_condition_max"] = max(r["metric_condition"] for r in diag)
        row["metric_transform_bytes_max"] = max(
            r["metric_transform_bytes"] for r in diag
        )
    else:
        d, m, J, p = int(case), 3, 32, 6
        rng = np.random.default_rng(seed)
        P = np.linalg.qr(rng.normal(size=(d, m)))[0].T
        U, g = rng.normal(size=(J, p, d)), rng.normal(size=(J, m))
        truth = np.linalg.qr(rng.normal(size=(d, m)))[0].T
        I = np.einsum("jpd,jd->jp", U, g @ truth) + 0.05 * rng.normal(size=(J, p))
        mass = np.geomspace(0.1, 10, J)
        row["data_sha256"] = hashlib.sha256(U.tobytes() + I.tobytes()).hexdigest()
        started = time.perf_counter()
        B, diagnostics = solve_fixed_coefficients(
            P,
            U,
            I,
            g,
            mass,
            rank=2,
            lambda_penalty=0.05,
            low_rank_target=target,
            metric_power=power,
            metric_floor=floor,
            metric_alpha=0.1,
            metric_eigenvalues=np.array([1.0, 0.4, 0.1]),
        )
        row["seconds"] = time.perf_counter() - started
        diag = [diagnostics]
        row["objective_last"] = diagnostics["rank_objective_history"][-1]
        row["coefficient_error"] = float(np.linalg.norm(B - truth))
        row["metric_condition_max"] = diagnostics["metric_condition"]
        row["metric_transform_bytes_max"] = diagnostics["metric_transform_bytes"]
    row["inner_converged"] = all(all(r["inner_converged"]) for r in diag)
    row["inner_iterations"] = sum(sum(r["inner_iterations"]) for r in diag)
    row["lsmr_iterations"] = sum(r["lsmr_iterations_total"] for r in diag)
    row["normal_residual_max"] = max(r["v_normal_residual_max"] for r in diag)
    row["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--kind", default="fit")
    parser.add_argument("--case", default="small")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--target", default="matrix")
    parser.add_argument("--variant", default="frobenius")
    parser.add_argument("--phase", choices=SEEDS, default="selection")
    parser.add_argument("--candidates", nargs="*", default=list(VARIANTS)[1:])
    args = parser.parse_args()
    if args.worker:
        try:
            row = worker(args.kind, args.case, args.seed, args.target, args.variant)
        except Exception:
            row = {
                "kind": args.kind,
                "case": args.case,
                "seed": args.seed,
                "target": args.target,
                "variant": args.variant,
                "status": "error",
                "error": traceback.format_exc(),
            }
        print(json.dumps(row))
        return
    import numpy as np
    import scipy

    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    manifest = {
        "phase": args.phase,
        "seeds": SEEDS[args.phase],
        "variants": VARIANTS,
        "config": CONFIG,
        "candidates": args.candidates,
        "fixed": {
            "d": [100, 300],
            "J": 32,
            "p": 6,
            "m": 3,
            "alpha": 0.1,
            "spectrum": [1.0, 0.4, 0.1],
            "ridge": 0.05,
        },
        "fit_cases": {"small": [500, 20, 3, 40, 60], "medium": [900, 60, 3, 64, 100]},
        "seed_scheme": "synthetic data seed; fit seed+1000 split by existing engine",
        "dtype": "float64",
        "threads": 1,
        "memory": "process peak RSS incl imports/data",
        "kernel": (
            "Epanechnikov squared distance; full tensor after first orthogonal step"
        ),
        "selection_gate": (
            "no failed fits; maximum quality worsening <=.02; median improvement >0"
        ),
        "commit": subprocess.check_output(
            ["rtk", "proxy", "git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": subprocess.check_output(
            ["rtk", "proxy", "git", "status", "--porcelain"], text=True
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sources": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in [
                "ADP/solver/SVD.py",
                "ADP/core/ADP_Solver.py",
                "ADP/core/multi/ADP_multi_index.py",
                "ADP/engine/common/index_fit.py",
                "experiments/svd_a_metric_2026-09-30/benchmark.py",
            ]
        },
    }
    path = OUT / (args.phase + ".jsonl")
    if path.exists():
        raise FileExistsError(path)
    (OUT / (args.phase + "_manifest.json")).write_text(json.dumps(manifest, indent=2))
    kinds = (
        [("fit", ["small", "medium"]), ("fixed", ["100", "300"])]
        if args.phase == "selection"
        else [("fit", ["small", "medium"])]
    )
    with path.open("w") as stream:
        for kind, cases in kinds:
            for case in cases:
                for target in ("matrix", "correction"):
                    for seed in SEEDS[args.phase]:
                        variants = ["frobenius", *args.candidates]
                        if seed % 2:
                            variants.reverse()
                        for variant in variants:
                            command = [
                                sys.executable,
                                str(Path(__file__).resolve()),
                                "--worker",
                                "--kind",
                                kind,
                                "--case",
                                case,
                                "--target",
                                target,
                                "--seed",
                                str(seed),
                                "--variant",
                                variant,
                            ]
                            try:
                                result = subprocess.run(
                                    command,
                                    env=env,
                                    cwd=ROOT,
                                    capture_output=True,
                                    text=True,
                                    timeout=240,
                                    check=True,
                                )
                                row = json.loads(result.stdout.strip().splitlines()[-1])
                            except Exception:
                                row = {
                                    "kind": kind,
                                    "case": case,
                                    "seed": seed,
                                    "target": target,
                                    "variant": variant,
                                    "status": "error",
                                    "error": traceback.format_exc(),
                                }
                            stream.write(json.dumps(row) + "\n")
                            stream.flush()
                            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
