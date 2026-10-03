"""Фиксированный paired R3 для H1; fresh serial workers, один BLAS thread.

Запуск: python -m experiments.glm_grassmann_2026_10_02.benchmark
Артефакты не перезаписываются. Held-out запуск намеренно отсутствует.
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
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = [73100, 73101, 73102]
THREADS = dict.fromkeys(
    (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ),
    "1",
)


def digest(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for array in arrays:
        h.update(str(array.shape).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value)}")


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, default=json_default, indent=2) + "\n")


def frozen_problem(seed: int, d: int):
    streams = np.random.SeedSequence(seed).spawn(4)
    data, truth_rng, noise, init = [np.random.default_rng(s) for s in streams]
    truth = np.linalg.qr(truth_rng.normal(size=(d, 2)))[0]
    Y = np.linalg.qr(truth + 0.15 * init.normal(size=(d, 2)) / np.sqrt(d))[0]
    U = data.normal(size=(120, 10, d)) * np.geomspace(1, 10, d)
    coefficients = truth_rng.normal(size=(120, 2))
    I = (U @ truth @ coefficients[..., None]).squeeze(-1)
    I += noise.normal(scale=0.1, size=I.shape)
    mass = np.geomspace(0.1, 10, 120)
    return Y.T, U, I, mass, truth


def fullfit_problem(seed: int, d: int):
    from ADP import ADP_Config

    streams = np.random.SeedSequence(seed).spawn(4)
    data, truth_rng, noise = [np.random.default_rng(s) for s in streams[:3]]
    n = 300 if d == 8 else 600
    X = data.normal(size=(n, d))
    truth = np.linalg.qr(truth_rng.normal(size=(d, 2)))[0]
    z = X @ truth
    y = np.sin(z[:, 0]) + 0.5 * z[:, 1] ** 2 + noise.normal(scale=0.15, size=n)
    config = ADP_Config(
        seed=int(streams[3].generate_state(1)[0]),
        N_loc=8,
        N_lin=100,
        N_J=100,
        N_phi=12,
        outer_steps=10,
        index_init="local",
        select_step="last",
        local_ridge=1e-8,
        lambda_penalty=0.05,
        h_min=1.0,
        batch_size=32,
    )
    return X, y, truth, config


def projector_distance(estimate: np.ndarray, truth: np.ndarray) -> float:
    residual = truth - estimate @ (estimate.T @ truth)
    return float(np.linalg.norm(residual) / np.sqrt(truth.shape[1]))


def worker(args: argparse.Namespace) -> None:
    from functools import partial

    from ADP import ADP_multi_index, ADP_solver
    from ADP.solver.grassman import solve as baseline
    from experiments.glm_grassmann_2026_10_02.full_gn import solve as candidate

    method = baseline if args.method == "core" else candidate
    row = {
        "stage": args.stage,
        "d": args.d,
        "m": 2,
        "seed": args.seed,
        "method": args.method,
    }
    if args.stage == "frozen":
        B, U, I, mass, truth = frozen_problem(args.seed, args.d)
        row.update(input_hash=digest(B, U, I, mass, truth), J=120, p=10)
        result = method(B, U, I, mass=mass)  # excluded warmup
        times = []
        for _ in range(5):
            started = time.perf_counter()
            result = method(B, U, I, mass=mass)
            times.append(time.perf_counter() - started)
        row.update(
            seconds=float(np.median(times)),
            time_samples=times,
            projector_distance=projector_distance(result.index.T, truth),
            diagnostics=[result.diagnostics],
        )
    else:
        X, y, truth, config = fullfit_problem(args.seed, args.d)
        row.update(input_hash=digest(X, y, truth), n=len(y), J=100, p=12)
        model = ADP_multi_index(
            2, config=config, solver=ADP_solver(partial(method, max_steps=5))
        )
        started = time.perf_counter()
        model.fit(X, y)
        assert model.basis_ is not None
        row.update(
            seconds=time.perf_counter() - started,
            projector_distance=projector_distance(model.basis_, truth),
            outer_stop=model.result_.stop_reason,
            outer_steps=len(model.trace_),
            diagnostics=model.solver_diagnostics_,
            schedule=[
                {k: r[k] for k in ("h", "factor", "err", "mean_mass")}
                for r in model.trace_
            ],
        )
    row["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    row["error"] = None
    write(args.output, row)


def manifest() -> dict:
    import scipy
    from threadpoolctl import threadpool_info

    sources = [
        "experiments/glm_grassmann_2026_10_02/full_gn.py",
        "experiments/glm_grassmann_2026_10_02/benchmark.py",
        "tests/test_glm_full_gn.py",
        "ADP/solver/grassman.py",
        "ADP/solver/grassman_exp.py",
        "ADP/solver/LSMR.py",
        "ADP/core/ADP_Config.py",
        "ADP/core/ADP_Solver.py",
        "ADP/core/multi/ADP_multi_index.py",
        *[
            f"ADP/engine/common/{s}.py"
            for s in (
                "index_fit",
                "weights",
                "statistic",
                "calculus",
                "initialize",
                "utils",
            )
        ],
    ]
    return {
        "selection_seeds": SEEDS,
        "held_out_unrun": list(range(73200, 73205)),
        "frozen_dimensions": [20, 100, 300],
        "fullfit_dimensions": [8, 30],
        "inner_settings": {
            "max_steps": 5,
            "lambda_prox": 0.05,
            "tol": 1e-6,
            "local_ridge": 0,
            "max_angle": 0.5,
        },
        "candidate": {"lsmr_maxiter": 30, "lsmr_tol": 1e-9, "certificate_tol": 1e-6},
        "baseline": {
            "method": "core_gn",
            "energy_tol": 0.05,
            "workspace_bytes": 16 * 1024**2,
        },
        "fullfit_config": vars_config(),
        "kernel": "max(1-q^2,0), q=(alpha^2*orthogonal_distance2+principal2)/h^2",
        "centers": "uniform without replacement, 100; no displacement",
        "directions": "current auto/localized multi law; redraw per outer iteration",
        "statistics": (
            "normalized finite sketch; external mass; current bandwidth search"
        ),
        "streams": (
            "SeedSequence(seed).spawn(4): data/truth/noise/init; "
            "ADP spawn(3) centers/init/directions"
        ),
        "thread_environment": THREADS,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas": threadpool_info(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "source_sha256": {
            s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources
        },
        "gates": {
            "per_pair_quality_atol": 1e-10,
            "median_time_ratio_max": 1.10,
            "per_pair_rss_extra_mib_max": 32,
            "advantage_min": 0.10,
        },
        "timing": (
            "fresh serial processes; import/startup excluded; "
            "5 frozen repetitions after warmup; full fits once"
        ),
        "memory": (
            "Linux ru_maxrss whole fresh worker incl input/imports, "
            "MiB; native allocations included"
        ),
        "timeout_seconds": 120,
    }


def vars_config() -> dict:
    from dataclasses import asdict

    config = asdict(fullfit_problem(73004, 8)[3])
    config["kernel"] = "ADP.core.ADP_Config.epanechnikov"
    config["seed"] = "derived per seed using fourth SeedSequence stream"
    return config


def driver() -> None:
    protocol_path = HERE / "protocol.json"
    if protocol_path.exists():
        raise RuntimeError("selection already frozen; do not overwrite or retune")
    write(protocol_path, manifest())
    rows = []
    for stage, dimensions in (("frozen", [20, 100, 300]), ("fullfit", [8, 30])):
        for d in dimensions:
            for seed in SEEDS:
                for method in ("core", "full"):
                    output = HERE / f"worker_{stage}_d{d}_seed{seed}_{method}.json"
                    command = [
                        sys.executable,
                        "-m",
                        "experiments.glm_grassmann_2026_10_02.benchmark",
                        "--worker",
                        "--stage",
                        stage,
                        "--d",
                        str(d),
                        "--seed",
                        str(seed),
                        "--method",
                        method,
                        "--output",
                        str(output),
                    ]
                    try:
                        proc = subprocess.run(
                            command,
                            env={**os.environ, **THREADS},
                            capture_output=True,
                            text=True,
                            timeout=120,
                        )
                        if proc.returncode:
                            raise RuntimeError(proc.stderr[-4000:])
                        row = json.loads(output.read_text())
                    except (RuntimeError, subprocess.TimeoutExpired) as error:
                        row = {
                            "stage": stage,
                            "d": d,
                            "seed": seed,
                            "method": method,
                            "error": str(error),
                        }
                        write(output, row)
                    rows.append(row)
                    with (HERE / "runs.jsonl").open("a") as f:
                        f.write(json.dumps(row, default=json_default) + "\n")
                    print(
                        stage,
                        d,
                        seed,
                        method,
                        row.get("seconds"),
                        row.get("error"),
                        flush=True,
                    )
    pairs = []
    for core in (r for r in rows if r["method"] == "core"):
        full = next(
            r
            for r in rows
            if r["method"] == "full"
            and (r["stage"], r["d"], r["seed"])
            == (core["stage"], core["d"], core["seed"])
        )
        pair = {
            "stage": core["stage"],
            "d": core["d"],
            "seed": core["seed"],
            "error": core["error"] or full["error"],
        }
        if not pair["error"]:
            if core["input_hash"] != full["input_hash"]:
                raise RuntimeError("paired input hash mismatch")
            pair.update(
                time_ratio=full["seconds"] / core["seconds"],
                rss_extra_mib=full["peak_rss_mib"] - core["peak_rss_mib"],
                quality_delta=full["projector_distance"] - core["projector_distance"],
            )
            if core["stage"] == "frozen":
                a, b = core["diagnostics"][0], full["diagnostics"][0]
                pair.update(
                    objective_ratio=b["profile_objective"] / a["profile_objective"],
                    gradient_ratio=b["riemannian_gradient"] / a["riemannian_gradient"],
                )
        pairs.append(pair)
    write(HERE / "pairs.json", pairs)
    fullfit = [p for p in pairs if p["stage"] == "fullfit" and not p["error"]]
    gates = {
        "complete_failure_free": not any(p["error"] for p in pairs),
        "quality": len(fullfit) == 6
        and all(p["quality_delta"] <= 1e-10 for p in fullfit),
        "time": len(fullfit) == 6
        and float(np.median([p["time_ratio"] for p in fullfit])) <= 1.10,
        "rss": len(fullfit) == 6 and all(p["rss_extra_mib"] <= 32 for p in fullfit),
        "advantage": any(
            float(
                np.median(
                    [p[key] for p in pairs if p["stage"] == "frozen" and not p["error"]]
                )
            )
            <= 0.9
            for key in ("objective_ratio", "gradient_ratio")
        ),
    }
    write(
        HERE / "gates.json",
        {**gates, "passed": all(gates.values()), "held_out_run": False},
    )
    print(gates, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--stage", choices=["frozen", "fullfit"], default="frozen")
    parser.add_argument("--method", choices=["core", "full"], default="full")
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--seed", type=int, default=73004)
    parser.add_argument("--output", type=Path, default=Path("/tmp/glm-smoke.json"))
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        driver()


if __name__ == "__main__":
    main()
