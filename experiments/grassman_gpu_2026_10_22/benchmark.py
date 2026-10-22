"""Paired full fits, immutable sources, actual R MEANMAVE and strict gates."""

from __future__ import annotations

import argparse
import cProfile
import json
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from experiments.r_sdr.design import Case, generate, metrics
from experiments.r_sdr.protocol import (
    ROOT,
    digest,
    ensure_manifest,
    environment,
    snapshot as snapshot_r_sdr,
    source_files,
)
from experiments.r_sdr.runner import adp_settings, read_worker_meta, run_process

HERE = Path(__file__).resolve().parent


def heavy_settings(case: Case, overrides: dict) -> dict:
    settings = adp_settings(case, {})
    settings["N_J"] = max(settings["N_J"], int(np.ceil(case.n / settings["N_loc"])))
    settings.update(overrides)
    return settings


def acceptance(rows: list[dict], method: str, *, expected_seeds: list[int]) -> dict:
    """Require all planned pairs, then evaluate frozen error/time thresholds."""
    lookup = {(r["seed"], r["method"]): r for r in rows}
    pairs = [
        (lookup.get((seed, method)), lookup.get((seed, "MEANMAVE")))
        for seed in expected_seeds
    ]
    valid = [
        (a, b)
        for a, b in pairs
        if a is not None and b is not None and a["status"] == b["status"] == "ok"
    ]
    result = {
        "passed": False,
        "planned_pairs": len(pairs),
        "valid_pairs": len(valid),
        "complete": len(valid) == len(pairs),
    }
    if not valid:
        return result
    log_error = np.array(
        [
            np.log(
                max(a["projector_distance"], 1e-15)
                / max(b["projector_distance"], 1e-15)
            )
            for a, b in valid
        ]
    )
    rng = np.random.default_rng(221022)
    samples = rng.integers(len(valid), size=(10000, len(valid)))
    upper = float(np.exp(np.quantile(log_error[samples].mean(axis=1), 0.975)))
    error_ratio = float(np.exp(log_error.mean()))
    time_ratio = float(
        np.median([a["fit_seconds"] / b["fit_seconds"] for a, b in valid])
    )
    worst = max(a["projector_distance"] - b["projector_distance"] for a, b in valid)
    result.update(
        error_geomean_ratio=error_ratio,
        error_bootstrap95_upper=upper,
        worst_error_increase=worst,
        median_time_ratio=time_ratio,
        passed=bool(
            result["complete"]
            and len(pairs) >= 10
            and error_ratio <= 1.05
            and upper <= 1.10
            and worst <= 0.05
            and time_ratio <= 1.10
        ),
    )
    return result


def worker(job: dict) -> None:
    from ADP import ADP_Config, ADP_multi_index, ADP_solver
    from ADP.solver.grassman import solve

    X = np.loadtxt(job["x"], delimiter=",", ndmin=2)
    y = np.loadtxt(job["y"], delimiter=",", ndmin=1)
    gpu = job["method"].startswith("GPU")
    cp = None
    if gpu:
        from ADP.gpu import require_cupy

        cp = require_cupy()
    config = ADP_Config(seed=job["seed"], **job["config"], gpu=gpu, gpu_solver=gpu)

    def fit():
        model = ADP_multi_index(
            job["m"], config=config, solver=ADP_solver(solve, **job["solver"])
        )
        model.fit(
            X,
            y,
            progress=lambda row: print(
                f"outer={row['iteration']} h={row['h']:.6g}", flush=True
            ),
        )
        if cp is not None:
            cp.cuda.get_current_stream().synchronize()
        return model

    warmup = 0.0
    if job["method"] == "GPU-warm":
        start = time.perf_counter()
        fit()
        warmup = time.perf_counter() - start
    profiler = cProfile.Profile() if job.get("profile") else None
    started = time.perf_counter()
    if profiler is not None:
        profiler.enable()
    model = fit()
    elapsed = time.perf_counter() - started
    if profiler is not None:
        profiler.disable()
        profiler.dump_stats(job["meta"] + ".prof")
    np.savetxt(job["basis"], model.basis_, delimiter=",", fmt="%.17g")
    diagnostics = model.solver_diagnostics_
    meta = {
        "fit_seconds": elapsed,
        "warmup_seconds": warmup,
        "iterations": len(model.trace_),
        "stop_reason": model.result_.stop_reason,
        "all_inner_certified": all(d.get("converged", False) for d in diagnostics),
        "n_inner_certified": sum(d.get("converged") is True for d in diagnostics),
        "n_inner_total": len(diagnostics),
        "inner_diagnostics": diagnostics,
        "initial_basis": model.result_.beta_init.tolist(),
        "trace_bases": [row["basis"].tolist() for row in model.trace_],
        "trace_scales": [
            {k: row[k] for k in ("h", "factor", "err")} for row in model.trace_
        ],
        "stage_profile": model.profile_,
        "cupy_pool_reserved_bytes": cp.get_default_memory_pool().total_bytes()
        if cp is not None
        else None,
        "cupy_version": cp.__version__ if cp is not None else None,
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion() if cp is not None else None,
        "cuda_driver": cp.cuda.runtime.driverGetVersion() if cp is not None else None,
    }
    Path(job["meta"]).write_text(json.dumps(meta, allow_nan=False, indent=2))


def run(args: argparse.Namespace) -> None:
    case = Case("gpu-heavy-additive", args.n, args.d, args.m)
    overrides = json.loads(args.config.read_text()) if args.config else {}
    if {"gpu", "gpu_solver", "seed", "kernel"} & overrides.keys():
        raise ValueError("config cannot override gpu/gpu_solver/seed/kernel")
    settings = heavy_settings(case, overrides)
    from ADP import ADP_Config
    from ADP.engine.common.index_fit import validate_sizes

    config = ADP_Config(**settings)
    validate_sizes(config, "multi", case.n, case.d, case.m)
    config_record = asdict(config)
    config_record["kernel"] = (
        "Epanechnikov max(1-s**2,0); s is squared localization/h**2"
    )
    solver = json.loads(args.solver.read_text()) if args.solver else {}
    info, env = environment(args.rscript, args.threads)
    info["nvidia_smi"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    files = sorted(set(source_files()) | set(HERE.glob("*.py")))
    protocol = {
        "case": asdict(case),
        "seed": args.seed,
        "runs": args.runs,
        "methods": args.methods,
        "config": settings,
        "effective_config": config_record,
        "solver": solver,
        "timeout": args.timeout,
        "memory_mib": args.memory_mib,
        "environment": info,
        "profile": args.profile,
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in files},
        "cuda_timing": "GPU includes lazy kernels; GPU-warm excludes a full warmup",
        "gpu_memory": "CuPy pool at fit end; excludes driver/libraries",
    }
    output = args.output.resolve()
    ensure_manifest(output, protocol, resume=False)
    snapshot_r_sdr(output)
    with zipfile.ZipFile(output / "sources.zip", "a", zipfile.ZIP_DEFLATED) as archive:
        for path in HERE.glob("*.py"):
            archive.write(path, path.relative_to(ROOT))
    source = output / "source"
    with zipfile.ZipFile(output / "sources.zip") as archive:
        archive.extractall(source)
    env["PYTHONPATH"] = str(source)
    rows: list[dict] = []
    for seed in range(args.seed, args.seed + args.runs):
        folder = output / str(seed)
        folder.mkdir()
        X, y, truth = generate(case, seed)
        np.savetxt(folder / "x.csv", X, delimiter=",", fmt="%.17g")
        np.savetxt(folder / "y.csv", y, delimiter=",", fmt="%.17g")
        np.save(folder / "truth.npy", truth)
        hashes = {key: digest(folder / key) for key in ("x.csv", "y.csv", "truth.npy")}
        for method in args.methods:
            basis, meta = folder / f"{method}.basis.csv", folder / f"{method}.meta.json"
            job = {
                "method": method,
                "m": case.m,
                "seed": seed,
                "config": settings,
                "solver": solver,
                "x": str(folder / "x.csv"),
                "y": str(folder / "y.csv"),
                "basis": str(basis),
                "meta": str(meta),
                "profile": args.profile,
            }
            path = folder / f"{method}.job.json"
            path.write_text(json.dumps(job, indent=2))
            if method == "MEANMAVE":
                command = [
                    args.rscript,
                    "--vanilla",
                    str(source / "experiments/r_sdr/worker.R"),
                    "MAVE-package",
                    job["x"],
                    job["y"],
                    str(basis),
                    str(meta),
                    str(seed),
                    str(case.m),
                    "50",
                    "1e-6",
                ]
            else:
                command = [
                    sys.executable,
                    "-m",
                    "experiments.grassman_gpu_2026_10_22.benchmark",
                    "worker",
                    str(path),
                ]
            print(f"start seed={seed} method={method}", flush=True)
            row = {"seed": seed, "method": method, "input_hashes": hashes}
            row.update(
                run_process(
                    command,
                    env,
                    folder / f"{method}.log",
                    timeout=args.timeout,
                    memory_mib=args.memory_mib,
                    cwd=source,
                )
            )
            if row["status"] == "ok":
                try:
                    row.update(
                        read_worker_meta("MAVE-package", meta)
                        if method == "MEANMAVE"
                        else json.loads(meta.read_text())
                    )
                    row.update(
                        metrics(np.loadtxt(basis, delimiter=",", ndmin=2), truth)
                    )
                except (OSError, ValueError, KeyError) as error:
                    row.update(status="invalid_result", error=str(error))
            rows.append(row)
            with (output / "runs.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(
                json.dumps(
                    {
                        k: row.get(k)
                        for k in (
                            "seed",
                            "method",
                            "status",
                            "fit_seconds",
                            "projector_distance",
                        )
                    }
                ),
                flush=True,
            )
    seeds = list(range(args.seed, args.seed + args.runs))
    summary = {
        m: acceptance(rows, m, expected_seeds=seeds)
        for m in args.methods
        if m != "MEANMAVE"
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(json.loads(Path(sys.argv[2]).read_text()))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=90800)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--d", type=int, default=50)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--memory-mib", type=float, default=12000)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--solver", type=Path)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("CPU", "GPU", "GPU-warm", "MEANMAVE"),
        default=["CPU", "GPU", "GPU-warm", "MEANMAVE"],
    )
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if min(args.runs, args.n, args.d, args.m, args.threads) <= 0 or args.m > args.d:
        parser.error("require positive counts and m<=d")
    run(args)


if __name__ == "__main__":
    main()
