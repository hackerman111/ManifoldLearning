"""Run a small LSMR recheck against saved MEANMAVE benchmark fits.

The default selects three existing paired seeds each for m=3 and m=5 from
the d=25, n=5000 benchmark and runs only the missing LSMR fits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

BENCHMARK_ROOT = Path("benchmarks/results/outputs/r_sdr/d25-n5000-20261023")
OUTPUT = Path("benchmarks/results/outputs/r_sdr/lsmr-meanmave-20261025")
THREAD_ENV = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _fit(job_path: Path) -> int:
    from ADP import ADP_Config, ADP_multi_index, ADP_solver
    from ADP.solver.LSMR import solve as solve_lsmr
    from experiments.r_sdr.design import metrics

    job = json.loads(job_path.read_text())
    with np.load(job["input"]) as archive:
        x, y, truth = archive["x"], archive["y"], archive["truth"]
    model = ADP_multi_index(
        job["m"],
        config=ADP_Config(seed=job["method_seed"], **job["config"]),
        solver=ADP_solver(solve_lsmr, **job["solver"]),
    )
    started = time.perf_counter()
    model.fit(x, y)
    fit_seconds = time.perf_counter() - started
    diagnostics = model.solver_diagnostics_
    row = {
        "status": "ok",
        "fit_seconds": fit_seconds,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "metrics": metrics(model.basis_, truth),
        "all_inner_certified": bool(diagnostics)
        and all(item.get("converged", False) for item in diagnostics),
        "n_inner_certified": sum(item.get("converged") is True for item in diagnostics),
        "n_inner_total": len(diagnostics),
        "stop_reason": model.result_.stop_reason,
        "inner_diagnostics": diagnostics,
        "effective_parameters": {
            key: value
            for key, value in model.effective_parameters_.items()
            if key not in {"trace", "beta_init", "beta_final"}
            and isinstance(value, (str, int, float, bool, type(None)))
        },
    }
    Path(job["result"]).write_text(
        json.dumps(_json_safe(row), indent=2, allow_nan=False) + "\n"
    )
    return 0


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for m in (3, 5):
        selected = [row for row in rows if row["m"] == m]
        if not selected:
            continue
        successful = [row for row in selected if row["lsmr_status"] == "ok"]
        summaries.append(
            {
                "m": m,
                "n_pairs": len(selected),
                "n_lsmr_ok": len(successful),
                "lsmr_recovered": sum(row["lsmr_recovered"] for row in successful),
                "meanmave_recovered": sum(
                    row["meanmave_recovered"] for row in selected
                ),
                "lsmr_median_edr": (
                    statistics.median(row["lsmr_edr"] for row in successful)
                    if successful else None
                ),
                "meanmave_median_edr": statistics.median(
                    row["meanmave_edr"] for row in selected
                ),
                "lsmr_median_fit_seconds": (
                    statistics.median(row["lsmr_fit_seconds"] for row in successful)
                    if successful else None
                ),
                "meanmave_median_fit_seconds": statistics.median(
                    row["meanmave_fit_seconds"] for row in selected
                ),
                "median_fit_time_ratio_lsmr_over_meanmave": (
                    statistics.median(
                        row["lsmr_fit_seconds"] / row["meanmave_fit_seconds"]
                        for row in successful
                    ) if successful else None
                ),
                "lsmr_all_inner_certified": sum(
                    row["lsmr_all_inner_certified"] is True for row in successful
                ),
            }
        )
    return summaries


def _run(args: argparse.Namespace) -> dict[str, Any]:
    source_manifest_hashes: dict[str, str] = {}
    rows_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for m in (3, 5):
        case_root = args.benchmark_root / f"m{m}"
        manifest_path = case_root / "manifest.json"
        case_manifest = json.loads(manifest_path.read_text())
        source_manifest_hashes[str(m)] = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
        case = case_manifest["protocol"]["cases"][0]
        baseline_rows = [
            json.loads(line)
            for line in (case_root / "runs.jsonl").read_text().splitlines()
        ]
        baseline_by_seed = {
            int(row["seed"]): row
            for row in baseline_rows
            if row["method"] == "MAVE-package" and row["status"] == "ok"
        }
        selected_seeds = sorted(baseline_by_seed)[: args.runs]
        if len(selected_seeds) != args.runs:
            raise ValueError(f"m={m}: only {len(selected_seeds)} successful MEANMAVE fits")

        for seed in selected_seeds:
            baseline = baseline_by_seed[seed]
            input_path = case_root / "inputs" / f"{case['name']}-{seed}.npz"
            if not input_path.is_file():
                raise FileNotFoundError(input_path)
            with np.load(input_path) as archive:
                x, y, truth = archive["x"], archive["y"], archive["truth"]
            data_hash = hashlib.sha256(x.tobytes() + y.tobytes() + truth.tobytes()).hexdigest()
            if data_hash != baseline["data_sha256"]:
                raise ValueError(f"saved input hash differs for m={m}, seed={seed}")

            row: dict[str, Any] = {
                "case": case["name"],
                "n": case["n"],
                "d": case["d"],
                "m": m,
                "seed": seed,
                "data_sha256": data_hash,
                "meanmave_status": baseline["status"],
                "meanmave_edr": float(baseline["projector_distance"]),
                "meanmave_fit_seconds": float(baseline["fit_seconds"]),
                "meanmave_process_seconds": float(baseline["process_seconds"]),
                "meanmave_peak_rss_mib": float(baseline["peak_rss_mib"]),
                "meanmave_recovered": bool(baseline["recovered"]),
                "meanmave_method_seed": int(baseline["method_seed"]),
            }
            job = {
                "input": str(input_path.resolve()),
                "result": str((args.output / "fits" / f"m{m}-{seed}.json").resolve()),
                "m": m,
                "method_seed": int(baseline["method_seed"]),
                "config": case_manifest["protocol"]["adp_config"][case["name"]],
                "solver": case_manifest["protocol"]["adp_solver"],
            }
            job_path = args.output / "jobs" / f"m{m}-{seed}.json"
            job_path.write_text(json.dumps(job, indent=2) + "\n")
            env = os.environ.copy()
            env.update({key: str(args.threads) for key in THREAD_ENV})
            env["OMP_DYNAMIC"] = "FALSE"
            env["MKL_DYNAMIC"] = "FALSE"
            started = time.perf_counter()
            try:
                if Path(job["result"]).exists():
                    fitted = json.loads(Path(job["result"]).read_text())
                    process_seconds = None
                else:
                    completed = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "benchmarks.lsmr_meanmave_recheck",
                            "--fit-job",
                            str(job_path),
                        ],
                        capture_output=True,
                        text=True,
                        env=env,
                        timeout=args.timeout,
                        check=False,
                    )
                    process_seconds = time.perf_counter() - started
                    if completed.returncode != 0:
                        raise RuntimeError(
                            completed.stderr[-4000:] or f"exit {completed.returncode}"
                        )
                    fitted = json.loads(Path(job["result"]).read_text())
                row.update(
                    lsmr_status="ok",
                    lsmr_fit_seconds=fitted["fit_seconds"],
                    lsmr_process_seconds=process_seconds,
                    lsmr_peak_rss_mib=fitted["peak_rss_mib"],
                    lsmr_edr=fitted["metrics"]["projector_distance"],
                    lsmr_trace_score=fitted["metrics"]["trace_score"],
                    lsmr_recovered=(
                        fitted["metrics"]["projector_distance"]
                        <= case_manifest["protocol"]["recovery_threshold"]
                    ),
                    lsmr_all_inner_certified=fitted["all_inner_certified"],
                    lsmr_inner_certified=fitted["n_inner_certified"],
                    lsmr_inner_total=fitted["n_inner_total"],
                    lsmr_stop_reason=fitted["stop_reason"],
                    lsmr_diagnostics=fitted["inner_diagnostics"],
                )
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                row.update(
                    lsmr_status="timeout" if isinstance(error, subprocess.TimeoutExpired) else "error",
                    lsmr_error=str(error),
                    lsmr_process_seconds=time.perf_counter() - started,
                )
            rows_by_key[(m, seed)] = row
            _result(args.output, rows_by_key, source_manifest_hashes, args)
            print(
                f"m={m} seed={seed} LSMR={row['lsmr_status']} "
                f"MEANMAVE EDR={row['meanmave_edr']:.4f} "
                f"LSMR EDR={row.get('lsmr_edr', 'NA')}",
                flush=True,
            )
    return _result(args.output, rows_by_key, source_manifest_hashes, args)


def _result(
    output: Path,
    rows_by_key: dict[tuple[int, int], dict[str, Any]],
    source_manifest_hashes: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    result = {
        "protocol": {
            "source_benchmark": str(args.benchmark_root),
            "source_manifest_sha256_by_rank": source_manifest_hashes,
            "ranks": (3, 5),
            "selected_successful_meanmave_seeds_per_rank": args.runs,
            "threads": args.threads,
            "lsmr_solver": "source benchmark HPAO settings, max_steps=80, tol=1e-6",
            "description": "Only LSMR fits are new; MEANMAVE records and exact saved inputs are reused.",
        },
        "summaries": _summaries(rows),
        "rows": rows,
    }
    (output / "results.json").write_text(
        json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=BENCHMARK_ROOT)
    parser.add_argument("--runs", type=int, default=3, help="existing paired seeds per rank")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fit-job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.fit_job:
        return _fit(args.fit_job)
    if args.runs < 1 or args.threads < 1 or args.timeout <= 0:
        parser.error("runs, threads, and timeout must be positive")
    if args.output.exists() and not args.resume:
        parser.error(f"refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    (args.output / "fits").mkdir(exist_ok=args.resume)
    (args.output / "jobs").mkdir(exist_ok=args.resume)
    _run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
