"""Pair LSMR with saved MAVE and MEANMAVE fits from the d=10 benchmark.

Only three new LSMR fits per rank are run by default; the other two methods
and the exact data are reused from the existing benchmark output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

BENCHMARK_ROOT = Path("benchmarks/results/outputs/r_sdr/highdim-dm-20261023")
CASE_DIRS = ("d10-m3", "d10-m5")
OUTPUT = Path("benchmarks/results/outputs/r_sdr/lsmr-mave-existing-20261025")
THREAD_ENV = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return statistics.median(values) if values else None


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for m in (3, 5):
        selected = [row for row in rows if row["m"] == m]
        if not selected:
            continue
        lsmr = [row for row in selected if row["lsmr_status"] == "ok"]
        summary: dict[str, Any] = {"m": m, "n_pairs": len(selected), "lsmr_ok": len(lsmr)}
        for label, prefix in (("MAVE", "mave"), ("MEANMAVE", "meanmave"), ("LSMR", "lsmr")):
            source = lsmr if prefix == "lsmr" else selected
            summary[f"{prefix}_median_edr"] = _median(source, f"{prefix}_edr")
            summary[f"{prefix}_median_fit_seconds"] = _median(
                source, f"{prefix}_fit_seconds"
            )
            summary[f"{prefix}_recovered"] = sum(
                row.get(f"{prefix}_recovered") is True for row in source
            )
        summary["lsmr_median_time_ratio_vs_mave"] = _median(
            [
                {"ratio": row["lsmr_fit_seconds"] / row["mave_fit_seconds"]}
                for row in lsmr
                if row.get("mave_fit_seconds", 0) > 0
            ],
            "ratio",
        )
        summary["lsmr_median_time_ratio_vs_meanmave"] = _median(
            [
                {"ratio": row["lsmr_fit_seconds"] / row["meanmave_fit_seconds"]}
                for row in lsmr
                if row.get("meanmave_fit_seconds", 0) > 0
            ],
            "ratio",
        )
        summary["lsmr_inner_fits_certified"] = sum(
            row.get("lsmr_all_inner_certified") is True for row in lsmr
        )
        result.append(summary)
    return result


def _write_result(
    output: Path,
    rows: list[dict[str, Any]],
    manifest_hashes: dict[str, str],
    runs: int,
    threads: int,
) -> dict[str, Any]:
    result = {
        "protocol": {
            "benchmark_root": str(BENCHMARK_ROOT),
            "source_manifest_sha256_by_case": manifest_hashes,
            "selected_seeds_per_rank": runs,
            "new_lsmr_fits": len(rows),
            "threads": threads,
            "lsmr_settings": "source ADP settings: max_steps=80, tol=1e-6",
            "note": "MAVE and MAVE::MEANMAVE rows and their exact inputs are reused.",
        },
        "summaries": _summaries(rows),
        "rows": rows,
    }
    (output / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    manifest_hashes: dict[str, str] = {}
    env = os.environ.copy()
    env.update({key: str(args.threads) for key in THREAD_ENV})
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"

    for case_dir in CASE_DIRS:
        case_root = args.benchmark_root / case_dir
        manifest_path = case_root / "manifest.json"
        case_manifest = json.loads(manifest_path.read_text())
        protocol = case_manifest["protocol"]
        case = protocol["cases"][0]
        m = int(case["m"])
        manifest_hashes[case_dir] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        baseline_rows = list(csv.DictReader((case_root / "runs.csv").open()))
        by_method_seed = {
            (row["method"], int(row["seed"])): row
            for row in baseline_rows
            if row["method"] in {"MAVE", "MAVE-package"} and row["status"] == "ok"
        }
        shared_seeds = sorted(
            seed
            for method, seed in by_method_seed
            if method == "MAVE"
            and ("MAVE-package", seed) in by_method_seed
        )[: args.runs]
        if len(shared_seeds) != args.runs:
            raise ValueError(f"{case_dir}: only {len(shared_seeds)} complete MAVE pairs")

        for seed in shared_seeds:
            mave = by_method_seed[("MAVE", seed)]
            meanmave = by_method_seed[("MAVE-package", seed)]
            input_path = case_root / "inputs" / f"{case['name']}-{seed}.npz"
            with np.load(input_path) as archive:
                x, y, truth = archive["x"], archive["y"], archive["truth"]
            data_hash = hashlib.sha256(x.tobytes() + y.tobytes() + truth.tobytes()).hexdigest()
            if data_hash != mave["data_sha256"] or data_hash != meanmave["data_sha256"]:
                raise ValueError(f"saved input hash mismatch for {case_dir}, seed={seed}")

            row: dict[str, Any] = {
                "case": case["name"],
                "n": int(case["n"]),
                "d": int(case["d"]),
                "m": m,
                "seed": seed,
                "data_sha256": data_hash,
                "mave_status": "ok",
                "mave_edr": float(mave["projector_distance"]),
                "mave_fit_seconds": float(mave["fit_seconds"]),
                "mave_process_seconds": float(mave["process_seconds"]),
                "mave_peak_rss_mib": float(mave["peak_rss_mib"]),
                "mave_recovered": mave["recovered"] == "True",
                "meanmave_status": "ok",
                "meanmave_edr": float(meanmave["projector_distance"]),
                "meanmave_fit_seconds": float(meanmave["fit_seconds"]),
                "meanmave_process_seconds": float(meanmave["process_seconds"]),
                "meanmave_peak_rss_mib": float(meanmave["peak_rss_mib"]),
                "meanmave_recovered": meanmave["recovered"] == "True",
            }
            fit_result = args.output / "fits" / f"m{m}-{seed}.json"
            job = {
                "input": str(input_path.resolve()),
                "result": str(fit_result.resolve()),
                "m": m,
                "method_seed": int(mave["method_seed"]),
                "config": protocol["adp_config"][case["name"]],
                "solver": protocol["adp_solver"],
            }
            job_path = args.output / "jobs" / f"m{m}-{seed}.json"
            job_path.write_text(json.dumps(job, indent=2) + "\n")
            started = time.perf_counter()
            try:
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
                fitted = json.loads(fit_result.read_text())
                row.update(
                    lsmr_status="ok",
                    lsmr_fit_seconds=fitted["fit_seconds"],
                    lsmr_process_seconds=process_seconds,
                    lsmr_peak_rss_mib=fitted["peak_rss_mib"],
                    lsmr_edr=fitted["metrics"]["projector_distance"],
                    lsmr_recovered=(
                        fitted["metrics"]["projector_distance"]
                        <= protocol["recovery_threshold"]
                    ),
                    lsmr_all_inner_certified=fitted["all_inner_certified"],
                    lsmr_inner_certified=fitted["n_inner_certified"],
                    lsmr_inner_total=fitted["n_inner_total"],
                    lsmr_stop_reason=fitted["stop_reason"],
                    lsmr_diagnostics=fitted["inner_diagnostics"],
                )
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                row.update(
                    lsmr_status=(
                        "timeout"
                        if isinstance(error, subprocess.TimeoutExpired)
                        else "error"
                    ),
                    lsmr_error=str(error),
                    lsmr_process_seconds=time.perf_counter() - started,
                )
            rows.append(row)
            _write_result(args.output, rows, manifest_hashes, args.runs, args.threads)
            print(
                f"m={m} seed={seed} LSMR={row['lsmr_status']} "
                f"MAVE EDR={row['mave_edr']:.4f} "
                f"MEANMAVE EDR={row['meanmave_edr']:.4f} "
                f"LSMR EDR={row.get('lsmr_edr', 'NA')}",
                flush=True,
            )
    return _write_result(args.output, rows, manifest_hashes, args.runs, args.threads)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=BENCHMARK_ROOT)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.runs < 1 or args.threads < 1 or args.timeout <= 0:
        parser.error("runs, threads, and timeout must be positive")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    (args.output / "fits").mkdir()
    (args.output / "jobs").mkdir()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
