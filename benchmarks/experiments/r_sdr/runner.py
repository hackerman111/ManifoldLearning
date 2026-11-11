"""Последовательные изолированные fits, лимиты и сохраняемые сырые строки."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .design import (
    ACCURATE_GRASSMANN_METHOD,
    ARMIJO_GRASSMANN_METHOD,
    COMBINED10_GRASSMANN_METHOD,
    COMBINED_GRASSMANN_METHOD,
    CONJUGATE_GRASSMANN_METHOD,
    CONSTANT_GRASSMANN_METHOD,
    GRASSMANN_METHOD,
    INTRINSIC_GRASSMANN_METHOD,
    QUADRATIC_GRASSMANN_METHOD,
    RELATIVE_GRASSMANN_METHOD,
    WARMUP_GRASSMANN_METHOD,
    Case,
    generate,
    metrics,
)
from .protocol import ROOT, digest, ensure_manifest, environment, snapshot, source_files
from .report import write_report

ADP_JSON_META_METHODS = {
    "ADP",
    GRASSMANN_METHOD,
    ACCURATE_GRASSMANN_METHOD,
    CONSTANT_GRASSMANN_METHOD,
    ARMIJO_GRASSMANN_METHOD,
    QUADRATIC_GRASSMANN_METHOD,
    INTRINSIC_GRASSMANN_METHOD,
    WARMUP_GRASSMANN_METHOD,
    CONJUGATE_GRASSMANN_METHOD,
    RELATIVE_GRASSMANN_METHOD,
    COMBINED10_GRASSMANN_METHOD,
    COMBINED_GRASSMANN_METHOD,
}


def read_worker_meta(method: str, path: Path) -> dict:
    if method in ADP_JSON_META_METHODS:
        return json.loads(path.read_text())
    with path.open() as stream:
        meta = next(csv.DictReader(stream, delimiter="\t"))
    meta["fit_seconds"] = float(meta["fit_seconds"])
    meta["iterations"] = int(meta["iterations"]) if meta["iterations"] else None
    meta["converged"] = {"TRUE": True, "FALSE": False, "": None}[meta["converged"]]
    return meta


def run_process(
    command: list[str],
    env: dict[str, str],
    log: Path,
    *,
    timeout: float,
    memory_mib: float,
    cwd: Path | None = None,
) -> dict:
    started = time.perf_counter()
    peak = 0.0
    status = "ok"
    with log.open("wb") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=cwd,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                try:
                    lines = Path(f"/proc/{process.pid}/status").read_text().splitlines()
                    values = [
                        float(line.split()[1]) / 1024
                        for line in lines
                        if line.startswith(("VmHWM:", "VmRSS:"))
                    ]
                    peak = max([peak, *values])
                except (FileNotFoundError, ProcessLookupError):
                    pass
                if time.perf_counter() - started > timeout:
                    status = "timeout"
                    break
                if peak > memory_mib:
                    status = "memory_limit"
                    break
                time.sleep(0.025)
        finally:
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    if status == "ok" and process.returncode:
        status = "error"
    return {
        "status": status,
        "returncode": process.returncode,
        "process_seconds": time.perf_counter() - started,
        "peak_rss_mib": peak or None,
    }


def adp_settings(case: Case, overrides: dict) -> dict:
    settings = {
        "N_lin": min(case.n, max(case.d + 2, 2 * case.d)),
        "N_J": min(case.n, 300),
        "N_phi": max(case.m + 1, min(case.d, 12)),
        "N_loc": 20,
        "index_init": "local",
        "select_step": "best",
    }
    settings.update(overrides)
    return settings


def load_rows(path: Path, protocol: dict) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    valid_cases = {c["name"] for c in protocol["cases"]}
    keys = set()
    for row in rows:
        key = (row["case"], row["seed"], row["method"])
        if (
            key in keys
            or row["case"] not in valid_cases
            or row["method"] not in protocol["methods"]
            or not protocol["seed"] <= row["seed"] < protocol["seed"] + protocol["runs"]
        ):
            raise ValueError("duplicate or foreign row in runs.jsonl")
        keys.add(key)
    return rows


def run(args, cases: list[Case], methods: list[str]) -> int:
    output = args.output.resolve()
    info, env = environment(args.rscript, args.threads)
    overrides = json.loads(args.adp_config.read_text()) if args.adp_config else {}
    if (
        not isinstance(overrides, dict)
        or {"seed", "kernel", "experimental_centers", "gpu", "gpu_solver"}
        & overrides.keys()
    ):
        raise ValueError(
            "--adp-config requires scalar CPU ADP_Config options, excluding"
            " seed/kernel/experimental_centers"
        )
    from ADP.core.ADP_Config import ADP_Config

    for case in cases:
        ADP_Config(**adp_settings(case, overrides))
    protocol = {
        "schema": 2,
        "cases": [asdict(c) for c in cases],
        "methods": methods,
        "runs": args.runs,
        "seed": args.seed,
        "planned_fits": len(cases) * len(methods) * args.runs,
        "timeout_seconds": args.timeout,
        "memory_limit_mib": args.memory_mib,
        "recovery_threshold": args.recovery_threshold,
        "r_iterations": args.r_iterations,
        "r_tolerance": args.r_tolerance,
        "sir_slices": args.sir_slices,
        "keep_inputs": args.keep_inputs,
        "rscript": args.rscript,
        "adp_config": {c.name: adp_settings(c, overrides) for c in cases},
        "adp_solver": {"max_steps": args.solver_steps, "tol": 1e-6},
        "grassman_solver": {
            "max_steps": args.grassman_steps,
            "tol": 1e-6,
            "local_ridge": 0.0,
            "relative_gn": False,
            "quadratic_trials": 0,
            "constant_centers": False,
            "smooth_max_steps": 5,
        },
        "accurate_solver": {"max_steps": 10, "tol": 1e-6, "local_ridge": 0.0},
        "constant_solver": {"max_steps": 5, "tol": 1e-6, "local_ridge": 0.0},
        "armijo_solver": {"max_steps": 10, "tol": 1e-6, "local_ridge": 0.0},
        "quadratic_solver": {
            "max_steps": 10,
            "tol": 1e-6,
            "local_ridge": 0.0,
            "smooth_max_steps": 5,
        },
        "intrinsic_solver": {
            "max_steps": 10,
            "tol": 1e-6,
            "local_ridge": 0.0,
            "smooth_max_steps": 5,
        },
        "warmup_solver": {
            "max_steps": 10,
            "tol": 1e-6,
            "local_ridge": 0.0,
            "smooth_max_steps": 5,
        },
        "conjugate_solver": {
            "max_steps": 10,
            "tol": 1e-6,
            "local_ridge": 0.0,
            "smooth_max_steps": 5,
        },
        "relative_solver": {
            "max_steps": 10,
            "tol": 1e-6,
            "local_ridge": 0.0,
            "relative_gn": True,
            "quadratic_trials": 0,
            "constant_centers": False,
            "smooth_max_steps": 5,
        },
        "combined10_solver": {
            "max_steps": 10,
            "tol": 1e-6,
            "local_ridge": 0.0,
            "relative_gn": True,
            "quadratic_trials": 3,
            "constant_centers": True,
            "smooth_max_steps": 5,
        },
        "combined_solver": {
            "max_steps": 20,
            "tol": 1e-6,
            "local_ridge": 0.0,
            "relative_gn": True,
            "quadratic_trials": 3,
            "constant_centers": True,
            "smooth_max_steps": None,
        },
        "environment": info,
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in source_files()},
        "implementations": {
            ("ADP"): (
                "current CPU multi-index HPAO LSMR; configured schedule; all "
                "inner certificates retained"
            ),
            (GRASSMANN_METHOD): (
                "ADP Grassmann core_gn with polar Armijo; local_ridge=0; "
                "same unpenalized finite-sketch profile; method-specific step budget"
            ),
            (ACCURATE_GRASSMANN_METHOD): (
                "experimental accurate Grassmann candidate; local_ridge=0; same "
                "unpenalized finite-sketch profile; fixed 10-step budget"
            ),
            (CONSTANT_GRASSMANN_METHOD): (
                "constant-center Grassmann: excludes exactly-zero U_j centers and "
                "restores their constant loss/coefficients/gradient scale; original "
                "rank guard and localized scalar fallback retained; 5-step budget"
            ),
            (ARMIJO_GRASSMANN_METHOD): (
                "true-profile rank-one Armijo; no bounded-scalar fallback; original "
                "GN guards; smooth-profile cap 5, nonsmooth cap 10"
            ),
            (QUADRATIC_GRASSMANN_METHOD): (
                "true-profile rank-one search with up to 3 quadratic angle proposals, "
                "then original Armijo fallback; original GN guards; smooth cap 5, "
                "nonsmooth cap 10"
            ),
            (INTRINSIC_GRASSMANN_METHOD): (
                "regular-center GN proposal for eligible rank-loss profiles; each "
                "trial and acceptance use the full all-center profile; original "
                "rank/workspace guards; max 10 steps, smooth cap 5"
            ),
            (WARMUP_GRASSMANN_METHOD): (
                "one original bounded-scalar step on each nonsmooth solve, then "
                "intrinsic GN; full-profile acceptance and original certificates; "
                "max 10 steps, smooth cap 5"
            ),
            CONJUGATE_GRASSMANN_METHOD: (
                "PR+ spectral rank-guard geodesics with projection transport and "
                "rank-reset guard; max 10 steps, smooth cap 5"
            ),
            RELATIVE_GRASSMANN_METHOD: (
                "intrinsic regular-center GN with relative damping "
                "lambda_prox*||J||F^2/(q*m); full all-center true-profile Armijo; "
                "smooth GN unchanged; max 10 steps, smooth cap 5"
            ),
            COMBINED10_GRASSMANN_METHOD: (
                "relative GN + true-profile quadratic3/Armijo + exact zero-U "
                "reduction; original gradient stopping scale; max10/smooth5"
            ),
            COMBINED_GRASSMANN_METHOD: (
                "relative GN + true-profile quadratic3/Armijo + exact zero-U "
                "reduction; original gradient stopping scale; max20, no smooth cap"
            ),
            ("MAVE"): (
                "R ordinary MAVE reference; full-X Gaussian kernel; fixed "
                "ambient bandwidth; MEANOPG init; QR solves"
            ),
            ("MAVE-package"): (
                "MAVE::mave.compute MEANMAVE; screen=d; max.dim=true m; "
                "internal defaults"
            ),
            ("SIR"): (
                "base R classical whitened SIR, tied quantile slices; no regularization"
            ),
            ("rMAVE"): (
                "R refined MAVE reference; MEANOPG init; Gaussian projected "
                "weights; QR solves; no ridge"
            ),
            ("dMAVE"): (
                "R untrimmed density MAVE reference; CSOPG init; joint all-Y "
                "targets; fixed Gaussian response bandwidth; QR solves"
            ),
        },
        "convergence_definitions": {
            "ADP": (
                "outer convergence unknown; all_inner_certified is separate; "
                "h_min/local_mass_limit are schedule stops"
            ),
            GRASSMANN_METHOD: (
                "outer convergence unknown; Grassmann inner stationarity is "
                "recorded per scale; schedule stops are not convergence certificates"
            ),
            ACCURATE_GRASSMANN_METHOD: (
                "outer convergence unknown; candidate inner diagnostics are "
                "recorded per scale; schedule stops are not convergence certificates"
            ),
            CONSTANT_GRASSMANN_METHOD: (
                "outer convergence unknown; candidate inner diagnostics are "
                "recorded per scale; schedule stops are not convergence certificates"
            ),
            ARMIJO_GRASSMANN_METHOD: (
                "outer convergence unknown; candidate inner diagnostics are "
                "recorded per scale; schedule stops are not convergence certificates"
            ),
            QUADRATIC_GRASSMANN_METHOD: (
                "outer convergence unknown; candidate inner diagnostics are "
                "recorded per scale; schedule stops are not convergence certificates"
            ),
            INTRINSIC_GRASSMANN_METHOD: (
                "full live all-center gradient and original stationarity applicability "
                "are retained; schedule stops are not convergence certificates"
            ),
            WARMUP_GRASSMANN_METHOD: (
                "full live all-center gradient and original stationarity applicability "
                "are retained; schedule stops are not convergence certificates"
            ),
            CONJUGATE_GRASSMANN_METHOD: (
                "full live all-center gradient and original stationarity applicability "
                "are retained; schedule stops are not convergence certificates"
            ),
            RELATIVE_GRASSMANN_METHOD: (
                "full live all-center gradient and original stationarity applicability "
                "are retained; schedule stops are not convergence certificates"
            ),
            COMBINED10_GRASSMANN_METHOD: (
                "live reduced-objective gradient; exact zero-U profiles are "
                "constant; full coefficient/loss/rank counts restored; schedule "
                "stops are not convergence certificates"
            ),
            COMBINED_GRASSMANN_METHOD: (
                "live reduced-objective gradient; exact zero-U profiles are "
                "constant; full coefficient/loss/rank counts restored; schedule "
                "stops are not convergence certificates"
            ),
            "MAVE": "outer subspace-step tolerance at fixed full-X bandwidth",
            "MAVE-package": "not exposed by package API",
            "SIR": "closed form; iterative convergence not applicable",
            "rMAVE": "outer subspace-step tolerance after reaching h_min",
            "dMAVE": "outer subspace-step tolerance after reaching h_min",
        },
        ("seed_scheme"): (
            "SeedSequence([run_seed,SHA256(case.name)[:8]]).spawn(3): "
            "basis,X,noise; method seed=SeedSequence([run_seed,1]) modulo "
            "2^31-1"
        ),
        ("quality"): (
            "||P_est-P_true||_F/sqrt(2m), stable principal angles; oracle m; "
            "original feature coordinates"
        ),
        ("kernel"): (
            "ADP K(q)=max(1-q^2,0), q=squared localized distance/bandwidth; R "
            "references Gaussian exp(-||B^T dx||^2/(2h^2))"
        ),
    }
    ensure_manifest(output, protocol, resume=args.resume)
    if not args.resume:
        snapshot(output)
    # Execute the saved sources. Audit them on resume before running another fit.
    frozen = output / "source"
    for relative, sha in protocol["sources"].items():
        if digest(frozen / relative) != sha:
            raise ValueError(f"frozen source differs: {relative}")
    env["PYTHONPATH"] = str(frozen)
    rows = load_rows(output / "runs.jsonl", protocol)
    done = {(r["case"], r["seed"], r["method"]) for r in rows}
    (output / "fits").mkdir(exist_ok=True)
    (output / "truth").mkdir(exist_ok=True)
    for case in cases:
        for seed in range(args.seed, args.seed + args.runs):
            todo = [m for m in methods if (case.name, seed, m) not in done]
            if not todo:
                continue
            x, y, truth = generate(case, seed)
            np.savetxt(
                output / "truth" / f"{case.name}-{seed}.csv",
                truth,
                delimiter=",",
                fmt="%.17g",
            )
            payload_sha = hashlib.sha256(
                x.tobytes() + y.tobytes() + truth.tobytes()
            ).hexdigest()
            order_rng = np.random.default_rng(
                np.random.SeedSequence([seed, int(payload_sha[:8], 16)])
            )
            order_rng.shuffle(todo)
            method_seed = int(
                np.random.SeedSequence([seed, 1]).generate_state(1)[0]
            ) % (2**31 - 1)
            with tempfile.TemporaryDirectory(prefix="r-sdr-") as scratch_dir:
                scratch = Path(scratch_dir)
                x_path, y_path = scratch / "x.csv", scratch / "y.csv"
                np.savetxt(x_path, x, delimiter=",", fmt="%.17g")
                np.savetxt(y_path, y, delimiter=",", fmt="%.17g")
                hashes = {
                    "x_csv_sha256": digest(x_path),
                    "y_csv_sha256": digest(y_path),
                }
                if args.keep_inputs:
                    inputs = output / "inputs"
                    inputs.mkdir(exist_ok=True)
                    np.savez_compressed(
                        inputs / f"{case.name}-{seed}.npz", x=x, y=y, truth=truth
                    )
                for position, method in enumerate(todo):
                    directory = output / "fits" / f"{case.name}-{seed}-{method}"
                    directory.mkdir(exist_ok=True)
                    basis_path, meta_path = (
                        directory / "basis.csv",
                        directory / "meta.txt",
                    )
                    row = {
                        "case": case.name,
                        "seed": seed,
                        "method": method,
                        "method_seed": method_seed,
                        "n": case.n,
                        "d": case.d,
                        "m": case.m,
                        "data_sha256": payload_sha,
                        "position_in_remaining_block": position,
                        "recovered": False,
                        **hashes,
                        "log": str((directory / "worker.log").relative_to(output)),
                    }
                    if method in ADP_JSON_META_METHODS:
                        job = {
                            "method": method,
                            "x": str(x_path),
                            "y": str(y_path),
                            "basis": str(basis_path),
                            "meta": str(meta_path),
                            "m": case.m,
                            "seed": method_seed,
                            "config": protocol["adp_config"][case.name],
                            "solver": protocol[
                                "accurate_solver"
                                if method == ACCURATE_GRASSMANN_METHOD
                                else "constant_solver"
                                if method == CONSTANT_GRASSMANN_METHOD
                                else "armijo_solver"
                                if method == ARMIJO_GRASSMANN_METHOD
                                else "quadratic_solver"
                                if method == QUADRATIC_GRASSMANN_METHOD
                                else "intrinsic_solver"
                                if method == INTRINSIC_GRASSMANN_METHOD
                                else "warmup_solver"
                                if method == WARMUP_GRASSMANN_METHOD
                                else "conjugate_solver"
                                if method == CONJUGATE_GRASSMANN_METHOD
                                else "relative_solver"
                                if method == RELATIVE_GRASSMANN_METHOD
                                else "combined10_solver"
                                if method == COMBINED10_GRASSMANN_METHOD
                                else "combined_solver"
                                if method == COMBINED_GRASSMANN_METHOD
                                else "grassman_solver"
                                if method == GRASSMANN_METHOD
                                else "adp_solver"
                            ],
                        }
                        job_path = scratch / "job.json"
                        job_path.write_text(json.dumps(job))
                        command = [
                            sys.executable,
                            "-m",
                            "experiments.r_sdr.worker",
                            str(job_path),
                        ]
                    else:
                        command = [
                            args.rscript,
                            "--vanilla",
                            str(frozen / "experiments/r_sdr/worker.R"),
                            method,
                            str(x_path),
                            str(y_path),
                            str(basis_path),
                            str(meta_path),
                            str(method_seed),
                            str(case.m),
                            str(args.r_iterations),
                            str(
                                args.sir_slices if method == "SIR" else args.r_tolerance
                            ),
                        ]
                    row.update(
                        run_process(
                            command,
                            env,
                            directory / "worker.log",
                            timeout=args.timeout,
                            memory_mib=args.memory_mib,
                            cwd=frozen,
                        )
                    )
                    if row["status"] == "ok":
                        try:
                            estimate = np.loadtxt(basis_path, delimiter=",", ndmin=2)
                            row.update(metrics(estimate, truth))
                            if method in ADP_JSON_META_METHODS:
                                meta = read_worker_meta(method, meta_path)
                                row["meta"] = str(meta_path.relative_to(output))
                                for key in (
                                    "all_inner_certified",
                                    "n_inner_certified",
                                    "n_inner_total",
                                ):
                                    row[key] = meta[key]
                            else:
                                meta = read_worker_meta(method, meta_path)
                            for key in (
                                "fit_seconds",
                                "iterations",
                                "converged",
                                "stop_reason",
                                "detail",
                                "solver_variant",
                            ):
                                row[key] = meta.get(key)
                            if (
                                not np.isfinite(row["fit_seconds"])
                                or row["fit_seconds"] < 0
                            ):
                                raise ValueError("invalid worker elapsed time")
                            row["recovered"] = (
                                row["projector_distance"] <= args.recovery_threshold
                            )
                        except (ValueError, KeyError, StopIteration, OSError) as error:
                            row.update(
                                status="error", error=str(error), recovered=False
                            )
                    if row["status"] != "ok":
                        row.setdefault(
                            "error", f"{row['status']}; inspect {row['log']}"
                        )
                    with (output / "runs.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    rows.append(row)
                    print(
                        f"[{len(rows)}/{protocol['planned_fits']}] {case.name} "
                        f"seed={seed} {method}: {row['status']} "
                        f"EDR={row.get('projector_distance', 'NA')} "
                        f"fit={row.get('fit_seconds', 'NA')}",
                        flush=True,
                    )
            write_report(output)
    write_report(output, plots=args.plots)
    complete = len(rows) == protocol["planned_fits"]
    state = {
        "complete": complete,
        "recorded_fits": len(rows),
        "failed_fits": sum(r["status"] != "ok" for r in rows),
    }
    (output / "completion.json").write_text(json.dumps(state, indent=2))
    return 0 if complete and not state["failed_fits"] else 1
