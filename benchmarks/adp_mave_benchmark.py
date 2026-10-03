#!/usr/bin/env python3
"""Paired recovery and fit-time benchmark for ADP and MAVE-family methods."""

# Long generated report/protocol descriptions are kept readable in source.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import scipy

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = Path(__file__).resolve().parent
sys.path[:] = [
    entry for entry in sys.path if not entry or Path(entry).resolve() != BENCHMARK_DIR
]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


METHODS = ("ADP", "MAVE", "dMAVE", "rMAVE")
DEFAULT_OUTPUT = "experiments/adp_mave_comparison_2026-10-02_anchors_final"
THREAD_ENV = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def set_single_thread_env() -> None:
    for name in THREAD_ENV:
        os.environ[name] = "1"


def read_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", ndmin=2, dtype=np.float64)


def peak_rss_kib() -> float:
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                return float(line.split()[1])
    # Linux reports ru_maxrss in KiB. Other platforms may use bytes.
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform.startswith("linux") else value / 1024.0


def write_basis(path: Path, basis: np.ndarray) -> None:
    np.savetxt(path, np.asarray(basis, dtype=np.float64), delimiter=",", fmt="%.17g")


def write_worker_metadata(
    path: Path,
    method: str,
    seconds: float,
    rss_kib: float,
    iterations: int | None,
    detail: str,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("method", "seconds", "peak_rss_kib", "iterations", "detail"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": method,
                "seconds": f"{seconds:.17g}",
                "peak_rss_kib": f"{rss_kib:.17g}",
                "iterations": "" if iterations is None else iterations,
                "detail": detail.replace("\t", " ").replace("\n", " "),
            }
        )


def run_adp_worker(args: argparse.Namespace) -> int:
    try:
        set_single_thread_env()
        x = read_matrix(args.x)
        y = np.loadtxt(args.y, delimiter=",", ndmin=1, dtype=np.float64)
        # Imports and process startup are outside fit_seconds.
        from ADP.core.ADP_Config import ADP_Config
        from ADP.core.multi.ADP_multi_index import ADP_multi_index

        config = ADP_Config(
            seed=args.seed,
            N_loc=8,
            N_lin=min(args.n, 100),
            N_J=min(args.n, 100),
            N_phi=12,
            outer_steps=20,
            lambda_penalty=0.05,
            local_ridge=1e-8,
            h_min=1.0,
            index_init="local",
            estimator="new",
            direction_mode="auto",
            select_step="best",
            batch_size=32,
        )
        model = ADP_multi_index(index_dim=args.dim, config=config)
        rss_before = peak_rss_kib()
        start = time.perf_counter()
        model.fit(x, y)
        seconds = time.perf_counter() - start
        rss = max(rss_before, peak_rss_kib())
        write_basis(args.basis, model.basis_)
        trace = model.trace_
        final_h = trace[-1].get("h") if trace else None
        detail = f"stop_reason={model.result_.stop_reason};final_h={final_h}"
        write_worker_metadata(args.meta, "ADP", seconds, rss, len(trace), detail)
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


def generate_data(seed: int, n: int, d: int, dim: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    truth = np.linalg.qr(rng.normal(size=(d, dim)))[0][:, :dim]
    z = x @ truth
    y = np.sin(z[:, 0]) + 0.5 * z[:, 1] ** 2
    for j in range(2, dim):
        y += 0.35 * np.tanh(z[:, j])
    y += rng.normal(scale=0.15, size=n)
    return x, y, truth


def subspace_metrics(
    estimate: np.ndarray, truth: np.ndarray, dim: int
) -> dict[str, float]:
    if estimate.ndim != 2 or estimate.shape != truth.shape:
        raise ValueError(
            f"basis shape {estimate.shape} does not match truth {truth.shape}"
        )
    if not np.all(np.isfinite(estimate)):
        raise ValueError("estimated basis contains non-finite values")
    q, r = np.linalg.qr(estimate)
    if np.linalg.matrix_rank(r) < dim:
        raise ValueError("estimated basis is rank deficient")
    singular_values = np.linalg.svd(q[:, :dim].T @ truth, compute_uv=False)
    cosines = np.clip(singular_values, 0.0, 1.0)
    overlap = float(np.mean(cosines**2))
    return {
        "projector_overlap": overlap,
        "projector_distance": math.sqrt(max(0.0, 1.0 - overlap)),
        "mean_principal_angle_deg": float(
            np.mean(np.arccos(cosines)) * 180.0 / math.pi
        ),
    }


def run_r_job(
    rscript: str,
    r_worker: Path,
    method: str,
    x_path: Path,
    y_path: Path,
    basis_path: Path,
    meta_path: Path,
    seed: int,
    dim: int,
    r_iterations: int,
    density_anchors: int,
) -> list[str]:
    command = [
        rscript,
        str(r_worker),
        method,
        str(x_path),
        str(y_path),
        str(basis_path),
        str(meta_path),
        str(seed),
        str(dim),
        str(r_iterations),
        str(density_anchors),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode == 0:
        return []
    return [
        completed.stderr.strip()
        or completed.stdout.strip()
        or f"R exited {completed.returncode}"
    ]


def parse_meta(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as stream:
        return next(csv.DictReader(stream, delimiter="\t"))


def write_jsonl(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def git_value(root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=False
    )
    if result.returncode:
        return None
    return result.stdout.strip()


def r_package_version(rscript: str) -> str:
    result = subprocess.run(
        [rscript, "-e", 'cat(as.character(packageVersion("MAVE")))'],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"R cannot load MAVE: {result.stderr.strip()}")
    return result.stdout.strip()


def summarize(
    rows: list[dict[str, object]],
    methods: list[str],
    seeds: list[int],
    n: int,
    d: int,
    dim: int,
    mave_version: str,
) -> str:
    lines = [
        "# ADP vs MAVE-family benchmark",
        "",
        "Synthetic paired benchmark; each method receives identical saved inputs for a given seed.",
        "Higher projector overlap is better; lower projector distance, principal angle, fit time, and RSS are better.",
        "Fit time excludes process startup, imports, and package loading. It includes method initialization and response-kernel construction.",
        "",
        "## Results",
        "",
        "| Method | Success | Median fit (s) | Median peak RSS (MiB) | Median overlap | Median projector distance | Median angle (deg) | Paired time ratio vs ADP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_method = {
        method: [row for row in rows if row["method"] == method] for method in methods
    }
    adp_seconds = {
        int(row["seed"]): float(row["fit_seconds"])
        for row in by_method.get("ADP", [])
        if row["status"] == "ok" and row["fit_seconds"] is not None
    }
    for method in methods:
        group = by_method[method]
        good = [row for row in group if row["status"] == "ok"]
        if not good:
            lines.append(f"| {method} | 0/{len(group)} | — | — | — | — | — | — |")
            continue
        times = [float(row["fit_seconds"]) for row in good]
        rss = [float(row["peak_rss_kib"]) / 1024 for row in good]
        overlap = [float(row["projector_overlap"]) for row in good]
        distance = [float(row["projector_distance"]) for row in good]
        angles = [float(row["mean_principal_angle_deg"]) for row in good]
        ratios = [
            float(row["fit_seconds"]) / adp_seconds[int(row["seed"])]
            for row in good
            if int(row["seed"]) in adp_seconds
        ]
        ratio = f"{float(np.median(ratios)):.3f}x" if ratios else "—"
        lines.append(
            f"| {method} | {len(good)}/{len(group)} | {float(np.median(times)):.3f} | "
            f"{float(np.median(rss)):.1f} | {float(np.median(overlap)):.4f} | "
            f"{float(np.median(distance)):.4f} | "
            f"{float(np.median(angles)):.2f} | {ratio} |"
        )

    lines.extend(
        [
            "",
            "## Protocol and interpretation",
            "",
            f"- Problem: `n={n}, d={d}, m={dim}`; seeds `{', '.join(map(str, seeds))}`; R MAVE `{mave_version}`. Raw per-fit rows, including failures, are in `runs.jsonl`.",
            "- Data: `X ~ N(0,I)`, `z = X B_true`, `y = sin(z1) + 0.5 z2^2 + N(0,0.15^2)` for `m=2` (additional active `tanh` terms for `m>2`).",
            "- `MAVE` is the package's `CSMAVE` method ([MAVE manual](https://stat.ethz.ch/CRAN/web/packages/MAVE/MAVE.pdf)). The installed MAVE package does not expose selectors named `dMAVE` or `rMAVE`.",
            "- `dMAVE` follows the kernelized-response construction for density MAVE ([review](https://pmc.ncbi.nlm.nih.gov/articles/PMC3685755/)): at five quantile anchors `t`, fit package `CSMAVE` to `K_h(y-t)` and aggregate projectors. `h=1.06 sd(y) n^(-1/5)`; all anchor fits are included in fit time.",
            "- `rMAVE` is a benchmark-only projected-kernel refinement initialized with package `CSOPG`; its local/global weighted least-squares iterations and bandwidth schedule are implemented in `benchmarks/adp_mave_methods.R`.",
            "- These are distinct estimators/implementations. This small synthetic suite measures recovery for the stated model and does not establish general superiority.",
            "- Peak RSS is process high-water mark from `/proc/self/status` when available. R/Python startup memory is included; startup time is reported separately as `process_seconds` in raw rows.",
            "",
            "## Reproduction",
            "",
            "```sh",
            "UV_CACHE_DIR=/tmp/adp-mave-uv-cache uv run --no-sync python benchmarks/adp_mave_benchmark.py --output experiments/adp_mave_comparison_reproduction",
            "```",
            "",
            "The output directory must be new; the driver refuses to overwrite previous measurements.",
        ]
    )
    return "\n".join(lines)


def run_benchmark(args: argparse.Namespace) -> int:
    set_single_thread_env()
    root = Path(__file__).resolve().parents[1]
    rscript = shutil.which(args.rscript)
    if rscript is None:
        raise RuntimeError(f"Rscript not found: {args.rscript}")
    package_version = r_package_version(rscript)
    r_version_result = subprocess.run(
        [rscript, "--version"], text=True, capture_output=True, check=False
    )
    r_version = (r_version_result.stdout + r_version_result.stderr).strip()
    r_worker = Path(__file__).with_name("adp_mave_methods.R").resolve()
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    input_dir = output / "inputs"
    input_dir.mkdir()
    runs_path = output / "runs.jsonl"

    protocol = {
        "benchmark": "ADP multi-index vs MAVE/CSMAVE, response-anchor dMAVE discretization, and projected-kernel rMAVE refinement",
        "date": "2026-10-02",
        "git_commit": git_value(root, "rev-parse", "HEAD"),
        "git_status": git_value(root, "status", "--short"),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numpy_build_configuration": "",
        "rscript": rscript,
        "r_version": r_version,
        "mave_version": package_version,
        "n": args.n,
        "d": args.d,
        "m": args.dim,
        "seeds": args.seeds,
        "methods": args.methods,
        "rMAVE_max_iterations": args.r_iterations,
        "dMAVE_response_anchors": args.density_anchors,
        "threads": {name: os.environ[name] for name in THREAD_ENV},
        "data_model": "X~N(0,I); y=sin(z1)+0.5*z2^2+0.35*sum(tanh(zj),j>=3)+N(0,0.15^2), z=X B_true",
        "data_seed_scheme": "numpy.default_rng(seed), X then QR Gaussian B_true then response noise",
        "input_hash": "SHA256 of contiguous float64 X bytes followed by y bytes; same hash required across methods within seed",
        "quality": {
            "projector_overlap": "mean squared cosine of principal angles; higher is better",
            "projector_distance": "sqrt(1 - projector_overlap), equal to ||P_hat-P_true||_F/sqrt(2m); lower is better",
            "mean_principal_angle_deg": "mean principal angle in degrees; lower is better",
        },
        "timing_boundary": "fit only; excludes process startup/import/package load, includes algorithm initialization and response-kernel construction",
        "memory_boundary": "per-process VmHWM high-water mark including startup and imports",
        "method_settings": {
            "ADP": {
                "N_loc": 8,
                "N_lin": min(args.n, 100),
                "N_J": min(args.n, 100),
                "N_phi": 12,
                "outer_steps": 20,
                "lambda_penalty": 0.05,
                "local_ridge": 1e-8,
                "h_min": 1.0,
                "index_init": "local",
                "estimator": "new",
                "select_step": "best",
            },
            "MAVE": {"method": "CSMAVE", "max.dim": args.dim, "screen": args.d},
            "dMAVE": {
                "method": "CSMAVE on K_h(y-t) responses",
                "response_bandwidth": "1.06 * sd(y) * n^(-1/5)",
                "anchor_probabilities": list(
                    np.linspace(0.1, 0.9, args.density_anchors)
                ),
            },
            "rMAVE": {
                "initialization": "MAVE::mave.compute(method=CSOPG)",
                "kernel": "exp(-0.5 * projected_squared_distance / h^2)",
                "initial_bandwidth": "2.34 * n^(-1/(max(d,3)+6))",
                "minimum_bandwidth": "2.34 * n^(-1/(m+4))",
                "bandwidth_shrink": "n^(-1/(2*(max(d,3)+6)))",
                "global_ridge_relative": 1e-8,
                "max_iterations": args.r_iterations,
            },
        },
    }
    build_config = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with contextlib.redirect_stdout(build_config):
            np.show_config()
    protocol["numpy_build_configuration"] = build_config.getvalue()
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        x, y, truth = generate_data(seed, args.n, args.d, args.dim)
        seed_dir = input_dir / f"seed_{seed}"
        seed_dir.mkdir()
        x_path, y_path = seed_dir / "X.csv", seed_dir / "y.csv"
        np.savetxt(x_path, x, delimiter=",", fmt="%.17g")
        np.savetxt(y_path, y, delimiter=",", fmt="%.17g")
        np.savetxt(seed_dir / "B_true.csv", truth, delimiter=",", fmt="%.17g")
        input_hash = hashlib.sha256(
            x.tobytes(order="C") + y.tobytes(order="C")
        ).hexdigest()
        # Rotate execution order across seeds to reduce systematic order effects.
        offset = args.seeds.index(seed) % len(args.methods)
        order = args.methods[offset:] + args.methods[:offset]

        for method in order:
            row: dict[str, object] = {
                "method": method,
                "seed": seed,
                "input_sha256": input_hash,
                "status": "error",
                "fit_seconds": None,
                "process_seconds": None,
                "peak_rss_kib": None,
                "projector_overlap": None,
                "projector_distance": None,
                "mean_principal_angle_deg": None,
                "iterations": None,
                "detail": "",
                "error": None,
            }
            with tempfile.TemporaryDirectory(prefix="adp-mave-") as temp:
                temp_dir = Path(temp)
                basis_path, meta_path = temp_dir / "basis.csv", temp_dir / "meta.tsv"
                start = time.perf_counter()
                if method == "ADP":
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker-adp",
                        "--method",
                        method,
                        "--x",
                        str(x_path),
                        "--y",
                        str(y_path),
                        "--basis",
                        str(basis_path),
                        "--meta",
                        str(meta_path),
                        "--seed",
                        str(seed),
                        "--n",
                        str(args.n),
                        "--dim",
                        str(args.dim),
                    ]
                    child_env = os.environ.copy()
                    completed = subprocess.run(
                        command,
                        text=True,
                        capture_output=True,
                        check=False,
                        env=child_env,
                    )
                    errors = (
                        []
                        if completed.returncode == 0
                        else [
                            completed.stderr.strip()
                            or f"Python worker exited {completed.returncode}"
                        ]
                    )
                else:
                    errors = run_r_job(
                        rscript,
                        r_worker,
                        method,
                        x_path,
                        y_path,
                        basis_path,
                        meta_path,
                        seed,
                        args.dim,
                        args.r_iterations,
                        args.density_anchors,
                    )
                row["process_seconds"] = time.perf_counter() - start
                if errors:
                    row["error"] = errors[-1][-4000:]
                elif basis_path.exists() and meta_path.exists():
                    try:
                        metadata = parse_meta(meta_path)
                        estimate = read_matrix(basis_path)
                        iteration_value = metadata["iterations"]
                        row.update(
                            {
                                "status": "ok",
                                "fit_seconds": float(metadata["seconds"]),
                                "peak_rss_kib": float(metadata["peak_rss_kib"]),
                                "iterations": int(iteration_value)
                                if iteration_value not in {"", "NA", "NaN"}
                                else None,
                                "detail": metadata["detail"],
                                **subspace_metrics(estimate, truth, args.dim),
                            }
                        )
                    except Exception as exc:
                        row["error"] = (
                            f"invalid worker output: {type(exc).__name__}: {exc}"
                        )
                else:
                    row["error"] = "worker exited without basis/metadata"
            rows.append(row)
            write_jsonl(runs_path, row)
            print(
                f"seed={seed} method={method} status={row['status']} "
                f"fit_s={row['fit_seconds']} overlap={row['projector_overlap']}"
            )

    report = summarize(
        rows, args.methods, args.seeds, args.n, args.d, args.dim, package_version
    )
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"report: {output / 'REPORT.md'}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--r-iterations", type=int, default=8)
    parser.add_argument("--d-anchors", dest="density_anchors", type=int, default=5)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--worker-adp", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--x", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--y", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--basis", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--meta", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    set_single_thread_env()
    args = make_parser().parse_args()
    if args.worker_adp:
        if any(
            value is None
            for value in (
                args.method,
                args.x,
                args.y,
                args.basis,
                args.meta,
                args.seed,
                args.n,
                args.dim,
            )
        ):
            raise SystemExit("incomplete ADP worker arguments")
        return run_adp_worker(args)
    if args.n <= args.d or args.d <= args.dim or args.dim < 2:
        raise SystemExit("require n > d > m >= 2")
    if args.r_iterations < 1:
        raise SystemExit("--r-iterations must be positive")
    if args.density_anchors < 1:
        raise SystemExit("--d-anchors must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("seeds must be unique")
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
