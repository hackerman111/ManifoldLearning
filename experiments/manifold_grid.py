"""Воспроизводимая сетка manifold из ``Manifold exp.md``."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .manifold_generalization import DiagnosticManifold, make_data
from .manifold_recovery_probe import _version
from .runner import _local_subspace_metrics


@dataclass(frozen=True, slots=True)
class Cell:
    n: int
    d: int
    m: int
    curvature: float
    noise: float
    series: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[int, int, int, float, float]:
        return (self.n, self.d, self.m, self.curvature, self.noise)


SERIES: dict[str, tuple[tuple[int, int, int, float, float], ...]] = {
    "dimension": tuple((200, d, 1, 0.35, 0.1) for d in (3, 4, 6, 10)),
    "sample-size": tuple((n, 10, 1, 0.35, 0.1) for n in (100, 200, 400, 800)),
    "noise": tuple((400, 10, 2, 0.35, s) for s in (0.05, 0.10, 0.20)),
    "intrinsic-dimension": tuple((800, 10, m, 0.35, 0.1) for m in (1, 2, 3)),
    "curvature": tuple((400, 10, 2, c, 0.1) for c in (0.0, 0.35, 0.8)),
    "hd-fixed-n": tuple((800, d, 2, 0.35, 0.1) for d in (10, 20, 50, 100, 200)),
    "hd-fixed-n-over-d": (
        (1600, 20, 2, 0.35, 0.1),
        (4000, 50, 2, 0.35, 0.1),
        (8000, 100, 2, 0.35, 0.1),
    ),
    "noiseless-sanity": ((400, 10, 2, 0.8, 0.0),),
}


def grid() -> tuple[Cell, ...]:
    """Deduplicate identical data and estimator configurations, retaining labels."""
    memberships: dict[tuple[int, int, int, float, float], list[str]] = {}
    for name, points in SERIES.items():
        for point in points:
            memberships.setdefault(point, []).append(name)
    return tuple(Cell(*key, tuple(names)) for key, names in memberships.items())


def _selected_cells(names: tuple[str, ...]) -> tuple[Cell, ...]:
    selected = set(names)
    return tuple(cell for cell in grid() if selected.intersection(cell.series))


def _n_lin_target(cell: Cell) -> int:
    """Closest target mass to 200 allowed by the live manifold validation."""
    return max(cell.d + 2, min(200, cell.n - 1))


def _run_case(
    cell: Cell, seed: int, scale_boundary: Literal["raise", "stop"]
) -> dict[str, Any]:
    started = time.perf_counter()
    X, Y, truth, queries, query_truth, model_seed = make_data(
        seed, cell.n, cell.d, cell.m, cell.curvature, cell.noise
    )
    n_lin = _n_lin_target(cell)
    h_min = 3.0 * float(np.std(X, axis=0).mean()) / math.sqrt(cell.n)
    model = DiagnosticManifold(
        cell.m,
        estimator="manifold",
        solver="cg",
        N_J=40,
        N_phi=40,
        N_loc=80,
        N_lin=n_lin,
        N_manifold=10,
        sync_steps=3,
        lambda_manifold=0.5,
        cg_tol=1e-6,
        a=2.0 ** (1.0 / cell.m),
        h_min=h_min,
        batch_size=32,
        seed=model_seed,
        scale_boundary=scale_boundary,
    )
    identifiable = cell.curvature == 0.0 or cell.m == 1
    row: dict[str, Any] = {
        **asdict(cell),
        "seed": seed,
        "model_seed": model_seed,
        "N_lin": n_lin,
        "identifiable_target": identifiable,
        "error": None,
        "recovered": False if identifiable else None,
    }
    fit_started = time.perf_counter()
    try:
        model.fit(X, Y)
        row["fit_seconds"] = time.perf_counter() - fit_started
        evaluation_started = time.perf_counter()
        center_rms, center_max, _ = _local_subspace_metrics(
            truth[model.center_indices_], model.projectors_
        )
        center_truth = truth[model.center_indices_]
        nearest = model._nearest_center_indices(model._prepare_queries(queries))
        chart_rms, chart_max, _ = _local_subspace_metrics(
            center_truth[nearest], model.projectors_[nearest]
        )
        query_rms, query_max, _ = _local_subspace_metrics(
            query_truth, model.projectors_[nearest]
        )
        oracle_rms, oracle_max, _ = _local_subspace_metrics(
            query_truth, center_truth[nearest]
        )
        center_pass = center_rms <= 0.2 and center_max <= 0.2
        chart_pass = chart_rms <= 0.2 and chart_max <= 0.2
        row.update(
            center_rms=center_rms,
            center_max=center_max,
            chart_estimation_rms=chart_rms,
            chart_estimation_max=chart_max,
            query_raw_rms=query_rms,
            query_raw_max=query_max,
            query_oracle_rms=oracle_rms,
            query_oracle_max=oracle_max,
            generator_center_gate=center_pass,
            stop_reason=model.stop_reason_,
            n_scales=model.n_scales_,
            effective_config=model.effective_config_,
            linear_residual_max=max(
                float(trace["linear_relative_residual_max"]) for trace in model.trace_
            ),
            trace=model.trace_,
        )
        if identifiable:
            row["recovered"] = center_pass and chart_pass
        row["evaluation_seconds"] = time.perf_counter() - evaluation_started
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row.setdefault("fit_seconds", time.perf_counter() - fit_started)
    row["support_diagnostics"] = getattr(model, "debug", {})
    row["total_seconds"] = time.perf_counter() - started
    row["process_cumulative_peak_rss_mib"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    )
    return row


def _summary(
    rows: list[dict[str, Any]], names: tuple[str, ...], runs: int
) -> dict[str, Any]:
    series_rows = {}
    for name in names:
        selected = [row for row in rows if name in row["series"]]
        planned = sum(name in cell.series for cell in grid()) * runs
        valid = [row for row in selected if row["error"] is None]
        identifiable = [row for row in selected if row["identifiable_target"]]
        series_rows[name] = {
            "planned": planned,
            "finished": len(selected),
            "errors": sum(row["error"] is not None for row in selected),
            "identifiable_finished": len(identifiable),
            "recovered": sum(row["recovered"] is True for row in selected),
            "function_boundary_stops": sum(
                row.get("stop_reason") == "function_mass_boundary" for row in selected
            ),
            "manifold_boundary_stops": sum(
                row.get("stop_reason") == "manifold_mass_boundary" for row in selected
            ),
            **{
                f"{metric}_median": (
                    float(np.median([row[metric] for row in valid if metric in row]))
                    if any(metric in row for row in valid)
                    else None
                )
                for metric in (
                    "center_rms",
                    "center_max",
                    "chart_estimation_rms",
                    "chart_estimation_max",
                    "query_raw_rms",
                    "query_raw_max",
                    "query_oracle_max",
                    "fit_seconds",
                )
            },
        }
    planned = sum(series_rows[name]["planned"] for name in names)
    # A cell shared by several selected series is fitted once and counted once.
    selected_cells = _selected_cells(names)
    planned_unique = len(selected_cells) * runs
    return {
        "complete": len(rows) == planned_unique,
        "planned_unique_fits": planned_unique,
        "finished_unique_fits": len(rows),
        "errors": sum(row["error"] is not None for row in rows),
        "series": series_rows,
        "series_fit_sum_with_reuse": planned,
    }


def _source_manifest() -> dict[str, Any]:
    version = _version()
    hashes = version.get("source_sha256")
    if isinstance(hashes, dict):
        hashes[__file__] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        generalization_path = Path("experiments/manifold_generalization.py")
        hashes[str(generalization_path)] = hashlib.sha256(
            generalization_path.read_bytes()
        ).hexdigest()
    return version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=("development", "full"), default="development"
    )
    parser.add_argument("--runs", type=int, help="Переопределить повторы профиля")
    parser.add_argument("--seed", type=int, default=81000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--scale-boundary", choices=("stop", "raise"), default="stop"
    )
    parser.add_argument("--series", help="Серии через запятую; по умолчанию все")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.list:
        for name, points in SERIES.items():
            print(f"{name}: {len(points)} rows")
        print(f"unique configurations: {len(grid())}")
        return 0

    names = (
        tuple(name.strip() for name in args.series.split(","))
        if args.series
        else tuple(SERIES)
    )
    if not names or any(not name for name in names):
        parser.error("--series содержит пустой селектор")
    unknown = tuple(name for name in names if name not in SERIES)
    if unknown:
        parser.error("неизвестные серии: " + ", ".join(unknown))
    if len(set(names)) != len(names):
        parser.error("--series содержит повторяющиеся имена")
    runs = (
        args.runs
        if args.runs is not None
        else (30 if args.profile == "development" else 250)
    )
    if runs < 1 or args.seed < 0 or args.threads < 1:
        parser.error(
            "runs и threads должны быть положительными, seed — неотрицательным"
        )

    cells = _selected_cells(names)
    planned = len(cells) * runs
    print(f"Серии: {', '.join(names)}")
    for name in names:
        count = sum(name in cell.series for cell in cells)
        print(f"{name}: {count} points x {runs} runs = {count * runs} fits")
    print(
        f"Итого: {len(cells)} уникальных конфигураций, {planned} fits; "
        f"profile={args.profile}; scale_boundary={args.scale_boundary}"
    )
    if args.dry_run:
        for cell in cells:
            print(
                f"n={cell.n}, d={cell.d}, m={cell.m}, c={cell.curvature:g}, "
                f"sigma={cell.noise:g}; N_lin={_n_lin_target(cell)}; "
                f"series={','.join(cell.series)}"
            )
        return 0

    output_dir = args.output_dir or Path("benchmark_outputs/experiments") / (
        datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f%z") + "-manifold-grid"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=args.threads):
        active_threadpools = threadpool_info()
    manifest = {
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(),
        "profile": args.profile,
        "series": list(names),
        "cells": [asdict(cell) for cell in cells],
        "runs": runs,
        "seeds": [args.seed, args.seed + runs - 1],
        "seed_scheme": "SeedSequence(seed).spawn(5): X, Q, noise, queries, model",
        "planned_unique_fits": planned,
        "dtype": "float64",
        "data_formula": (
            "X~N(0,I_d); u=X@Q; z_k=u_k+c*u_(m+k)^2/2; "
            "f=sum(z_k^2)/2; Y=(f-mean(f))/std(f)+sigma*N(0,1)"
        ),
        "truth": (
            "row(Dz), analytic full-rank generator chart; "
            "descriptive for curved m>=2"
        ),
        "kernel": "max(1 - (distance_squared / h**2)**2, 0)",
        "bandwidth_rule": (
            "production target-mass search; h_min=3*mean(std(X))/sqrt(n); "
            "a=2**(1/m)"
        ),
        "anisotropy": "production structure-adaptive alpha mass search",
        "centers": "40 uniform samples without replacement; production seed stream",
        "directions": "independent normalized Gaussian directions, refreshed by scale",
        "local_linear_solver": (
            "production weighted least squares with unchanged rank guards"
        ),
        "rss_measurement": (
            "Linux process ru_maxrss high-water mark, cumulative; not per-fit peak"
        ),
        "estimator": {
            "name": "manifold",
            "solver": "cg",
            "N_J": 40,
            "N_phi": 40,
            "N_loc": 80,
            "N_lin_policy": "max(d+2, min(200, n-1))",
            "N_manifold": 10,
            "sync_steps": 3,
            "lambda_manifold": 0.5,
            "cg_tol": 1e-6,
            "support": "local",
            "scale_boundary": args.scale_boundary,
            "batch_size": 32,
            "a": "2**(1/m)",
            "h_min": "3*mean(std(X, axis=0))/sqrt(n)",
        },
        "recovery": (
            "identifiable if c=0 or m=1; center AND same-center chart-estimation "
            "RMS/max principal sine <= 0.2; raw and oracle nearest-chart query "
            "errors are descriptive"
        ),
        "threadpool": {
            "requested_threads": args.threads,
            "active": active_threadpools,
        },
        "version": _source_manifest(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    summary_path = output_dir / "summary.json"
    runs_path = output_dir / "runs.jsonl"
    with (
        threadpool_limits(limits=args.threads),
        runs_path.open("w", encoding="utf-8") as stream,
    ):
        (output_dir / "summary.json").write_text(
            json.dumps(_summary(rows, names, runs), indent=2), encoding="utf-8"
        )
        for cell in cells:
            for seed in range(args.seed, args.seed + runs):
                row = _run_case(cell, seed, args.scale_boundary)
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                summary_path.write_text(
                    json.dumps(_summary(rows, names, runs), indent=2),
                    encoding="utf-8",
                )
                print(
                    f"{len(rows)}/{planned} {cell.key} seed={seed} "
                    f"stop={row.get('stop_reason')} "
                    f"recovered={row['recovered']} error={row['error']}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
