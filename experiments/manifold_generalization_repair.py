"""Paired spectrum check with identifiable recovery and separate chart errors.

Run from the repository root: ``python -m experiments.manifold_generalization_repair``.
The original manifold_generalization.py and its result directories remain frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .manifold_generalization import DiagnosticManifold, make_data
from .manifold_recovery_probe import _version
from .runner import _local_subspace_metrics


def run_case(
    case: dict[str, Any], spectrum: Literal["relative", "unit"]
) -> dict[str, Any]:
    """Fit once; truth is used only after fit for diagnostic distances."""
    X, Y, truth, queries, query_truth, model_seed = make_data(
        case["seed"], 600, 8, case["m"], case["curvature"], case["noise"]
    )
    broad = case["support"] == "broad"
    identifiable = case["curvature"] == 0.0 or case["m"] == 1
    model = DiagnosticManifold(
        case["m"],
        estimator="manifold",
        localization_spectrum=spectrum,
        N_loc=300 if broad else 80,
        N_lin=300 if broad else 200,
        N_J=40,
        N_phi=40,
        N_manifold=30 if broad else 10,
        sync_steps=3,
        lambda_manifold=0.5,
        cg_tol=1e-6,
        solver="cg",
        seed=model_seed,
        scale_boundary="stop",
    )
    row: dict[str, Any] = {
        **case,
        "spectrum": spectrum,
        "model_seed": model_seed,
        "identifiable_target": identifiable,
        "error": None,
        "recovered": False if identifiable else None,
    }
    started = time.perf_counter()
    try:
        model.fit(X, Y)
        center_truth = truth[model.center_indices_]
        nearest = model._nearest_center_indices(model._prepare_queries(queries))
        metrics = {
            "center": _local_subspace_metrics(center_truth, model.projectors_),
            "chart_estimation": _local_subspace_metrics(
                center_truth[nearest], model.projectors_[nearest]
            ),
            "query_raw": _local_subspace_metrics(
                query_truth, model.projectors_[nearest]
            ),
            "query_oracle": _local_subspace_metrics(
                query_truth, center_truth[nearest]
            ),
        }
        for name, (rms, maximum, _) in metrics.items():
            row[name + "_rms"] = rms
            row[name + "_max"] = maximum
        center_pass = row["center_rms"] <= 0.2 and row["center_max"] <= 0.2
        chart_pass = (
            row["chart_estimation_rms"] <= 0.2
            and row["chart_estimation_max"] <= 0.2
        )
        row["generator_center_gate"] = center_pass
        if identifiable:
            row["recovered"] = center_pass and chart_pass
        row.update(
            stop_reason=model.stop_reason_,
            n_scales=model.n_scales_,
            effective_config=model.effective_config_,
            linear_residual_max=max(
                float(step["linear_relative_residual_max"])
                for step in model.trace_
            ),
            support_diagnostics=getattr(model, "debug", {}),
        )
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["support_diagnostics"] = getattr(model, "debug", {})
    row["fit_seconds"] = time.perf_counter() - started
    row["process_peak_rss_mib"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    )
    return row


def summarize(
    rows: list[dict[str, Any]], cases: list[dict[str, Any]]
) -> dict[str, Any]:
    profiles = []
    for support in ("local", "broad"):
        for m in (1, 2, 3):
            for curvature in (0.0, 0.35, 0.8):
                for noise in (0.0, 0.1):
                    case_key = (support, m, curvature, noise)
                    planned = sum(
                        (c["support"], c["m"], c["curvature"], c["noise"])
                        == case_key
                        for c in cases
                    )
                    if planned == 0:
                        continue
                    for spectrum in ("relative", "unit"):
                        selected = [
                            r
                            for r in rows
                            if (
                                r["support"],
                                r["m"],
                                r["curvature"],
                                r["noise"],
                                r["spectrum"],
                            )
                            == (*case_key, spectrum)
                        ]
                        entry = {
                            "support": support,
                            "m": m,
                            "curvature": curvature,
                            "noise": noise,
                            "spectrum": spectrum,
                            "planned": planned,
                            "finished": len(selected),
                            "numerical_failures": sum(
                                r["error"] is not None for r in selected
                            ),
                            "identifiable_target": curvature == 0.0 or m == 1,
                            "recovered": sum(r["recovered"] is True for r in selected),
                            "generator_center_gate": sum(
                                r.get("generator_center_gate") is True for r in selected
                            ),
                        }
                        for name in (
                            "center_rms",
                            "center_max",
                            "chart_estimation_max",
                            "query_raw_max",
                            "query_oracle_max",
                            "fit_seconds",
                        ):
                            values = [r[name] for r in selected if name in r]
                            entry[name + "_median"] = (
                                float(np.median(values)) if values else None
                            )
                        profiles.append(entry)
    return {
        "complete": len(rows) == 2 * len(cases),
        "planned": 2 * len(cases),
        "finished": len(rows),
        "profiles": profiles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("selection", "validation-flat"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--budget-seconds", type=float, default=600.0)
    args = parser.parse_args()
    if (
        args.profile is None
        or args.out is None
        or not math.isfinite(args.budget_seconds)
        or args.budget_seconds <= 0
    ):
        parser.error("--profile, --out and a positive --budget-seconds are required")
    cases = [
        {"seed": seed, "support": support, "m": m, "curvature": c, "noise": noise}
        for seed in (
            range(81000, 81005)
            if args.profile == "selection"
            else range(82000, 82005)
        )
        for support in ("local", "broad")
        for m in (1, 2, 3)
        for c in ((0.0, 0.35, 0.8) if args.profile == "selection" else (0.0,))
        for noise in (0.0, 0.1)
    ]
    np.random.default_rng(20260928).shuffle(cases)
    args.out.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=1):
        version = _version()
        hashes = version["source_sha256"]
        assert isinstance(hashes, dict)
        for source in (
            "experiments/manifold_generalization.py",
            "experiments/manifold_generalization_repair.py",
            "experiments/runner.py",
        ):
            hashes[source] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        manifest = {
            "profile": args.profile,
            "cases": cases,
            "n": 600,
            "d": 8,
            "query_count": 512,
            "dtype": "float64",
            "version": version,
            "threads": threadpool_info(),
            "budget_seconds": args.budget_seconds,
            "model": {
                "estimator": "manifold",
                "localization_spectrum": ["relative", "unit"],
                "solver": "cg",
                "N_loc": "local:80, broad:300",
                "N_lin": "local:200, broad:300",
                "N_J": 40,
                "N_phi": 40,
                "N_manifold": "local:10, broad:30",
                "sync_steps": 3,
                "lambda_manifold": 0.5,
                "cg_tol": 1e-6,
                "scale_boundary": "stop",
                "batch_size": 32,
            },
            "kernel": "max(1-(distance_squared/h^2)^2,0)",
            "directions": "independent normalized Gaussian, refreshed each scale",
            "centers": "uniform without replacement, production seed stream",
            "anisotropy": "production alpha mass search",
            "seed_scheme": "SeedSequence(seed).spawn(5): X,Q,noise,queries,model",
            "recovery": (
                "flat or m=1 curved only; center and nearest-chart estimation "
                "RMS/max sine <=0.2; raw query and exact-chart oracle separate"
            ),
            "curved_m_ge_2": (
                "generator row(Dz) descriptive; not identified from scalar Y"
            ),
            "rss": "cumulative process ru_maxrss MiB, not per-fit peak",
        }
        (args.out / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        rows: list[dict[str, Any]] = []
        started = time.perf_counter()
        with (args.out / "runs.jsonl").open("w", encoding="utf-8") as stream:
            for case in cases:
                for spectrum in ("relative", "unit"):
                    if time.perf_counter() - started >= args.budget_seconds:
                        break
                    row = run_case(case, spectrum)
                    rows.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    (args.out / "summary.json").write_text(
                        json.dumps(summarize(rows, cases), indent=2), encoding="utf-8"
                    )
                    print(
                        f"{len(rows)}/{2 * len(cases)} {case} {spectrum} "
                        f"center={row.get('center_rms')} "
                        f"recovered={row['recovered']} error={row['error']}",
                        flush=True,
                    )
                if time.perf_counter() - started >= args.budget_seconds:
                    break


if __name__ == "__main__":
    main()
