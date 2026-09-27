"""Paired production-model validation for the frozen manifold recovery protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from ADP import ADP_Manifold

from .data import _generate_data, _make_seed_bundle
from .diagnostic import make_case
from .manifold_recovery_probe import _version
from .runner import _local_subspace_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=("selection", "validation", "noiseless"), required=True
    )
    parser.add_argument("--solver", choices=("cg", "hybrid"), default="cg")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    case = make_case("manifold", "base")
    point = case.experiment.smoke
    assert (
        point.h_min_factor is not None
        and point.N_loc is not None
        and point.sync_steps is not None
        and point.lambda_manifold is not None
    )
    seeds = (
        range(10)
        if args.split == "selection"
        else range(100, 120)
        if args.split == "validation"
        else range(200, 205)
    )
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        if args.split == "noiseless":
            rng = np.random.default_rng(seed)
            X = rng.normal(size=(point.n, point.d))
            Y = np.sin(X[:, 0]) + 0.5 * np.square(X[:, 1])
            Y = (Y - Y.mean()) / Y.std()
            truth = np.zeros_like(X)
            truth[:, 0] = np.cos(X[:, 0])
            truth[:, 1] = X[:, 1]
            truth /= np.linalg.norm(truth, axis=1)[:, None]
            model_seed = int(np.random.SeedSequence([seed, 999]).generate_state(1)[0])
        else:
            bundle = _make_seed_bundle(
                case.experiment.selector,
                point,
                seed,
                common_random_fields=case.experiment.common_random_fields,
            )
            generated = _generate_data(case.experiment.selector, point, bundle, seed)
            X, Y = generated.X, generated.Y
            truth = generated.beta[:, :, 0]
            model_seed = bundle.init
        for estimator in (
            ("manifold", "local_quadratic")
            if seed % 2 == 0
            else ("local_quadratic", "manifold")
        ):
            model = ADP_Manifold(
                1,
                estimator=estimator,
                N_loc=point.N_loc,
                N_lin=point.N_lin,
                N_J=point.N_J,
                N_phi=point.N_phi,
                N_manifold=point.N_manifold,
                sync_steps=point.sync_steps,
                lambda_manifold=point.lambda_manifold,
                a=math.sqrt(2),
                h_min=point.h_min_factor * point.sigma_x / math.sqrt(point.n),
                seed=model_seed,
                cg_tol=1e-6,
                solver=args.solver,
            )
            started = time.perf_counter()
            error = None
            try:
                model.fit(X, Y)
            except (RuntimeError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - started
            if error is None:
                selected_truth = truth[model.center_indices_, None, :]
                quality, whole_manifold_distance, _ = _local_subspace_metrics(
                    selected_truth, model.projectors_
                )
                residual = max(
                    float(entry["linear_relative_residual_max"])
                    for entry in model.trace_
                )
                completed = model.stop_reason_ == "h_min" and residual <= 1e-5
                stop_reason = model.stop_reason_
                iterations = sum(
                    int(entry["linear_iterations"]) for entry in model.trace_
                )
            else:
                quality = None
                whole_manifold_distance = None
                residual = None
                completed = False
                stop_reason = None
                iterations = None
            rows.append(
                {
                    "seed": seed,
                    "model_seed": model_seed,
                    "estimator": estimator,
                    "solver": args.solver,
                    "quality": quality,
                    "whole_manifold_max_local_projector_distance": (
                        whole_manifold_distance
                    ),
                    "whole_manifold_pass": (
                        whole_manifold_distance is not None
                        and whole_manifold_distance <= 0.2
                    ),
                    "completion_pass": completed,
                    "recovered": (
                        error is None
                        and quality is not None
                        and quality <= 0.2
                        and whole_manifold_distance is not None
                        and whole_manifold_distance <= 0.2
                    ),
                    "stop_reason": stop_reason,
                    "linear_residual_max": residual,
                    "linear_iterations": iterations,
                    "fit_seconds": elapsed,
                    "error": error,
                }
            )
            print(
                f"{seed} {estimator} quality={quality} "
                f"completed={completed} error={error}",
                flush=True,
            )
    if args.split == "selection" and args.solver == "cg":
        reference = json.loads(
            (
                Path("docs/experiments/manifold_recovery_2026-09-23")
                / "quad_moments_knn1_local_mass6_selection/runs.json"
            ).read_text(encoding="utf-8")
        )
        for row, expected in zip(
            (item for item in rows if item["estimator"] == "local_quadratic"),
            reference,
            strict=True,
        ):
            assert row["error"] == expected["error"]
            assert abs(float(row["quality"]) - expected["quality"]) < 1e-12
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        estimator: {
            "runs": sum(row["estimator"] == estimator for row in rows),
            "recovered": sum(
                row["estimator"] == estimator and bool(row["recovered"]) for row in rows
            ),
            "numerical_failures": sum(
                row["estimator"] == estimator and row["error"] is not None
                for row in rows
            ),
            "quality_median_successful": float(
                np.median(
                    [
                        row["quality"]
                        for row in rows
                        if row["estimator"] == estimator and row["quality"] is not None
                    ]
                )
            ),
            "fit_seconds_median": float(
                np.median(
                    [
                        row["fit_seconds"]
                        for row in rows
                        if row["estimator"] == estimator
                    ]
                )
            ),
        }
        for estimator in ("manifold", "local_quadratic")
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "seeds": list(seeds),
                "solver": args.solver,
                "version": _version(),
                "point": {"n": point.n, "d": point.d, "sigma_eps": point.sigma_eps},
                "quality": "full-center RMS principal sine",
                "threshold": 0.2,
                "whole_manifold_metric": (
                    "maximum local principal sine over all centers"
                ),
                "whole_manifold_threshold": 0.2,
                "recovery_rule": (
                    "both geometric checks pass; completion and inner residuals "
                    "are diagnostics only"
                ),
                "completion": "h_min and every certified inner residual <= 1e-5",
                "outer_convergence_certified": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
