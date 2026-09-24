"""Isolated fixed-seed time/RSS and traced-peak comparison for manifold recovery."""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import time
import tracemalloc
from pathlib import Path

import numpy as np

from ADP import ADP_Manifold
from experiments.data import _generate_data, _make_seed_bundle
from experiments.diagnostic import make_case
from experiments.runner import _local_subspace_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--estimator", choices=("manifold", "local_quadratic"), required=True
    )
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    case = make_case("manifold", "base")
    point = case.experiment.smoke
    assert (
        point.N_loc is not None
        and point.sync_steps is not None
        and point.lambda_manifold is not None
        and point.h_min_factor is not None
    )
    bundle = _make_seed_bundle(
        case.experiment.selector,
        point,
        0,
        common_random_fields=case.experiment.common_random_fields,
    )
    data = _generate_data(case.experiment.selector, point, bundle, 0)
    n_loc = point.N_loc
    sync_steps = point.sync_steps
    lambda_manifold = point.lambda_manifold
    h_min_factor = point.h_min_factor
    assert (
        n_loc is not None
        and sync_steps is not None
        and lambda_manifold is not None
        and h_min_factor is not None
    )

    def fit() -> float:
        model = ADP_Manifold(
            1,
            estimator=args.estimator,
            N_loc=n_loc,
            N_lin=point.N_lin,
            N_J=point.N_J,
            N_phi=point.N_phi,
            N_manifold=point.N_manifold,
            sync_steps=sync_steps,
            lambda_manifold=lambda_manifold,
            a=math.sqrt(2),
            h_min=h_min_factor * point.sigma_x / math.sqrt(point.n),
            seed=bundle.init,
            cg_tol=1e-6,
        ).fit(data.X, data.Y)
        truth = np.swapaxes(data.beta[model.center_indices_], 1, 2)
        return _local_subspace_metrics(truth, model.projectors_)[0]

    fit()  # BLAS/import warm-up excluded from fit times.
    seconds = []
    quality = 0.0
    for _ in range(args.repeat):
        started = time.perf_counter()
        quality = fit()
        seconds.append(time.perf_counter() - started)
    tracemalloc.start()
    fit()
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result = {
        "estimator": args.estimator,
        "seed": 0,
        "shape": {"n": point.n, "d": point.d, "J": point.N_J, "P": point.N_phi, "m": 1},
        "dtype": str(data.X.dtype),
        "repeat": args.repeat,
        "fit_seconds": seconds,
        "fit_seconds_median": float(np.median(seconds)),
        "tracemalloc_fit_peak_mib": traced_peak / 1024**2,
        "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "quality": quality,
        "threads": {
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "memory_method": "tracemalloc fit peak; Linux per-process ru_maxrss high-water",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
