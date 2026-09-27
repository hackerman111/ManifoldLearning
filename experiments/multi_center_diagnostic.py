"""C1: свойства случайных центров до выбора правила отбора."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy

from ADP.core.ADP_Config import ADP_Config
from ADP.engine.common.calculus import pairwise_distance2, search_bandwidth

from .data import _generate_data, _make_seed_bundle
from .diagnostic import _code_fingerprint
from .multiv2_quality import SENTINELS, _baseline
from .runner import _effective_config

SEEDS = range(4000, 4010)
CENTER_FIELDS: tuple[str, ...] = (
    "point",
    "seed",
    "center_position",
    "center_id",
    "mass_lin",
    "n_eff_lin",
    "support_lin",
    "rank_lin",
    "smin_over_smax_lin",
    "mass_outer",
    "n_eff_outer",
    "support_outer",
    "nearest_weight_cosine",
    "nearest_center_id",
)
SEED_FIELDS: tuple[str, ...] = (
    "point",
    "seed",
    "status",
    "error",
    "seed_bundle",
    "wall_sec",
    "peak_rss_mib",
    "n_centers",
    "h_lin",
    "h_outer",
    "n_eff_ratio",
    "rank_deficient_frac",
    "condition_ratio",
    "duplicate_spread",
    "informative",
)


def _weight_metrics(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mass = weights.sum(axis=1)
    if not np.all(np.isfinite(mass)) or np.any(mass <= 0):
        raise RuntimeError("pilot contains an empty or non-finite neighborhood")
    return (
        mass,
        np.square(mass) / np.square(weights).sum(axis=1),
        np.count_nonzero(weights, axis=1),
    )


def _local_spectrum(
    X: np.ndarray, weights: np.ndarray, mass: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Диагностика raw design: одна локальная (support,d) SVD за раз."""
    n, d = X.shape
    cutoff = np.finfo(float).eps * max(n + d, d + 1)
    rank = np.empty(len(weights), dtype=int)
    condition = np.empty(len(weights))
    for j, row in enumerate(weights):
        support = row > 0
        local_X = X[support]
        local_w = row[support]
        mean = (local_w @ local_X) / mass[j]
        singular = np.linalg.svd(
            (local_X - mean) * np.sqrt(local_w)[:, None], compute_uv=False
        )
        largest = float(singular[0]) if len(singular) else 0.0
        rank[j] = int(np.count_nonzero(singular > cutoff * largest)) if largest else 0
        condition[j] = (
            float(singular[d - 1] / largest) if largest and len(singular) >= d else 0.0
        )
    return rank, condition


def _ratio(values: np.ndarray) -> float:
    low, high = np.quantile(values, (0.1, 0.9))
    return float(high / low) if low > 0 else 0.0


def _diagnose(
    name: str, seed: int
) -> tuple[list[dict[str, object]], dict[str, object]]:
    point = replace(_baseline(name), center_displacement=0.0, training_set="all")
    seeds = _make_seed_bundle(f"multi-center-{name}", point, seed)
    generated = _generate_data(f"multi-center-{name}", point, seeds, seed)
    config = _effective_config(ADP_Config(), point, seeds.init)
    if config.center_displacement != 0 or config.training_set != "all":
        raise RuntimeError("C1 requires unsmoothed centers and all training rows")
    if config.N_J is None or config.N_lin is None or config.h_min is None:
        raise RuntimeError("C1 requires explicit center count, pilot mass and h_min")
    X = generated.X
    n, d = X.shape
    center_seed = np.random.SeedSequence(config.seed).spawn(3)[0]
    indices = np.random.default_rng(center_seed).choice(n, config.N_J, replace=False)
    distance2 = pairwise_distance2(X, X[indices])
    h_lin = search_bandwidth(
        distance2, config.N_lin, config.kernel, lower=np.finfo(float).eps
    )
    h_outer = search_bandwidth(
        distance2, config.N_loc, config.kernel, lower=config.h_min
    )
    weights_lin = config.kernel(distance2 / h_lin**2)
    weights_outer = config.kernel(distance2 / h_outer**2)
    mass_lin, n_eff_lin, support_lin = _weight_metrics(weights_lin)
    mass_outer, n_eff_outer, support_outer = _weight_metrics(weights_outer)
    rank, condition = _local_spectrum(X, weights_lin, mass_lin)

    normalized = weights_outer / np.linalg.norm(weights_outer, axis=1)[:, None]
    similarities = normalized @ normalized.T
    np.fill_diagonal(similarities, -np.inf)
    nearest_position = np.argmax(similarities, axis=1)
    nearest_cosine = similarities[np.arange(len(indices)), nearest_position]
    rows = [
        {
            "point": name,
            "seed": seed,
            "center_position": j,
            "center_id": int(center_id),
            "mass_lin": float(mass_lin[j]),
            "n_eff_lin": float(n_eff_lin[j]),
            "support_lin": int(support_lin[j]),
            "rank_lin": int(rank[j]),
            "smin_over_smax_lin": float(condition[j]),
            "mass_outer": float(mass_outer[j]),
            "n_eff_outer": float(n_eff_outer[j]),
            "support_outer": int(support_outer[j]),
            "nearest_weight_cosine": float(nearest_cosine[j]),
            "nearest_center_id": int(indices[nearest_position[j]]),
        }
        for j, center_id in enumerate(indices)
    ]
    n_eff_ratio = _ratio(n_eff_outer)
    rank_fraction = float(np.mean(rank < d))
    condition_ratio = _ratio(condition)
    duplicate_spread = float(
        np.quantile(nearest_cosine, 0.9) - np.quantile(nearest_cosine, 0.1)
    )
    informative = (
        n_eff_ratio >= 1.5
        or rank_fraction >= 0.05
        or condition_ratio >= 2.0
        or duplicate_spread >= 0.1
    )
    return rows, {
        "seed_bundle": json.dumps(asdict(seeds), sort_keys=True),
        "n_centers": len(indices),
        "n_eff_ratio": n_eff_ratio,
        "rank_deficient_frac": rank_fraction,
        "condition_ratio": condition_ratio,
        "duplicate_spread": duplicate_spread,
        "informative": informative,
        "h_lin": h_lin,
        "h_outer": h_outer,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_outputs/diagnostic/multi_center_c1_20260925"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print("C1: d10,n1000 x seeds 4000-4009; 20 pilots, 10000 centers")
        return 0
    root = args.output_dir
    if root.exists():
        parser.error(f"output directory already exists: {root}")
    root.mkdir(parents=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    manifest = {
        "status": "running",
        "revision": revision,
        "dirty": dirty,
        "code_fingerprint": _code_fingerprint(),
        "points": {
            name: asdict(replace(_baseline(name), center_displacement=0.0))
            for name in SENTINELS
        },
        "seeds": list(SEEDS),
        "kernel": "epanechnikov: max(1-(distance2/h**2)**2,0)",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total_started = perf_counter()
    with (
        (root / "centers.csv").open("w", newline="", encoding="utf-8") as centers_file,
        (root / "seeds.csv").open("w", newline="", encoding="utf-8") as seeds_file,
    ):
        center_writer = csv.DictWriter(centers_file, fieldnames=CENTER_FIELDS)
        seed_writer = csv.DictWriter(seeds_file, fieldnames=SEED_FIELDS)
        center_writer.writeheader()
        seed_writer.writeheader()
        for name in SENTINELS:
            for seed in SEEDS:
                started = perf_counter()
                row: dict[str, object] = {
                    "point": name,
                    "seed": seed,
                    "status": "success",
                }
                try:
                    centers, summary = _diagnose(name, seed)
                    center_writer.writerows(centers)
                    row.update(
                        {key: summary[key] for key in SEED_FIELDS if key in summary}
                    )
                except Exception as error:  # Каждый неуспешный seed остаётся в таблице.
                    row.update(
                        status="numerical_failure",
                        error=f"{type(error).__name__}: {error}",
                    )
                row["wall_sec"] = perf_counter() - started
                row["peak_rss_mib"] = (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                )
                seed_writer.writerow(row)
                centers_file.flush()
                seeds_file.flush()
                print(
                    f"{name} seed={seed} {row['status']} {row['wall_sec']:.1f}s",
                    flush=True,
                )
    manifest["status"] = "complete"
    manifest["total_wall_sec"] = perf_counter() - total_started
    manifest["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
