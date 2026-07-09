from __future__ import annotations

# Must be set before importing numpy/scipy-like numerical libraries.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class H0ResearchConfig:
    d: int = 100
    n_multiplier: int = 10
    n_seeds: int = 300
    base_seed: int = 20260709

    n_min_values: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0)
    sigma_x_values: tuple[float, ...] = (0.7, 1.0, 1.3)
    corr_values: tuple[float, ...] = (0.0, 0.3, 0.7)
    center_fractions: tuple[float, ...] = (0.25, 1.0)

    kernel: str = "epanechnikov"

    # "mean": condition uses average local mass.
    # "q05": condition uses 5% quantile of local masses.
    mass_rule: str = "mean"
    q_mass: float = 0.05

    search_steps: int = 60
    grid_points: int = 161
    h_grid_left: float = 0.25
    h_grid_right: float = 4.0

    local_tolerance_rel: float = 0.05
    bootstrap_reps: int = 1000

    save_example_curves: bool = True
    example_curve_seeds: tuple[int, ...] = (0, 1, 2)

    @property
    def n(self) -> int:
        return self.n_multiplier * self.d


# ---------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------


def ar1_cholesky(d: int, corr: float) -> np.ndarray:
    if abs(corr) < 1e-15:
        return np.eye(d)

    idx = np.arange(d)
    cov = corr ** np.abs(idx[:, None] - idx[None, :])
    cov += 1e-12 * np.eye(d)
    return np.linalg.cholesky(cov)


def generate_x_base(n: int, d: int, corr: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, d))
    chol = ar1_cholesky(d, corr)
    return z @ chol.T


def choose_centers(X: np.ndarray, center_fraction: float, seed: int) -> np.ndarray:
    n = X.shape[0]
    n_centers = max(1, min(n, int(round(center_fraction * n))))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_centers, replace=False)
    return X[idx]


# ---------------------------------------------------------------------
# Distances and mass
# ---------------------------------------------------------------------


def pairwise_sq_dists(centers: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Returns D2[j, i] = ||X_i - center_j||^2.

    Memory: O(J n), not O(J n d).
    """
    x_norm = np.sum(X * X, axis=1)
    c_norm = np.sum(centers * centers, axis=1)

    d2 = c_norm[:, None] + x_norm[None, :] - 2.0 * (centers @ X.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


@dataclass
class EpanechnikovMassCache:
    sorted_d2: np.ndarray
    cumsum_d2: np.ndarray

    @classmethod
    def from_distances(cls, d2: np.ndarray) -> "EpanechnikovMassCache":
        sorted_d2 = np.sort(d2, axis=1)
        cumsum_d2 = np.cumsum(sorted_d2, axis=1)
        return cls(sorted_d2=sorted_d2, cumsum_d2=cumsum_d2)

    def mass_per_center(self, h: float) -> np.ndarray:
        """
        For K(q) = max(1 - q, 0):

            sum_i K(D2_ji / h^2)
            =
            count(D2_ji < h^2) - sum(D2_ji < h^2) / h^2.

        This avoids recomputing all kernel weights for every h.
        """
        if not np.isfinite(h) or h <= 0:
            return np.zeros(self.sorted_d2.shape[0], dtype=float)

        t = h * h
        J = self.sorted_d2.shape[0]

        counts = np.empty(J, dtype=np.int64)
        sums = np.empty(J, dtype=float)

        for j in range(J):
            count = np.searchsorted(self.sorted_d2[j], t, side="left")
            counts[j] = count
            sums[j] = self.cumsum_d2[j, count - 1] if count > 0 else 0.0

        mass = counts.astype(float) - sums / t
        np.maximum(mass, 0.0, out=mass)
        return mass


def mass_statistic(
    mass_per_center: np.ndarray,
    *,
    rule: str,
    q_mass: float,
) -> float:
    if rule == "mean":
        return float(np.mean(mass_per_center))
    if rule == "q05":
        return float(np.quantile(mass_per_center, q_mass))
    raise ValueError(f"Unknown mass_rule: {rule}")


# ---------------------------------------------------------------------
# h0 selection
# ---------------------------------------------------------------------


@dataclass
class H0Selection:
    h0: float
    h_low: float
    h_high: float
    mass_h0: float
    mass_low: float
    mass_high: float
    expand_steps_used: int
    search_steps_used: int
    failed: bool
    error: str


def select_h0(
    cache: EpanechnikovMassCache,
    n_min: float,
    *,
    mass_rule: str,
    q_mass: float,
    search_steps: int,
) -> H0Selection:
    """
    Selects the minimal h0 such that mass_stat(h0) >= n_min.
    """
    try:
        max_d2 = float(np.max(cache.sorted_d2))
        if not np.isfinite(max_d2) or max_d2 <= 0:
            raise ValueError("Pairwise distances are degenerate.")

        h_low = 0.0
        h_high = math.sqrt(max_d2) * 1e-3
        expand_steps = 0

        def stat(h: float) -> float:
            return mass_statistic(
                cache.mass_per_center(h),
                rule=mass_rule,
                q_mass=q_mass,
            )

        while stat(h_high) < n_min:
            h_high *= 2.0
            expand_steps += 1
            if expand_steps > 80:
                raise RuntimeError("Could not bracket h0.")

        for _ in range(search_steps):
            h_mid = 0.5 * (h_low + h_high)
            if stat(h_mid) >= n_min:
                h_high = h_mid
            else:
                h_low = h_mid

        h0 = h_high

        return H0Selection(
            h0=h0,
            h_low=h_low,
            h_high=h_high,
            mass_h0=stat(h0),
            mass_low=stat(h_low),
            mass_high=stat(h_high),
            expand_steps_used=expand_steps,
            search_steps_used=search_steps,
            failed=False,
            error="",
        )

    except Exception as exc:
        return H0Selection(
            h0=np.nan,
            h_low=np.nan,
            h_high=np.nan,
            mass_h0=np.nan,
            mass_low=np.nan,
            mass_high=np.nan,
            expand_steps_used=0,
            search_steps_used=0,
            failed=True,
            error=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------


def h0_local_checks(
    cache: EpanechnikovMassCache,
    h0: float,
    n_min: float,
    *,
    mass_rule: str,
    q_mass: float,
    tol_rel: float,
) -> dict[str, float | bool]:
    tol = tol_rel * n_min

    def stat(h: float) -> float:
        return mass_statistic(
            cache.mass_per_center(h),
            rule=mass_rule,
            q_mass=q_mass,
        )

    m_099 = stat(0.99 * h0)
    m_100 = stat(h0)
    m_101 = stat(1.01 * h0)

    center_mass = cache.mass_per_center(h0)

    return {
        "mass_099_h0": m_099,
        "mass_h0_recheck": m_100,
        "mass_101_h0": m_101,
        "center_mass_min_h0": float(np.min(center_mass)),
        "center_mass_q05_h0": float(np.quantile(center_mass, 0.05)),
        "center_mass_mean_h0": float(np.mean(center_mass)),
        "center_mass_median_h0": float(np.median(center_mass)),
        "center_mass_max_h0": float(np.max(center_mass)),
        "pass_mass_at_h0": bool(m_100 + 1e-10 >= n_min),
        "pass_mass_below_099": bool(m_099 <= n_min + tol),
        "pass_mass_above_101": bool(m_101 + 1e-10 >= n_min),
    }


def h0_grid_curve(
    cache: EpanechnikovMassCache,
    h0: float,
    *,
    grid_points: int,
    left: float,
    right: float,
    mass_rule: str,
    q_mass: float,
) -> pd.DataFrame:
    multipliers = np.geomspace(left, right, grid_points)

    rows = []
    prev_mass = -np.inf
    monotone_ok = True

    for mult in multipliers:
        h = mult * h0
        masses = cache.mass_per_center(h)
        stat = mass_statistic(masses, rule=mass_rule, q_mass=q_mass)

        if stat + 1e-10 < prev_mass:
            monotone_ok = False

        prev_mass = stat

        rows.append(
            {
                "h_multiplier": float(mult),
                "h": float(h),
                "mass_stat": float(stat),
                "mass_mean": float(np.mean(masses)),
                "mass_q05": float(np.quantile(masses, 0.05)),
                "mass_min": float(np.min(masses)),
                "monotone_ok_so_far": bool(monotone_ok),
            }
        )

    return pd.DataFrame(rows)


def bootstrap_ci_median(
    values: np.ndarray,
    *,
    reps: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    medians = np.empty(reps, dtype=float)

    for b in range(reps):
        sample = rng.choice(values, size=values.size, replace=True)
        medians[b] = np.median(sample)

    low, high = np.quantile(medians, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def failed_check_fields() -> dict[str, float | bool]:
    return {
        "mass_099_h0": np.nan,
        "mass_h0_recheck": np.nan,
        "mass_101_h0": np.nan,
        "center_mass_min_h0": np.nan,
        "center_mass_q05_h0": np.nan,
        "center_mass_mean_h0": np.nan,
        "center_mass_median_h0": np.nan,
        "center_mass_max_h0": np.nan,
        "pass_mass_at_h0": False,
        "pass_mass_below_099": False,
        "pass_mass_above_101": False,
        "pass_grid_monotone": False,
    }


# ---------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------


def run_h0_distance_job(
    job: tuple[int, float, float],
    config: H0ResearchConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """
    One independent parallel job.

    Unit of work:
        seed_id, corr, center_fraction

    Inside one job:
        - generate X once;
        - choose centers once;
        - compute base distances once;
        - run all sigma_x and n_min values.
    """
    seed_id, corr, center_fraction = job

    n = config.n
    d = config.d

    records: list[dict[str, object]] = []
    curve_records: list[dict[str, object]] = []

    data_seed = config.base_seed + 1_000_003 * seed_id + int(10_000 * corr)
    center_seed = data_seed + 17

    X_base = generate_x_base(n=n, d=d, corr=corr, seed=data_seed)
    centers_base = choose_centers(X_base, center_fraction, seed=center_seed)

    d2_base = pairwise_sq_dists(centers_base, X_base)

    for sigma_x in config.sigma_x_values:
        d2 = (sigma_x * sigma_x) * d2_base
        cache = EpanechnikovMassCache.from_distances(d2)

        for n_min in config.n_min_values:
            selected = select_h0(
                cache,
                n_min=n_min,
                mass_rule=config.mass_rule,
                q_mass=config.q_mass,
                search_steps=config.search_steps,
            )

            row: dict[str, object] = {
                "seed_id": seed_id,
                "data_seed": data_seed,
                "center_seed": center_seed,
                "d": d,
                "n": n,
                "n_over_d": n / d,
                "corr": corr,
                "sigma_x": sigma_x,
                "center_fraction": center_fraction,
                "n_centers": centers_base.shape[0],
                "n_min": n_min,
                "kernel": config.kernel,
                "mass_rule": config.mass_rule,
                "q_mass": config.q_mass,
                "h0": selected.h0,
                "h0_over_sigma_x": selected.h0 / sigma_x
                if np.isfinite(selected.h0)
                else np.nan,
                "h_low": selected.h_low,
                "h_high": selected.h_high,
                "mass_h0": selected.mass_h0,
                "mass_low": selected.mass_low,
                "mass_high": selected.mass_high,
                "expand_steps_used": selected.expand_steps_used,
                "search_steps_used": selected.search_steps_used,
                "failed": selected.failed,
                "error": selected.error,
            }

            if not selected.failed:
                row.update(
                    h0_local_checks(
                        cache,
                        selected.h0,
                        n_min,
                        mass_rule=config.mass_rule,
                        q_mass=config.q_mass,
                        tol_rel=config.local_tolerance_rel,
                    )
                )

                if (
                    config.save_example_curves
                    and seed_id in set(config.example_curve_seeds)
                ):
                    curve = h0_grid_curve(
                        cache,
                        selected.h0,
                        grid_points=config.grid_points,
                        left=config.h_grid_left,
                        right=config.h_grid_right,
                        mass_rule=config.mass_rule,
                        q_mass=config.q_mass,
                    )

                    row["pass_grid_monotone"] = bool(
                        curve["monotone_ok_so_far"].iloc[-1]
                    )

                    for _, c_row in curve.iterrows():
                        curve_records.append(
                            {
                                "seed_id": seed_id,
                                "corr": corr,
                                "sigma_x": sigma_x,
                                "center_fraction": center_fraction,
                                "n_centers": centers_base.shape[0],
                                "n_min": n_min,
                                **c_row.to_dict(),
                            }
                        )
                else:
                    row["pass_grid_monotone"] = True

            else:
                row.update(failed_check_fields())

            row["pass_h0_core"] = bool(
                (not row["failed"])
                and row["pass_mass_at_h0"]
                and row["pass_mass_below_099"]
                and row["pass_mass_above_101"]
                and row["pass_grid_monotone"]
            )

            records.append(row)

    return records, curve_records


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------


def run_h0_research(
    config: H0ResearchConfig,
    output_dir: Path,
    *,
    n_jobs: int | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    start_time = time.perf_counter()

    jobs = [
        (seed_id, corr, center_fraction)
        for seed_id in range(config.n_seeds)
        for corr in config.corr_values
        for center_fraction in config.center_fractions
    ]

    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)

    if n_jobs < 1:
        raise ValueError("--jobs must be >= 1")

    records: list[dict[str, object]] = []
    curve_records: list[dict[str, object]] = []

    if n_jobs == 1:
        iterator = tqdm(
            jobs,
            total=len(jobs),
            desc="h0 research sequential",
            dynamic_ncols=True,
        )

        for job in iterator:
            job_records, job_curves = run_h0_distance_job(job, config)
            records.extend(job_records)
            curve_records.extend(job_curves)

            seed_id, corr, center_fraction = job
            iterator.set_postfix(
                seed=seed_id,
                corr=corr,
                centers=center_fraction,
                refresh=True,
            )

    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(run_h0_distance_job, job, config): job
                for job in jobs
            }

            iterator = tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"h0 research parallel, workers={n_jobs}",
                dynamic_ncols=True,
            )

            for future in iterator:
                job = futures[future]
                job_records, job_curves = future.result()
                records.extend(job_records)
                curve_records.extend(job_curves)

                seed_id, corr, center_fraction = job
                iterator.set_postfix(
                    seed=seed_id,
                    corr=corr,
                    centers=center_fraction,
                    refresh=True,
                )

    records_df = pd.DataFrame(records)
    curves_df = pd.DataFrame(curve_records)

    records_path = output_dir / "h0_records.csv"
    curves_path = output_dir / "h0_example_curves.csv"
    summary_path = output_dir / "h0_summary.csv"
    extra_tests_path = output_dir / "h0_extra_tests.json"
    manifest_path = output_dir / "h0_manifest.json"

    records_df.to_csv(records_path, index=False)
    curves_df.to_csv(curves_path, index=False)

    summary = summarize_records(records_df, config)
    summary.to_csv(summary_path, index=False)

    extra_tests = run_extra_tests(records_df, config)
    extra_tests_path.write_text(json.dumps(extra_tests, ensure_ascii=False, indent=2))

    plot_paths = save_plots(records_df, curves_df, summary, plots_dir)

    elapsed = time.perf_counter() - start_time

    manifest = {
        "experiment": "ADP h0 isotropic bandwidth research",
        "d": config.d,
        "n": config.n,
        "n_over_d": config.n / config.d,
        "n_seeds": config.n_seeds,
        "n_jobs": n_jobs,
        "n_min_values": list(config.n_min_values),
        "sigma_x_values": list(config.sigma_x_values),
        "corr_values": list(config.corr_values),
        "center_fractions": list(config.center_fractions),
        "kernel": config.kernel,
        "mass_rule": config.mass_rule,
        "q_mass": config.q_mass,
        "total_h0_checks": int(len(records_df)),
        "total_parallel_jobs": int(len(jobs)),
        "elapsed_sec": elapsed,
        "outputs": {
            "records": str(records_path),
            "curves": str(curves_path),
            "summary": str(summary_path),
            "extra_tests": str(extra_tests_path),
            "plots": {name: str(path) for name, path in plot_paths.items()},
        },
    }

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    saved = {
        "records": records_path,
        "curves": curves_path,
        "summary": summary_path,
        "extra_tests": extra_tests_path,
        "manifest": manifest_path,
    }
    saved.update(plot_paths)

    return saved


# ---------------------------------------------------------------------
# Summary and statistical checks
# ---------------------------------------------------------------------


def summarize_records(records: pd.DataFrame, config: H0ResearchConfig) -> pd.DataFrame:
    group_cols = [
        "d",
        "n",
        "n_min",
        "sigma_x",
        "corr",
        "center_fraction",
        "n_centers",
    ]

    rows: list[dict[str, object]] = []

    for keys, group in records.groupby(group_cols, dropna=False):
        key_dict = dict(zip(group_cols, keys))

        h0 = group["h0"].to_numpy(dtype=float)
        h0_scaled = group["h0_over_sigma_x"].to_numpy(dtype=float)

        h0 = h0[np.isfinite(h0)]
        h0_scaled = h0_scaled[np.isfinite(h0_scaled)]

        if h0.size > 1:
            h0_mean = float(np.mean(h0))
            h0_std = float(np.std(h0, ddof=1))
            h0_mean_ci_low = h0_mean - 1.96 * h0_std / math.sqrt(h0.size)
            h0_mean_ci_high = h0_mean + 1.96 * h0_std / math.sqrt(h0.size)
        elif h0.size == 1:
            h0_mean = float(h0[0])
            h0_std = 0.0
            h0_mean_ci_low = h0_mean
            h0_mean_ci_high = h0_mean
        else:
            h0_mean = np.nan
            h0_std = np.nan
            h0_mean_ci_low = np.nan
            h0_mean_ci_high = np.nan

        if h0.size > 0:
            h0_median = float(np.median(h0))
            h0_median_ci_low, h0_median_ci_high = bootstrap_ci_median(
                h0,
                reps=config.bootstrap_reps,
                seed=config.base_seed + 991,
            )
            h0_q25 = float(np.quantile(h0, 0.25))
            h0_q75 = float(np.quantile(h0, 0.75))
        else:
            h0_median = np.nan
            h0_median_ci_low = np.nan
            h0_median_ci_high = np.nan
            h0_q25 = np.nan
            h0_q75 = np.nan

        row = {
            **key_dict,
            "count": int(len(group)),
            "failed_rate": float(np.mean(group["failed"].astype(bool))),
            "pass_h0_core_rate": float(np.mean(group["pass_h0_core"].astype(bool))),
            "pass_mass_at_h0_rate": float(
                np.mean(group["pass_mass_at_h0"].astype(bool))
            ),
            "pass_mass_below_099_rate": float(
                np.mean(group["pass_mass_below_099"].astype(bool))
            ),
            "pass_mass_above_101_rate": float(
                np.mean(group["pass_mass_above_101"].astype(bool))
            ),
            "pass_grid_monotone_rate": float(
                np.mean(group["pass_grid_monotone"].astype(bool))
            ),
            "h0_mean": h0_mean,
            "h0_std": h0_std,
            "h0_mean_ci95_low": h0_mean_ci_low,
            "h0_mean_ci95_high": h0_mean_ci_high,
            "h0_median": h0_median,
            "h0_median_ci95_low": h0_median_ci_low,
            "h0_median_ci95_high": h0_median_ci_high,
            "h0_q25": h0_q25,
            "h0_q75": h0_q75,
            "h0_over_sigma_x_median": float(np.median(h0_scaled))
            if h0_scaled.size
            else np.nan,
            "mass_h0_median": float(np.nanmedian(group["mass_h0"])),
            "mass_099_h0_median": float(np.nanmedian(group["mass_099_h0"])),
            "mass_101_h0_median": float(np.nanmedian(group["mass_101_h0"])),
            "center_mass_q05_h0_median": float(
                np.nanmedian(group["center_mass_q05_h0"])
            ),
            "center_mass_min_h0_median": float(
                np.nanmedian(group["center_mass_min_h0"])
            ),
        }

        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def run_extra_tests(records: pd.DataFrame, config: H0ResearchConfig) -> dict[str, object]:
    tests: dict[str, object] = {}

    # Test 1: h0 should be nondecreasing in n_min for fixed X, centers, sigma_x, corr.
    monotone_rows = []
    group_cols = ["seed_id", "corr", "sigma_x", "center_fraction", "n_centers"]

    for keys, group in records.groupby(group_cols):
        g = group.sort_values("n_min")
        h0_values = g["h0"].to_numpy(dtype=float)
        n_min_values = g["n_min"].to_numpy(dtype=float)

        ok = bool(np.all(np.diff(h0_values) >= -1e-10))

        monotone_rows.append(
            {
                **dict(zip(group_cols, keys)),
                "n_min_values": n_min_values.tolist(),
                "h0_values": h0_values.tolist(),
                "pass_monotone_n_min": ok,
            }
        )

    monotone_df = pd.DataFrame(monotone_rows)

    tests["monotone_n_min"] = {
        "description": "For fixed X, centers, sigma_x and corr, selected h0 should be nondecreasing in n_min.",
        "count": int(len(monotone_df)),
        "pass_rate": float(monotone_df["pass_monotone_n_min"].mean())
        if len(monotone_df)
        else np.nan,
    }

    # Test 2: h0 / sigma_x should be stable across sigma_x for the same base cloud.
    scale_rows = []
    group_cols = ["seed_id", "corr", "center_fraction", "n_centers", "n_min"]

    for keys, group in records.groupby(group_cols):
        vals = group["h0_over_sigma_x"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]

        if vals.size <= 1:
            rel_spread = np.nan
            ok = False
        else:
            med = np.median(vals)
            rel_spread = float(np.max(np.abs(vals - med)) / max(abs(med), 1e-12))
            ok = bool(rel_spread <= 0.02)

        scale_rows.append(
            {
                **dict(zip(group_cols, keys)),
                "relative_spread_h0_over_sigma_x": rel_spread,
                "pass_scale_invariance": ok,
            }
        )

    scale_df = pd.DataFrame(scale_rows)

    tests["scale_invariance"] = {
        "description": "For the same base cloud, h0 / sigma_x should be stable across sigma_x.",
        "count": int(len(scale_df)),
        "pass_rate": float(scale_df["pass_scale_invariance"].mean())
        if len(scale_df)
        else np.nan,
        "median_relative_spread": float(
            np.nanmedian(scale_df["relative_spread_h0_over_sigma_x"])
        )
        if len(scale_df)
        else np.nan,
    }

    tests["scientific_replicates"] = {
        "d": config.d,
        "n": config.n,
        "n_over_d": config.n / config.d,
        "n_seeds": config.n_seeds,
        "total_h0_checks": int(len(records)),
        "recommended_minimum_pass_rate_core": 0.99,
        "recommended_minimum_pass_rate_monotone_n_min": 0.99,
        "recommended_minimum_pass_rate_scale_invariance": 0.99,
    }

    return tests


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------


def subset_if_possible(df: pd.DataFrame, column: str, value: float) -> pd.DataFrame:
    if column not in df.columns:
        return df
    mask = df[column] == value
    if mask.any():
        return df[mask]
    return df


def save_plots(
    records: pd.DataFrame,
    curves: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    saved: dict[str, Path] = {}

    # Plot 1: M_iso(h) versus h / h0.
    if not curves.empty:
        example = curves.copy()
        example = subset_if_possible(example, "seed_id", float(example["seed_id"].min()))
        example = subset_if_possible(example, "corr", 0.3)
        example = subset_if_possible(example, "sigma_x", 1.0)

        if "center_fraction" in example.columns:
            example = example[example["center_fraction"] == example["center_fraction"].max()]

        fig, ax = plt.subplots(figsize=(9.0, 5.2))

        for n_min, group in example.groupby("n_min"):
            group = group.sort_values("h_multiplier")
            ax.plot(
                group["h_multiplier"],
                group["mass_stat"],
                linewidth=2.0,
                label=f"n_min={n_min:g}",
            )
            ax.axhline(n_min, linestyle="--", linewidth=1.0)

        ax.set_xscale("log")
        ax.set_xlabel(r"$h / h_0$")
        ax.set_ylabel(r"$M_{\mathrm{iso}}(h)$")
        ax.set_title(r"Проверка выбора $h_0$: локальная масса против масштаба")
        ax.grid(True, alpha=0.25)
        ax.legend()

        path = output_dir / "h0_mass_curves_examples.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved["mass_curves_examples"] = path

    # Plot 2: h0 / sigma_x by n_min.
    plot_df = records.copy()
    plot_df = subset_if_possible(plot_df, "corr", 0.3)
    plot_df = subset_if_possible(plot_df, "sigma_x", 1.0)

    if "center_fraction" in plot_df.columns:
        plot_df = plot_df[plot_df["center_fraction"] == plot_df["center_fraction"].max()]

    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    data = []
    labels = []
    for n_min, group in plot_df.groupby("n_min"):
        vals = group["h0_over_sigma_x"].dropna().to_numpy()
        if vals.size:
            data.append(vals)
            labels.append(f"{n_min:g}")

    if data:
        ax.boxplot(data, labels=labels, showfliers=False)

    ax.set_xlabel(r"$n_{\min}$")
    ax.set_ylabel(r"$h_0 / \sigma_X$")
    ax.set_title(r"Распределение выбранного $h_0$ при $d=100,\ n=1000$")
    ax.grid(True, alpha=0.25)

    path = output_dir / "h0_boxplot_by_n_min.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved["boxplot_by_n_min"] = path

    # Plot 3: pass rate by n_min.
    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    pass_by_nmin = (
        records.groupby("n_min")
        .agg(pass_rate=("pass_h0_core", "mean"))
        .reset_index()
        .sort_values("n_min")
    )

    ax.bar(pass_by_nmin["n_min"].astype(str), pass_by_nmin["pass_rate"])
    ax.axhline(0.99, linestyle="--", linewidth=1.2, label="0.99 target")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"$n_{\min}$")
    ax.set_ylabel("pass rate")
    ax.set_title(r"Доля успешных проверок $h_0$")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    path = output_dir / "h0_pass_rate_by_n_min.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved["pass_rate_by_n_min"] = path

    # Plot 4: median h0 by corr and n_min.
    plot_summary = summary.copy()
    plot_summary = subset_if_possible(plot_summary, "sigma_x", 1.0)

    if "center_fraction" in plot_summary.columns:
        plot_summary = plot_summary[
            plot_summary["center_fraction"] == plot_summary["center_fraction"].max()
        ]

    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    for corr, group in plot_summary.groupby("corr"):
        group = group.sort_values("n_min")
        ax.plot(
            group["n_min"],
            group["h0_median"],
            marker="o",
            linewidth=2.0,
            label=f"corr={corr:g}",
        )

    ax.set_xlabel(r"$n_{\min}$")
    ax.set_ylabel(r"median $h_0$")
    ax.set_title(r"Зависимость $h_0$ от $n_{\min}$ и корреляции признаков")
    ax.grid(True, alpha=0.25)
    ax.legend()

    path = output_dir / "h0_median_by_corr.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved["median_by_corr"] = path

    return saved


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research pipeline for ADP isotropic bandwidth h0 at d=100, n=10d."
    )

    parser.add_argument("--out", type=Path, default=Path("outputs/adp_h0_d100"))

    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--n-multiplier", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=300)
    parser.add_argument("--base-seed", type=int, default=20260709)

    parser.add_argument("--n-min", type=str, default="5,10,20,40")
    parser.add_argument("--sigma-x", type=str, default="0.7,1.0,1.3")
    parser.add_argument("--corr", type=str, default="0.0,0.3,0.7")
    parser.add_argument("--center-fractions", type=str, default="0.25,1.0")

    parser.add_argument("--mass-rule", choices=("mean", "q05"), default="mean")
    parser.add_argument("--q-mass", type=float, default=0.05)

    parser.add_argument("--search-steps", type=int, default=60)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)

    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Number of worker processes. Default: cpu_count - 1.",
    )

    parser.add_argument(
        "--no-example-curves",
        action="store_true",
        help="Do not save detailed M_iso(h) example curves.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = H0ResearchConfig(
        d=args.d,
        n_multiplier=args.n_multiplier,
        n_seeds=args.seeds,
        base_seed=args.base_seed,
        n_min_values=parse_float_tuple(args.n_min),
        sigma_x_values=parse_float_tuple(args.sigma_x),
        corr_values=parse_float_tuple(args.corr),
        center_fractions=parse_float_tuple(args.center_fractions),
        mass_rule=args.mass_rule,
        q_mass=args.q_mass,
        search_steps=args.search_steps,
        bootstrap_reps=args.bootstrap_reps,
        save_example_curves=not args.no_example_curves,
    )

    print("ADP h0 research")
    print(f"d = {config.d}")
    print(f"n = {config.n}")
    print(f"n/d = {config.n / config.d:g}")
    print(f"seeds = {config.n_seeds}")
    print(f"jobs = {args.jobs if args.jobs is not None else 'cpu_count - 1'}")
    print(f"n_min = {config.n_min_values}")
    print(f"sigma_x = {config.sigma_x_values}")
    print(f"corr = {config.corr_values}")
    print(f"center_fractions = {config.center_fractions}")
    print(f"mass_rule = {config.mass_rule}")
    print(f"output = {args.out}")

    saved = run_h0_research(config, args.out, n_jobs=args.jobs)

    print("\nSaved files:")
    for name, path in saved.items():
        print(f"{name:28s} {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
