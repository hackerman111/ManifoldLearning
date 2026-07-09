from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import time
import tracemalloc
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

from adp.common.plotting import (ADP_COLORS, apply_adp_axis_style,
                                 configure_adp_matplotlib, save_figure,
                                 set_adp_figure_size)
from adp.core import ADP, ADPConfig, ADPResult
from adp.evaluation.metrics import direction_metrics

Q_DEFINITION = (
    "fraction of active coordinates in beta* used by this stress runner; "
    "algorithm-internal q is the kernel quadratic-form argument"
)
LOCALIZING_TENSOR_FORM = "T_k^2 = h_k^-2 * (rho_k^2 I + beta_{k-1} beta_{k-1}^T)"
STEP0_DIRECTION_DISTRIBUTION = "normalized N(0, I_d), uniform on the sphere"
STEPK_DIRECTION_DISTRIBUTION = "normalized rho_k * N(0, I_d) + eta * beta_{k-1}"
LATEX_PREAMBLE = "\n".join(
    [
        r"\usepackage[T2A,T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[english,russian]{babel}",
        r"\usepackage{amsmath}",
    ]
)


def linear_plus_sin_link(z: np.ndarray) -> np.ndarray:
    return z + np.sin(z)


@dataclass(frozen=True, slots=True)
class StressProfile:
    name: str
    scale_label: str
    n_values: tuple[int, ...]
    d_values: tuple[int, ...]
    seeds: tuple[int, ...]
    links: tuple[str, ...]
    noise_levels: tuple[float, ...]
    sigma_x_values: tuple[float, ...]
    corr_values: tuple[float, ...]
    q_values: tuple[float, ...]
    n_directions: int
    min_neighbors: float
    outer_steps: int
    inner_steps: int
    n_centers: int | None = None
    center_fraction: float = 0.25
    max_centers: int | None = None
    lambda_penalty: float | None = None
    bandwidth_decay: float = math.sqrt(2.0)
    anisotropy_min: float | None = None
    kernel: str = "epanechnikov"
    center_noise_scale: float = 0.1
    renew_directions: bool = True
    chunk_size: int = 32
    tol: float = 1e-6
    ridge: float = 1e-10
    dtype: str = "float64"
    local_mass_quantile: float = 0.05
    scale_expand_steps: int = 12
    scale_search_steps: int = 12
    anisotropy_search_steps: int = 12
    objective_check_every: int = 2
    use_neighbor_index: bool = True
    min_cosine_abs: float = 0.70

    @property
    def max_n(self) -> int:
        return max(self.n_values)

    @property
    def max_d(self) -> int:
        return max(self.d_values)

    def centers_for_n(self, n: int) -> int:
        if self.n_centers is not None:
            return min(max(1, int(self.n_centers)), n)
        centers = max(1, int(round(self.center_fraction * n)))
        if self.max_centers is not None:
            centers = min(centers, self.max_centers)
        return min(centers, n)


@dataclass(frozen=True, slots=True)
class StressCase:
    profile: str
    scale_label: str
    ordinal: int
    seed: int
    data_seed: int
    fit_seed: int
    beta_seed: int
    n: int
    d: int
    link: str
    sigma_eps: float
    sigma_x: float
    corr: float
    q: float
    n_centers: int
    n_directions: int
    min_neighbors: float
    lambda_penalty: float | None
    outer_steps: int
    inner_steps: int
    bandwidth_decay: float
    anisotropy_min: float | None
    kernel: str
    center_noise_scale: float
    renew_directions: bool
    chunk_size: int
    tol: float
    ridge: float
    dtype: str
    local_mass_quantile: float
    scale_expand_steps: int
    scale_search_steps: int
    anisotropy_search_steps: int
    objective_check_every: int
    use_neighbor_index: bool
    min_cosine_abs: float

    @property
    def beta_support_size(self) -> int:
        return max(1, min(self.d, int(math.ceil(self.q * self.d))))

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "scale_label": self.scale_label,
            "case_ordinal": self.ordinal,
            "seed": self.seed,
            "data_seed": self.data_seed,
            "fit_seed": self.fit_seed,
            "beta_seed": self.beta_seed,
            "n": self.n,
            "d": self.d,
            "sigma_eps": self.sigma_eps,
            "sigma_x": self.sigma_x,
            "corr": self.corr,
            "q": self.q,
            "q_definition": Q_DEFINITION,
            "beta_support_size": self.beta_support_size,
            "link": self.link,
            "kernel": self.kernel,
            "n_centers": self.n_centers,
            "theta_centers": self.n_centers / self.n,
            "centers_source": "sample_without_replacement_plus_gaussian_noise",
            "center_noise_scale": self.center_noise_scale,
            "n_directions": self.n_directions,
            "directions_distribution_step0": STEP0_DIRECTION_DISTRIBUTION,
            "directions_distribution_stepk": STEPK_DIRECTION_DISTRIBUTION,
            "min_neighbors": self.min_neighbors,
            "lambda_penalty": self.lambda_penalty if self.lambda_penalty is not None else self.min_neighbors,
            "outer_steps": self.outer_steps,
            "inner_steps": self.inner_steps,
            "bandwidth_decay": self.bandwidth_decay,
            "anisotropy_min": self.anisotropy_min,
            "renew_directions": self.renew_directions,
            "chunk_size": self.chunk_size,
            "tol": self.tol,
            "ridge": self.ridge,
            "dtype": self.dtype,
            "local_mass_quantile": self.local_mass_quantile,
            "scale_expand_steps": self.scale_expand_steps,
            "scale_search_steps": self.scale_search_steps,
            "anisotropy_search_steps": self.anisotropy_search_steps,
            "objective_check_every": self.objective_check_every,
            "use_neighbor_index": self.use_neighbor_index,
            "algorithm_variant": "single-index-random-projection",
            "localizing_tensor_form": LOCALIZING_TENSOR_FORM,
            "target_dim": 1,
            "min_cosine_abs": self.min_cosine_abs,
        }

    def adp_config(self, random_state: int, show_progress: bool) -> ADPConfig:
        return ADPConfig(
            n_centers=self.n_centers,
            n_directions=self.n_directions,
            target_dim=1,
            min_neighbors=self.min_neighbors,
            lambda_penalty=self.lambda_penalty,
            outer_steps=self.outer_steps,
            inner_steps=self.inner_steps,
            tol=self.tol,
            bandwidth_decay=self.bandwidth_decay,
            anisotropy_min=self.anisotropy_min,
            kernel=self.kernel,  # type: ignore[arg-type]
            center_noise_scale=self.center_noise_scale,
            renew_directions=self.renew_directions,
            chunk_size=self.chunk_size,
            ridge=self.ridge,
            dtype=self.dtype,
            local_mass_quantile=self.local_mass_quantile,
            scale_expand_steps=self.scale_expand_steps,
            scale_search_steps=self.scale_search_steps,
            anisotropy_search_steps=self.anisotropy_search_steps,
            objective_check_every=self.objective_check_every,
            show_progress=show_progress,
            random_state=random_state,
            use_neighbor_index=self.use_neighbor_index,
        )


def stress_profiles() -> dict[str, StressProfile]:
    return {
        "smoke": StressProfile(
            name="smoke",
            scale_label="very fast syntax and plumbing check",
            n_values=(60,),
            d_values=(4,),
            seeds=(0,),
            links=("linear",),
            noise_levels=(0.02,),
            sigma_x_values=(1.0,),
            corr_values=(0.0,),
            q_values=(1.0,),
            n_centers=12,
            n_directions=3,
            min_neighbors=4.0,
            outer_steps=1,
            inner_steps=2,
            chunk_size=8,
            min_cosine_abs=0.40,
        ),
        "quick": StressProfile(
            name="quick",
            scale_label="fast multi-seed sanity grid",
            n_values=(120,),
            d_values=(6, 10),
            seeds=(0, 1),
            links=("linear", "sin"),
            noise_levels=(0.03,),
            sigma_x_values=(1.0,),
            corr_values=(0.0, 0.4),
            q_values=(0.5, 1.0),
            n_centers=28,
            n_directions=5,
            min_neighbors=6.0,
            outer_steps=2,
            inner_steps=4,
            chunk_size=12,
            min_cosine_abs=0.55,
        ),
        "medium": StressProfile(
            name="medium",
            scale_label="moderate tuning grid",
            n_values=(320,),
            d_values=(10, 25, 50),
            seeds=(0, 1, 2),
            links=("linear", "sin", "quadratic", "linear_plus_sin"),
            noise_levels=(0.03, 0.08),
            sigma_x_values=(0.7, 1.3),
            corr_values=(0.0, 0.55),
            q_values=(0.25, 0.5, 1.0),
            center_fraction=0.25,
            max_centers=90,
            n_directions=10,
            min_neighbors=10.0,
            outer_steps=3,
            inner_steps=8,
            chunk_size=24,
            min_cosine_abs=0.65,
        ),
        "large": StressProfile(
            name="large",
            scale_label="large dimension and localization stress",
            n_values=(800, 1200),
            d_values=(100, 200),
            seeds=(0, 1, 2),
            links=("linear", "sin", "quadratic"),
            noise_levels=(0.05, 0.15),
            sigma_x_values=(0.7, 1.5),
            corr_values=(0.0, 0.8),
            q_values=(0.2, 0.5, 1.0),
            center_fraction=0.20,
            max_centers=220,
            n_directions=16,
            min_neighbors=16.0,
            outer_steps=4,
            inner_steps=10,
            chunk_size=32,
            min_cosine_abs=0.60,
        ),
        "extreme": StressProfile(
            name="extreme",
            scale_label="extremely huge breakdown-dimension search",
            n_values=(2000, 5000),
            d_values=(300, 700, 1000),
            seeds=(0, 1, 2, 3, 4),
            links=("linear", "sin", "quadratic", "linear_plus_sin"),
            noise_levels=(0.05, 0.20),
            sigma_x_values=(0.5, 2.0),
            corr_values=(0.0, 0.5, 0.9),
            q_values=(0.1, 0.25, 1.0),
            center_fraction=0.15,
            max_centers=700,
            n_directions=32,
            min_neighbors=25.0,
            outer_steps=5,
            inner_steps=12,
            chunk_size=32,
            min_cosine_abs=0.50,
        ),
    }


def build_cases(
    profile_names: Iterable[str],
    *,
    base_seed: int = 0,
    max_cases: int | None = None,
) -> list[StressCase]:
    profiles = stress_profiles()
    cases: list[StressCase] = []
    ordinal = 0
    for profile_name in profile_names:
        if profile_name == "all":
            selected_profiles = profiles.values()
        else:
            try:
                selected_profiles = (profiles[profile_name],)
            except KeyError as exc:
                raise ValueError(f"Unknown profile: {profile_name}") from exc

        for profile in selected_profiles:
            for n in profile.n_values:
                n_centers = profile.centers_for_n(n)
                for d in profile.d_values:
                    for seed in profile.seeds:
                        for link in profile.links:
                            for sigma_eps in profile.noise_levels:
                                for sigma_x in profile.sigma_x_values:
                                    for corr in profile.corr_values:
                                        for q_value in profile.q_values:
                                            if not (0.0 < q_value <= 1.0):
                                                raise ValueError("q must be in (0, 1]")
                                            case_seed = base_seed + seed * 1_000_003 + ordinal * 97
                                            cases.append(
                                                StressCase(
                                                    profile=profile.name,
                                                    scale_label=profile.scale_label,
                                                    ordinal=ordinal,
                                                    seed=seed,
                                                    data_seed=case_seed,
                                                    fit_seed=case_seed + 1,
                                                    beta_seed=case_seed + 2,
                                                    n=n,
                                                    d=d,
                                                    link=link,
                                                    sigma_eps=sigma_eps,
                                                    sigma_x=sigma_x,
                                                    corr=corr,
                                                    q=q_value,
                                                    n_centers=n_centers,
                                                    n_directions=profile.n_directions,
                                                    min_neighbors=profile.min_neighbors,
                                                    lambda_penalty=profile.lambda_penalty,
                                                    outer_steps=profile.outer_steps,
                                                    inner_steps=profile.inner_steps,
                                                    bandwidth_decay=profile.bandwidth_decay,
                                                    anisotropy_min=profile.anisotropy_min,
                                                    kernel=profile.kernel,
                                                    center_noise_scale=profile.center_noise_scale,
                                                    renew_directions=profile.renew_directions,
                                                    chunk_size=profile.chunk_size,
                                                    tol=profile.tol,
                                                    ridge=profile.ridge,
                                                    dtype=profile.dtype,
                                                    local_mass_quantile=profile.local_mass_quantile,
                                                    scale_expand_steps=profile.scale_expand_steps,
                                                    scale_search_steps=profile.scale_search_steps,
                                                    anisotropy_search_steps=profile.anisotropy_search_steps,
                                                    objective_check_every=profile.objective_check_every,
                                                    use_neighbor_index=profile.use_neighbor_index,
                                                    min_cosine_abs=profile.min_cosine_abs,
                                                )
                                            )
                                            ordinal += 1
                                            if max_cases is not None and len(cases) >= max_cases:
                                                return cases
    return cases


def resolve_link(link: str) -> str | Any:
    if link == "linear_plus_sin":
        return linear_plus_sin_link
    return link


def make_sparse_beta(d: int, q: float, seed: int) -> np.ndarray:
    support_size = max(1, min(d, int(math.ceil(q * d))))
    rng = np.random.default_rng(seed)
    support = rng.choice(d, size=support_size, replace=False)
    beta = np.zeros(d, dtype=float)
    beta[support] = rng.normal(size=support_size)
    norm = np.linalg.norm(beta)
    if norm == 0:
        beta[support[0]] = 1.0
        norm = 1.0
    return beta / norm


@contextlib.contextmanager
def capture_cg_iterations() -> Iterator[list[dict[str, int]]]:
    from adp.variants import random_projection

    original_cg = random_projection.cg
    calls: list[dict[str, int]] = []

    def counted_cg(*args: Any, **kwargs: Any) -> Any:
        iterations = 0
        user_callback = kwargs.get("callback")

        def callback(xk: np.ndarray) -> None:
            nonlocal iterations
            iterations += 1
            if user_callback is not None:
                user_callback(xk)

        kwargs["callback"] = callback
        result = original_cg(*args, **kwargs)
        info = int(result[1])
        calls.append({"iterations": iterations, "info": info})
        return result

    random_projection.cg = counted_cg
    try:
        yield calls
    finally:
        random_projection.cg = original_cg


def y_diagnostics(y: np.ndarray) -> dict[str, float]:
    y_arr = np.asarray(y, dtype=float)
    mean = float(np.mean(y_arr))
    std = float(np.std(y_arr))
    if std > 0:
        outlier_frac = float(np.mean(np.abs(y_arr - mean) > 3.0 * std))
    else:
        outlier_frac = 0.0
    return {
        "y_mean": mean,
        "y_std": std,
        "y_min": float(np.min(y_arr)),
        "y_max": float(np.max(y_arr)),
        "y_outlier_frac_3sigma": outlier_frac,
    }


def result_diagnostics(case: StressCase, result: ADPResult, cg_calls: list[dict[str, int]]) -> dict[str, Any]:
    stats = result.statistics
    beta = np.asarray(result.beta, dtype=float)
    beta_norm = float(np.linalg.norm(beta))
    h_values = [step.h for step in result.history]
    beta_deltas = [step.beta_delta for step in result.history]
    rho_values = [step.anisotropy for step in result.history if step.anisotropy is not None]

    record: dict[str, Any] = {
        "h0": float(h_values[0]) if h_values else float(stats.h),
        "h_final": float(stats.h),
        "rho_final": float(stats.anisotropy) if stats.anisotropy is not None else math.nan,
        "rho_min": float(min(rho_values)) if rho_values else math.nan,
        "rho_max": float(max(rho_values)) if rho_values else math.nan,
        "weights_mean": float(stats.weights_mean),
        "beta_norm": beta_norm,
        "beta_delta_last": float(beta_deltas[-1]) if beta_deltas else math.nan,
        "beta_delta_max": float(max(beta_deltas)) if beta_deltas else math.nan,
        "inner_iterations_total": len(result.history),
        "inner_iterations_max_per_outer": max(_history_counts_by_outer(result.history).values(), default=0),
        "objective": float(result.objective),
        "statistics_time_sec": float(result.timings.get("statistics", math.nan)),
        "solve_time_sec": float(result.timings.get("solve", math.nan)),
        "fit_time_sec": float(result.timings.get("total", math.nan)),
        "cg_calls": len(cg_calls),
        "cg_iterations_total": int(sum(item["iterations"] for item in cg_calls)),
        "cg_iterations_max": int(max((item["iterations"] for item in cg_calls), default=0)),
        "cg_info_failures": int(sum(1 for item in cg_calls if item["info"] != 0)),
        "has_dxd_normal_matrix": False,
        "has_local_d_plus_1_regression": False,
        "uses_matrix_free_beta_update": True,
        "stores_full_weights_matrix": False,
        "stores_all_U": stats.U is not None,
        "localizing_tensor_materialized": False,
        "condition_number": math.nan,
    }

    if stats.N is not None:
        n_values = np.asarray(stats.N, dtype=float)
        record.update(
            {
                "N_min": float(np.min(n_values)),
                "N_mean": float(np.mean(n_values)),
                "N_max": float(np.max(n_values)),
                "N_frac_near_zero": float(np.mean(n_values <= np.finfo(float).eps)),
                "N_frac_below_min_neighbors": float(np.mean(n_values < case.min_neighbors)),
            }
        )
    else:
        record.update({"N_min": math.nan, "N_mean": math.nan, "N_max": math.nan, "N_frac_near_zero": math.nan, "N_frac_below_min_neighbors": math.nan})

    if stats.imav is not None:
        imav = np.asarray(stats.imav, dtype=float)
        record.update(
            {
                "imav_shape": json.dumps(list(imav.shape)),
                "imav_abs_mean": float(np.mean(np.abs(imav))),
                "imav_abs_max": float(np.max(np.abs(imav))),
                "imav_all_finite": bool(np.all(np.isfinite(imav))),
            }
        )

    if stats.U is not None:
        u = np.asarray(stats.U, dtype=float)
        ubeta = np.einsum("jpd,d->jp", u, beta)
        denominator = np.einsum("jp,jp->j", ubeta, ubeta)
        pred = result.slopes[:, None] * ubeta
        residual = float(np.sum((stats.imav - pred) ** 2))
        record.update(
            {
                "U_shape": json.dumps(list(u.shape)),
                "U_fro_norm": float(np.linalg.norm(u)),
                "U_all_finite": bool(np.all(np.isfinite(u))),
                "U_beta_abs_min": float(np.min(np.abs(ubeta))),
                "U_beta_abs_mean": float(np.mean(np.abs(ubeta))),
                "U_beta_abs_max": float(np.max(np.abs(ubeta))),
                "U_beta_denominator_min": float(np.min(denominator)),
                "U_beta_denominator_mean": float(np.mean(denominator)),
                "U_beta_denominator_frac_tiny": float(np.mean(denominator <= np.finfo(float).tiny)),
                "residual": residual,
                "estimated_U_storage_kib": float(u.nbytes / 1024.0),
            }
        )
    else:
        record.update(
            {
                "U_shape": json.dumps([]),
                "U_beta_abs_mean": math.nan,
                "U_beta_denominator_min": math.nan,
                "residual": math.nan,
                "estimated_U_storage_kib": 0.0,
            }
        )

    if stats.n_directions is not None:
        direction_shape = [stats.centers.shape[0], stats.n_directions]
        if stats.U is not None:
            direction_shape.append(int(stats.U.shape[-1]))
        record.update(
            {
                "directions_shape": json.dumps(direction_shape),
                "directions_norm_min": 1.0,
                "directions_norm_mean": 1.0,
                "directions_norm_max": 1.0,
            }
        )
    elif stats.directions is not None:
        directions = np.asarray(stats.directions, dtype=float)
        norms = np.linalg.norm(directions, axis=-1)
        record.update(
            {
                "directions_shape": json.dumps(list(directions.shape)),
                "directions_norm_min": float(np.min(norms)),
                "directions_norm_mean": float(np.mean(norms)),
                "directions_norm_max": float(np.max(norms)),
            }
        )
    else:
        record["directions_shape"] = json.dumps([])

    slopes = np.asarray(result.slopes, dtype=float)
    record.update(
        {
            "ell_min": float(np.min(slopes)),
            "ell_mean": float(np.mean(slopes)),
            "ell_max": float(np.max(slopes)),
            "ell_std": float(np.std(slopes)),
            "ell_all_finite": bool(np.all(np.isfinite(slopes))),
            "estimated_weights_matrix_kib": float(case.n_centers * case.n * 8 / 1024.0),
            "estimated_dxd_matrix_kib": float(case.d * case.d * 8 / 1024.0),
            "estimated_local_regression_matrix_kib_per_center": float((case.d + 1) * (case.d + 1) * 8 / 1024.0),
            "complexity_proxy_nJ_n_d_P": int(case.n_centers * case.n * case.d * case.n_directions),
        }
    )
    return record


def _history_counts_by_outer(history: Iterable[Any]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for step in history:
        counts[step.outer] = counts.get(step.outer, 0) + 1
    return counts


def run_case(case: StressCase, *, show_progress: bool = False) -> dict[str, Any]:
    record = case.to_manifest_record()
    record.update({"failed": False, "error": ""})
    started_total = time.perf_counter()

    try:
        beta_true = make_sparse_beta(case.d, case.q, case.beta_seed)
        data_model = ADP.create("new", case.adp_config(case.data_seed, show_progress=False))
        data_started = time.perf_counter()
        data = data_model.generate_data(
            n=case.n,
            d=case.d,
            n_centers=case.n_centers,
            n_directions=case.n_directions,
            beta=beta_true,
            noise=case.sigma_eps,
            sigma_x=case.sigma_x,
            corr=case.corr,
            link=resolve_link(case.link),
        )
        record["data_generation_time_sec"] = time.perf_counter() - data_started
        record.update(y_diagnostics(data.y))

        model = ADP.create("new", case.adp_config(case.fit_seed, show_progress=show_progress))
        started_tracing = not tracemalloc.is_tracing()
        if started_tracing:
            tracemalloc.start()
        tracemalloc.reset_peak()
        fit_started = time.perf_counter()
        try:
            with capture_cg_iterations() as cg_calls:
                result = model.fit(data.X, data.y, centers=data.centers, directions=data.directions)
            _, peak_memory = tracemalloc.get_traced_memory()
        finally:
            if started_tracing:
                tracemalloc.stop()

        metrics = direction_metrics(result.beta, data.beta)
        record.update(
            {
                "fit_wall_time_sec": time.perf_counter() - fit_started,
                "total_case_time_sec": time.perf_counter() - started_total,
                "peak_memory_kib": peak_memory / 1024.0,
                "cosine": metrics["cosine"],
                "cosine_abs": metrics["cosine_abs"],
                "angle_deg": metrics["angle_deg"],
                "signed_l2": metrics["signed_l2"],
                "case_passed_quality_gate": bool(metrics["cosine_abs"] >= case.min_cosine_abs),
            }
        )
        record.update(result_diagnostics(case, result, cg_calls))
    except Exception as exc:
        record.update(_failure_defaults())
        record["failed"] = True
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["total_case_time_sec"] = time.perf_counter() - started_total
    return record


def _failure_defaults() -> dict[str, Any]:
    numeric_nan = {
        "data_generation_time_sec",
        "fit_wall_time_sec",
        "peak_memory_kib",
        "cosine",
        "cosine_abs",
        "angle_deg",
        "signed_l2",
        "y_mean",
        "y_std",
        "y_min",
        "y_max",
        "y_outlier_frac_3sigma",
        "h0",
        "h_final",
        "rho_final",
        "rho_min",
        "rho_max",
        "N_min",
        "N_mean",
        "N_max",
        "N_frac_near_zero",
        "N_frac_below_min_neighbors",
        "U_beta_abs_mean",
        "U_beta_denominator_min",
        "residual",
        "beta_norm",
        "beta_delta_last",
        "statistics_time_sec",
        "solve_time_sec",
        "fit_time_sec",
        "fit_wall_time_sec",
        "estimated_U_storage_kib",
        "estimated_weights_matrix_kib",
    }
    defaults = {key: math.nan for key in numeric_nan}
    defaults.update(
        {
            "imav_shape": json.dumps([]),
            "U_shape": json.dumps([]),
            "directions_shape": json.dumps([]),
            "inner_iterations_total": 0,
            "cg_iterations_total": 0,
            "cg_info_failures": 0,
            "has_dxd_normal_matrix": False,
            "has_local_d_plus_1_regression": False,
            "uses_matrix_free_beta_update": True,
            "case_passed_quality_gate": False,
        }
    )
    return defaults


def summarize_records(records: list[dict[str, Any]], *, breakdown_threshold: float) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    if "complexity_proxy_nJ_n_d_P" not in frame.columns and {"n_centers", "n", "d", "n_directions"}.issubset(frame.columns):
        frame["complexity_proxy_nJ_n_d_P"] = frame["n_centers"] * frame["n"] * frame["d"] * frame["n_directions"]
    metric_defaults: dict[str, Any] = {
        "failed": False,
        "case_passed_quality_gate": math.nan,
        "cosine_abs": math.nan,
        "angle_deg": math.nan,
        "N_min": math.nan,
        "N_frac_below_min_neighbors": math.nan,
        "U_beta_denominator_min": math.nan,
        "cg_iterations_total": math.nan,
        "fit_time_sec": math.nan,
        "statistics_time_sec": math.nan,
        "solve_time_sec": math.nan,
        "peak_memory_kib": math.nan,
        "complexity_proxy_nJ_n_d_P": math.nan,
    }
    for column, default in metric_defaults.items():
        if column not in frame.columns:
            frame[column] = default
    group_cols = ["profile", "n", "d", "link", "sigma_eps", "sigma_x", "corr", "q", "n_directions", "n_centers"]
    summary = (
        frame.groupby(group_cols, dropna=False)
        .agg(
            count=("case_ordinal", "count"),
            failed_count=("failed", "sum"),
            quality_pass_rate=("case_passed_quality_gate", "mean"),
            cosine_abs_mean=("cosine_abs", "mean"),
            cosine_abs_min=("cosine_abs", "min"),
            angle_deg_mean=("angle_deg", "mean"),
            N_min_mean=("N_min", "mean"),
            N_frac_below_min_neighbors_mean=("N_frac_below_min_neighbors", "mean"),
            U_beta_denominator_min_mean=("U_beta_denominator_min", "mean"),
            cg_iterations_total_mean=("cg_iterations_total", "mean"),
            fit_time_sec_mean=("fit_time_sec", "mean"),
            statistics_time_sec_mean=("statistics_time_sec", "mean"),
            solve_time_sec_mean=("solve_time_sec", "mean"),
            peak_memory_kib_mean=("peak_memory_kib", "mean"),
            complexity_proxy_mean=("complexity_proxy_nJ_n_d_P", "mean"),
        )
        .reset_index()
    )
    fit_time = summary["fit_time_sec_mean"].replace(0.0, math.nan)
    statistics_time = summary["statistics_time_sec_mean"].replace(0.0, math.nan)
    summary["statistics_share_mean"] = summary["statistics_time_sec_mean"] / fit_time
    summary["solve_share_mean"] = summary["solve_time_sec_mean"] / fit_time
    summary["statistics_throughput_proxy_mean"] = summary["complexity_proxy_mean"] / statistics_time
    summary["breakdown_dimension"] = _breakdown_dimension(frame, breakdown_threshold)
    return summary


def _breakdown_dimension(frame: pd.DataFrame, threshold: float) -> float:
    if "cosine_abs" not in frame or frame.empty:
        return math.nan
    by_d = frame.groupby("d", dropna=False)["cosine_abs"].mean().sort_index()
    below = by_d[by_d < threshold]
    if below.empty:
        return math.nan
    return float(below.index[0])


def write_outputs(
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    breakdown_threshold: float,
    use_latex: bool = False,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "adp_single_index_stress_records.csv"
    summary_path = output_dir / "adp_single_index_stress_summary.csv"
    manifest_path = output_dir / "adp_single_index_stress_manifest.json"

    if records:
        all_keys = sorted({key for record in records for key in record})
        with records_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(records)
    else:
        records_path.write_text("")

    summary = summarize_records(records, breakdown_threshold=breakdown_threshold)
    summary.to_csv(summary_path, index=False)
    frame = pd.DataFrame(records)
    plot_paths = save_stress_plots(frame, summary, output_dir / "plots", breakdown_threshold=breakdown_threshold, use_latex=use_latex)
    saved: dict[str, Path] = {"records_csv": records_path, "summary_csv": summary_path}
    saved.update(plot_paths)
    manifest_path.write_text(
        json.dumps(
            {
                "records": len(records),
                "profiles": list(stress_profiles()),
                "q_definition": Q_DEFINITION,
                "localizing_tensor_form": LOCALIZING_TENSOR_FORM,
                "breakdown_threshold": breakdown_threshold,
                "outputs": {
                    "records_csv": str(records_path),
                    "summary_csv": str(summary_path),
                },
                "plots": {key: str(path) for key, path in plot_paths.items()},
                "latex_plots": bool(use_latex),
                "latex_preamble": LATEX_PREAMBLE if use_latex else "",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    saved["manifest_json"] = manifest_path
    return saved


def save_stress_plots(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    *,
    breakdown_threshold: float,
    use_latex: bool = False,
    dpi: int = 150,
) -> dict[str, Path]:
    if frame.empty or summary.empty:
        return {}
    configure_stress_matplotlib(use_latex=use_latex)
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    quality = _finite_rows(summary, ["d", "cosine_abs_mean"])
    if not quality.empty:
        fig, ax = plt.subplots()
        set_adp_figure_size(fig, width=8.2, height=4.8)
        for index, (profile, group) in enumerate(quality.groupby("profile", sort=False)):
            group = group.sort_values("d")
            ax.plot(
                group["d"],
                group["cosine_abs_mean"],
                marker="o",
                linewidth=2.2,
                label=profile,
                color=ADP_COLORS[index % len(ADP_COLORS)],
            )
        ax.axhline(breakdown_threshold, color="#dc2626", linestyle="--", linewidth=1.4, label="порог breakdown")
        ax.set_ylim(0.0, 1.05)
        apply_adp_axis_style(
            ax,
            xlabel=r"размерность $d$",
            ylabel=r"среднее $|\cos(\beta,\hat{\beta})|$",
            title="Качество восстановления направления",
            legend_title="профиль",
        )
        saved["quality_plot"] = save_figure(fig, output_dir / "stress_quality_by_dimension.png", dpi=dpi, close=True)

    timings = _finite_rows(summary, ["complexity_proxy_mean", "fit_time_sec_mean"])
    if not timings.empty:
        fig, ax = plt.subplots()
        set_adp_figure_size(fig, width=8.2, height=4.8)
        for index, (profile, group) in enumerate(timings.groupby("profile", sort=False)):
            ax.scatter(
                group["complexity_proxy_mean"],
                group["fit_time_sec_mean"],
                s=54,
                label=profile,
                color=ADP_COLORS[index % len(ADP_COLORS)],
                edgecolor="#ffffff",
                linewidth=0.8,
            )
        if (timings["complexity_proxy_mean"] > 0).all():
            ax.set_xscale("log")
        apply_adp_axis_style(
            ax,
            xlabel=r"proxy сложности $n_J\,n\,d\,n_S$",
            ylabel="среднее время обучения, сек",
            title="Время работы ADP",
            legend_title="профиль",
        )
        saved["time_plot"] = save_figure(fig, output_dir / "stress_time_by_complexity.png", dpi=dpi, close=True)

    memory = _finite_rows(summary, ["d", "peak_memory_kib_mean"])
    if not memory.empty:
        fig, ax = plt.subplots()
        set_adp_figure_size(fig, width=8.2, height=4.8)
        for index, (profile, group) in enumerate(memory.groupby("profile", sort=False)):
            group = group.sort_values("d")
            ax.plot(
                group["d"],
                group["peak_memory_kib_mean"],
                marker="o",
                linewidth=2.2,
                label=profile,
                color=ADP_COLORS[index % len(ADP_COLORS)],
            )
        apply_adp_axis_style(
            ax,
            xlabel=r"размерность $d$",
            ylabel="пиковая память, КиБ",
            title="Память stress-запуска",
            legend_title="профиль",
        )
        saved["memory_plot"] = save_figure(fig, output_dir / "stress_memory_by_dimension.png", dpi=dpi, close=True)

    localization = _finite_rows(summary, ["N_min_mean", "N_frac_below_min_neighbors_mean"])
    if not localization.empty:
        plot_frame = localization.copy()
        plot_frame["scenario_label"] = _scenario_labels(plot_frame)
        fig, ax = plt.subplots()
        set_adp_figure_size(fig, width=max(8.2, 0.6 * len(plot_frame)), height=4.8)
        positions = np.arange(len(plot_frame))
        ax.bar(positions, plot_frame["N_min_mean"], color=ADP_COLORS[2], edgecolor="#ffffff", linewidth=0.8, label=r"$\min_j N_j$")
        ax.set_xticks(positions)
        ax.set_xticklabels(plot_frame["scenario_label"], rotation=30, ha="right")
        apply_adp_axis_style(
            ax,
            xlabel="сценарий",
            ylabel=r"среднее $\min_j N_j$",
            title="Локальная масса окрестностей",
            legend_title=None,
        )
        ax2 = ax.twinx()
        ax2.plot(
            positions,
            100.0 * plot_frame["N_frac_below_min_neighbors_mean"],
            color=ADP_COLORS[1],
            marker="o",
            linewidth=2.0,
            label=r"доля ниже $n_{\min}$, %",
        )
        ax2.set_ylabel(r"доля ниже $n_{\min}$, %", labelpad=8)
        ax2.tick_params(axis="both", length=0, pad=6)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        fig.subplots_adjust(right=0.76)
        ax.legend(
            lines + lines2,
            labels + labels2,
            title="метрика",
            loc="upper left",
            bbox_to_anchor=(1.08, 1.0),
            borderaxespad=0.0,
        )
        saved["localization_plot"] = save_figure(fig, output_dir / "stress_localization_mass.png", dpi=dpi, close=True)

    optimization = _finite_rows(summary, ["U_beta_denominator_min_mean", "cg_iterations_total_mean"])
    if not optimization.empty:
        plot_frame = optimization.copy()
        plot_frame["scenario_label"] = _scenario_labels(plot_frame)
        fig, ax = plt.subplots()
        set_adp_figure_size(fig, width=max(8.2, 0.6 * len(plot_frame)), height=4.8)
        positions = np.arange(len(plot_frame))
        ax.bar(
            positions,
            plot_frame["U_beta_denominator_min_mean"],
            color=ADP_COLORS[0],
            edgecolor="#ffffff",
            linewidth=0.8,
            label=r"$\min |U\beta|^2$",
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(plot_frame["scenario_label"], rotation=30, ha="right")
        apply_adp_axis_style(
            ax,
            xlabel="сценарий",
            ylabel=r"средний $\min |U\beta|^2$",
            title="Стабильность beta-обновления",
            legend_title=None,
        )
        ax2 = ax.twinx()
        ax2.plot(
            positions,
            plot_frame["cg_iterations_total_mean"],
            color=ADP_COLORS[4],
            marker="o",
            linewidth=2.0,
            label="итерации CG",
        )
        ax2.set_ylabel("итерации CG", labelpad=8)
        ax2.tick_params(axis="both", length=0, pad=6)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        fig.subplots_adjust(right=0.76)
        ax.legend(
            lines + lines2,
            labels + labels2,
            title="метрика",
            loc="upper left",
            bbox_to_anchor=(1.08, 1.0),
            borderaxespad=0.0,
        )
        saved["optimization_plot"] = save_figure(fig, output_dir / "stress_beta_update_stability.png", dpi=dpi, close=True)

    return saved


def configure_stress_matplotlib(*, use_latex: bool) -> None:
    configure_adp_matplotlib()
    import matplotlib as mpl

    mpl.rcParams["text.usetex"] = bool(use_latex)
    if use_latex:
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["text.latex.preamble"] = LATEX_PREAMBLE
    else:
        mpl.rcParams["text.latex.preamble"] = ""


def _finite_rows(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        return pd.DataFrame()
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna(subset=columns)


def _scenario_labels(frame: pd.DataFrame) -> list[str]:
    labels = []
    for _, row in frame.iterrows():
        labels.append(f"{row['profile']}: d={int(row['d'])}, P={int(row['n_directions'])}")
    return labels


def print_profile_table() -> None:
    for profile in stress_profiles().values():
        case_count = len(build_cases([profile.name]))
        print(
            f"{profile.name:8s} cases={case_count:5d} "
            f"n={profile.n_values} d={profile.d_values} P={profile.n_directions} "
            f"seeds={profile.seeds} links={profile.links} -- {profile.scale_label}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stress-test single-index ADP random-projection implementation.",
    )
    parser.add_argument(
        "--profile",
        action="append",
        choices=tuple(stress_profiles()) + ("all",),
        default=None,
        help="Scale profile to run. Can be repeated. Default: smoke.",
    )
    parser.add_argument("--list-profiles", action="store_true", help="Print profile sizes and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Write/print manifest without fitting ADP.")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit number of generated cases.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed used to derive data/fit/beta seeds.")
    parser.add_argument("--output", type=Path, default=Path("stress_outputs"), help="Output directory.")
    parser.add_argument("--show-progress", action="store_true", help="Show ADP progress bars during fit.")
    parser.add_argument("--breakdown-threshold", type=float, default=0.70, help="Mean |cos| threshold for breakdown dimension.")
    parser.add_argument("--fail-on-quality", action="store_true", help="Return non-zero if any completed case misses its quality gate.")
    parser.add_argument("--dtype", choices=("float64", "float32"), default=None, help="Override ADP numeric dtype for stress cases.")
    parser.add_argument("--local-mass-quantile", type=float, default=None, help="Override lower quantile used for local mass bandwidth selection.")
    parser.add_argument("--scale-expand-steps", type=int, default=None, help="Override bandwidth scale expansion budget.")
    parser.add_argument("--scale-search-steps", type=int, default=None, help="Override bandwidth scale binary-search budget.")
    parser.add_argument("--anisotropy-search-steps", type=int, default=None, help="Override rho binary-search budget.")
    parser.add_argument("--objective-check-every", type=int, default=None, help="Check full objective every N inner iterations.")
    parser.add_argument(
        "--latex",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render plot text through LaTeX. Use --no-latex to disable.",
    )
    return parser.parse_args(argv)


def apply_case_overrides(cases: list[StressCase], args: argparse.Namespace) -> list[StressCase]:
    overrides: dict[str, Any] = {}
    for arg_name, field_name in (
        ("dtype", "dtype"),
        ("local_mass_quantile", "local_mass_quantile"),
        ("scale_expand_steps", "scale_expand_steps"),
        ("scale_search_steps", "scale_search_steps"),
        ("anisotropy_search_steps", "anisotropy_search_steps"),
        ("objective_check_every", "objective_check_every"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[field_name] = value
    if not overrides:
        return cases
    return [replace(case, **overrides) for case in cases]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_profiles:
        print_profile_table()
        return 0

    profile_names = args.profile or ["smoke"]
    cases = build_cases(profile_names, base_seed=args.seed, max_cases=args.max_cases)
    cases = apply_case_overrides(cases, args)
    if args.dry_run:
        records = [case.to_manifest_record() for case in cases]
    else:
        records = [run_case(case, show_progress=args.show_progress) for case in cases]

    paths = write_outputs(records, args.output, breakdown_threshold=args.breakdown_threshold, use_latex=args.latex)
    print(f"cases: {len(records)}")
    print(f"records: {paths['records_csv']}")
    print(f"summary: {paths['summary_csv']}")
    print(f"manifest: {paths['manifest_json']}")

    if not args.dry_run and any(record.get("failed") for record in records):
        return 1
    if args.fail_on_quality and any(not record.get("case_passed_quality_gate", False) for record in records):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
