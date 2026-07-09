from __future__ import annotations

# Keep process-level parallelism from multiplying BLAS threads in every worker.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adp import ADP, ADPConfig, ADPData
from adp.evaluation.metrics import direction_metrics


ExperimentName = Literal["4", "5", "6"]
MethodName = Literal[
    "full_adp",
    "step0_only",
    "no_anisotropy",
    "fixed_h",
    "no_regularization",
    "random_beta_init",
]


@dataclass(frozen=True)
class ConfirmatoryConfig:
    d_values: tuple[int, ...] = (50, 100, 200)
    n_over_d_values: tuple[int, ...] = (10, 20)
    corr_values: tuple[float, ...] = (0.0, 0.3, 0.7)
    snr_values: tuple[float, ...] = (20.0, 10.0, 5.0)
    link_values: tuple[str, ...] = ("linear", "tanh", "sin")
    q_values: tuple[float, ...] = (0.3, 1.0)
    seeds: int = 100
    base_seed: int = 20260709

    n_directions: int = 64
    min_neighbors: float = 256.0
    center_fraction: float = 0.5
    h0_inflation: float = 1.1
    lambda_rel: float = 1e-2
    outer_steps: int = 8
    inner_steps: int = 20
    gamma_h: float = 0.8
    local_mass_quantile: float = 0.05
    scale_search_steps: int = 12
    anisotropy_search_steps: int = 12
    objective_check_every: int = 2
    kernel: str = "epanechnikov"
    dtype: str = "float64"
    center_noise_scale: float = 1.0
    renew_directions: bool = True
    backend: str = "numpy"
    methods: tuple[MethodName, ...] = (
        "full_adp",
        "step0_only",
        "no_anisotropy",
        "fixed_h",
        "no_regularization",
        "random_beta_init",
    )
    experiments: tuple[ExperimentName, ...] = ("4", "5", "6")
    max_scenarios: int | None = None
    bootstrap_reps: int = 1000
    show_progress: bool = False
    progress_log_every: int = 1


@dataclass(frozen=True)
class ScenarioSpec:
    experiment: ExperimentName
    scenario_id: str
    scenario_index: int
    d: int
    n: int
    n_over_d: int
    corr: float
    snr: float
    link: str
    q: float


@dataclass(frozen=True)
class RunJob:
    experiment: ExperimentName
    scenario: ScenarioSpec
    seed_id: int
    method: MethodName


def parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def parse_float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def parse_str_tuple(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in text.split(",") if item.strip())


def build_scenarios(
    config: ConfirmatoryConfig,
    *,
    experiment: ExperimentName,
) -> list[ScenarioSpec]:
    scenarios: list[ScenarioSpec] = []
    q_values = config.q_values if experiment in {"5", "6"} else (config.q_values[0],)

    for d in config.d_values:
        for n_over_d in config.n_over_d_values:
            for corr in config.corr_values:
                for snr in config.snr_values:
                    for link in config.link_values:
                        for q in q_values:
                            n = int(d * n_over_d)
                            index = len(scenarios)
                            scenarios.append(
                                ScenarioSpec(
                                    experiment=experiment,
                                    scenario_id=(
                                        f"exp{experiment}_d{d}_nod{n_over_d}_"
                                        f"corr{corr:g}_snr{snr:g}_{link}_q{q:g}"
                                    ),
                                    scenario_index=index,
                                    d=d,
                                    n=n,
                                    n_over_d=n_over_d,
                                    corr=corr,
                                    snr=snr,
                                    link=link,
                                    q=q,
                                )
                            )

    if config.max_scenarios is not None:
        return balanced_subset(scenarios, config.max_scenarios)
    return scenarios


def balanced_subset(scenarios: list[ScenarioSpec], limit: int) -> list[ScenarioSpec]:
    if limit <= 0 or len(scenarios) <= limit:
        return scenarios
    positions = np.linspace(0, len(scenarios) - 1, num=limit)
    indices = sorted({int(round(pos)) for pos in positions})
    return [scenarios[i] for i in indices[:limit]]


def make_sparse_beta(d: int, q: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    support_size = min(d, max(1, int(round(q * d))))
    support = rng.choice(d, size=support_size, replace=False)
    beta = np.zeros(d, dtype=float)
    beta[support] = rng.normal(size=support_size)
    norm = np.linalg.norm(beta)
    if norm < np.finfo(float).eps:
        beta[support[0]] = 1.0
        norm = 1.0
    return beta / norm


def make_adp_config(
    config: ConfirmatoryConfig,
    scenario: ScenarioSpec,
    *,
    random_state: int,
    method: MethodName,
) -> ADPConfig:
    n_centers = max(1, min(scenario.n, int(round(config.center_fraction * scenario.n))))
    outer_steps = 1 if method == "step0_only" else config.outer_steps
    lambda_penalty = config.lambda_rel * float(config.min_neighbors)
    if method == "no_regularization":
        lambda_penalty = 0.0
    bandwidth_decay = 1.0 / max(config.gamma_h, np.finfo(float).eps)
    if method == "fixed_h":
        bandwidth_decay = 1.0

    return ADPConfig(
        n_centers=n_centers,
        n_directions=config.n_directions,
        min_neighbors=config.min_neighbors,
        lambda_penalty=lambda_penalty,
        outer_steps=outer_steps,
        inner_steps=config.inner_steps,
        bandwidth_decay=bandwidth_decay,
        initial_bandwidth_inflation=config.h0_inflation,
        local_mass_quantile=config.local_mass_quantile,
        scale_search_steps=config.scale_search_steps,
        anisotropy_search_steps=config.anisotropy_search_steps,
        objective_check_every=config.objective_check_every,
        kernel=config.kernel,  # type: ignore[arg-type]
        dtype=config.dtype,
        center_noise_scale=config.center_noise_scale,
        renew_directions=config.renew_directions,
        backend=config.backend,  # type: ignore[arg-type]
        show_progress=config.show_progress,
        random_state=random_state,
    )


def generate_case_data(
    config: ConfirmatoryConfig,
    scenario: ScenarioSpec,
    *,
    data_seed: int,
) -> ADPData:
    beta = make_sparse_beta(scenario.d, scenario.q, data_seed + 17)
    gen_config = make_adp_config(
        config,
        scenario,
        random_state=data_seed,
        method="full_adp",
    )
    model = ADP.create("new", gen_config)
    data = model.generate_data(
        n=scenario.n,
        d=scenario.d,
        n_centers=gen_config.n_centers,
        n_directions=gen_config.n_directions,
        beta=beta,
        noise=0.0,
        corr=scenario.corr,
        link=scenario.link,
    )

    rng = np.random.default_rng(data_seed + 31)
    signal = np.asarray(data.y, dtype=float)
    signal_scale = float(np.std(signal))
    if not np.isfinite(signal_scale) or signal_scale <= np.finfo(float).eps:
        signal_scale = 1.0
    noise_scale = signal_scale / max(float(scenario.snr), np.finfo(float).eps)
    eps = rng.normal(scale=noise_scale, size=scenario.n)
    return ADPData(
        X=np.asarray(data.X, dtype=float),
        y=signal + eps,
        beta=np.asarray(data.beta, dtype=float),
        centers=np.asarray(data.centers, dtype=float),
        directions=np.asarray(data.directions, dtype=float) if data.directions is not None else None,
        noise=eps,
        link_name=data.link_name,
    )


def methods_for_experiment(
    config: ConfirmatoryConfig,
    experiment: ExperimentName,
) -> tuple[MethodName, ...]:
    if experiment in {"4", "6"}:
        return ("full_adp",)
    return config.methods


def build_jobs(config: ConfirmatoryConfig) -> list[RunJob]:
    jobs: list[RunJob] = []
    for experiment in config.experiments:
        for scenario in build_scenarios(config, experiment=experiment):
            for seed_id in range(config.seeds):
                for method in methods_for_experiment(config, experiment):
                    jobs.append(
                        RunJob(
                            experiment=experiment,
                            scenario=scenario,
                            seed_id=seed_id,
                            method=method,
                        )
                    )
    return jobs


def run_job(job: RunJob, config: ConfirmatoryConfig) -> list[dict[str, object]]:
    scenario = job.scenario
    data_seed = (
        config.base_seed
        + 10_000_019 * int(job.experiment)
        + 100_003 * scenario.scenario_index
        + 1_009 * job.seed_id
    )
    fit_seed = data_seed + 101
    data = generate_case_data(config, scenario, data_seed=data_seed)
    adp_config = make_adp_config(config, scenario, random_state=fit_seed, method=job.method)
    model = ADP.create("new", adp_config)

    if job.method == "no_anisotropy":
        model._select_new_anisotropy = lambda X, centers, h, beta: 1.0  # type: ignore[method-assign]

    beta0 = None
    if job.method == "random_beta_init":
        beta0 = make_sparse_beta(scenario.d, 1.0, fit_seed + 97)

    started = time.perf_counter()
    try:
        result = model.fit(
            data.X,
            data.y,
            centers=data.centers,
            directions=data.directions,
            beta0=beta0,
        )
        runtime_sec = time.perf_counter() - started
        return rows_from_result(job, config, data, result, runtime_sec, failed=False, error="")
    except Exception as exc:
        return [
            failed_row(
                job,
                config,
                data_seed=data_seed,
                fit_seed=fit_seed,
                runtime_sec=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}",
            )
        ]


def rows_from_result(
    job: RunJob,
    config: ConfirmatoryConfig,
    data: ADPData,
    result: object,
    runtime_sec: float,
    *,
    failed: bool,
    error: str,
) -> list[dict[str, object]]:
    progress = list(getattr(result, "progress", []))
    beta_path = list(getattr(result, "beta_path", []))
    if not beta_path:
        beta_path = [np.asarray(getattr(result, "beta"), dtype=float)]

    rows: list[dict[str, object]] = []
    cos_values = [
        direction_metrics(np.asarray(beta, dtype=float), data.beta)["cosine_abs"]
        for beta in beta_path
    ]
    cos0 = cos_values[0] if cos_values else math.nan

    for index, beta in enumerate(beta_path):
        record = progress[index] if index < len(progress) else {}
        cos_beta = direction_metrics(np.asarray(beta, dtype=float), data.beta)["cosine_abs"]
        local_mass_q05 = float(record.get("local_mass_q05", math.nan))
        row = base_row(job, config)
        row.update(
            {
                "outer_k": int(record.get("outer", index + 1)) - 1,
                "h_k": float(record.get("h", math.nan)),
                "rho_k": float(record.get("rho", math.nan)),
                "local_mass_mean": float(record.get("local_mass_mean", record.get("weights", math.nan))),
                "local_mass_q05": local_mass_q05,
                "local_mass_min": float(record.get("local_mass_min", math.nan)),
                "cos_beta_k": float(cos_beta),
                "cos_delta_from_k0": float(cos_beta - cos0) if np.isfinite(cos0) else math.nan,
                "success_08": bool(cos_beta >= 0.8),
                "success_09": bool(cos_beta >= 0.9),
                "beta_delta_outer": float(record.get("delta", math.nan)),
                "objective_final_inner": float(record.get("objective", math.nan)),
                "runtime_sec": float(runtime_sec),
                "failed": bool(failed),
                "error": error,
                "local_mass_gate": bool(local_mass_q05 >= config.n_directions + 4)
                if np.isfinite(local_mass_q05)
                else False,
            }
        )
        rows.append(row)
    return rows


def base_row(job: RunJob, config: ConfirmatoryConfig) -> dict[str, object]:
    scenario = job.scenario
    return {
        "experiment": job.experiment,
        "seed": job.seed_id,
        "scenario_id": scenario.scenario_id,
        "method": job.method,
        "d": scenario.d,
        "n": scenario.n,
        "n_over_d": scenario.n_over_d,
        "corr": scenario.corr,
        "snr": scenario.snr,
        "q": scenario.q,
        "link": scenario.link,
        "n_directions": config.n_directions,
        "n_min": config.min_neighbors,
        "center_fraction": config.center_fraction,
        "h0_inflation": config.h0_inflation,
        "lambda_rel": config.lambda_rel,
        "outer_steps": config.outer_steps,
        "inner_steps": config.inner_steps,
        "gamma_h": config.gamma_h,
        "backend": config.backend,
    }


def failed_row(
    job: RunJob,
    config: ConfirmatoryConfig,
    *,
    data_seed: int,
    fit_seed: int,
    runtime_sec: float,
    error: str,
) -> dict[str, object]:
    row = base_row(job, config)
    row.update(
        {
            "data_seed": data_seed,
            "fit_seed": fit_seed,
            "outer_k": -1,
            "h_k": math.nan,
            "rho_k": math.nan,
            "local_mass_mean": math.nan,
            "local_mass_q05": math.nan,
            "local_mass_min": math.nan,
            "cos_beta_k": math.nan,
            "cos_delta_from_k0": math.nan,
            "success_08": False,
            "success_09": False,
            "beta_delta_outer": math.nan,
            "objective_final_inner": math.nan,
            "runtime_sec": runtime_sec,
            "failed": True,
            "error": error,
            "local_mass_gate": False,
        }
    )
    return row


def summarize_records(records: pd.DataFrame, config: ConfirmatoryConfig) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    group_cols = ["experiment", "scenario_id", "method"]
    for keys, group in records.groupby(group_cols, dropna=False):
        final = final_outer_rows(group)
        cos = final["cos_beta_k"].to_numpy(dtype=float)
        cos = cos[np.isfinite(cos)]
        delta = final["cos_delta_from_k0"].to_numpy(dtype=float)
        delta = delta[np.isfinite(delta)]
        row = {
            **dict(zip(group_cols, keys)),
            "records": int(len(group)),
            "final_runs": int(len(final)),
            "failure_rate": float(final["failed"].astype(bool).mean()) if len(final) else math.nan,
            "cos_beta_final_median": float(np.median(cos)) if cos.size else math.nan,
            "cos_beta_final_q25": float(np.quantile(cos, 0.25)) if cos.size else math.nan,
            "cos_beta_final_q75": float(np.quantile(cos, 0.75)) if cos.size else math.nan,
            "delta_cos_median": float(np.median(delta)) if delta.size else math.nan,
            "success_08_rate": float(final["success_08"].astype(bool).mean()) if len(final) else math.nan,
            "success_09_rate": float(final["success_09"].astype(bool).mean()) if len(final) else math.nan,
            "local_mass_q05_median": float(np.nanmedian(final["local_mass_q05"])) if len(final) else math.nan,
            "runtime_sec_median": float(np.nanmedian(final["runtime_sec"])) if len(final) else math.nan,
        }
        low, high = bootstrap_ci_median(delta, reps=config.bootstrap_reps, seed=config.base_seed + 701)
        row["delta_cos_median_ci95_low"] = low
        row["delta_cos_median_ci95_high"] = high
        add_rho_checks(row, group, config)
        add_growth_checks(row, final)
        rows.append(row)

    result = pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)
    bool_cols = [
        "rho_median_trend_ok",
        "growth_median_positive",
        "growth_ci95_low_positive",
        "growth_pass",
        "strong_success",
        "medium_success",
    ]
    for col in bool_cols:
        if col in result:
            result[col] = result[col].map(lambda value: bool(value) if pd.notna(value) else value).astype(object)
    return result


def add_rho_checks(
    row: dict[str, object],
    group: pd.DataFrame,
    config: ConfirmatoryConfig,
) -> None:
    rho = group["rho_k"].to_numpy(dtype=float) if "rho_k" in group else np.array([], dtype=float)
    finite_rho = rho[np.isfinite(rho)]
    if finite_rho.size:
        row["rho_in_range_rate"] = float(np.mean((0.0 <= finite_rho) & (finite_rho <= 1.0)))
    else:
        row["rho_in_range_rate"] = math.nan

    if "local_mass_mean" in group:
        local_mass_mean = group["local_mass_mean"].to_numpy(dtype=float)
        finite_mean = local_mass_mean[np.isfinite(local_mass_mean)]
        row["local_mass_mean_gate_rate"] = float(np.mean(finite_mean >= config.min_neighbors)) if finite_mean.size else math.nan
    else:
        row["local_mass_mean_gate_rate"] = math.nan

    if "local_mass_q05" in group:
        local_mass_q05 = group["local_mass_q05"].to_numpy(dtype=float)
        finite_q05 = local_mass_q05[np.isfinite(local_mass_q05)]
        q05_threshold = float(config.n_directions + 4)
        row["local_mass_q05_gate_threshold"] = q05_threshold
        row["local_mass_q05_gate_rate"] = float(np.mean(finite_q05 >= q05_threshold)) if finite_q05.size else math.nan
    else:
        row["local_mass_q05_gate_threshold"] = float(config.n_directions + 4)
        row["local_mass_q05_gate_rate"] = math.nan

    rho_by_outer = (
        group[["outer_k", "rho_k"]]
        .dropna()
        .groupby("outer_k")["rho_k"]
        .median()
        .sort_index()
        if {"outer_k", "rho_k"}.issubset(group.columns)
        else pd.Series(dtype=float)
    )
    if len(rho_by_outer) >= 2:
        max_increase = float(np.max(np.diff(rho_by_outer.to_numpy(dtype=float))))
        row["rho_median_trend_max_increase"] = max_increase
        row["rho_median_trend_ok"] = bool(max_increase <= 0.05)
    else:
        row["rho_median_trend_max_increase"] = 0.0 if len(rho_by_outer) == 1 else math.nan
        row["rho_median_trend_ok"] = bool(len(rho_by_outer) == 1)

    if {"rho_k", "cos_beta_k"}.issubset(group.columns):
        corr_frame = group[["rho_k", "cos_beta_k"]].dropna()
        row["rho_cos_spearman"] = spearman_corr(
            corr_frame["rho_k"].to_numpy(dtype=float),
            corr_frame["cos_beta_k"].to_numpy(dtype=float),
        )
    else:
        row["rho_cos_spearman"] = math.nan


def add_growth_checks(
    row: dict[str, object],
    final: pd.DataFrame,
) -> None:
    delta = final["cos_delta_from_k0"].to_numpy(dtype=float) if "cos_delta_from_k0" in final else np.array([], dtype=float)
    finite_delta = delta[np.isfinite(delta)]
    row["improvement_rate"] = float(np.mean(finite_delta > 0.0)) if finite_delta.size else math.nan
    row["growth_median_positive"] = bool(row["delta_cos_median"] > 0.0) if pd.notna(row["delta_cos_median"]) else math.nan
    row["growth_ci95_low_positive"] = bool(row["delta_cos_median_ci95_low"] > 0.0) if pd.notna(row["delta_cos_median_ci95_low"]) else math.nan
    row["strong_success"] = bool(row["success_09_rate"] >= 0.7) if pd.notna(row["success_09_rate"]) else math.nan
    row["medium_success"] = bool(row["success_08_rate"] >= 0.8) if pd.notna(row["success_08_rate"]) else math.nan
    row["growth_pass"] = bool(
        row["growth_median_positive"]
        and row["growth_ci95_low_positive"]
        and pd.notna(row["improvement_rate"])
        and float(row["improvement_rate"]) >= 0.75
    )


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return math.nan
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if np.std(x_rank) <= np.finfo(float).eps or np.std(y_rank) <= np.finfo(float).eps:
        return math.nan
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def final_outer_rows(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return records.copy()
    if "seed" not in records.columns:
        sort_cols = [col for col in ("scenario_id", "method", "outer_k") if col in records.columns]
        return records.sort_values(sort_cols).reset_index(drop=True) if sort_cols else records.copy()
    group_cols = [col for col in ("experiment", "scenario_id", "method", "seed") if col in records.columns]
    rows = []
    for _, group in records.groupby(group_cols, dropna=False):
        group = group.sort_values("outer_k")
        rows.append(group.iloc[-1])
    return pd.DataFrame(rows).reset_index(drop=True)


def final_success_summary(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()
    frame = records.copy()
    if "method" in frame:
        frame = frame[frame["method"] == "full_adp"]
    final = final_outer_rows(frame)
    rows: list[dict[str, object]] = []
    for scenario_id, group in final.groupby("scenario_id", dropna=False):
        cos = group["cos_beta_k"].to_numpy(dtype=float)
        finite_cos = cos[np.isfinite(cos)]
        success_08 = group["success_08"].astype(bool) if "success_08" in group else group["cos_beta_k"] >= 0.8
        success_09 = group["success_09"].astype(bool) if "success_09" in group else group["cos_beta_k"] >= 0.9
        n_directions = float(group["n_directions"].iloc[0]) if len(group) else math.nan
        local_mass_gate_threshold = n_directions + 4.0
        row = {
            "scenario_id": scenario_id,
            "runs": int(len(group)),
            "median_cos": float(np.median(finite_cos)) if finite_cos.size else math.nan,
            "success_08_rate": float(success_08.mean()) if len(group) else math.nan,
            "success_09_rate": float(success_09.mean()) if len(group) else math.nan,
            "failure_rate": float(group["failed"].astype(bool).mean()) if len(group) else math.nan,
            "local_mass_q05_median": float(np.nanmedian(group["local_mass_q05"])) if len(group) else math.nan,
            "local_mass_gate_threshold": local_mass_gate_threshold,
        }
        row["median_cos_ge_08"] = bool(row["median_cos"] >= 0.8)
        row["success_08_rate_ge_08"] = bool(row["success_08_rate"] >= 0.8)
        row["failure_rate_le_005"] = bool(row["failure_rate"] <= 0.05)
        row["local_mass_gate"] = bool(row["local_mass_q05_median"] >= local_mass_gate_threshold)
        row["strong_mode"] = bool(row["median_cos"] >= 0.9)
        row["borderline_mode"] = bool(0.7 <= row["median_cos"] < 0.8)
        row["breakdown"] = bool(row["median_cos"] < 0.7 or row["failure_rate"] > 0.1)
        row["protocol_pass"] = bool(
            row["median_cos_ge_08"]
            and row["success_08_rate_ge_08"]
            and row["failure_rate_le_005"]
            and row["local_mass_gate"]
        )
        rows.append(row)

    result = pd.DataFrame(rows).sort_values("scenario_id").reset_index(drop=True)
    bool_cols = [
        "median_cos_ge_08",
        "success_08_rate_ge_08",
        "failure_rate_le_005",
        "local_mass_gate",
        "strong_mode",
        "borderline_mode",
        "breakdown",
        "protocol_pass",
    ]
    for col in bool_cols:
        if col in result:
            result[col] = result[col].map(bool).astype(object)
    return result


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
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    medians = np.empty(max(1, reps), dtype=float)
    for i in range(medians.size):
        sample = rng.choice(values, size=values.size, replace=True)
        medians[i] = np.median(sample)
    low, high = np.quantile(medians, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def run_confirmatory_experiments(
    config: ConfirmatoryConfig,
    output_dir: Path,
    *,
    n_jobs: int | None = None,
    output_prefix: str = "confirmatory_456",
    experiment_label: str = "ADP confirmatory experiments 4, 5, 6 from Tests.md",
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)
    if n_jobs < 1:
        raise ValueError("--jobs must be >= 1")

    jobs = build_jobs(config)
    started = time.perf_counter()
    records: list[dict[str, object]] = []
    requested_n_jobs = n_jobs
    actual_n_jobs = n_jobs
    parallel_fallback_error = ""
    total_jobs = len(jobs)
    completed_jobs = 0

    def update_progress_postfix(progress: object, job: RunJob) -> None:
        progress.set_postfix(
            experiment=job.experiment,
            scenario=job.scenario.scenario_id,
            seed=job.seed_id,
            method=job.method,
            refresh=True,
        )

    def emit_progress_line(desc: str, completed: int, job: RunJob) -> None:
        every = config.progress_log_every
        if every <= 0:
            return
        if completed != total_jobs and completed % every != 0:
            return

        print(
            (
                f"{desc}: {completed}/{total_jobs} jobs "
                f"experiment={job.experiment} "
                f"scenario={job.scenario.scenario_id} "
                f"seed={job.seed_id} "
                f"method={job.method}"
            ),
            file=sys.stderr,
            flush=True,
        )

    def mark_job_done(desc: str, progress: object, job: RunJob) -> None:
        nonlocal completed_jobs
        completed_jobs += 1
        update_progress_postfix(progress, job)
        emit_progress_line(desc, completed_jobs, job)

    def run_jobs_sequential(desc: str) -> None:
        iterator: Iterable[RunJob] = tqdm(
            jobs,
            total=total_jobs,
            desc=desc,
            unit="job",
            dynamic_ncols=True,
        )
        for job in iterator:
            records.extend(run_job(job, config))
            mark_job_done(desc, iterator, job)

    if n_jobs == 1:
        run_jobs_sequential(f"{output_prefix} sequential")
    else:
        try:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = {
                    executor.submit(run_job, job, config): job
                    for job in jobs
                }
                iterator = tqdm(
                    as_completed(futures),
                    total=total_jobs,
                    desc=f"{output_prefix} parallel",
                    unit="job",
                    dynamic_ncols=True,
                )
                for future in iterator:
                    job = futures[future]
                    records.extend(future.result())
                    mark_job_done(f"{output_prefix} parallel", iterator, job)
        except OSError as exc:
            records.clear()
            completed_jobs = 0
            actual_n_jobs = 1
            parallel_fallback_error = f"{type(exc).__name__}: {exc}"
            run_jobs_sequential(f"{output_prefix} sequential fallback")

    records_df = pd.DataFrame(records)
    sort_cols = ["experiment", "scenario_id", "method", "seed", "outer_k"]
    if not records_df.empty:
        records_df = records_df.sort_values(sort_cols).reset_index(drop=True)

    summary_df = summarize_records(records_df, config)
    final_df = final_success_summary(records_df[records_df["experiment"].astype(str) == "6"]) if not records_df.empty else pd.DataFrame()

    records_path = output_dir / f"{output_prefix}_records.csv"
    summary_path = output_dir / f"{output_prefix}_summary.csv"
    final_path = output_dir / f"{output_prefix}_final_success.csv"
    manifest_path = output_dir / f"{output_prefix}_manifest.json"
    records_df.to_csv(records_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    final_df.to_csv(final_path, index=False)

    plot_paths = save_plots(records_df, summary_df, final_df, plots_dir, output_prefix=output_prefix)
    elapsed = time.perf_counter() - started
    manifest = {
        "experiment": experiment_label,
        "experiments": list(config.experiments),
        "synthetic_experiment_6_from_final_success_protocol": True,
        "n_jobs": actual_n_jobs,
        "requested_n_jobs": requested_n_jobs,
        "parallel_fallback_error": parallel_fallback_error,
        "jobs": len(jobs),
        "records": int(len(records_df)),
        "elapsed_sec": elapsed,
        "config": config_to_json(config),
        "outputs": {
            "records": str(records_path),
            "summary": str(summary_path),
            "final_success": str(final_path),
            "plots": {name: str(path) for name, path in plot_paths.items()},
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    saved = {
        "records": records_path,
        "summary": summary_path,
        "final_success": final_path,
        "manifest": manifest_path,
    }
    saved.update(plot_paths)
    return saved


def config_to_json(config: ConfirmatoryConfig) -> dict[str, object]:
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in config.__dict__.items()
    }


def save_plots(
    records: pd.DataFrame,
    summary: pd.DataFrame,
    final_success: pd.DataFrame,
    output_dir: Path,
    *,
    output_prefix: str = "confirmatory_456",
) -> dict[str, Path]:
    saved: dict[str, Path] = {}
    saved["rho_plot"] = save_rho_plot(records, output_dir / f"{output_prefix}_rho_by_outer.png")
    saved["h_plot"] = save_h_plot(records, output_dir / f"{output_prefix}_h_by_outer.png")
    saved["mass_plot"] = save_mass_plot(records, output_dir / f"{output_prefix}_local_mass_by_outer.png")
    saved["rho_cos_scatter_plot"] = save_rho_cos_scatter_plot(records, output_dir / f"{output_prefix}_rho_vs_cos.png")
    saved["cos_plot"] = save_cos_plot(records, output_dir / f"{output_prefix}_cos_by_outer.png")
    saved["success_plot"] = save_success_plot(records, output_dir / f"{output_prefix}_success08_by_outer.png")
    saved["failure_plot"] = save_failure_plot(records, output_dir / f"{output_prefix}_failure_by_d.png")
    saved["ablation_plot"] = save_ablation_plot(summary, output_dir / f"{output_prefix}_ablation_final_cos.png")
    saved["final_success_plot"] = save_final_success_plot(final_success, output_dir / f"{output_prefix}_final_success.png")
    return saved


def save_rho_plot(records: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    frame = records[(records["experiment"].astype(str) == "4") & records["rho_k"].notna()] if not records.empty else records
    if frame is not None and not frame.empty:
        grouped = (
            frame.groupby("outer_k")
            .agg(rho_median=("rho_k", "median"), rho_q25=("rho_k", lambda x: np.quantile(x, 0.25)), rho_q75=("rho_k", lambda x: np.quantile(x, 0.75)))
            .reset_index()
            .sort_values("outer_k")
        )
        ax.plot(grouped["outer_k"], grouped["rho_median"], marker="o", linewidth=2.0)
        ax.fill_between(grouped["outer_k"], grouped["rho_q25"], grouped["rho_q75"], alpha=0.2)
    ax.set_xlabel("outer k")
    ax.set_ylabel(r"median $\rho_k$")
    ax.set_title(r"Эксперимент 4: динамика $\rho_k$")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_h_plot(records: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    frame = records[records["experiment"].astype(str) == "4"] if not records.empty else records
    if frame is not None and not frame.empty:
        grouped = (
            frame.groupby("outer_k")
            .agg(h_median=("h_k", "median"), h_q25=("h_k", lambda x: np.nanquantile(x, 0.25)), h_q75=("h_k", lambda x: np.nanquantile(x, 0.75)))
            .reset_index()
            .sort_values("outer_k")
        )
        ax.plot(grouped["outer_k"], grouped["h_median"], marker="o", linewidth=2.0)
        ax.fill_between(grouped["outer_k"], grouped["h_q25"], grouped["h_q75"], alpha=0.2)
    ax.set_xlabel("outer k")
    ax.set_ylabel(r"median $h_k$")
    ax.set_title(r"Эксперимент 4: динамика $h_k$")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_mass_plot(records: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    frame = records[records["experiment"].astype(str) == "4"] if not records.empty else records
    if frame is not None and not frame.empty:
        grouped = (
            frame.groupby("outer_k")
            .agg(q05=("local_mass_q05", "median"), mean_mass=("local_mass_mean", "median"), n_min=("n_min", "median"), n_directions=("n_directions", "median"))
            .reset_index()
            .sort_values("outer_k")
        )
        ax.plot(grouped["outer_k"], grouped["q05"], marker="o", linewidth=2.0, label="median Q05 mass")
        ax.plot(grouped["outer_k"], grouped["mean_mass"], marker="s", linewidth=1.5, label="median mean mass")
        if not grouped.empty:
            ax.axhline(float(grouped["n_min"].iloc[0]), linestyle="--", linewidth=1.0, color="black", alpha=0.45, label="n_min")
            ax.axhline(float(grouped["n_directions"].iloc[0]) + 4.0, linestyle=":", linewidth=1.0, color="black", alpha=0.45, label="n_phi + 4")
        ax.legend()
    ax.set_xlabel("outer k")
    ax.set_ylabel("local mass")
    ax.set_title("Эксперимент 4: локальная масса по итерациям")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_rho_cos_scatter_plot(records: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    frame = records[(records["experiment"].astype(str) == "4") & records["rho_k"].notna() & records["cos_beta_k"].notna()] if not records.empty else records
    if frame is not None and not frame.empty:
        scatter = ax.scatter(frame["rho_k"], frame["cos_beta_k"], c=frame["outer_k"], cmap="viridis", alpha=0.7)
        fig.colorbar(scatter, ax=ax, label="outer k")
    ax.set_xlabel(r"$\rho_k$")
    ax.set_ylabel(r"$|\cos(\beta_k,\beta^*)|$")
    ax.set_title(r"Эксперимент 4: связь $\rho_k$ и качества")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_cos_plot(records: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    frame = records[records["experiment"].astype(str) == "5"] if not records.empty else records
    if frame is not None and not frame.empty:
        grouped = (
            frame.groupby(["method", "outer_k"])
            .agg(cos_median=("cos_beta_k", "median"))
            .reset_index()
            .sort_values(["method", "outer_k"])
        )
        for method, group in grouped.groupby("method"):
            ax.plot(group["outer_k"], group["cos_median"], marker="o", linewidth=2.0, label=method)
        ax.legend()
    ax.axhline(0.8, linestyle="--", linewidth=1.0, color="black", alpha=0.45)
    ax.axhline(0.9, linestyle=":", linewidth=1.0, color="black", alpha=0.45)
    ax.set_xlabel("outer k")
    ax.set_ylabel(r"median $|\cos(\beta_k,\beta^*)|$")
    ax.set_title(r"Эксперимент 5: рост качества по итерациям")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_success_plot(records: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    frame = records[records["experiment"].astype(str) == "5"] if not records.empty else records
    if frame is not None and not frame.empty:
        grouped = (
            frame.groupby(["method", "outer_k"])
            .agg(success=("success_08", "mean"))
            .reset_index()
            .sort_values(["method", "outer_k"])
        )
        for method, group in grouped.groupby("method"):
            ax.plot(group["outer_k"], group["success"], marker="o", linewidth=2.0, label=method)
        ax.legend()
    ax.axhline(0.8, linestyle="--", linewidth=1.0, color="black", alpha=0.45)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("outer k")
    ax.set_ylabel(r"$Pr(c_k \geq 0.8)$")
    ax.set_title("Эксперимент 5: success rate по итерациям")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_failure_plot(records: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    frame = final_outer_rows(records[records["experiment"].astype(str) == "5"]) if not records.empty else records
    if frame is not None and not frame.empty:
        grouped = (
            frame.groupby("d")
            .agg(failure=("failed", "mean"))
            .reset_index()
            .sort_values("d")
        )
        ax.bar(grouped["d"].astype(str), grouped["failure"])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("d")
    ax.set_ylabel("failure rate")
    ax.set_title("Эксперимент 5: failure rate по размерности")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_ablation_plot(summary: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    frame = summary[summary["experiment"].astype(str) == "5"] if not summary.empty else summary
    if frame is not None and not frame.empty:
        grouped = (
            frame.groupby("method")
            .agg(cos=("cos_beta_final_median", "median"))
            .reset_index()
            .sort_values("cos", ascending=False)
        )
        ax.bar(grouped["method"], grouped["cos"])
        ax.tick_params(axis="x", rotation=25)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(r"median final $|\cos|$")
    ax.set_title("Эксперимент 5: full ADP и ablation-режимы")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_final_success_plot(final_success: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    if not final_success.empty:
        counts = final_success["protocol_pass"].map(bool).value_counts()
        labels = ["pass", "fail"]
        values = [int(counts.get(True, 0)), int(counts.get(False, 0))]
        ax.bar(labels, values)
    ax.set_ylabel("scenario count")
    ax.set_title("Эксперимент 6: финальный протокол успеха")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def build_common_parser(
    *,
    description: str,
    default_out: Path,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument("--d", type=str, default="50,100,200")
    parser.add_argument("--n-over-d", type=str, default="10,20")
    parser.add_argument("--corr", type=str, default="0.0,0.3,0.7")
    parser.add_argument("--snr", type=str, default="20,10,5")
    parser.add_argument("--links", type=str, default="linear,tanh,sin")
    parser.add_argument("--q", type=str, default="0.3,1.0")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=20260709)
    parser.add_argument("--n-directions", type=int, default=64)
    parser.add_argument("--min-neighbors", type=float, default=256.0)
    parser.add_argument("--center-fraction", type=float, default=0.5)
    parser.add_argument("--lambda-rel", type=float, default=1e-2)
    parser.add_argument("--outer-steps", type=int, default=8)
    parser.add_argument("--inner-steps", type=int, default=20)
    parser.add_argument("--gamma-h", type=float, default=0.8)
    parser.add_argument("--methods", type=str, default="full_adp,step0_only,no_anisotropy,fixed_h,no_regularization,random_beta_init")
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--backend", choices=("numpy", "cupy"), default="numpy")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument(
        "--progress-log-every",
        type=int,
        default=1,
        help="Write one newline progress record to stderr every N completed jobs; use 0 to disable.",
    )
    return parser


def config_from_args(
    args: argparse.Namespace,
    *,
    experiments: tuple[ExperimentName, ...],
) -> ConfirmatoryConfig:
    return ConfirmatoryConfig(
        d_values=parse_int_tuple(args.d),
        n_over_d_values=parse_int_tuple(args.n_over_d),
        corr_values=parse_float_tuple(args.corr),
        snr_values=parse_float_tuple(args.snr),
        link_values=parse_str_tuple(args.links),
        q_values=parse_float_tuple(args.q),
        seeds=args.seeds,
        base_seed=args.base_seed,
        n_directions=args.n_directions,
        min_neighbors=args.min_neighbors,
        center_fraction=args.center_fraction,
        lambda_rel=args.lambda_rel,
        outer_steps=args.outer_steps,
        inner_steps=args.inner_steps,
        gamma_h=args.gamma_h,
        methods=parse_str_tuple(args.methods),  # type: ignore[arg-type]
        experiments=experiments,
        max_scenarios=args.max_scenarios,
        bootstrap_reps=args.bootstrap_reps,
        backend=args.backend,
        show_progress=args.show_progress,
        progress_log_every=args.progress_log_every,
    )


def configure_live_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True, write_through=True)


def run_experiment_cli(
    *,
    description: str,
    default_out: Path,
    experiments: tuple[ExperimentName, ...],
    output_prefix: str,
    experiment_label: str,
) -> int:
    configure_live_output()
    parser = build_common_parser(description=description, default_out=default_out)
    args = parser.parse_args()
    config = config_from_args(args, experiments=experiments)

    print(experiment_label, flush=True)
    print(f"experiments = {config.experiments}", flush=True)
    print(f"d = {config.d_values}", flush=True)
    print(f"n/d = {config.n_over_d_values}", flush=True)
    print(f"seeds = {config.seeds}", flush=True)
    print(f"methods = {config.methods}", flush=True)
    print(f"jobs = {args.jobs if args.jobs is not None else 'cpu_count - 1'}", flush=True)
    print(f"output = {args.out}", flush=True)
    print(f"progress_log_every = {config.progress_log_every}", flush=True)

    saved = run_confirmatory_experiments(
        config,
        args.out,
        n_jobs=args.jobs,
        output_prefix=output_prefix,
        experiment_label=experiment_label,
    )
    print("\nSaved files:", flush=True)
    for name, path in saved.items():
        print(f"{name:24s} {path}", flush=True)
    return 0
