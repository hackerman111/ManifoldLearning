from __future__ import annotations

import math

import numpy as np

from ...common.resource_monitor import ResourceMonitor
from ...common.types import ADPConfig
from ...engine.base import ADP
from ..metrics import direction_metrics
from .baselines import fit_baseline
from .correctness import run_correctness
from .datasets import DatasetUnavailable, generate_synthetic_data, load_cached_real_dataset
from .types import RunOutcome, SingleIndexJob, SingleIndexSeriesConfig


def execute_job(
    job: SingleIndexJob,
    config: SingleIndexSeriesConfig,
) -> RunOutcome:
    if job.scenario.executor == "correctness":
        return run_correctness(job)
    if job.scenario.executor == "real_data":
        return _execute_real_data(job, config)
    return _execute_recovery(job, config)


def _execute_recovery(
    job: SingleIndexJob,
    config: SingleIndexSeriesConfig,
) -> RunOutcome:
    data = generate_synthetic_data(job)
    if job.method in {
        "random_direction",
        "ols",
        "statsmodels_sir",
        "statsmodels_save",
        "statsmodels_phd",
        "sklearn_pls",
        "opg",
        "ade",
        "mave",
        "rmave",
    }:
        monitor = ResourceMonitor()
        with monitor:
            beta = fit_baseline(job.method, data.X, data.y, seed=job.seeds.init)
        return RunOutcome(
            metrics=direction_metrics(beta, data.beta),
            iterations=(),
            solver_iterations=(),
            stop_reason="complete",
            algorithm_usage=monitor.usage.to_dict("algorithm"),
        )
    if job.method == "negative_control":
        beta = fit_baseline("random_direction", data.X, data.y, seed=job.seeds.init)
        return RunOutcome(
            metrics=direction_metrics(beta, data.beta),
            iterations=(),
            solver_iterations=(),
            stop_reason="negative_control",
        )

    adp_config, directions, stage_factories = _adp_configuration(job, config, data)
    model = ADP.create("new", adp_config, stage_factories=stage_factories)
    result = model.fit(
        data.X,
        data.y,
        centers=data.centers,
        directions=directions,
    )
    metrics = {
        **direction_metrics(result.beta, data.beta),
        "objective": float(result.objective),
    }
    iterations = _iteration_rows(result, data.beta)
    solver_iterations = _solver_rows(result) if job.scenario.record_solver_trace else ()
    return RunOutcome(
        metrics=metrics,
        iterations=iterations,
        solver_iterations=solver_iterations,
        stop_reason="complete",
        algorithm_usage=dict(result.resource_usage),
    )


def _execute_real_data(
    job: SingleIndexJob,
    config: SingleIndexSeriesConfig,
) -> RunOutcome:
    if config.data_dir is None:
        raise DatasetUnavailable("real-data executor requires --data-dir")
    dataset = load_cached_real_dataset(
        job.scenario.scenario_id,
        config.data_dir,
        allow_download=config.allow_download,
    )
    monitor = ResourceMonitor()
    with monitor:
        if job.method == "full_adp":
            n, d = dataset.X.shape
            model = ADP.create(
                "new",
                ADPConfig(
                    n_centers=min(n, max(10, n // 4)),
                    n_directions=min(32, max(4, d)),
                    min_neighbors=min(64.0, max(5.0, n / 10.0)),
                    outer_steps=4,
                    inner_steps=10,
                    statistics_workers=config.statistics_workers,
                    random_state=job.seeds.init,
                    show_progress=False,
                ),
            )
            result = model.fit(dataset.X, dataset.y)
            beta = result.beta
            objective = float(result.objective)
        else:
            beta = fit_baseline(job.method, dataset.X, dataset.y, seed=job.seeds.init)
            objective = math.nan
    return RunOutcome(
        metrics={
            "n": int(dataset.X.shape[0]),
            "d": int(dataset.X.shape[1]),
            "beta_norm": float(np.linalg.norm(beta)),
            "objective": objective,
            "dataset_sha256": dataset.sha256,
        },
        iterations=(),
        solver_iterations=(),
        stop_reason="complete",
        algorithm_usage=monitor.usage.to_dict("algorithm"),
    )


def _adp_configuration(job, config, data):
    scenario = job.scenario
    algorithm = scenario.algorithm
    solver = scenario.solver
    method = job.method
    outer_steps = int(solver.get("outer_steps", 4))
    inner_steps = int(solver.get("inner_steps", 20))
    bandwidth_decay = float(algorithm.get("bandwidth_decay", math.sqrt(2.0)))
    renew_directions = bool(algorithm.get("renew_directions", True))
    local_mass_mode = str(algorithm.get("local_mass_mode", "quantile"))
    lambda_penalty = algorithm.get("lambda_penalty")
    directions = data.directions
    stage_factories = None

    if method == "step0_only":
        outer_steps = 1
    elif method == "fixed_h":
        bandwidth_decay = 1.0
    elif method == "fixed_directions":
        renew_directions = False
    elif method == "no_regularization":
        lambda_penalty = 0.0
    elif method == "mean_mass":
        local_mass_mode = "mean"
    elif method == "full_directional_basis":
        dimension = data.X.shape[1]
        directions = np.broadcast_to(
            np.eye(dimension)[None, :, :],
            (data.centers.shape[0], dimension, dimension),
        ).copy()
    elif method == "no_anisotropy":
        stage_factories = {"bandwidth_selector": _isotropic_selector_factory}

    n_directions = directions.shape[1] if directions is not None else int(
        algorithm.get("n_directions", 8)
    )
    adp_config = ADPConfig(
        n_centers=int(algorithm.get("n_centers", data.centers.shape[0])),
        n_directions=n_directions,
        min_neighbors=float(algorithm.get("min_neighbors", 10.0)),
        lambda_penalty=None if lambda_penalty is None else float(lambda_penalty),
        outer_steps=outer_steps,
        inner_steps=inner_steps,
        bandwidth_decay=bandwidth_decay,
        local_mass_mode=local_mass_mode,
        renew_directions=renew_directions,
        center_noise_scale=0.0,
        statistics_workers=config.statistics_workers,
        random_state=job.seeds.init,
        show_progress=False,
    )
    return adp_config, directions, stage_factories


class _IsotropicBandwidthSelector:
    def __init__(self, context) -> None:
        self.model = context.model

    def select_initial(self, X, centers, index=None):
        return self.model._select_isotropic_bandwidth_default(X, centers, index)

    def select_anisotropy(self, X, centers, h, beta):
        return 1.0


def _isotropic_selector_factory(context):
    return _IsotropicBandwidthSelector(context)


def _iteration_rows(result, beta_true):
    rows = []
    for index, beta in enumerate(result.beta_path):
        record = result.progress[index] if index < len(result.progress) else {}
        rows.append(
            {
                "outer_k": int(record.get("outer", index + 1)) - 1,
                "h_k": float(record.get("h", math.nan)),
                "rho_k": float(record.get("rho", math.nan)),
                "local_mass_mean": float(record.get("local_mass_mean", math.nan)),
                "local_mass_q05": float(record.get("local_mass_q05", math.nan)),
                "local_mass_min": float(record.get("local_mass_min", math.nan)),
                "objective": float(record.get("objective", math.nan)),
                "cosine_abs": direction_metrics(beta, beta_true)["cosine_abs"],
                "beta_delta": float(record.get("delta", math.nan)),
                "statistics_time_sec": float(result.timings.get("statistics", math.nan)),
                "solve_time_sec": float(result.timings.get("solve", math.nan)),
                "runtime_sec": float(record.get("elapsed", math.nan)),
            }
        )
    return tuple(rows)


def _solver_rows(result):
    initial = abs(float(result.history[0].objective)) if result.history else 1.0
    scale = max(initial, np.finfo(float).eps)
    return tuple(
        {
            "outer_k": int(step.outer),
            "inner_k": int(step.inner),
            "cg_k": -1,
            "relative_objective": float(step.objective) / scale,
            "relative_residual": math.nan,
            "projective_delta": float(step.beta_delta),
            "cg_info": 0,
        }
        for step in result.history
    )


__all__ = ["RunOutcome", "execute_job"]
