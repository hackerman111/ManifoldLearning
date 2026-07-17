from __future__ import annotations

import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from tqdm.auto import tqdm

from ...common.experiment_log import configuration_fingerprint, stable_run_id
from ...common.resource_monitor import ResourceMonitor
from .baselines import BaselineUnavailable
from .datasets import DatasetUnavailable
from .executors import execute_job
from .reports import write_single_index_reports
from .scenarios import full_parameter_grid, smoke_parameter_grid
from .storage import SingleIndexSeriesStore
from .types import (
    EXPERIMENT_SELECTORS,
    ExperimentParameters,
    RunOutcome,
    SeedBundle,
    SingleIndexJob,
    SingleIndexSeriesConfig,
)


def build_single_index_jobs(
    config: SingleIndexSeriesConfig,
) -> tuple[SingleIndexJob, ...]:
    """Expand canonical experiment grids into deterministic independent jobs."""

    selected = set(config.experiments)
    seeds = config.seeds
    if seeds is None:
        seeds = tuple(range(100)) if config.profile == "full" else (0,)
    diagnostic_seeds = set(config.diagnostic_seeds)
    jobs: list[SingleIndexJob] = []
    grid_builder = (
        full_parameter_grid if config.profile == "full" else smoke_parameter_grid
    )
    for experiment in EXPERIMENT_SELECTORS:
        if experiment not in selected:
            continue
        for base_parameters in grid_builder(experiment):
            parameters = replace(
                base_parameters,
                center_fraction=config.center_fraction,
            )
            for seed in seeds:
                jobs.append(
                    _build_single_index_job(
                        experiment,
                        parameters,
                        seed,
                        diagnostic=seed in diagnostic_seeds,
                    )
                )
    expanded = tuple(jobs)
    if config.max_runs is not None:
        return expanded[: config.max_runs]
    return expanded


def _build_single_index_job(
    experiment: str,
    parameters: ExperimentParameters,
    seed: int,
    *,
    diagnostic: bool,
) -> SingleIndexJob:
    identity_fingerprint = configuration_fingerprint(
        {
            "experiment": experiment,
            "parameters": asdict(parameters),
            "seed": seed,
        }
    )
    entropy = int(identity_fingerprint.removeprefix("cfg-"), 16)
    state = np.random.SeedSequence(entropy).generate_state(10)
    sub_seeds = SeedBundle(*(int(value) for value in state))
    return SingleIndexJob(
        experiment=experiment,
        parameters=parameters,
        seed=seed,
        seeds=sub_seeds,
        run_id=stable_run_id(
            "single_index",
            experiment,
            "adp",
            seed,
            config_fingerprint=identity_fingerprint,
        ),
        diagnostic=diagnostic,
    )


def run_single_index_benchmark(
    config: SingleIndexSeriesConfig,
    output_root: str | Path,
    *,
    resume: str | Path | None = None,
) -> Mapping[str, Path]:
    """Run or resume a normalized, crash-safe single-index benchmark series."""

    jobs = build_single_index_jobs(config)
    store = (
        SingleIndexSeriesStore.resume(Path(resume), config)
        if resume is not None
        else SingleIndexSeriesStore.create(Path(output_root), config, jobs)
    )
    pending = list(store.pending_jobs(jobs))
    total = len(pending)
    completed = 0

    with tqdm(
        total=total,
        desc="single-index",
        unit="job",
        dynamic_ncols=True,
    ) as progress:
        if config.jobs == 1:
            completed = _run_serial(
                store,
                pending,
                config,
                completed,
                total,
                progress,
            )
        elif pending:
            _limit_worker_threads()
            try:
                with ProcessPoolExecutor(max_workers=config.jobs) as pool:
                    futures = {
                        pool.submit(_execute_and_persist, store, job, config): job
                        for job in pending
                    }
                    for future in as_completed(futures):
                        job = futures[future]
                        future.result()
                        completed = _mark_job_done(
                            progress,
                            completed,
                            total,
                            job,
                        )
            except OSError as exc:
                print(
                    f"parallel fallback: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                remaining = list(store.pending_jobs(jobs))
                completed = total - len(remaining)
                if progress.n < completed:
                    progress.update(completed - progress.n)
                completed = _run_serial(
                    store,
                    remaining,
                    config,
                    completed,
                    total,
                    progress,
                )

    statuses = store._committed_statuses()
    final_status = (
        "complete"
        if jobs and all(statuses.get(job.run_id) == "success" for job in jobs)
        else "partial"
    )
    saved = dict(store.finalize(status=final_status))
    reports = write_single_index_reports(
        store.series_dir,
        random_state=config.base_seed,
    )
    return {**saved, **reports}


def _run_serial(
    store: SingleIndexSeriesStore,
    jobs: Sequence[SingleIndexJob],
    config: SingleIndexSeriesConfig,
    completed: int,
    total: int,
    progress: Any,
) -> int:
    for job in jobs:
        _execute_and_persist(store, job, config)
        completed = _mark_job_done(progress, completed, total, job)
    return completed


def _mark_job_done(
    progress: Any,
    completed: int,
    total: int,
    job: SingleIndexJob,
) -> int:
    completed += 1
    progress.set_postfix(
        {"scenario": job.scenario.scenario_id, "method": job.method},
        refresh=True,
    )
    progress.update(1)
    _log_progress(completed, total, job)
    return completed


def _execute_and_persist(
    store: SingleIndexSeriesStore,
    job: SingleIndexJob,
    config: SingleIndexSeriesConfig,
) -> None:
    outcome: RunOutcome | None = None
    status = "success"
    error = ""
    stage = "execute"
    persist_started = time.perf_counter()
    full_monitor = ResourceMonitor()

    with full_monitor:
        try:
            outcome = execute_job(job, config)
            stage = "persist"
            persist_started = time.perf_counter()
            iteration_rows = outcome.iterations or (_summary_iteration(outcome),)
            store.append_worker_rows(
                "iterations",
                (_iteration_row(job, row) for row in iteration_rows),
            )
            store.append_worker_rows(
                "solver_iterations",
                (_solver_iteration_row(job, row) for row in outcome.solver_iterations),
            )
        except (BaselineUnavailable, DatasetUnavailable) as exc:
            status = "unavailable"
            error = f"{type(exc).__name__}: {exc}"
            persist_started = time.perf_counter()
            _append_failure(store, job, status, exc, stage)
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            persist_started = time.perf_counter()
            _append_failure(store, job, status, exc, stage)

    metrics = outcome.metrics if outcome is not None and status == "success" else {}
    algorithm_usage = _resource_defaults("algorithm")
    if outcome is not None:
        algorithm_usage.update(outcome.algorithm_usage)
    full_usage = full_monitor.usage.to_dict("full_run")
    persist_time = max(time.perf_counter() - persist_started, np.finfo(float).eps)
    run_row: dict[str, Any] = {
        "run_id": job.run_id,
        "scenario_id": job.scenario.scenario_id,
        "family": job.scenario.family,
        "executor": job.scenario.executor,
        "method": job.method,
        "repeat": job.repeat,
        "data_seed": job.seeds.data,
        "beta_seed": job.seeds.beta,
        "centers_seed": job.seeds.centers,
        "directions_seed": job.seeds.directions,
        "init_seed": job.seeds.init,
        "status": status,
        "failed": status == "failed",
        "error": error,
        "stage": stage,
        "stop_reason": outcome.stop_reason if outcome is not None else status,
        "iteration_rows": len(outcome.iterations or ((),)) if outcome is not None else 0,
        "solver_iteration_rows": len(outcome.solver_iterations) if outcome is not None else 0,
        "result_persist_time_sec": persist_time,
        "cosine": metrics.get("cosine", math.nan),
        "cosine_abs": metrics.get("cosine_abs", math.nan),
        "angle_deg": metrics.get("angle_deg", math.nan),
        "signed_l2": metrics.get("signed_l2", math.nan),
        "objective": metrics.get("objective", math.nan),
        "dataset_source": metrics.get("dataset_source", ""),
        "dataset_path": metrics.get("dataset_path", ""),
        "dataset_size_bytes": metrics.get("dataset_size_bytes", math.nan),
        "dataset_sha256": metrics.get("dataset_sha256", ""),
        "dataset_rows": metrics.get("dataset_rows", math.nan),
        "dataset_features": metrics.get("dataset_features", math.nan),
        **algorithm_usage,
        **full_usage,
    }
    # The run row is the commit marker and must be the final write for this job.
    store.append_worker_rows("runs", (run_row,))


def _summary_iteration(outcome: RunOutcome) -> dict[str, Any]:
    return {
        "outer_k": -1,
        "objective": outcome.metrics.get("objective", math.nan),
        "cosine_abs": outcome.metrics.get("cosine_abs", math.nan),
        "runtime_sec": outcome.algorithm_usage.get("algorithm_time_sec", math.nan),
    }


def _iteration_row(job: SingleIndexJob, row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_id": job.run_id,
        "scenario_id": job.scenario.scenario_id,
        "method": job.method,
        **row,
    }


def _solver_iteration_row(
    job: SingleIndexJob,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": job.run_id,
        "scenario_id": job.scenario.scenario_id,
        "method": job.method,
        **row,
    }


def _append_failure(
    store: SingleIndexSeriesStore,
    job: SingleIndexJob,
    status: str,
    exc: Exception,
    stage: str,
) -> None:
    store.append_worker_rows(
        "failures",
        (
            {
                "run_id": job.run_id,
                "scenario_id": job.scenario.scenario_id,
                "method": job.method,
                "status": status,
                "category": "optional_dependency" if status == "unavailable" else "runtime",
                "exception_type": type(exc).__name__,
                "error": f"{type(exc).__name__}: {exc}",
                "stage": stage,
                "last_outer_k": -1,
                "last_inner_k": -1,
            },
        ),
    )


def _resource_defaults(prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_time_sec": math.nan,
        f"{prefix}_rss_start_mib": math.nan,
        f"{prefix}_rss_min_mib": math.nan,
        f"{prefix}_rss_mean_mib": math.nan,
        f"{prefix}_rss_max_mib": math.nan,
        f"{prefix}_rss_peak_delta_mib": math.nan,
        f"{prefix}_memory_samples": 0,
        f"{prefix}_memory_source": "unavailable",
    }


def _limit_worker_threads() -> None:
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"


def _log_progress(
    completed: int,
    total: int,
    job: SingleIndexJob,
) -> None:
    print(
        f"{completed}/{total} scenario={job.scenario.scenario_id} "
        f"method={job.method} seed={job.seeds.data}",
        file=sys.stderr,
        flush=True,
    )


__all__ = ["build_single_index_jobs", "run_single_index_benchmark"]
