from __future__ import annotations

import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ...common.experiment_log import configuration_fingerprint, stable_run_id
from ...common.resource_monitor import ResourceMonitor
from .baselines import BaselineUnavailable
from .datasets import DatasetUnavailable
from .executors import execute_job
from .scenarios import scenarios_for_profile
from .storage import SingleIndexSeriesStore
from .types import RunOutcome, SeedBundle, SingleIndexJob, SingleIndexSeriesConfig


def build_single_index_jobs(
    config: SingleIndexSeriesConfig,
) -> tuple[SingleIndexJob, ...]:
    """Expand a profile into deterministic jobs with paired method seeds."""

    scenarios = scenarios_for_profile(config.profile)
    if config.max_scenarios is not None:
        scenarios = scenarios[: config.max_scenarios]
    fingerprint_values = asdict(config)
    fingerprint_values.pop("retry_failed", None)
    fingerprint_values.pop("allow_download", None)
    config_fingerprint = configuration_fingerprint(fingerprint_values)
    jobs: list[SingleIndexJob] = []
    for scenario_index, scenario in enumerate(scenarios):
        for repeat in range(scenario.repeats):
            state = np.random.SeedSequence(
                [config.base_seed, scenario_index, repeat]
            ).generate_state(5)
            seeds = SeedBundle(*(int(value) for value in state))
            for method in scenario.methods:
                run_fingerprint = configuration_fingerprint(
                    {
                        "series": config_fingerprint,
                        "repeat": repeat,
                        "seeds": asdict(seeds),
                    }
                )
                jobs.append(
                    SingleIndexJob(
                        scenario=scenario,
                        method=method,
                        repeat=repeat,
                        seeds=seeds,
                        run_id=stable_run_id(
                            "single_index",
                            scenario.scenario_id,
                            method,
                            seeds.data,
                            config_fingerprint=run_fingerprint,
                        ),
                    )
                )
    return tuple(jobs)


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

    if config.jobs == 1:
        completed = _run_serial(store, pending, config, completed, total)
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
                    completed += 1
                    _log_progress(completed, total, job)
        except OSError as exc:
            print(
                f"parallel fallback: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            remaining = list(store.pending_jobs(jobs))
            completed = total - len(remaining)
            _run_serial(store, remaining, config, completed, total)

    statuses = store._committed_statuses()
    final_status = (
        "complete"
        if jobs and all(statuses.get(job.run_id) == "success" for job in jobs)
        else "partial"
    )
    return store.finalize(status=final_status)


def _run_serial(
    store: SingleIndexSeriesStore,
    jobs: Sequence[SingleIndexJob],
    config: SingleIndexSeriesConfig,
    completed: int,
    total: int,
) -> int:
    for job in jobs:
        _execute_and_persist(store, job, config)
        completed += 1
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
