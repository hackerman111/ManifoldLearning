from __future__ import annotations

import os
import sys
from collections import Counter
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from tqdm.auto import tqdm

from ...common.experiment_log import configuration_fingerprint, stable_run_id
from .executors import execute_job
from .reports import write_single_index_reports
from .scenarios import full_parameter_grid, smoke_parameter_grid
from .storage import SingleIndexSeriesStore
from .types import (
    EXPERIMENT_SELECTORS,
    ExperimentParameters,
    SeedBundle,
    SingleIndexJob,
    SingleIndexSeriesConfig,
)


_THREAD_LIMIT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
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
    dry_run: bool = False,
    reports_only: bool = False,
) -> Mapping[str, Path]:
    """Run, resume, inspect, or rerender one benchmark series."""

    if dry_run and reports_only:
        raise ValueError("dry-run and reports-only are mutually exclusive")
    if reports_only and resume is None:
        raise ValueError("reports-only requires a resume series")

    jobs = build_single_index_jobs(config)
    if dry_run:
        _print_dry_run(jobs)
        return {}

    store = (
        SingleIndexSeriesStore.resume(Path(resume), config)
        if resume is not None
        else SingleIndexSeriesStore.create(Path(output_root), config, jobs)
    )
    pending = list(store.pending_jobs(jobs))
    if reports_only:
        saved = dict(store.finalize(status=_series_status(store, jobs)))
        write_single_index_reports(store.series_dir)
        return saved

    total = len(pending)
    completed = 0
    with tqdm(
        total=total,
        desc="single-index",
        unit="fit",
        dynamic_ncols=True,
    ) as progress:
        process_jobs = _resolve_process_jobs(config.jobs)
        if process_jobs == 1:
            completed = _run_serial(
                store,
                pending,
                config,
                completed=completed,
                total=total,
                progress=progress,
            )
        elif pending:
            completed = _run_parallel(
                store,
                pending,
                config,
                process_jobs=process_jobs,
                completed=completed,
                total=total,
                progress=progress,
            )

    saved = dict(store.finalize(status=_series_status(store, jobs)))
    write_single_index_reports(store.series_dir)
    return saved


def _run_serial(
    store: SingleIndexSeriesStore,
    jobs: Sequence[SingleIndexJob],
    config: SingleIndexSeriesConfig,
    *,
    completed: int,
    total: int,
    progress: Any,
) -> int:
    _limit_worker_threads()
    for job in jobs:
        outcome = execute_job(job, config)
        store.commit(outcome)
        completed = _mark_job_done(
            store,
            progress,
            completed,
            total,
            job,
            str(outcome.run_row["status"]),
        )
    return completed


def _run_parallel(
    store: SingleIndexSeriesStore,
    jobs: Sequence[SingleIndexJob],
    config: SingleIndexSeriesConfig,
    *,
    process_jobs: int,
    completed: int,
    total: int,
    progress: Any,
) -> int:
    _limit_worker_threads()
    pool: ProcessPoolExecutor | None = None
    futures: dict[Future[tuple[str, str]], SingleIndexJob] = {}
    try:
        pool = ProcessPoolExecutor(
            max_workers=process_jobs,
            initializer=_initialize_worker,
        )
        for job in jobs:
            future = pool.submit(
                _execute_and_commit,
                store.series_dir,
                job,
                config,
            )
            futures[future] = job
    except OSError as exc:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        print(
            f"parallel fallback: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        remaining = list(store.pending_jobs(jobs))
        return _run_serial(
            store,
            remaining,
            config,
            completed=total - len(remaining),
            total=total,
            progress=progress,
        )

    assert pool is not None
    try:
        for future in as_completed(futures):
            job = futures[future]
            run_id, status = future.result()
            if run_id != job.run_id:
                raise RuntimeError(
                    f"worker returned run_id {run_id!r}, expected {job.run_id!r}"
                )
            completed = _mark_job_done(
                store,
                progress,
                completed,
                total,
                job,
                status,
            )
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return completed


def _execute_and_commit(
    series_dir: Path,
    job: SingleIndexJob,
    config: SingleIndexSeriesConfig,
) -> tuple[str, str]:
    store = SingleIndexSeriesStore.resume(series_dir, config)
    outcome = execute_job(job, config)
    store.commit(outcome)
    status = str(outcome.run_row["status"])
    if job.run_id not in store.completed_run_ids():
        raise RuntimeError(f"commit marker was not published for {job.run_id}")
    return job.run_id, status


def _mark_job_done(
    store: SingleIndexSeriesStore,
    progress: Any,
    completed: int,
    total: int,
    job: SingleIndexJob,
    status: str,
) -> int:
    if job.run_id not in store.completed_run_ids():
        raise RuntimeError(f"progress advanced before commit marker for {job.run_id}")
    completed += 1
    progress.set_postfix(
        {
            "experiment": job.experiment,
            "seed": job.seed,
            "status": status,
        },
        refresh=True,
    )
    progress.update(1)
    print(
        f"{completed}/{total} experiment={job.experiment} "
        f"seed={job.seed} status={status}",
        file=sys.stderr,
        flush=True,
    )
    return completed


def _series_status(
    store: SingleIndexSeriesStore,
    jobs: Sequence[SingleIndexJob],
) -> str:
    completed = store.completed_run_ids()
    return (
        "complete"
        if all(job.run_id in completed for job in jobs)
        else "partial"
    )


def _resolve_process_jobs(value: int | str) -> int:
    if value == "auto":
        return max(1, os.cpu_count() or 1)
    return int(value)


def _initialize_worker() -> None:
    _limit_worker_threads()


def _limit_worker_threads() -> None:
    for variable in _THREAD_LIMIT_VARIABLES:
        os.environ[variable] = "1"


def _print_dry_run(jobs: Sequence[SingleIndexJob]) -> None:
    counts = Counter(job.experiment for job in jobs)
    for experiment in EXPERIMENT_SELECTORS:
        if counts[experiment]:
            print(f"{experiment}: {counts[experiment]}")
    print(f"total: {len(jobs)}")


__all__ = [
    "build_single_index_jobs",
    "run_single_index_benchmark",
]
