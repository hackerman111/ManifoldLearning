from collections import Counter
from dataclasses import replace
import os

import pandas as pd

import adp.evaluation.single_index.runner as single_index_runner
from adp.evaluation.single_index.runner import (
    _initialize_worker,
    _mark_job_done,
    build_single_index_jobs,
    run_single_index_benchmark,
)
from adp.evaluation.single_index.scenarios import EXPERIMENT_COUNTS
from adp.evaluation.single_index.types import (
    ExperimentParameters,
    SingleIndexSeriesConfig,
)


def test_full_profile_expands_to_exactly_24000_jobs():
    jobs = build_single_index_jobs(SingleIndexSeriesConfig(profile="full"))

    assert len(jobs) == 24_000
    assert Counter(job.experiment for job in jobs) == EXPERIMENT_COUNTS
    assert {job.seed for job in jobs} == set(range(100))
    assert all(job.parameters.n_centers == job.parameters.n for job in jobs)


def test_parameter_families_are_not_cross_multiplied():
    jobs = build_single_index_jobs(
        SingleIndexSeriesConfig(profile="full", experiments=("3",), seeds=(7,))
    )

    assert len(jobs) == 42
    assert {job.experiment for job in jobs} == {"3"}
    assert {job.parameters.rho_corr for job in jobs} == {0.0}
    assert {job.parameters.sigma_x for job in jobs} == {1.0}
    assert {job.parameters.link for job in jobs} == {"quadratic"}
    assert {job.parameters.sigma_eps for job in jobs} == {
        0.0,
        0.316,
        0.5,
        0.707,
        1.0,
        1.414,
        2.0,
    }


def test_job_ids_and_subseeds_do_not_depend_on_process_count_or_order():
    config = SingleIndexSeriesConfig(
        profile="full",
        experiments=("1", "8.3"),
        seeds=(2, 9),
        jobs=1,
    )
    serial = build_single_index_jobs(config)
    parallel = build_single_index_jobs(replace(config, jobs=8))
    reversed_selectors = build_single_index_jobs(
        replace(config, experiments=("8.3", "1"))
    )

    serial_identity = {
        (job.experiment, job.parameters, job.seed): (job.run_id, job.seeds)
        for job in serial
    }
    parallel_identity = {
        (job.experiment, job.parameters, job.seed): (job.run_id, job.seeds)
        for job in parallel
    }
    reversed_identity = {
        (job.experiment, job.parameters, job.seed): (job.run_id, job.seeds)
        for job in reversed_selectors
    }
    assert serial_identity == parallel_identity == reversed_identity


def test_job_identity_canonicalizes_signed_zero_parameters():
    positive_zero = ExperimentParameters(
        d=25,
        n_over_d=5,
        rho_corr=0.0,
        sigma_eps=0.0,
        outlier_fraction=0.0,
        delta=0.0,
    )
    negative_zero = ExperimentParameters(
        d=25,
        n_over_d=5,
        rho_corr=-0.0,
        sigma_eps=-0.0,
        outlier_fraction=-0.0,
        delta=-0.0,
    )

    positive_job = single_index_runner._build_single_index_job(
        "3",
        positive_zero,
        7,
        diagnostic=False,
    )
    negative_job = single_index_runner._build_single_index_job(
        "3",
        negative_zero,
        7,
        diagnostic=True,
    )

    assert negative_zero == positive_zero
    assert negative_job.run_id == positive_job.run_id
    assert negative_job.seeds == positive_job.seeds


def test_center_fraction_overrides_jobs_without_expanding_matrix():
    base = build_single_index_jobs(
        SingleIndexSeriesConfig(profile="full", experiments=("2",), seeds=(0,))
    )
    quarter = build_single_index_jobs(
        SingleIndexSeriesConfig(
            profile="full",
            experiments=("2",),
            seeds=(0,),
            center_fraction=0.25,
        )
    )

    assert len(base) == len(quarter) == 20
    assert all(job.parameters.center_fraction == 0.25 for job in quarter)
    assert all(job.parameters.n_centers <= job.parameters.n for job in quarter)


def test_smoke_and_max_runs_are_deterministic_post_expansion_limits():
    smoke = build_single_index_jobs(SingleIndexSeriesConfig(profile="smoke"))
    limited = build_single_index_jobs(
        SingleIndexSeriesConfig(profile="smoke", max_runs=3)
    )

    assert len(smoke) == 11
    assert limited == smoke[:3]
    assert {job.seed for job in smoke} == {0}


def test_worker_initializer_caps_every_supported_runtime(monkeypatch):
    variables = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    for name in variables:
        monkeypatch.delenv(name, raising=False)

    _initialize_worker()

    for name in variables:
        assert os.environ[name] == "1"


class _ProgressRecorder:
    def __init__(self, *, disable):
        self.disable = disable
        self.postfix = None
        self.updated = 0

    def set_postfix(self, values, refresh=True):
        self.postfix = values

    def update(self, amount):
        self.updated += amount


class _CommittedStore:
    def __init__(self, run_id):
        self.run_id = run_id

    def completed_run_ids(self):
        return {self.run_id}


def test_interactive_tqdm_updates_without_printing_a_new_line(capsys):
    job = build_single_index_jobs(
        SingleIndexSeriesConfig(profile="smoke", max_runs=1)
    )[0]
    progress = _ProgressRecorder(disable=False)

    completed = _mark_job_done(
        _CommittedStore(job.run_id), progress, 0, 1, job, "success"
    )

    assert completed == 1
    assert progress.updated == 1
    assert capsys.readouterr().err == ""


def test_disabled_tqdm_keeps_line_oriented_progress_for_redirected_logs(capsys):
    job = build_single_index_jobs(
        SingleIndexSeriesConfig(profile="smoke", max_runs=1)
    )[0]
    progress = _ProgressRecorder(disable=True)

    _mark_job_done(
        _CommittedStore(job.run_id), progress, 0, 1, job, "success"
    )

    assert capsys.readouterr().err == (
        "1/1 experiment=1 seed=0 status=success\n"
    )


def test_parallel_runner_commits_one_normalized_outcome_per_fit(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        single_index_runner,
        "write_single_index_reports",
        lambda series_dir: pd.DataFrame(),
    )
    config = SingleIndexSeriesConfig(
        profile="smoke",
        experiments=("1", "3"),
        jobs=2,
        seeds=(0,),
        diagnostic_seeds=(0,),
        center_fraction=0.25,
        max_runs=2,
    )

    saved = run_single_index_benchmark(config, tmp_path)

    runs = pd.read_csv(saved["run_summary"])
    assert len(runs) == 2
    assert set(runs["run_id"]) == {
        job.run_id for job in build_single_index_jobs(config)
    }
    assert set(runs["statistics_workers"]) == {1}
    assert set(runs["status"]) <= {
        "success",
        "nonconverged",
        "numerical_failure",
    }
