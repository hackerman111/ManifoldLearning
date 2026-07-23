from collections import Counter
from dataclasses import replace
import os
from types import SimpleNamespace

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


def test_local_solver_axis_pairs_run_ids_and_reuses_seed_bundles():
    jobs = build_single_index_jobs(
        SingleIndexSeriesConfig(
            profile="full",
            experiments=("2",),
            seeds=(7,),
            local_solvers=("zero_intercept", "least_squares"),
        )
    )

    assert len(jobs) == 40
    for index in range(0, len(jobs), 2):
        zero_intercept, least_squares = jobs[index : index + 2]
        assert zero_intercept.parameters == least_squares.parameters
        assert zero_intercept.seed == least_squares.seed == 7
        assert zero_intercept.local_solver == "zero_intercept"
        assert least_squares.local_solver == "least_squares"
        assert zero_intercept.seeds == least_squares.seeds
        assert zero_intercept.run_id != least_squares.run_id


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


def test_parameter_sweeps_reuse_common_random_substreams_within_experiment():
    jobs = build_single_index_jobs(
        SingleIndexSeriesConfig(
            profile="full",
            experiments=("3",),
            seeds=(7,),
        )
    )

    assert len({job.seeds for job in jobs}) == 1
    assert len({job.run_id for job in jobs}) == len(jobs)


def test_experiment_two_keeps_original_random_projection_grid():
    jobs = build_single_index_jobs(
        SingleIndexSeriesConfig(
            profile="full",
            experiments=("2",),
            seeds=(7,),
        )
    )

    assert len(jobs) == 20
    assert {job.parameters.statistics_builder for job in jobs} == {
        "random_projection"
    }


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


def test_process_pool_reuses_workers_for_pool_lifetime(monkeypatch):
    captured = {}

    class RecordingPool:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def shutdown(self, **kwargs):
            captured["shutdown"] = kwargs

    monkeypatch.setattr(single_index_runner, "ProcessPoolExecutor", RecordingPool)

    completed = single_index_runner._run_parallel(
        store=None,
        jobs=(),
        config=SingleIndexSeriesConfig(profile="smoke"),
        process_jobs=3,
        completed=0,
        total=0,
        progress=None,
    )

    assert completed == 0
    assert captured["max_workers"] == 3
    assert "max_tasks_per_child" not in captured


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
        "1/1 experiment=1 local_solver=least_squares seed=0 status=success\n"
    )


def test_jobs_one_uses_serial_path_without_process_pool(tmp_path, monkeypatch):
    executed = []

    class RecordingStore:
        def __init__(self):
            self.series_dir = tmp_path
            self.completed = set()
            self.final_status = None

        def pending_jobs(self, jobs):
            return jobs

        def commit(self, outcome):
            self.completed.add(outcome.run_row["run_id"])

        def completed_run_ids(self):
            return set(self.completed)

        def finalize(self, *, status):
            self.final_status = status
            return {}

    store = RecordingStore()

    class RecordingStoreFactory:
        @staticmethod
        def create(output_root, config, jobs):
            return store

    class ForbiddenPool:
        def __init__(self, **kwargs):
            raise AssertionError("process pool selected for jobs=1")

    def execute_without_fit(job, config):
        executed.append(job.run_id)
        return SimpleNamespace(
            run_row={"run_id": job.run_id, "status": "success"}
        )

    monkeypatch.setattr(
        single_index_runner,
        "SingleIndexSeriesStore",
        RecordingStoreFactory,
    )
    monkeypatch.setattr(single_index_runner, "ProcessPoolExecutor", ForbiddenPool)
    monkeypatch.setattr(single_index_runner, "execute_job", execute_without_fit)
    monkeypatch.setattr(
        single_index_runner,
        "write_single_index_reports",
        lambda series_dir: pd.DataFrame(),
    )
    config = SingleIndexSeriesConfig(
        profile="smoke",
        experiments=("1",),
        jobs=1,
        seeds=(0,),
        diagnostic_seeds=(),
        center_fraction=0.25,
        max_runs=1,
    )

    run_single_index_benchmark(config, tmp_path)

    assert len(executed) == 1
    assert store.completed == set(executed)
    assert store.final_status == "complete"


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
