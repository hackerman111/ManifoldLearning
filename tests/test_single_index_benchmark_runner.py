from dataclasses import replace

import pandas as pd

import adp.evaluation.single_index.runner as single_runner
from adp.evaluation.single_index.runner import (
    build_single_index_jobs,
    run_single_index_benchmark,
)
from adp.evaluation.single_index.types import SingleIndexSeriesConfig


class RecordingProgress:
    def __init__(self, calls, **kwargs):
        self.kwargs = kwargs
        self.n = 0
        self.postfixes = []
        calls.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def update(self, amount=1):
        self.n += amount

    def set_postfix(self, values, refresh=True):
        self.postfixes.append((values, refresh))


def record_progress(monkeypatch):
    calls = []
    monkeypatch.setattr(
        single_runner,
        "tqdm",
        lambda **kwargs: RecordingProgress(calls, **kwargs),
        raising=False,
    )
    return calls


def make_config(**overrides):
    values = {
        "profile": "smoke",
        "base_seed": 123,
        "jobs": 1,
        "statistics_workers": 1,
        "max_scenarios": 2,
    }
    values.update(overrides)
    return SingleIndexSeriesConfig(**values)


def test_build_jobs_is_deterministic_and_pairs_method_seeds():
    config = make_config(max_scenarios=4)

    first = build_single_index_jobs(config)
    repeated = build_single_index_jobs(config)

    assert first == repeated
    grouped = {}
    for job in first:
        grouped.setdefault((job.scenario.scenario_id, job.repeat), []).append(job)
    for jobs in grouped.values():
        assert len({job.seeds for job in jobs}) == 1
        assert len({job.run_id for job in jobs}) == len(jobs)


def test_runner_writes_resource_rows_payload_and_resume_without_duplicates(tmp_path):
    config = make_config()

    saved = run_single_index_benchmark(config, tmp_path)
    runs = pd.read_csv(saved["runs"])
    iterations = pd.read_csv(saved["iterations"])

    assert len(runs) == 2
    assert set(runs["status"]) == {"success"}
    assert (runs["full_run_time_sec"] > 0.0).all()
    finite_algorithm = runs["algorithm_time_sec"].dropna()
    assert (finite_algorithm > 0.0).all()
    assert (
        runs.loc[finite_algorithm.index, "full_run_time_sec"] >= finite_algorithm
    ).all()
    assert (runs["result_persist_time_sec"] > 0.0).all()
    assert set(iterations["run_id"]) == set(runs["run_id"])
    assert saved["summary"].exists()
    assert saved["scaling"].exists()
    assert saved["paired"].exists()
    assert saved["worst_five"].exists()
    artifacts = pd.read_csv(saved["artifacts"])
    assert {"summary", "scaling", "paired", "worst_five"}.issubset(
        set(artifacts["name"])
    )
    assert {f"G{index:02d}" for index in range(1, 22)}.issubset(
        set(artifacts["name"])
    )
    assert set(artifacts["status"]) <= {"created", "error"}

    resumed = run_single_index_benchmark(config, tmp_path, resume=saved["series"].parent)
    resumed_runs = pd.read_csv(resumed["runs"])
    assert len(resumed_runs) == len(runs)
    assert resumed_runs["run_id"].is_unique


def test_runner_records_failure_without_stopping_series(tmp_path, monkeypatch):
    def fail_job(job, config):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(single_runner, "execute_job", fail_job)

    saved = run_single_index_benchmark(make_config(max_scenarios=1), tmp_path)
    runs = pd.read_csv(saved["runs"])
    failures = pd.read_csv(saved["failures"])

    assert list(runs["status"]) == ["failed"]
    assert bool(runs.loc[0, "failed"])
    assert "forced failure" in runs.loc[0, "error"]
    assert list(failures["run_id"]) == list(runs["run_id"])
    assert runs.loc[0, "full_run_time_sec"] > 0.0


def test_runner_reports_persisted_jobs_with_tqdm(tmp_path, monkeypatch):
    calls = record_progress(monkeypatch)

    run_single_index_benchmark(make_config(max_scenarios=1), tmp_path)

    progress = calls[0]
    assert progress.kwargs == {
        "total": 1,
        "desc": "single-index",
        "unit": "job",
        "dynamic_ncols": True,
    }
    assert progress.n == 1
    assert progress.postfixes[-1][0] == {
        "scenario": "C01",
        "method": "full_adp",
    }


def test_process_pool_completion_updates_tqdm(tmp_path, monkeypatch):
    calls = record_progress(monkeypatch)

    class ImmediateFuture:
        def result(self):
            return None

    class ImmediatePool:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, *args):
            function(*args)
            return ImmediateFuture()

    monkeypatch.setattr(single_runner, "ProcessPoolExecutor", ImmediatePool)
    monkeypatch.setattr(single_runner, "as_completed", lambda futures: iter(futures))

    run_single_index_benchmark(
        make_config(jobs=2, max_scenarios=1),
        tmp_path,
    )

    assert calls[0].n == 1
    assert calls[0].postfixes[-1][0] == {
        "scenario": "C01",
        "method": "full_adp",
    }


def test_process_pool_oserror_falls_back_and_logs_progress(
    tmp_path,
    monkeypatch,
    capsys,
):
    calls = record_progress(monkeypatch)

    class BrokenPool:
        def __init__(self, *args, **kwargs):
            raise OSError("pool unavailable")

    monkeypatch.setattr(single_runner, "ProcessPoolExecutor", BrokenPool)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(variable, "8")

    saved = run_single_index_benchmark(
        make_config(jobs=2, max_scenarios=1),
        tmp_path,
    )
    captured = capsys.readouterr()

    assert len(pd.read_csv(saved["runs"])) == 1
    assert "parallel fallback" in captured.err
    assert "1/1" in captured.err
    assert "scenario=C01" in captured.err
    assert calls[0].n == 1
    assert calls[0].postfixes[-1][0] == {
        "scenario": "C01",
        "method": "full_adp",
    }
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert single_runner.os.environ[variable] == "1"
