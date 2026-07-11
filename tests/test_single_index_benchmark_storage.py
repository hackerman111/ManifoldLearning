from dataclasses import replace

import pandas as pd
import pytest

from adp.common.experiment_log import stable_run_id
from adp.evaluation.single_index.scenarios import scenarios_for_profile
from adp.evaluation.single_index.storage import SingleIndexSeriesStore
from adp.evaluation.single_index.types import (
    SeedBundle,
    SingleIndexJob,
    SingleIndexSeriesConfig,
)


def make_jobs(count=2):
    scenario = scenarios_for_profile("smoke")[1]
    jobs = []
    for repeat in range(count):
        seeds = SeedBundle(
            data=100 + repeat,
            beta=200 + repeat,
            centers=300 + repeat,
            directions=400 + repeat,
            init=500 + repeat,
        )
        jobs.append(
            SingleIndexJob(
                scenario=scenario,
                method=scenario.methods[0],
                repeat=repeat,
                seeds=seeds,
                run_id=stable_run_id(
                    "single_index",
                    scenario.scenario_id,
                    scenario.methods[0],
                    repeat,
                    config_fingerprint=f"job-{repeat}",
                ),
            )
        )
    return tuple(jobs)


def make_config(**overrides):
    values = {
        "profile": "smoke",
        "base_seed": 17,
        "jobs": 1,
        "statistics_workers": 1,
    }
    values.update(overrides)
    return SingleIndexSeriesConfig(**values)


def run_row(job, status="success"):
    return {
        "run_id": job.run_id,
        "scenario_id": job.scenario.scenario_id,
        "family": job.scenario.family,
        "executor": job.scenario.executor,
        "method": job.method,
        "repeat": job.repeat,
        "status": status,
        "failed": status == "failed",
        "error": "boom" if status == "failed" else "",
    }


def iteration_row(job, outer_k=0):
    return {
        "run_id": job.run_id,
        "scenario_id": job.scenario.scenario_id,
        "method": job.method,
        "outer_k": outer_k,
        "cosine_abs": 0.9,
    }


def test_create_publishes_running_series_and_all_initial_parameters(tmp_path):
    jobs = make_jobs()
    store = SingleIndexSeriesStore.create(tmp_path, make_config(), jobs)

    series_path = store.series_dir / "single_index_series.csv"
    parameters_path = store.series_dir / "single_index_initial_parameters.csv"
    series = pd.read_csv(series_path).iloc[0]
    parameters = pd.read_csv(parameters_path)

    assert store.series_dir.parent == tmp_path
    assert series["series_id"] == store.series_id
    assert series["status"] == "running"
    assert series["config_fingerprint"].startswith("cfg-")
    assert series["requested_jobs"] == len(jobs)
    assert set(parameters["run_id"]) == {job.run_id for job in jobs}
    assert set(parameters["series_id"]) == {store.series_id}
    assert parameters.loc[0, "algorithm_statistics_workers"] == 1


def test_finalize_merges_shards_filters_orphans_and_writes_artifacts(tmp_path):
    first, orphan = make_jobs()
    store = SingleIndexSeriesStore.create(tmp_path, make_config(), (first, orphan))
    store.append_worker_rows("iterations", [iteration_row(first)])
    store.append_worker_rows("iterations", [iteration_row(orphan)])
    store.append_worker_rows("runs", [run_row(first)])

    saved = store.finalize(status="partial")

    runs = pd.read_csv(saved["runs"])
    iterations = pd.read_csv(saved["iterations"])
    artifacts = pd.read_csv(saved["artifacts"])
    series = pd.read_csv(saved["series"]).iloc[0]
    assert list(runs["run_id"]) == [first.run_id]
    assert list(iterations["run_id"]) == [first.run_id]
    assert series["status"] == "partial"
    assert series["completed_jobs"] == 1
    assert not store.shard_dir.exists()
    assert set(artifacts["series_id"]) == {store.series_id}
    assert (artifacts["size_bytes"] >= 0).all()
    assert all(not path.startswith("/") for path in artifacts["path"])


def test_resume_uses_unmerged_run_shard_as_commit_marker(tmp_path):
    first, second = make_jobs()
    config = make_config()
    store = SingleIndexSeriesStore.create(tmp_path, config, (first, second))
    store.append_worker_rows("iterations", [iteration_row(first)])
    store.append_worker_rows("runs", [run_row(first)])

    resumed = SingleIndexSeriesStore.resume(store.series_dir, config)

    assert [job.run_id for job in resumed.pending_jobs((first, second))] == [
        second.run_id
    ]


def test_resume_removes_orphan_iteration_rows_before_dispatch(tmp_path):
    orphan, committed = make_jobs()
    config = make_config()
    store = SingleIndexSeriesStore.create(tmp_path, config, (orphan, committed))
    store.append_worker_rows(
        "iterations",
        [iteration_row(orphan), iteration_row(committed)],
    )
    store.append_worker_rows("runs", [run_row(committed)])

    resumed = SingleIndexSeriesStore.resume(store.series_dir, config)
    shard = next(resumed.shard_dir.glob("iterations-*.csv"))

    assert list(pd.read_csv(shard)["run_id"]) == [committed.run_id]


def test_resume_retries_failed_only_when_requested(tmp_path):
    first, second = make_jobs()
    config = make_config()
    store = SingleIndexSeriesStore.create(tmp_path, config, (first, second))
    store.append_worker_rows("runs", [run_row(first, "success"), run_row(second, "failed")])
    store.finalize(status="partial")

    default_resume = SingleIndexSeriesStore.resume(store.series_dir, config)
    retry_resume = SingleIndexSeriesStore.resume(
        store.series_dir,
        replace(config, retry_failed=True),
    )

    assert list(default_resume.pending_jobs((first, second))) == []
    assert [job.run_id for job in retry_resume.pending_jobs((first, second))] == [
        second.run_id
    ]


def test_resume_rejects_schema_or_configuration_mismatch(tmp_path):
    jobs = make_jobs(1)
    config = make_config()
    store = SingleIndexSeriesStore.create(tmp_path, config, jobs)

    with pytest.raises(ValueError, match="configuration fingerprint mismatch"):
        SingleIndexSeriesStore.resume(
            store.series_dir,
            replace(config, base_seed=config.base_seed + 1),
        )

    series_path = store.series_dir / "single_index_series.csv"
    series = pd.read_csv(series_path)
    series.loc[0, "schema_version"] = 999
    series.to_csv(series_path, index=False)
    with pytest.raises(ValueError, match="schema version mismatch"):
        SingleIndexSeriesStore.resume(store.series_dir, config)


def test_finalize_rejects_duplicate_run_ids(tmp_path):
    job = make_jobs(1)[0]
    store = SingleIndexSeriesStore.create(tmp_path, make_config(), (job,))
    store.append_worker_rows("runs", [run_row(job), run_row(job)])

    with pytest.raises(ValueError, match="duplicate runs key"):
        store.finalize(status="complete")
