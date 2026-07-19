from __future__ import annotations

import csv
from dataclasses import replace

import pandas as pd
import pytest

from adp.evaluation.single_index.schema import (
    INNER_ITERATION_COLUMNS,
    LOCAL_DIAGNOSTIC_COLUMNS,
    OUTER_ITERATION_COLUMNS,
    RUN_SUMMARY_COLUMNS,
    SCHEMA_VERSION,
    SOLVER_ITERATION_COLUMNS,
)
from adp.evaluation.single_index.storage import SingleIndexSeriesStore
from adp.evaluation.single_index.types import (
    ExperimentParameters,
    RunOutcome,
    SeedBundle,
    SingleIndexJob,
    SingleIndexSeriesConfig,
)


def make_jobs(count: int = 2) -> tuple[SingleIndexJob, ...]:
    jobs = []
    for index in range(count):
        jobs.append(
            SingleIndexJob(
                experiment="1",
                parameters=ExperimentParameters(
                    d=2 + index,
                    n_over_d=10,
                    sigma_eps=0.0,
                    link="linear",
                ),
                seed=index,
                seeds=SeedBundle(*(100 * offset + index for offset in range(1, 11))),
                run_id=f"run-storage-{index}",
                diagnostic=index == 0,
            )
        )
    return tuple(jobs)


def make_config(**updates: object) -> SingleIndexSeriesConfig:
    values: dict[str, object] = {
        "profile": "smoke",
        "experiments": ("1",),
        "jobs": 1,
        "seeds": (0, 1),
        "diagnostic_seeds": (0,),
        "center_fraction": 1.0,
    }
    values.update(updates)
    return SingleIndexSeriesConfig(**values)


def make_outcome(
    job: SingleIndexJob,
    *,
    status: str = "success",
    outer_k: int = 0,
) -> RunOutcome:
    return RunOutcome(
        run_row={
            "run_id": job.run_id,
            "experiment": job.experiment,
            "seed": job.seed,
            "d": job.parameters.d,
            "n": job.parameters.n,
            "n_over_d": job.parameters.n_over_d,
            "n_centers": job.parameters.n_centers,
            "center_fraction": job.parameters.center_fraction,
            "statistics_workers": 1,
            "status": status,
            "stop_reason": "complete" if status == "success" else "boom",
            "error_type": "" if status == "success" else "FloatingPointError",
            "error_message": "" if status == "success" else "boom",
        },
        outer_rows=(
            {
                "run_id": job.run_id,
                "experiment": job.experiment,
                "seed": job.seed,
                "outer_k": outer_k,
                "cosine_abs": 0.9,
            },
        ),
        inner_rows=(
            {
                "run_id": job.run_id,
                "experiment": job.experiment,
                "seed": job.seed,
                "outer_k": outer_k,
                "inner_k": 0,
                "objective": 1.0,
            },
        ),
        local_rows=(
            {
                "run_id": job.run_id,
                "experiment": job.experiment,
                "seed": job.seed,
                "outer_k": outer_k,
                "center_j": 0,
                "local_mass": 2.0,
            },
        ),
        solver_rows=(
            {
                "run_id": job.run_id,
                "experiment": job.experiment,
                "seed": job.seed,
                "outer_k": outer_k,
                "inner_k": 0,
                "solver_k": 1,
                "relative_residual": 0.1,
            },
        ),
    )


def test_run_summary_is_written_last_and_is_the_only_commit_marker(
    tmp_path,
    monkeypatch,
):
    job = make_jobs(1)[0]
    store = SingleIndexSeriesStore.create(tmp_path, make_config(), (job,))
    original = store._atomic_write_rows

    def fail_on_run_summary(path, columns, rows):
        if path.name == "run_summary.csv":
            raise OSError("marker write failed")
        return original(path, columns, rows)

    monkeypatch.setattr(store, "_atomic_write_rows", fail_on_run_summary)

    with pytest.raises(OSError, match="marker write failed"):
        store.commit(make_outcome(job))

    assert store.completed_run_ids() == set()
    run_dir = store.shard_dir / job.run_id
    assert (run_dir / "outer_iterations.csv").exists()
    assert not (run_dir / "run_summary.csv").exists()


def test_create_writes_running_series_and_resume_uses_only_markers(tmp_path):
    first, second = make_jobs()
    config = make_config()
    store = SingleIndexSeriesStore.create(tmp_path, config, (first, second))
    store.commit(make_outcome(first))
    orphan_dir = store.shard_dir / second.run_id
    orphan_dir.mkdir()
    store._atomic_write_rows(
        orphan_dir / "outer_iterations.csv",
        OUTER_ITERATION_COLUMNS,
        make_outcome(second).outer_rows,
    )

    resumed = SingleIndexSeriesStore.resume(store.series_dir, config)

    assert resumed.completed_run_ids() == {first.run_id}
    assert [job.run_id for job in resumed.pending_jobs((first, second))] == [
        second.run_id
    ]
    series = pd.read_csv(store.series_dir / "series.csv").iloc[0]
    assert series["status"] == "running"
    assert series["requested_jobs"] == 2


def test_retry_replaces_failed_marker_and_every_payload_fragment(tmp_path):
    job = make_jobs(1)[0]
    config = make_config()
    store = SingleIndexSeriesStore.create(tmp_path, config, (job,))
    store.commit(make_outcome(job, status="numerical_failure", outer_k=0))
    store.finalize(status="partial")

    default_resume = SingleIndexSeriesStore.resume(store.series_dir, config)
    retry_resume = SingleIndexSeriesStore.resume(
        store.series_dir,
        replace(config, retry_failed=True, jobs=3),
    )

    assert list(default_resume.pending_jobs((job,))) == []
    assert list(retry_resume.pending_jobs((job,))) == [job]

    retry_resume.commit(make_outcome(job, status="success", outer_k=4))
    saved = retry_resume.finalize(status="complete")

    assert list(pd.read_csv(saved["run_summary"])["status"]) == ["success"]
    assert list(pd.read_csv(saved["outer_iterations"])["outer_k"]) == [4]


def test_resume_fingerprint_excludes_jobs_retry_and_rejects_scientific_changes(
    tmp_path,
):
    jobs = make_jobs(1)
    config = make_config()
    store = SingleIndexSeriesStore.create(tmp_path, config, jobs)

    SingleIndexSeriesStore.resume(
        store.series_dir,
        replace(config, jobs="auto", retry_failed=True, max_runs=1),
    )
    with pytest.raises(ValueError, match="configuration fingerprint mismatch"):
        SingleIndexSeriesStore.resume(
            store.series_dir,
            replace(config, center_fraction=0.5),
        )

    series_path = store.series_dir / "series.csv"
    series = pd.read_csv(series_path)
    series.loc[0, "schema_version"] = 1
    series.to_csv(series_path, index=False)
    with pytest.raises(ValueError, match="schema version mismatch"):
        SingleIndexSeriesStore.resume(store.series_dir, config)


def test_finalize_streams_committed_shards_in_planned_order_with_stable_headers(
    tmp_path,
):
    first, second = make_jobs()
    store = SingleIndexSeriesStore.create(tmp_path, make_config(), (first, second))
    store.commit(make_outcome(second, outer_k=2))
    store.commit(make_outcome(first, outer_k=1))

    saved = store.finalize(status="complete")

    assert list(pd.read_csv(saved["run_summary"])["run_id"]) == [
        first.run_id,
        second.run_id,
    ]
    assert list(pd.read_csv(saved["outer_iterations"])["outer_k"]) == [1, 2]
    expected_headers = {
        "run_summary": RUN_SUMMARY_COLUMNS,
        "outer_iterations": OUTER_ITERATION_COLUMNS,
        "inner_iterations": INNER_ITERATION_COLUMNS,
        "local_diagnostics": LOCAL_DIAGNOSTIC_COLUMNS,
        "solver_iterations": SOLVER_ITERATION_COLUMNS,
    }
    for name, columns in expected_headers.items():
        with saved[name].open(newline="", encoding="utf-8") as handle:
            assert tuple(next(csv.reader(handle))) == columns


def test_finalize_writes_empty_detail_tables_with_public_headers(tmp_path):
    job = make_jobs(1)[0]
    store = SingleIndexSeriesStore.create(tmp_path, make_config(), (job,))
    outcome = replace(
        make_outcome(job),
        outer_rows=(),
        inner_rows=(),
        local_rows=(),
        solver_rows=(),
    )
    store.commit(outcome)

    saved = store.finalize(status="complete")

    for name in (
        "outer_iterations",
        "inner_iterations",
        "local_diagnostics",
        "solver_iterations",
    ):
        frame = pd.read_csv(saved[name])
        assert frame.empty
