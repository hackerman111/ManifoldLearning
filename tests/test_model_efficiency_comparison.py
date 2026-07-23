from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from adp.evaluation.single_index.types import ExperimentParameters


class TinyModel:
    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale
        self.config = SimpleNamespace(
            n_centers=None,
            n_directions=2,
            renew_directions=True,
        )

    def fit(self, X, y, *, centers, beta0, directions):
        scratch = np.ones((256, 256), dtype=float) * self.scale
        time.sleep(0.02)
        beta = np.asarray(beta0, dtype=float)
        return SimpleNamespace(
            beta=beta,
            objective=float(np.dot(beta, beta) + scratch[0, 0] * 0.0),
        )


def assert_recorded_cpu_constraint(runs: pd.DataFrame) -> None:
    get_affinity = callable(getattr(os, "sched_getaffinity", None))
    set_affinity = callable(getattr(os, "sched_setaffinity", None))
    affinity_supported = get_affinity and set_affinity
    assert runs["cpu_affinity_supported"].eq(affinity_supported).all()
    if affinity_supported:
        assert runs["worker_cpu_count"].eq(1).all()
        assert runs["cpu_affinity_pinned"].all()
        assert runs["worker_cpu_affinity"].eq(
            runs["assigned_cpu"].astype(str)
        ).all()
    else:
        assert runs["worker_cpu_count"].eq(0).all()
        assert ~runs["cpu_affinity_pinned"].any()


def test_comparison_module_exists_in_experiments():
    assert importlib.util.find_spec("experiments.compare_model_efficiency") is not None


def test_compare_models_runs_both_models_in_fresh_isolated_processes_on_paired_data():
    from experiments.compare_model_efficiency import compare_models

    runs = compare_models(
        TinyModel(1.0),
        TinyModel(2.0),
        model_names=("first", "second"),
        parameter_grid=(ExperimentParameters(d=3, n_over_d=4.0),),
        seeds=(7,),
        sample_interval_sec=0.002,
    )

    assert list(runs["model"]) == ["first", "second"]
    assert runs["case_id"].nunique() == 1
    assert set(runs["seed"]) == {7}
    assert set(runs["d"]) == {3}
    assert set(runs["n"]) == {12}
    assert set(runs["requested_n_over_d"]) == {4.0}
    assert set(runs["actual_n_over_d"]) == {4.0}
    assert set(runs["n_directions"]) == {4}
    assert runs["fit_time_sec"].gt(0.0).all()
    assert runs["rss_max_mib"].ge(runs["rss_start_mib"]).all()
    assert runs["rss_peak_delta_mib"].ge(0.0).all()
    assert runs["worker_pid"].nunique() == 2
    assert os.getpid() not in set(runs["worker_pid"])
    assert list(runs["actual_fit_order"]) == [0, 1]
    assert runs.loc[1, "fit_started_ns"] >= runs.loc[0, "fit_finished_ns"]
    assert_recorded_cpu_constraint(runs)
    assert runs["comparison_total_fits"].eq(2).all()
    assert runs["comparison_wall_time_sec"].nunique() == 1
    assert runs["comparison_wall_time_sec"].iloc[0] >= runs["fit_time_sec"].max()
    assert runs["comparison_fits_per_sec"].nunique() == 1
    assert runs["comparison_fits_per_sec"].iloc[0] == pytest.approx(
        2.0 / runs["comparison_wall_time_sec"].iloc[0]
    )
    assert runs["cosine_abs"].between(0.0, 1.0).all()
    assert runs["cosine_abs"].nunique() == 1
    assert runs["result_finite"].all()
    assert runs["beta_encoded"].str.count(r"\|").eq(2).all()


def test_compare_models_supports_parallel_workers_and_tqdm(monkeypatch):
    import experiments.compare_model_efficiency as comparison

    progress_state = {"total": None, "updates": 0, "disable": None}

    class RecordingProgress:
        def __init__(self, *, total, disable, **kwargs):
            progress_state["total"] = total
            progress_state["disable"] = disable

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def update(self, amount):
            progress_state["updates"] += amount

    monkeypatch.setattr(comparison, "tqdm", RecordingProgress)
    runs = comparison.compare_models(
        TinyModel(1.0),
        TinyModel(2.0),
        model_names=("first", "second"),
        parameter_grid=(ExperimentParameters(d=3, n_over_d=4.0),),
        seeds=(7, 8),
        jobs=2,
        show_progress=True,
        sample_interval_sec=0.002,
    )

    assert list(runs["model"]) == ["first", "second", "second", "first"]
    assert set(runs["jobs"]) == {2}
    assert runs["worker_pid"].nunique() == 4
    assert runs["parallel_pairs"].eq(2).all()
    for seed, pair in runs.groupby("seed", sort=True):
        actual = pair.sort_values("actual_fit_order")
        assert list(actual["actual_fit_order"]) == [0, 1]
        assert actual.iloc[1]["fit_started_ns"] >= actual.iloc[0]["fit_finished_ns"]
        assert actual["worker_pid"].nunique() == 2
        assert_recorded_cpu_constraint(actual)
        expected = ["first", "second"] if seed == 7 else ["second", "first"]
        assert list(actual["model"]) == expected
    assert progress_state == {"total": 4, "updates": 4, "disable": False}


def test_compare_models_rejects_nonpositive_parallelism():
    from experiments.compare_model_efficiency import compare_models

    with pytest.raises(ValueError, match="jobs"):
        compare_models(
            TinyModel(),
            TinyModel(),
            parameter_grid=(ExperimentParameters(d=3, n_over_d=4.0),),
            seeds=(0,),
            jobs=0,
        )


def test_pair_model_runs_computes_speed_and_memory_ratios_per_seed():
    from experiments.compare_model_efficiency import pair_model_runs

    runs = pd.DataFrame(
        {
            "case_id": ["case", "case", "case", "case"],
            "seed": [0, 0, 1, 1],
            "d": [3, 3, 3, 3],
            "n": [15, 15, 15, 15],
            "n_over_d": [5.0, 5.0, 5.0, 5.0],
            "model": ["baseline", "candidate", "baseline", "candidate"],
            "fit_time_sec": [4.0, 2.0, 6.0, 3.0],
            "rss_peak_delta_mib": [8.0, 4.0, 0.0, 0.0],
            "rss_max_mib": [108.0, 104.0, 110.0, 105.0],
            "cosine_abs": [0.9, 0.9, 0.95, 0.95],
            "objective": [2.0, 2.0, 3.0, 3.0],
            "beta_encoded": [
                "1|0|0",
                "-1|0|0",
                "1|0|0",
                "0|1|0",
            ],
            "result_finite": [True, True, True, True],
        }
    )

    paired = pair_model_runs(runs, model_names=("baseline", "candidate"))

    assert list(paired["time_speedup"]) == [2.0, 2.0]
    assert list(paired["peak_delta_memory_ratio"]) == [2.0, 1.0]
    assert list(paired["seed"]) == [0, 1]
    assert list(paired["beta_cosine_abs"]) == pytest.approx([1.0, 0.0])
    assert list(paired["beta_sign_invariant_error"]) == pytest.approx(
        [0.0, np.sqrt(2.0)]
    )
    assert list(paired["projector_frobenius_error"]) == pytest.approx(
        [0.0, np.sqrt(2.0)]
    )
    assert list(paired["objective_abs_gap"]) == [0.0, 0.0]
    assert list(paired["numerically_equivalent"]) == [True, False]
    assert paired["beta_atol"].eq(1e-5).all()
    assert paired["projector_atol"].eq(1e-5).all()
    assert paired["objective_rtol"].eq(1e-5).all()
    assert paired["objective_atol"].eq(1e-8).all()
    assert list(paired["requested_n_over_d"]) == [5.0, 5.0]
    assert list(paired["actual_n_over_d"]) == [5.0, 5.0]


def test_pair_model_runs_rejects_nonfinite_result_and_objective_mismatch():
    from experiments.compare_model_efficiency import pair_model_runs

    runs = pd.DataFrame(
        {
            "case_id": ["case", "case", "case", "case"],
            "seed": [0, 0, 1, 1],
            "d": [2, 2, 2, 2],
            "n": [5, 5, 5, 5],
            "n_over_d": [2.4, 2.4, 2.4, 2.4],
            "model": ["baseline", "candidate", "baseline", "candidate"],
            "fit_time_sec": [1.0, 1.0, 1.0, 1.0],
            "rss_peak_delta_mib": [0.0, 0.0, 0.0, 0.0],
            "rss_max_mib": [100.0, 100.0, 100.0, 100.0],
            "cosine_abs": [1.0, np.nan, 1.0, 1.0],
            "objective": [1.0, np.nan, 1.0, 1.1],
            "beta_encoded": ["1|0", "nan|0", "1|0", "1|0"],
            "result_finite": [True, False, True, True],
        }
    )

    paired = pair_model_runs(runs, model_names=("baseline", "candidate"))

    assert list(paired["requested_n_over_d"]) == [2.4, 2.4]
    assert list(paired["actual_n_over_d"]) == [2.5, 2.5]
    assert list(paired["result_pair_finite"]) == [False, True]
    assert list(paired["numerically_equivalent"]) == [False, False]
    assert paired.loc[1, "objective_abs_gap"] == pytest.approx(0.1)


def test_beta_equivalence_is_sign_invariant_but_detects_scale_changes():
    from experiments.compare_model_efficiency import _compare_encoded_betas

    cosine, beta_error, projector_error = _compare_encoded_betas(
        "1|0",
        "-2|0",
        2,
    )

    assert cosine == pytest.approx(1.0)
    assert beta_error == pytest.approx(0.5)
    assert projector_error == pytest.approx(0.0)


def test_summary_reports_pair_counts_and_finite_spread():
    from experiments.compare_model_efficiency import (
        pair_model_runs,
        summarize_paired_runs,
    )

    runs = pd.DataFrame(
        {
            "case_id": ["case", "case", "case", "case"],
            "seed": [0, 0, 1, 1],
            "d": [2, 2, 2, 2],
            "n": [5, 5, 5, 5],
            "n_over_d": [2.4, 2.4, 2.4, 2.4],
            "model": ["baseline", "candidate", "baseline", "candidate"],
            "fit_time_sec": [2.0, 1.0, 4.0, 2.0],
            "rss_peak_delta_mib": [0.0, 0.0, 4.0, 2.0],
            "rss_max_mib": [100.0, 100.0, 104.0, 102.0],
            "cosine_abs": [1.0, 1.0, 1.0, 1.0],
            "objective": [1.0, 1.0, 1.0, 1.0],
            "beta_encoded": ["1|0", "-1|0", "1|0", "-1|0"],
            "result_finite": [True, True, True, True],
        }
    )

    paired = pair_model_runs(runs, model_names=("baseline", "candidate"))
    summary = summarize_paired_runs(paired)

    assert summary.loc[0, "pair_count"] == 2
    assert summary.loc[0, "valid_time_speedup_count"] == 2
    assert summary.loc[0, "finite_time_speedup_count"] == 2
    assert summary.loc[0, "equivalent_pair_count"] == 2
    assert summary.loc[0, "q25_time_speedup"] == pytest.approx(2.0)
    assert summary.loc[0, "q75_time_speedup"] == pytest.approx(2.0)


def test_fit_tasks_counterbalance_model_order_and_log_actual_ratio():
    from experiments.compare_model_efficiency import _iter_fit_tasks

    tasks = list(
        _iter_fit_tasks(
            ("baseline", "candidate"),
            (b"baseline", b"candidate"),
            (ExperimentParameters(d=3, n_over_d=1.1),),
            (0, 1),
            0.01,
            1,
        )
    )

    assert [task.row["model"] for task in tasks] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    ]
    assert [task.row["fit_order"] for task in tasks] == [0, 1, 0, 1]
    assert [task.row["pair_order"] for task in tasks] == ["AB", "AB", "BA", "BA"]
    assert {task.row["requested_n_over_d"] for task in tasks} == {1.1}
    assert [task.row["actual_n_over_d"] for task in tasks] == pytest.approx(
        [4.0 / 3.0] * 4
    )


def test_configure_child_model_updates_existing_backend_worker_count():
    from adp import ADP, ADPConfig
    from experiments.compare_model_efficiency import _configure_child_model

    model = ADP.create(
        "new",
        ADPConfig(statistics_workers=3, show_progress=False),
    )

    _configure_child_model(model, n_centers=4, n_directions=3)

    assert model.config.statistics_workers == 1
    assert model.backend.statistics_workers == 1


def test_compare_models_rejects_duplicate_names_and_non_experiment_two_grid():
    from experiments.compare_model_efficiency import compare_models

    with pytest.raises(ValueError, match="distinct"):
        compare_models(
            TinyModel(),
            TinyModel(),
            model_names=("same", "same"),
            parameter_grid=(ExperimentParameters(d=3, n_over_d=4.0),),
            seeds=(0,),
        )
    with pytest.raises(ValueError, match="experiment 2"):
        compare_models(
            TinyModel(),
            TinyModel(),
            model_names=("first", "second"),
            parameter_grid=(
                ExperimentParameters(d=3, n_over_d=4.0, sigma_eps=1.0),
            ),
            seeds=(0,),
        )


def test_comparison_artifacts_include_csv_tables_and_time_memory_plots(tmp_path):
    from experiments.compare_model_efficiency import (
        compare_models,
        write_comparison_artifacts,
    )

    runs = compare_models(
        TinyModel(),
        TinyModel(),
        model_names=("first", "second"),
        parameter_grid=(ExperimentParameters(d=3, n_over_d=4.0),),
        seeds=(3,),
        sample_interval_sec=0.002,
    )
    artifacts = write_comparison_artifacts(
        runs,
        tmp_path,
        model_names=("first", "second"),
        dpi=40,
    )

    assert {
        "runs.csv",
        "paired.csv",
        "summary.csv",
        "runtime_vs_dimension.png",
        "memory_vs_dimension.png",
        "time_speedup_heatmap.png",
        "memory_ratio_heatmap.png",
    } == set(artifacts)
    assert all(path.exists() for path in artifacts.values())
    assert len(pd.read_csv(artifacts["runs.csv"])) == 2
    assert len(pd.read_csv(artifacts["paired.csv"])) == 1


def test_comparison_script_can_run_directly():
    completed = subprocess.run(
        [sys.executable, "experiments/compare_model_efficiency.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Compare two ADP model implementations" in completed.stdout
    assert "--jobs" in completed.stdout
    assert "--no-progress" in completed.stdout


def test_default_cli_models_are_explicit_random_projection_self_check():
    import experiments.compare_model_efficiency as comparison

    baseline, candidate = comparison._default_models()

    assert comparison.DEFAULT_MODEL_NAMES == (
        "random_projection_baseline",
        "random_projection_candidate",
    )
    assert baseline.algorithm.stage_names["statistics_builder"] == (
        "random_projection"
    )
    assert candidate.algorithm.stage_names["statistics_builder"] == (
        "random_projection"
    )


def test_main_returns_nonzero_when_candidate_is_not_numerically_equivalent(
    monkeypatch,
    tmp_path,
    capsys,
):
    import experiments.compare_model_efficiency as comparison

    def fake_compare_models(*models, model_names, **kwargs):
        baseline, candidate = model_names
        return pd.DataFrame(
            {
                "case_id": ["case", "case"],
                "seed": [0, 0],
                "d": [2, 2],
                "n": [4, 4],
                "n_over_d": [2.0, 2.0],
                "model": [baseline, candidate],
                "fit_time_sec": [1.0, 1.0],
                "rss_peak_delta_mib": [0.0, 0.0],
                "rss_max_mib": [100.0, 100.0],
                "cosine_abs": [1.0, 1.0],
                "objective": [1.0, 1.0],
                "beta_encoded": ["1|0", "0|1"],
                "result_finite": [True, True],
                "comparison_fits_per_sec": [2.0, 2.0],
            }
        )

    monkeypatch.setattr(comparison, "compare_models", fake_compare_models)
    monkeypatch.setattr(
        comparison,
        "write_comparison_artifacts",
        lambda *args, **kwargs: {},
    )

    exit_code = comparison.main(
        ["--profile", "smoke", "--no-progress", "--output", str(tmp_path)]
    )

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "not numerically equivalent" in captured.err
