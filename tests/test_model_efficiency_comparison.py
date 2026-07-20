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


def test_comparison_module_exists_in_experiments():
    assert importlib.util.find_spec("experiments.compare_model_efficiency") is not None


def test_compare_models_runs_both_models_in_isolated_processes_on_paired_data():
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
    assert set(runs["n_directions"]) == {4}
    assert runs["fit_time_sec"].gt(0.0).all()
    assert runs["rss_max_mib"].ge(runs["rss_start_mib"]).all()
    assert runs["rss_peak_delta_mib"].ge(0.0).all()
    assert runs["worker_pid"].nunique() == 2
    assert os.getpid() not in set(runs["worker_pid"])
    assert runs["cosine_abs"].between(0.0, 1.0).all()
    assert runs["cosine_abs"].nunique() == 1


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
        seeds=(7,),
        jobs=2,
        show_progress=True,
        sample_interval_sec=0.002,
    )

    assert list(runs["model"]) == ["first", "second"]
    assert set(runs["jobs"]) == {2}
    assert runs["worker_pid"].nunique() == 2
    assert progress_state == {"total": 2, "updates": 2, "disable": False}


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
            "d": [25, 25, 25, 25],
            "n": [125, 125, 125, 125],
            "n_over_d": [5.0, 5.0, 5.0, 5.0],
            "model": ["baseline", "candidate", "baseline", "candidate"],
            "fit_time_sec": [4.0, 2.0, 6.0, 3.0],
            "rss_peak_delta_mib": [8.0, 4.0, 10.0, 5.0],
            "rss_max_mib": [108.0, 104.0, 110.0, 105.0],
            "cosine_abs": [0.9, 0.9, 0.95, 0.95],
        }
    )

    paired = pair_model_runs(runs, model_names=("baseline", "candidate"))

    assert list(paired["time_speedup"]) == [2.0, 2.0]
    assert list(paired["peak_delta_memory_ratio"]) == [2.0, 2.0]
    assert list(paired["seed"]) == [0, 1]


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


def test_compact_factored_comparison_builds_two_selectable_models():
    from experiments.compare_compact_factored_efficiency import build_models

    baseline, candidate = build_models()

    assert baseline.algorithm.stage_names["statistics_builder"] == (
        "random_projection"
    )
    assert candidate.algorithm.stage_names["statistics_builder"] == (
        "cpu_compact_factored"
    )


def test_compact_factored_full_profile_plans_complete_experiment_two_grid(
    monkeypatch,
    tmp_path,
):
    import experiments.compare_compact_factored_efficiency as experiment

    captured = {}

    def fake_compare_models(*models, **kwargs):
        captured["models"] = models
        captured.update(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(experiment, "compare_models", fake_compare_models)
    monkeypatch.setattr(experiment, "write_comparison_artifacts", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        experiment,
        "pair_model_runs",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "time_speedup": [1.0],
                "peak_delta_memory_ratio": [1.0],
            }
        ),
    )

    exit_code = experiment.main(
        [
            "--profile",
            "full",
            "--jobs",
            "3",
            "--no-progress",
            "--output",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert captured["model_names"] == (
        "random_projection",
        "cpu_compact_factored",
    )
    assert len(captured["parameter_grid"]) == 20
    assert captured["seeds"] == tuple(range(100))
    assert captured["jobs"] == 3
    assert captured["show_progress"] is False


def test_compact_factored_comparison_script_runs_real_smoke(tmp_path):
    output = tmp_path / "compact-factored"
    completed = subprocess.run(
        [
            sys.executable,
            "experiments/compare_compact_factored_efficiency.py",
            "--profile",
            "smoke",
            "--seeds",
            "0",
            "--jobs",
            "1",
            "--no-progress",
            "--sample-interval",
            "0.002",
            "--dpi",
            "40",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    runs = pd.read_csv(output / "runs.csv")
    paired = pd.read_csv(output / "paired.csv")
    assert set(runs["model"]) == {
        "random_projection",
        "cpu_compact_factored",
    }
    assert len(runs) == 2
    assert len(paired) == 1
