from __future__ import annotations

import os
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from adp.evaluation.single_index.scenarios import smoke_parameter_grid
from adp.evaluation.single_index.types import ExperimentParameters


class TinyComparisonModel:
    def __init__(self, marker: str, *, fail: bool = False) -> None:
        self.marker = marker
        self.fail = fail
        self.config = SimpleNamespace(
            n_centers=None,
            n_directions=2,
            renew_directions=True,
        )

    def fit(self, X, y, *, centers, beta0, directions):
        del X, y, centers, directions
        time.sleep(0.01)
        if self.fail:
            raise RuntimeError(f"{self.marker} failed")
        beta = np.asarray(beta0, dtype=float)
        return SimpleNamespace(
            beta=beta,
            objective=float(beta @ beta),
            stage_timings={
                "statistics_builder": 0.4,
                "local_solver": 0.2,
                "beta_solver": 0.3,
            },
            stage_calls={
                "statistics_builder": 4,
                "local_solver": 2,
                "beta_solver": 3,
            },
        )


class InternalInitializerComparisonModel:
    def __init__(self, axis: int) -> None:
        self.axis = axis
        self.config = SimpleNamespace(
            n_centers=None,
            n_directions=2,
            renew_directions=True,
        )

    def fit(self, X, y, *, centers, beta0, directions):
        del X, y, centers, directions
        if beta0 is not None:
            raise AssertionError("external beta0 must be disabled")
        beta = np.zeros(3, dtype=float)
        beta[self.axis] = 1.0
        beta_ref = beta.copy()
        beta_hat0 = np.roll(beta, 1)
        return SimpleNamespace(
            beta=beta,
            objective=float(self.axis),
            beta_ref=beta_ref,
            beta_hat0=beta_hat0,
            stage_names={
                "beta_initializer": f"axis_{self.axis}",
                "bandwidth_selector": "local_mass_q10",
            },
            outer_telemetry=[
                {
                    "h": 1.5 + self.axis,
                    "local_mass_min": 2.0,
                    "local_mass_q05": 2.5,
                    "local_mass_q10": 3.0,
                    "local_mass_q25": 4.0,
                }
            ],
        )


def test_load_model_specs_preserves_mapping_order_and_builds_models(tmp_path):
    from experiments.model_comparison_specs import load_model_specs

    models_file = tmp_path / "models.py"
    models_file.write_text(
        """
from types import SimpleNamespace


class Model:
    def __init__(self, marker):
        self.marker = marker

    def fit(self, X, y, *, centers, beta0, directions):
        return SimpleNamespace(beta=beta0, objective=1.0)


MODELS = {
    "baseline": lambda: Model("first"),
    "candidate": lambda: Model("second"),
}
""",
        encoding="utf-8",
    )

    specs = load_model_specs(models_file)

    assert [spec.name for spec in specs] == ["baseline", "candidate"]
    assert [spec.model.marker for spec in specs] == ["first", "second"]


def test_load_model_specs_requires_at_least_two_models(tmp_path):
    from experiments.model_comparison_specs import load_model_specs

    models_file = tmp_path / "models.py"
    models_file.write_text(
        "MODELS = {'only': lambda: object()}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least two"):
        load_model_specs(models_file)


def test_load_model_specs_rejects_duplicate_normalized_names(tmp_path):
    from experiments.model_comparison_specs import load_model_specs

    models_file = tmp_path / "models.py"
    models_file.write_text(
        "MODELS = {'same': lambda: object(), ' same ': lambda: object()}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="distinct"):
        load_model_specs(models_file)


def test_load_model_specs_rejects_noncallable_factory(tmp_path):
    from experiments.model_comparison_specs import load_model_specs

    models_file = tmp_path / "models.py"
    models_file.write_text(
        "MODELS = {'baseline': object(), 'candidate': lambda: object()}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="baseline.*callable"):
        load_model_specs(models_file)


def test_load_model_specs_adds_factory_name_to_failure(tmp_path):
    from experiments.model_comparison_specs import load_model_specs

    models_file = tmp_path / "models.py"
    models_file.write_text(
        """
def fail():
    raise RuntimeError("broken setup")


MODELS = {'baseline': fail, 'candidate': lambda: object()}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="baseline.*broken setup"):
        load_model_specs(models_file)


def test_load_model_specs_requires_compatible_serializable_model(tmp_path):
    from experiments.model_comparison_specs import load_model_specs

    models_file = tmp_path / "models.py"
    models_file.write_text(
        """
class MissingFit:
    pass


MODELS = {
    'baseline': MissingFit,
    'candidate': MissingFit,
}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="baseline.*fit"):
        load_model_specs(models_file)


def test_multi_model_fit_tasks_rotate_three_model_order():
    from experiments.compare_model_efficiency import _iter_model_fit_tasks

    tasks = list(
        _iter_model_fit_tasks(
            ("baseline", "candidate_a", "candidate_b"),
            (b"baseline", b"candidate_a", b"candidate_b"),
            (ExperimentParameters(d=3, n_over_d=2.0),),
            (0, 1, 2),
            0.01,
            1,
        )
    )

    by_group = {}
    for task in tasks:
        by_group.setdefault(task.row["comparison_group_index"], []).append(
            task.row["model"]
        )
    assert by_group == {
        0: ["baseline", "candidate_a", "candidate_b"],
        1: ["candidate_a", "candidate_b", "baseline"],
        2: ["candidate_b", "baseline", "candidate_a"],
    }
    assert all(
        [task.row["fit_order"] for task in tasks if task.row["comparison_group_index"] == group]
        == [0, 1, 2]
        for group in by_group
    )


def test_multi_model_fit_tasks_cover_experiments_two_through_six():
    from experiments.compare_model_efficiency import _iter_model_fit_tasks

    experiments = ("2", "3", "4", "5", "6")
    grid = tuple(
        smoke_parameter_grid(experiment)[0]
        for experiment in experiments
    )
    tasks = list(
        _iter_model_fit_tasks(
            ("baseline", "candidate"),
            (b"baseline", b"candidate"),
            grid,
            (0,),
            0.01,
            1,
            experiment_selectors=experiments,
        )
    )

    first_per_group = tasks[::2]
    assert [task.row["experiment"] for task in first_per_group] == list(
        experiments
    )
    assert len({task.row["case_id"] for task in first_per_group}) == 5
    assert first_per_group[1].row["sigma_eps"] == 1.0
    assert first_per_group[2].row["rho_corr"] == 0.5
    assert first_per_group[3].row["sigma_x"] == 2.0
    assert first_per_group[4].row["link"] == "sin"


def test_compare_model_set_runs_three_models_in_fresh_sequential_processes():
    from experiments.compare_model_efficiency import compare_model_set

    runs = compare_model_set(
        (
            TinyComparisonModel("baseline"),
            TinyComparisonModel("candidate_a"),
            TinyComparisonModel("candidate_b"),
        ),
        model_names=("baseline", "candidate_a", "candidate_b"),
        parameter_grid=(ExperimentParameters(d=3, n_over_d=2.0),),
        seeds=(5,),
        sample_interval_sec=0.002,
        show_progress=False,
    )

    assert list(runs["model"]) == [
        "baseline",
        "candidate_a",
        "candidate_b",
    ]
    assert runs["worker_pid"].nunique() == 3
    assert os.getpid() not in set(runs["worker_pid"])
    ordered = runs.sort_values("actual_fit_order")
    assert list(ordered["actual_fit_order"]) == [0, 1, 2]
    assert (
        ordered["fit_started_ns"].iloc[1:].to_numpy()
        >= ordered["fit_finished_ns"].iloc[:-1].to_numpy()
    ).all()
    assert runs["input_fingerprint"].nunique() == 1
    assert runs["status"].eq("ok").all()
    assert runs["statistics_builder_time_sec"].eq(0.4).all()
    assert runs["local_solver_time_sec"].eq(0.2).all()
    assert runs["local_solver_calls"].eq(2).all()
    assert runs["beta_solver_time_sec"].eq(0.3).all()
    assert runs["beta_solver_calls"].eq(3).all()


def test_compare_model_set_can_use_each_models_internal_beta_initializer():
    from experiments.compare_model_efficiency import compare_model_set

    runs = compare_model_set(
        (
            InternalInitializerComparisonModel(0),
            InternalInitializerComparisonModel(1),
        ),
        model_names=("e1", "pca"),
        parameter_grid=(ExperimentParameters(d=3, n_over_d=2.0),),
        seeds=(5,),
        sample_interval_sec=0.002,
        show_progress=False,
        use_model_initializers=True,
    )

    assert runs["status"].eq("ok").all()
    assert runs["beta0_source"].eq("model_initializer").all()
    assert runs["input_fingerprint"].nunique() == 1
    assert list(runs["beta_initializer"]) == ["axis_0", "axis_1"]
    assert runs["bandwidth_selector"].eq("local_mass_q10").all()
    np.testing.assert_allclose(runs["initial_bandwidth"], [1.5, 2.5])
    assert runs["initial_local_mass_q10"].eq(3.0).all()
    assert runs["beta_ref_encoded"].str.len().gt(0).all()
    assert runs["beta_hat0_encoded"].str.len().gt(0).all()


def test_compare_model_set_accepts_experiment_parameter_grids():
    from experiments.compare_model_efficiency import compare_model_set

    runs = compare_model_set(
        (
            TinyComparisonModel("baseline"),
            TinyComparisonModel("candidate"),
        ),
        model_names=("baseline", "candidate"),
        experiment_parameter_grids={
            experiment: smoke_parameter_grid(experiment)
            for experiment in ("3", "4")
        },
        seeds=(5,),
        sample_interval_sec=0.002,
        show_progress=False,
    )

    assert list(runs["experiment"]) == ["3", "3", "4", "4"]
    assert runs["case_id"].nunique() == 2
    assert runs.groupby("comparison_group_index")[
        "input_fingerprint"
    ].nunique().eq(1).all()


def test_compare_model_set_records_failure_and_continues_group():
    from experiments.compare_model_efficiency import compare_model_set

    runs = compare_model_set(
        (
            TinyComparisonModel("baseline"),
            TinyComparisonModel("broken", fail=True),
            TinyComparisonModel("candidate"),
        ),
        model_names=("baseline", "broken", "candidate"),
        parameter_grid=(ExperimentParameters(d=3, n_over_d=2.0),),
        seeds=(5,),
        sample_interval_sec=0.002,
        show_progress=False,
    )

    assert list(runs["status"]) == ["ok", "failed", "ok"]
    failed = runs.loc[runs["model"].eq("broken")].iloc[0]
    assert failed["error_type"] == "RuntimeError"
    assert "broken failed" in failed["error_message"]
    assert runs.loc[runs["status"].eq("ok"), "result_finite"].all()


def _three_model_runs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": ["case"] * 6,
            "seed": [0, 0, 0, 1, 1, 1],
            "d": [2] * 6,
            "n": [4] * 6,
            "n_over_d": [2.0] * 6,
            "requested_n_over_d": [2.0] * 6,
            "actual_n_over_d": [2.0] * 6,
            "model": [
                "baseline",
                "candidate_a",
                "candidate_b",
                "baseline",
                "candidate_a",
                "candidate_b",
            ],
            "status": ["ok"] * 6,
            "fit_time_sec": [4.0, 2.0, 1.0, 8.0, 4.0, 2.0],
            "rss_start_mib": [100.0] * 6,
            "rss_min_mib": [100.0] * 6,
            "rss_mean_mib": [102.0] * 6,
            "rss_max_mib": [108.0, 104.0, 102.0, 108.0, 104.0, 102.0],
            "rss_peak_delta_mib": [8.0, 4.0, 2.0, 8.0, 4.0, 2.0],
            "cosine_abs": [1.0] * 6,
            "objective": [1.0] * 6,
            "beta_encoded": ["1|0", "-1|0", "1|0", "1|0", "-1|0", "1|0"],
            "result_finite": [True] * 6,
        }
    )


def test_pair_model_set_runs_compares_every_candidate_only_to_baseline():
    from experiments.compare_model_efficiency import pair_model_set_runs

    comparisons = pair_model_set_runs(
        _three_model_runs(),
        model_names=("baseline", "candidate_a", "candidate_b"),
    )

    assert len(comparisons) == 4
    assert set(comparisons["candidate_model"]) == {
        "candidate_a",
        "candidate_b",
    }
    assert comparisons["baseline_model"].eq("baseline").all()
    assert set(
        zip(
            comparisons["candidate_model"],
            comparisons["time_speedup"],
            strict=True,
        )
    ) == {("candidate_a", 2.0), ("candidate_b", 4.0)}


def test_model_summary_keeps_model_identity_and_failure_counts():
    from experiments.compare_model_efficiency import summarize_model_runs

    runs = _three_model_runs()
    runs.loc[1, "status"] = "failed"
    runs.loc[1, "fit_time_sec"] = np.nan
    summary = summarize_model_runs(runs)

    assert set(summary["model"]) == {
        "baseline",
        "candidate_a",
        "candidate_b",
    }
    candidate = summary.loc[summary["model"].eq("candidate_a")].iloc[0]
    assert candidate["run_count"] == 2
    assert candidate["success_count"] == 1
    assert candidate["failure_count"] == 1
    assert candidate["median_fit_time_sec"] == pytest.approx(4.0)


def test_write_model_set_artifacts_creates_tables_manifest_and_candidate_plots(
    tmp_path,
):
    from experiments.compare_model_efficiency import write_model_set_artifacts

    stale_runtime = tmp_path / "plots" / "runtime_vs_dimension.png"
    stale_runtime.parent.mkdir(parents=True)
    stale_runtime.write_text("obsolete", encoding="utf-8")
    stale_dimension_plot = (
        tmp_path / "plots" / "runtime_by_dimension" / "d_100.png"
    )
    stale_dimension_plot.parent.mkdir(parents=True)
    stale_dimension_plot.write_text("obsolete", encoding="utf-8")

    runs = _three_model_runs()
    higher_dimension = runs.copy()
    higher_dimension["case_id"] = "case-d4"
    higher_dimension["d"] = 4
    higher_dimension["n"] = 16
    combined_runs = pd.concat([runs, higher_dimension], ignore_index=True)

    artifacts = write_model_set_artifacts(
        combined_runs,
        tmp_path,
        model_names=("baseline", "candidate_a", "candidate_b"),
        configuration={
            "models_file": "experiments/my_models.py",
            "profile": "smoke",
            "seeds": [0, 1],
            "jobs": 1,
            "sample_interval_sec": 0.01,
        },
        dpi=40,
    )

    expected = {
        "runs.csv",
        "model_summary.csv",
        "comparisons.csv",
        "comparison_summary.csv",
        "manifest.json",
        "plots/runtime_by_dimension/d_2.png",
        "plots/runtime_by_dimension/d_4.png",
        "plots/memory_by_dimension/d_2.png",
        "plots/memory_by_dimension/d_4.png",
        "plots/cosine_abs_by_dimension/d_2.png",
        "plots/cosine_abs_by_dimension/d_4.png",
        "plots/candidate_a/time_speedup_heatmap.png",
        "plots/candidate_a/memory_ratio_heatmap.png",
        "plots/candidate_b/time_speedup_heatmap.png",
        "plots/candidate_b/memory_ratio_heatmap.png",
    }
    assert expected == set(artifacts)
    assert all(path.exists() for path in artifacts.values())
    assert not stale_runtime.exists()
    assert not stale_dimension_plot.exists()
    comparisons = pd.read_csv(artifacts["comparisons.csv"])
    assert len(comparisons) == 8
    manifest = __import__("json").loads(
        artifacts["manifest.json"].read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 1
    assert manifest["model_names"] == [
        "baseline",
        "candidate_a",
        "candidate_b",
    ]
    assert manifest["baseline_model"] == "baseline"
    assert set(manifest["artifacts"]) == expected - {"manifest.json"}


def test_multi_experiment_artifacts_separate_candidate_heatmaps(tmp_path):
    from experiments.compare_model_efficiency import write_model_set_artifacts

    experiment_two = _three_model_runs().copy()
    experiment_two["experiment"] = "2"
    experiment_three = _three_model_runs().copy()
    experiment_three["experiment"] = "3"
    experiment_three["case_id"] = "case-3"
    runs = pd.concat([experiment_two, experiment_three], ignore_index=True)

    artifacts = write_model_set_artifacts(
        runs,
        tmp_path,
        model_names=("baseline", "candidate_a", "candidate_b"),
        configuration={"experiments": ["2", "3"]},
        dpi=40,
    )

    assert (
        "plots/experiment_2/candidate_a/time_speedup_heatmap.png"
        in artifacts
    )
    assert (
        "plots/experiment_3/candidate_b/memory_ratio_heatmap.png"
        in artifacts
    )
    assert "plots/experiment_2/runtime_by_dimension/d_2.png" in artifacts
    assert "plots/experiment_3/cosine_abs_by_dimension/d_2.png" in artifacts
    assert all(path.exists() for path in artifacts.values())
    model_summary = pd.read_csv(artifacts["model_summary.csv"])
    comparison_summary = pd.read_csv(
        artifacts["comparison_summary.csv"]
    )
    assert set(model_summary["experiment"].astype(str)) == {"2", "3"}
    assert set(comparison_summary["experiment"].astype(str)) == {"2", "3"}


def test_top_level_cli_exposes_compare_subcommand():
    completed = subprocess.run(
        [sys.executable, "run_benchmarks.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "compare" in completed.stdout


def test_compare_cli_writes_complete_three_model_smoke_artifacts(tmp_path):
    models_file = tmp_path / "models.py"
    models_file.write_text(
        """
from types import SimpleNamespace
import numpy as np


class Model:
    def __init__(self, marker):
        self.marker = marker

    def fit(self, X, y, *, centers, beta0, directions):
        del X, y, centers, directions
        beta = np.asarray(beta0, dtype=float)
        return SimpleNamespace(beta=beta, objective=float(beta @ beta))


MODELS = {
    "baseline": lambda: Model("baseline"),
    "candidate_a": lambda: Model("candidate_a"),
    "candidate_b": lambda: Model("candidate_b"),
}
""",
        encoding="utf-8",
    )
    output = tmp_path / "comparison"

    completed = subprocess.run(
        [
            sys.executable,
            "run_benchmarks.py",
            "compare",
            "--models",
            str(models_file),
            "--profile",
            "smoke",
            "--seeds",
            "0",
            "--jobs",
            "1",
            "--sample-interval",
            "0.002",
            "--dpi",
            "40",
            "--no-progress",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "baseline" in completed.stdout
    assert "candidate_a" in completed.stdout
    assert "candidate_b" in completed.stdout
    assert len(pd.read_csv(output / "runs.csv")) == 3
    assert len(pd.read_csv(output / "comparisons.csv")) == 2
    assert (output / "manifest.json").is_file()
    assert (output / "plots/candidate_a/time_speedup_heatmap.png").is_file()
    assert (output / "plots/candidate_b/memory_ratio_heatmap.png").is_file()


def test_compare_cli_runs_smoke_experiments_two_through_six(tmp_path):
    models_file = tmp_path / "models.py"
    models_file.write_text(
        """
from types import SimpleNamespace
import numpy as np


class Model:
    def fit(self, X, y, *, centers, beta0, directions):
        del X, y, centers, directions
        beta = np.asarray(beta0, dtype=float)
        return SimpleNamespace(beta=beta, objective=float(beta @ beta))


MODELS = {
    "baseline": Model,
    "candidate": Model,
}
""",
        encoding="utf-8",
    )
    output = tmp_path / "comparison-2-6"

    completed = subprocess.run(
        [
            sys.executable,
            "run_benchmarks.py",
            "compare",
            "--models",
            str(models_file),
            "--experiments",
            "2:6",
            "--profile",
            "smoke",
            "--seeds",
            "0",
            "--jobs",
            "1",
            "--dpi",
            "40",
            "--no-progress",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    runs = pd.read_csv(output / "runs.csv")
    assert list(dict.fromkeys(runs["experiment"].astype(str))) == [
        "2",
        "3",
        "4",
        "5",
        "6",
    ]
    assert len(runs) == 10
    manifest = __import__("json").loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["experiments"] == ["2", "3", "4", "5", "6"]
    assert (
        output
        / "plots/experiment_6/candidate/time_speedup_heatmap.png"
    ).is_file()


def test_compare_cli_runs_manual_experiment_two_dimension_ratio_grid(tmp_path):
    models_file = tmp_path / "models.py"
    models_file.write_text(
        """
from types import SimpleNamespace
import numpy as np


class Model:
    def fit(self, X, y, *, centers, beta0, directions):
        del X, y, centers, directions
        beta = np.asarray(beta0, dtype=float)
        return SimpleNamespace(beta=beta, objective=float(beta @ beta))


MODELS = {
    "baseline": Model,
    "candidate": Model,
}
""",
        encoding="utf-8",
    )
    output = tmp_path / "manual-experiment-2"

    completed = subprocess.run(
        [
            sys.executable,
            "run_benchmarks.py",
            "compare",
            "--models",
            str(models_file),
            "--experiments",
            "2",
            "--d",
            "5,10",
            "--n-over-d",
            "5,10",
            "--seeds",
            "0",
            "--jobs",
            "1",
            "--dpi",
            "40",
            "--no-progress",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    runs = pd.read_csv(output / "runs.csv")
    assert len(runs) == 8
    assert set(runs["experiment"].astype(str)) == {"2"}
    assert set(runs["d"]) == {5, 10}
    assert set(runs["n_over_d"]) == {5.0, 10.0}
    assert (output / "plots/runtime_by_dimension/d_5.png").is_file()
    assert (output / "plots/runtime_by_dimension/d_10.png").is_file()
    assert (output / "plots/cosine_abs_by_dimension/d_5.png").is_file()
    assert (output / "plots/cosine_abs_by_dimension/d_10.png").is_file()
    manifest = __import__("json").loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["manual_d"] == [5, 10]
    assert manifest["manual_n_over_d"] == [5.0, 10.0]


def test_compare_cli_requires_complete_manual_experiment_two_grid(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "run_benchmarks.py",
            "compare",
            "--models",
            "examples/model_comparison_models.py",
            "--d",
            "5,10",
            "--output",
            str(tmp_path / "comparison"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "--d and --n-over-d must be provided together" in completed.stderr


def test_compare_cli_rejects_invalid_model_file_before_creating_output(
    tmp_path,
):
    models_file = tmp_path / "models.py"
    models_file.write_text(
        "MODELS = {'only': lambda: object()}\n",
        encoding="utf-8",
    )
    output = tmp_path / "comparison"

    completed = subprocess.run(
        [
            sys.executable,
            "run_benchmarks.py",
            "compare",
            "--models",
            str(models_file),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "at least two" in completed.stderr
    assert not output.exists()


def test_compare_cli_strict_mode_reports_numerical_disagreement(tmp_path):
    models_file = tmp_path / "models.py"
    models_file.write_text(
        """
from types import SimpleNamespace
import numpy as np


class Model:
    def __init__(self, axis):
        self.axis = axis

    def fit(self, X, y, *, centers, beta0, directions):
        del X, y, centers, beta0, directions
        beta = np.zeros(4, dtype=float)
        beta[self.axis] = 1.0
        return SimpleNamespace(beta=beta, objective=1.0)


MODELS = {
    "baseline": lambda: Model(0),
    "candidate": lambda: Model(1),
}
""",
        encoding="utf-8",
    )
    output = tmp_path / "comparison"

    completed = subprocess.run(
        [
            sys.executable,
            "run_benchmarks.py",
            "compare",
            "--models",
            str(models_file),
            "--profile",
            "smoke",
            "--seeds",
            "0",
            "--jobs",
            "1",
            "--dpi",
            "40",
            "--no-progress",
            "--require-equivalent",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "not numerically equivalent" in completed.stderr
    comparisons = pd.read_csv(output / "comparisons.csv")
    assert not comparisons["numerically_equivalent"].all()


def test_compare_cli_persists_failed_model_and_finishes_other_models(tmp_path):
    models_file = tmp_path / "models.py"
    models_file.write_text(
        """
from types import SimpleNamespace
import numpy as np


class Model:
    def __init__(self, fail=False):
        self.fail = fail

    def fit(self, X, y, *, centers, beta0, directions):
        del X, y, centers, directions
        if self.fail:
            raise RuntimeError("candidate exploded")
        beta = np.asarray(beta0, dtype=float)
        return SimpleNamespace(beta=beta, objective=float(beta @ beta))


MODELS = {
    "baseline": lambda: Model(),
    "broken": lambda: Model(fail=True),
    "candidate": lambda: Model(),
}
""",
        encoding="utf-8",
    )
    output = tmp_path / "comparison"

    completed = subprocess.run(
        [
            sys.executable,
            "run_benchmarks.py",
            "compare",
            "--models",
            str(models_file),
            "--profile",
            "smoke",
            "--seeds",
            "0",
            "--jobs",
            "1",
            "--dpi",
            "40",
            "--no-progress",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1, completed.stderr
    assert "model fits failed: 1/3" in completed.stderr
    runs = pd.read_csv(output / "runs.csv")
    assert list(runs["status"]) == ["ok", "failed", "ok"]
    assert "candidate exploded" in runs.loc[1, "error_message"]
    assert (output / "manifest.json").is_file()
    assert (output / "plots/broken/time_speedup_heatmap.png").is_file()


def test_ready_model_comparison_example_exposes_three_valid_factories():
    from experiments.model_comparison_specs import load_model_specs

    specs = load_model_specs("examples/model_comparison_models.py")

    assert [spec.name for spec in specs] == [
        "baseline",
        "zero_intercept",
        "direct_beta_solver",
    ]


@pytest.mark.parametrize(
    ("path", "expected_names"),
    [
        (
            "examples/initial_direction_models.py",
            [
                "e1_control",
                "pca",
                "ridge_eta_0",
                "ridge_eta_1e-4",
                "ridge_eta_1e-5",
                "ridge_eta_1e-6",
                "ridge_eta_1e-7",
                "ridge_eta_1e-2",
            ],
        ),
        (
            "examples/initial_bandwidth_models.py",
            [
                "mean_mass_control",
                "local_mass_q0",
                "local_mass_q05",
                "local_mass_q10",
                "local_mass_q25",
                "knn_q90_k1",
                "knn_q90_k2",
                "knn_q90_k4",
            ],
        ),
    ],
)
def test_ready_initialization_experiment_models(path, expected_names):
    from experiments.model_comparison_specs import load_model_specs

    specs = load_model_specs(path)

    assert [spec.name for spec in specs] == expected_names
    assert all(spec.model.config.min_neighbors == 4.0 for spec in specs)


def test_ready_ridge_eta_comparison_config_exposes_control_and_three_scales():
    from experiments.model_comparison_specs import load_model_specs

    specs = load_model_specs("examples/ridge_eta_comparison_models.py")

    assert [spec.name for spec in specs] == [
        "e1_control",
        "ridge_eta_1e-4",
        "ridge_eta_1e-5",
        "ridge_eta_1e-6",
    ]
    assert [
        spec.model.algorithm.stage_names["beta_initializer"]
        for spec in specs
    ] == [
        "e1",
        "ridge_1e-4",
        "ridge_1e-5",
        "ridge_1e-6",
    ]


def test_ready_small_ridge_comparison_config_exposes_requested_models():
    from experiments.model_comparison_specs import load_model_specs

    specs = load_model_specs("examples/small_ridge_comparison_models.py")

    assert [spec.name for spec in specs] == [
        "e1_control",
        "ridge_eta_1e-6",
        "ridge_eta_1e-7",
    ]
    assert [
        spec.model.algorithm.stage_names["beta_initializer"]
        for spec in specs
    ] == [
        "e1",
        "ridge_1e-6",
        "ridge_1e-7",
    ]


def test_ready_final_ridge_comparison_config_exposes_control_and_ridge_1e_6():
    from experiments.model_comparison_specs import load_model_specs

    specs = load_model_specs("examples/final_ridge_comparison_models.py")

    assert [spec.name for spec in specs] == [
        "e1_control",
        "ridge_eta_1e-6",
    ]
    assert [
        spec.model.algorithm.stage_names["beta_initializer"]
        for spec in specs
    ] == [
        "e1",
        "ridge_1e-6",
    ]
