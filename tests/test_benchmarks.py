import numpy as np
import pandas as pd
from typing import get_args

from adp.benchmarks import BenchmarkScenario, benchmark_summary, grid_scenarios, run_benchmark_suite, save_benchmark_report
from adp.evaluation.scenarios import BenchmarkMethod
from adp.evaluation import reports


def test_grid_scenarios_builds_dimension_direction_grid_with_shared_budget():
    scenarios = grid_scenarios(
        d_values=(10, 25),
        direction_values=(5, 10, 20),
        n=120,
        n_centers=30,
        outer_steps=2,
        inner_steps=4,
        trials=3,
    )

    assert len(scenarios) == 6
    assert {(scenario.d, scenario.n_directions) for scenario in scenarios} == {
        (10, 5),
        (10, 10),
        (10, 20),
        (25, 5),
        (25, 10),
        (25, 20),
    }
    assert {scenario.n for scenario in scenarios} == {120}
    assert {scenario.n_centers for scenario in scenarios} == {30}
    assert {scenario.outer_steps for scenario in scenarios} == {2}
    assert {scenario.inner_steps for scenario in scenarios} == {4}
    assert {scenario.trials for scenario in scenarios} == {3}


def test_benchmark_suite_compares_adp_with_ready_edr_baselines(tmp_path):
    scenario = BenchmarkScenario(
        name="quick_linear",
        n=90,
        d=4,
        n_centers=14,
        n_directions=4,
        min_neighbors=5,
        outer_steps=1,
        inner_steps=3,
        noise=0.02,
        corr=0.2,
        link="linear",
        trials=1,
    )

    frame = run_benchmark_suite(
        [scenario],
        methods=("adp_new", "statsmodels_sir", "sklearn_pls"),
        random_state=10,
        show_progress=False,
    )

    assert {"adp_new", "statsmodels_sir", "sklearn_pls"} == set(frame["method"])
    assert {
        "scenario",
        "trial",
        "method",
        "n_directions",
        "n_centers",
        "outer_steps",
        "inner_steps",
        "cosine_abs",
        "angle_deg",
        "fit_time_sec",
        "peak_memory_kib",
    }.issubset(frame.columns)
    assert np.all(np.isfinite(frame["fit_time_sec"]))
    assert np.all(np.isfinite(frame["peak_memory_kib"]))
    assert np.all(frame["peak_memory_kib"] >= 0.0)
    assert np.all((0.0 <= frame["cosine_abs"]) & (frame["cosine_abs"] <= 1.0))
    ready_baselines = frame[frame["method"].isin(["statsmodels_sir", "sklearn_pls"])]
    assert np.all(ready_baselines["cosine_abs"] > 0.7)

    saved = save_benchmark_report(frame, tmp_path, prefix="quick")

    assert saved["csv"].exists()
    assert saved["quality_plot"].exists()
    assert saved["time_plot"].exists()
    assert saved["csv"].read_text().startswith("scenario,trial,method")


def test_benchmark_methods_keep_only_new_adp_variant():
    methods = set(get_args(BenchmarkMethod))

    assert "adp_new" in methods
    assert "adp_old" not in methods


def test_benchmark_summary_includes_confidence_intervals_and_memory():
    frame = pd.DataFrame(
        [
            {
                "scenario": "grid_d10_p5",
                "method": "adp_new",
                "cosine_abs": 0.90,
                "angle_deg": 4.0,
                "fit_time_sec": 1.0,
                "peak_memory_kib": 100.0,
                "failed": False,
            },
            {
                "scenario": "grid_d10_p5",
                "method": "adp_new",
                "cosine_abs": 0.95,
                "angle_deg": 2.0,
                "fit_time_sec": 1.4,
                "peak_memory_kib": 140.0,
                "failed": False,
            },
        ]
    )

    summary = benchmark_summary(frame)

    expected_columns = {
        "count",
        "cosine_abs_ci95_low",
        "cosine_abs_ci95_high",
        "angle_deg_ci95_low",
        "angle_deg_ci95_high",
        "fit_time_sec_ci95_low",
        "fit_time_sec_ci95_high",
        "peak_memory_kib_mean",
        "peak_memory_kib_std",
        "peak_memory_kib_ci95_low",
        "peak_memory_kib_ci95_high",
    }
    assert expected_columns.issubset(summary.columns)
    new_row = summary[(summary["scenario"] == "grid_d10_p5") & (summary["method"] == "adp_new")].iloc[0]
    assert new_row["count"] == 2
    assert new_row["cosine_abs_ci95_low"] < new_row["cosine_abs_mean"] < new_row["cosine_abs_ci95_high"]


def test_benchmark_report_uses_russian_plot_labels(tmp_path, monkeypatch):
    captured = []
    original_plot_grouped_bars = reports.plot_grouped_bars

    def capture_plot(ax, frame, *, value, ylabel, title):
        captured.append((value, ylabel, title))
        original_plot_grouped_bars(ax, frame, value=value, ylabel=ylabel, title=title)

    monkeypatch.setattr(reports, "plot_grouped_bars", capture_plot)

    frame = pd.DataFrame(
        [
            {
                "scenario": "grid_d10_p5",
                "method": "adp_new",
                "cosine_abs": 0.95,
                "fit_time_sec": 1.2,
                "peak_memory_kib": 128.0,
            }
        ]
    )

    save_benchmark_report(frame, tmp_path, prefix="ru")

    labels = {(value, ylabel, title) for value, ylabel, title in captured}
    assert ("cosine_abs", "среднее |cos(beta, beta_hat)|", "Качество восстановления EDR") in labels
    assert ("fit_time_sec", "среднее время обучения, сек", "Время обучения EDR") in labels
