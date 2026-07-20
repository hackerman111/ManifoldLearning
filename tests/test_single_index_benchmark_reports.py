from __future__ import annotations

import math
import inspect

import numpy as np
import pandas as pd
import pytest

import adp.evaluation.single_index.executors as executors
import adp.evaluation.single_index.reports as reports
import adp.evaluation.single_index.plots as report_plots
from adp.evaluation.single_index.reports import (
    add_report_metrics,
    prepare_quantile_band,
    prepare_wilson_interval,
    write_single_index_reports,
)
from adp.evaluation.single_index.schema import (
    ARTIFACT_COLUMNS,
    INNER_ITERATION_COLUMNS,
    LOCAL_DIAGNOSTIC_COLUMNS,
    OUTER_ITERATION_COLUMNS,
    RUN_SUMMARY_COLUMNS,
    SCHEMA_VERSION,
    SERIES_COLUMNS,
    SOLVER_ITERATION_COLUMNS,
)
from adp.evaluation.single_index.types import EXPERIMENT_SELECTORS


REQUIRED_PLOTS = {
    "quality_vs_outer_iteration.png",
    "bandwidth_vs_outer_iteration.png",
    "rho_vs_outer_iteration.png",
    "beta_step_vs_outer_iteration.png",
    "objective_vs_outer_iteration.png",
    "objective_vs_inner_iteration.png",
    "beta_step_vs_inner_iteration.png",
    "solver_residual_vs_iteration.png",
    "local_mass_by_outer_iteration.png",
    "effective_neighbors_by_outer_iteration.png",
    "local_condition_by_outer_iteration.png",
    "mass_vs_condition.png",
    "local_slopes_by_outer_iteration.png",
    "quality_heatmap_d_nd_ratio.png",
    "success_rate_heatmap.png",
    "runtime_vs_dimension.png",
    "memory_vs_dimension.png",
    "iterations_heatmap_d_nd_ratio.png",
    "quality_vs_sigma_eps.png",
    "success_rate_vs_sigma_eps.png",
    "runtime_vs_sigma_eps.png",
    "outer_iterations_vs_sigma_eps.png",
    "final_objective_vs_sigma_eps.png",
    "quality_vs_correlation.png",
    "success_rate_vs_correlation.png",
    "local_condition_vs_correlation.png",
    "solver_iterations_vs_correlation.png",
    "runtime_vs_correlation.png",
    "quality_vs_sigma_x.png",
    "h0_vs_sigma_x.png",
    "final_bandwidth_vs_sigma_x.png",
    "local_mass_vs_sigma_x.png",
    "runtime_vs_sigma_x.png",
    "quality_by_link_function.png",
    "success_rate_by_link_function.png",
    "outer_iterations_by_link_function.png",
    "objective_by_link_function.png",
    "local_slopes_by_link_function.png",
    "quality_by_x_distribution.png",
    "quality_by_noise_distribution.png",
    "failure_rate_by_distribution.png",
    "runtime_by_distribution.png",
    "quality_by_heteroscedasticity.png",
    "quality_vs_outlier_fraction.png",
    "failure_rate_vs_outliers.png",
    "quality_vs_model_misspecification.png",
    "objective_vs_model_misspecification.png",
    "runtime_breakdown.png",
    "projector_error_vs_outer_iteration.png",
    "singular_fraction_vs_correlation.png",
    "bandwidth_ratio_vs_sigma_x.png",
    "status_breakdown.png",
    "runtime_share_breakdown.png",
    "correctness_rate.png",
}


def _write_frame(path, rows, columns):
    pd.DataFrame(rows).reindex(columns=columns).to_csv(path, index=False)


def write_fixture_tables(tmp_path, selectors=EXPERIMENT_SELECTORS):
    run_rows = []
    outer_rows = []
    inner_rows = []
    local_rows = []
    solver_rows = []
    for selector_index, selector in enumerate(selectors):
        for seed in range(3):
            run_id = f"run-{selector}-{seed}"
            failed = seed == 2
            run_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "series_id": "series-report-test",
                    "run_id": run_id,
                    "experiment": selector,
                    "seed": seed,
                    "diagnostic": seed < 2,
                    "d": 5 + 20 * (seed % 2),
                    "n": 50 + 50 * seed,
                    "n_over_d": 2.0 + 3.0 * (seed % 2),
                    "statistics_builder": (
                        "cpu_batched" if seed % 2 else "random_projection"
                    ),
                    "n_centers": 20,
                    "center_fraction": 1.0,
                    "sigma_x": (0.5, 1.0, 2.0)[seed],
                    "rho_corr": (0.0, 0.5, 0.9)[seed],
                    "sigma_eps": (0.0, 0.5, 1.0)[seed],
                    "snr": math.inf if seed == 0 else 4.0 / seed,
                    "link": ("linear", "quadratic", "sin")[seed],
                    "x_distribution": ("gaussian", "uniform", "student_t5")[seed],
                    "noise_distribution": ("gaussian", "student_t5", "student_t3")[seed],
                    "heteroscedastic": bool(seed % 2),
                    "outlier_fraction": (0.0, 0.01, 0.05)[seed],
                    "effective_outlier_fraction": (0.0, 0.02, 0.06)[seed],
                    "outlier_scale": 5.0,
                    "delta": (0.0, 0.1, 0.5)[seed],
                    "h_initial": 2.0 + seed,
                    "h_final": 1.0 + seed,
                    "rho_final": 0.25 * seed,
                    "outer_iterations": 2 + seed,
                    "inner_iterations_total": 4 + seed,
                    "cosine_abs": np.nan if failed else 0.995 - 0.03 * seed,
                    "projector_error": 0.1 + 0.1 * seed,
                    "objective": 1.0 + seed,
                    "algorithm_time_sec": 0.5 + selector_index * 0.1 + seed,
                    "statistics_builder_time_sec": 0.2 + 0.1 * seed,
                    "statistics_builder_calls": 2 + seed,
                    "algorithm_rss_max_mib": 100.0 + 10.0 * seed,
                    "status": "numerical_failure" if failed else "success",
                    "stop_reason": "numerical_exception" if failed else "tolerance",
                }
            )
            for outer_k in range(2):
                outer_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "series_id": "series-report-test",
                        "run_id": run_id,
                        "experiment": selector,
                        "seed": seed,
                        "outer_k": outer_k,
                        "h_k": 2.0 / (outer_k + 1) + seed,
                        "rho_k": 0.1 * outer_k,
                        "beta_k": "1|0",
                        "beta_norm": 1.0,
                        "cosine_abs": 0.9 + 0.02 * outer_k - 0.01 * seed,
                        "projector_error": 0.2,
                        "beta_delta": 0.2 / (outer_k + 1),
                        "objective_before": 2.0 + seed,
                        "objective_after": 1.0 + 0.2 * seed,
                        "relative_objective_decrease": 0.5,
                        "inner_iterations": 2,
                        "local_mass_mean": 10.0 + seed,
                        "local_mass_q05": 5.0 + seed,
                        "local_mass_median": 9.0 + seed,
                        "local_mass_q95": 15.0 + seed,
                        "ess_mean": 8.0 + seed,
                        "condition_median": 2.0 + seed,
                        "weights_time_sec": 0.01 + 0.001 * seed,
                        "statistics_time_sec": 0.02 + 0.001 * seed,
                        "optimization_time_sec": 0.03 + 0.001 * seed,
                        "service_overhead_sec": 0.005,
                        "iteration_time_sec": 0.065 + 0.003 * seed,
                    }
                )
                for inner_k in range(2):
                    inner_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "series_id": "series-report-test",
                            "run_id": run_id,
                            "experiment": selector,
                            "seed": seed,
                            "outer_k": outer_k,
                            "inner_k": inner_k,
                            "objective": 1.0 / (inner_k + 1) + seed,
                            "beta_delta": 0.1 / (inner_k + 1),
                            "linear_solver_iterations": 2 + seed,
                            "relative_linear_residual": 0.01,
                        }
                    )
                for center_j in range(2):
                    local_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "series_id": "series-report-test",
                            "run_id": run_id,
                            "experiment": selector,
                            "seed": seed,
                            "outer_k": outer_k,
                            "center_j": center_j,
                            "local_mass": 5.0 + center_j + seed,
                            "ess": 4.0 + center_j,
                            "condition": 2.0 + center_j + seed,
                            "slope": 0.5 + 0.1 * center_j,
                        }
                    )
                for solver_k in range(1, 3):
                    solver_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "series_id": "series-report-test",
                            "run_id": run_id,
                            "experiment": selector,
                            "seed": seed,
                            "outer_k": outer_k,
                            "inner_k": 0,
                            "solver_k": solver_k,
                            "relative_residual": 0.1**solver_k,
                        }
                    )
    _write_frame(tmp_path / "run_summary.csv", run_rows, RUN_SUMMARY_COLUMNS)
    _write_frame(tmp_path / "outer_iterations.csv", outer_rows, OUTER_ITERATION_COLUMNS)
    _write_frame(tmp_path / "inner_iterations.csv", inner_rows, INNER_ITERATION_COLUMNS)
    _write_frame(tmp_path / "local_diagnostics.csv", local_rows, LOCAL_DIAGNOSTIC_COLUMNS)
    _write_frame(tmp_path / "solver_iterations.csv", solver_rows, SOLVER_ITERATION_COLUMNS)
    _write_frame(
        tmp_path / "series.csv",
        [
            {
                "schema_version": SCHEMA_VERSION,
                "series_id": "series-report-test",
                "status": "complete",
            }
        ],
        SERIES_COLUMNS,
    )
    _write_frame(tmp_path / "artifacts.csv", [], ARTIFACT_COLUMNS)


def test_fixture_csvs_render_every_applicable_plot(tmp_path):
    write_fixture_tables(tmp_path)

    artifacts = write_single_index_reports(tmp_path, dpi=40)

    created = {
        path.name
        for path in artifacts.loc[artifacts.status == "created", "path"].map(tmp_path.__truediv__)
        if path.suffix == ".png"
    }
    assert REQUIRED_PLOTS <= created
    assert not artifacts.loc[artifacts.path.str.endswith(".png"), "path"].str.startswith("/").any()


def test_manifest_has_russian_labels_explicit_scales_and_stable_54_names():
    assert len(reports.PLOT_MANIFEST) == 54
    assert {spec.filename for spec in reports.PLOT_MANIFEST} == REQUIRED_PLOTS
    assert all(spec.title.startswith(("Эксперимент ", "Сводка —")) for spec in reports.PLOT_MANIFEST)
    assert all(spec.subtitle for spec in reports.PLOT_MANIFEST)
    assert all(any("а" <= char.lower() <= "я" or char in "ёЁ" for char in spec.xlabel) for spec in reports.PLOT_MANIFEST)
    assert all(any("а" <= char.lower() <= "я" or char in "ёЁ" for char in spec.ylabel) for spec in reports.PLOT_MANIFEST)

    by_name = {spec.filename: spec for spec in reports.PLOT_MANIFEST}
    assert by_name["runtime_vs_dimension.png"].y == "algorithm_time_sec"
    assert by_name["runtime_vs_dimension.png"].xscale == "log"
    assert by_name["runtime_vs_dimension.png"].yscale == "log"
    assert by_name["runtime_vs_dimension.png"].groups == ("n_over_d",)
    assert by_name["memory_vs_dimension.png"].y == "algorithm_rss_max_mib"
    assert by_name["memory_vs_dimension.png"].ylabel.endswith("МиБ")
    assert by_name["quality_vs_sigma_eps.png"].xscale == "linear"
    assert by_name["quality_vs_sigma_eps.png"].ylim == (0.0, 1.0)
    assert by_name["quality_vs_sigma_x.png"].xscale == "log2"
    assert by_name["objective_vs_model_misspecification.png"].yscale == "symlog"
    assert by_name["rho_vs_outer_iteration.png"].ylim == (0.0, 1.0)
    assert "n_over_d" in by_name["quality_vs_outer_iteration.png"].groups
    assert "n_over_d" in by_name["local_mass_by_outer_iteration.png"].facet
    assert "n_over_d" in by_name["quality_by_link_function.png"].facet
    assert by_name["correctness_rate.png"].selectors == ("1",)
    assert by_name["correctness_rate.png"].y == "success_value"
    assert by_name["correctness_rate.png"].facet == ("d",)
    for filename in (
        "failure_rate_by_distribution.png",
        "runtime_by_distribution.png",
    ):
        assert "experiment" in by_name[filename].groups
        assert "n_over_d" in by_name[filename].groups
    assert by_name["quality_vs_outlier_fraction.png"].x == (
        "effective_outlier_fraction"
    )
    assert by_name["failure_rate_vs_outliers.png"].x == (
        "effective_outlier_fraction"
    )


def test_default_single_index_report_dpi_is_300():
    assert inspect.signature(write_single_index_reports).parameters["dpi"].default == 300


def test_wilson_interval_matches_reference_values_and_keeps_groups_separate():
    frame = pd.DataFrame(
        {
            "experiment": ["3"] * 10 + ["4"] * 4,
            "sigma_eps": [0.5] * 14,
            "success_value": [1.0] * 8 + [0.0] * 2 + [1.0, 0.0, 0.0, 0.0],
        }
    )

    prepared = prepare_wilson_interval(
        frame,
        x="sigma_eps",
        y="success_value",
        groups=("experiment",),
    )

    first = prepared.loc[prepared.experiment == "3"].iloc[0]
    second = prepared.loc[prepared.experiment == "4"].iloc[0]
    assert first["estimate"] == 0.8
    assert first["low"] == pytest.approx(0.49016247153664183)
    assert first["high"] == pytest.approx(0.9433178485456247)
    assert first["n"] == 10
    assert second["estimate"] == 0.25


def test_quantile_plot_uses_external_russian_legends_and_quality_limits(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def capture(fig, path, *, dpi, close=False):
        captured["fig"] = fig
        captured["dpi"] = dpi
        return path

    monkeypatch.setattr(report_plots, "save_figure", capture)
    frame = pd.DataFrame(
        {
            "outer_k": [0, 1],
            "median": [0.96, 0.995],
            "q05": [0.90, 0.98],
            "q95": [0.99, 1.0],
            "group": ["n/d = 5", "n/d = 5"],
        }
    )

    report_plots.line_with_quantile_band(
        frame,
        x="outer_k",
        median="median",
        q05="q05",
        q95="q95",
        path=tmp_path / "plot.png",
        xlabel="Внешняя итерация",
        ylabel="Абсолютный косинус",
        title="Эксперимент 1 — качество направления",
        subtitle="медиана и интервал 5–95%",
        group="group",
        ylim=(0.0, 1.0),
        reference_y=0.99,
        dpi=37,
    )

    fig = captured["fig"]
    ax = fig.axes[0]
    assert captured["dpi"] == 37
    assert ax.get_ylim() == pytest.approx((0.0, 1.0))
    assert "медиана и интервал 5–95%" in [text.get_text() for text in fig.texts]
    legends = fig.legends
    assert [legend.get_title().get_text() for legend in legends] == ["Параметры", "Статистика"]
    assert all(legend.get_bbox_to_anchor().x0 >= 1.0 for legend in legends)
    statistic_labels = [text.get_text() for text in legends[1].get_texts()]
    assert statistic_labels == ["линия с маркером — медиана", "полоса — интервал 5–95%"]


def test_group_styles_are_global_and_distinct_across_facets(tmp_path, monkeypatch):
    captured = {}

    def capture(fig, path, *, dpi, close=False):
        captured["fig"] = fig
        return path

    monkeypatch.setattr(report_plots, "save_figure", capture)
    frame = pd.DataFrame(
        {
            "facet": ["d = 5", "d = 5", "d = 25"],
            "group": [
                "n/d = 2, масштаб выбросов = 5",
                "n/d = 2, масштаб выбросов = 10",
                "n/d = 5, масштаб выбросов = 5",
            ],
            "x": [1.0, 1.0, 1.0],
            "median": [0.95, 0.94, 0.96],
            "q05": [0.90, 0.89, 0.91],
            "q95": [0.99, 0.98, 0.99],
        }
    )

    report_plots.line_with_quantile_band(
        frame,
        x="x",
        median="median",
        q05="q05",
        q95="q95",
        path=tmp_path / "facets.png",
        xlabel="Параметр",
        ylabel="Качество",
        title="Эксперимент 3 — тест фасетов",
        group="group",
        facet="facet",
        dpi=30,
    )

    parameter_legend = captured["fig"].legends[0]
    colors = [handle.get_color() for handle in parameter_legend.legend_handles]
    styles = [handle.get_linestyle() for handle in parameter_legend.legend_handles]
    assert colors[0] == colors[1]
    assert colors[0] != colors[2]
    assert styles[0] != styles[1]
    assert styles[0] == styles[2]


def test_detail_summary_collapses_solver_iterations_to_one_value_per_run():
    spec = next(
        item for item in reports.PLOT_MANIFEST
        if item.filename == "solver_iterations_vs_correlation.png"
    )
    source = pd.DataFrame(
        {
            "run_id": ["long", "long", "long", "short"],
            "linear_solver_iterations": [2, 3, 5, 7],
            "rho_corr": [0.5, 0.5, 0.5, 0.5],
            "d": [25, 25, 25, 25],
            "n_over_d": [5, 5, 5, 5],
        }
    )

    collapsed = reports._prepare_plot_source(spec, source)

    assert collapsed.set_index("run_id")["linear_solver_iterations"].to_dict() == {
        "long": 10,
        "short": 7,
    }


def test_success_heatmap_uses_sample_fraction_in_each_cell():
    spec = next(
        item
        for item in reports.PLOT_MANIFEST
        if item.filename == "success_rate_heatmap.png"
    )
    source = pd.DataFrame(
        {
            "d": [25, 25, 25],
            "n_over_d": [5, 5, 5],
            "success_value": [1.0, 0.0, 0.0],
        }
    )

    prepared = reports._prepare_plot_source(spec, source)

    assert len(prepared) == 1
    assert prepared.iloc[0]["success_value"] == pytest.approx(1 / 3)


def test_heatmap_labels_scale_and_missing_cells_are_explicit(tmp_path, monkeypatch):
    captured = {}

    def capture(fig, path, *, dpi, close=False):
        captured["fig"] = fig
        return path

    monkeypatch.setattr(report_plots, "save_figure", capture)
    report_plots.heatmap(
        pd.DataFrame(
            {
                "ratio": [2, 5],
                "dimension": [5, 25],
                "quality": [0.75, 1.0],
            }
        ),
        x="ratio",
        y="dimension",
        value="quality",
        path=tmp_path / "heatmap.png",
        xlabel="Отношение n/d",
        ylabel="Размерность",
        title="Эксперимент 2 — качество",
        colorbar_label="Абсолютный косинус направления",
        value_limits=(0.0, 1.0),
        dpi=30,
    )

    fig = captured["fig"]
    assert fig.axes[0].images[0].get_clim() == (0.0, 1.0)
    assert fig.axes[1].get_ylabel() == "Абсолютный косинус направления"
    assert [tick.get_text() for tick in fig.axes[0].get_xticklabels()] == ["2", "5"]
    assert [tick.get_text() for tick in fig.axes[0].get_yticklabels()] == ["5", "25"]
    assert [text.get_text() for text in fig.axes[0].texts].count("нет данных") == 2


def test_all_infinite_values_have_exact_russian_annotation(tmp_path, monkeypatch):
    captured = {}

    def capture(fig, path, *, dpi, close=False):
        captured["fig"] = fig
        return path

    monkeypatch.setattr(report_plots, "save_figure", capture)
    report_plots.boxplot(
        pd.DataFrame({"outer": [0, 0, 0], "condition": [np.inf, np.inf, np.inf]}),
        x="outer",
        y="condition",
        path=tmp_path / "condition.png",
        xlabel="Внешняя итерация",
        ylabel="Число обусловленности",
        title="Эксперимент 1 — обусловленность",
        yscale="log",
        dpi=30,
    )

    assert "все 3 значения равны +∞" in [
        text.get_text() for text in captured["fig"].axes[0].texts
    ]


def test_symlog_axis_reports_actual_linear_threshold(tmp_path, monkeypatch):
    captured = {}

    def capture(fig, path, *, dpi, close=False):
        captured["fig"] = fig
        return path

    monkeypatch.setattr(report_plots, "save_figure", capture)
    report_plots.grouped_line(
        pd.DataFrame({"x": [0, 1, 2], "value": [-2.0, 0.0, 4.0]}),
        x="x",
        y="value",
        path=tmp_path / "symlog.png",
        xlabel="Итерация",
        ylabel="Значение (симлог-шкала)",
        title="Эксперимент 1 — знакопеременная метрика",
        yscale="symlog",
        dpi=30,
    )

    labels = [text.get_text() for text in captured["fig"].axes[0].texts]
    assert "симлог: linthresh = 0.2" in labels


def test_nonfinite_notes_are_preserved_per_facet_before_aggregation():
    source = pd.DataFrame(
        {
            "_facet": ["d = 5", "d = 5", "d = 25"],
            "condition": [np.inf, np.inf, np.nan],
        }
    )

    notes = reports._invalid_value_notes(source, "condition")

    assert notes == {
        "d = 5": "все 2 значения равны +∞",
        "d = 25": "нет конечных значений: NaN = 1, +∞ = 0, −∞ = 0",
    }


def test_empty_aggregate_keeps_facet_for_nonfinite_annotation(tmp_path, monkeypatch):
    captured = {}

    def capture(fig, path, *, dpi, close=False):
        captured["fig"] = fig
        return path

    monkeypatch.setattr(report_plots, "save_figure", capture)
    report_plots.line_with_quantile_band(
        pd.DataFrame(columns=["facet", "x", "median", "q05", "q95"]),
        x="x",
        median="median",
        q05="q05",
        q95="q95",
        path=tmp_path / "infinite.png",
        xlabel="Корреляция",
        ylabel="Число обусловленности",
        title="Эксперимент 4 — обусловленность",
        facet="facet",
        data_notes={"d = 25": "все 3 значения равны +∞"},
        dpi=30,
    )

    ax = captured["fig"].axes[0]
    assert ax.get_title() == "d = 25"
    assert "все 3 значения равны +∞" in [text.get_text() for text in ax.texts]


def test_log2_axis_uses_original_experimental_values_as_ticks(tmp_path, monkeypatch):
    captured = {}

    def capture(fig, path, *, dpi, close=False):
        captured["fig"] = fig
        return path

    monkeypatch.setattr(report_plots, "save_figure", capture)
    values = [0.25, 0.5, 1.0, 2.0, 4.0]
    report_plots.grouped_line(
        pd.DataFrame({"sigma_x": values, "quality": np.linspace(0.8, 1.0, 5)}),
        x="sigma_x",
        y="quality",
        path=tmp_path / "log2.png",
        xlabel="Масштаб признаков (логарифмическая шкала, основание 2)",
        ylabel="Качество",
        title="Эксперимент 5 — масштаб признаков",
        xscale="log2",
        dpi=30,
    )

    labels = [tick.get_text() for tick in captured["fig"].axes[0].get_xticklabels()]
    assert labels == ["0.25", "0.5", "1", "2", "4"]


def test_quantile_bands_keep_experiments_separate_and_use_5_50_95_percentiles():
    frame = pd.DataFrame(
        {
            "experiment": ["1"] * 5 + ["3"] * 5,
            "outer_k": [0] * 10,
            "cosine_abs": [0, 1, 2, 3, 100, 10, 11, 12, 13, 14],
        }
    )

    prepared = prepare_quantile_band(
        frame,
        x="outer_k",
        y="cosine_abs",
        groups=("experiment",),
    )

    first = prepared.loc[prepared.experiment == "1"].iloc[0]
    second = prepared.loc[prepared.experiment == "3"].iloc[0]
    assert first["q05"] == np.quantile([0, 1, 2, 3, 100], 0.05)
    assert first["median"] == 2
    assert first["q95"] == np.quantile([0, 1, 2, 3, 100], 0.95)
    assert second["median"] == 12


def test_success_metrics_count_failures_and_use_strict_experiment_one_threshold():
    runs = pd.DataFrame(
        {
            "experiment": ["1", "1", "2", "2", "2"],
            "status": ["success", "success", "success", "nonconverged", "numerical_failure"],
            "cosine_abs": [0.995, 0.95, 0.91, 0.905, np.nan],
        }
    )

    enriched = add_report_metrics(runs)

    assert list(enriched["success_value"]) == [1.0, 0.0, 1.0, 1.0, 0.0]
    assert list(enriched["failure_value"]) == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_report_rerender_never_executes_fit_and_isolates_plot_failures(
    tmp_path,
    monkeypatch,
):
    write_fixture_tables(tmp_path, selectors=("1",))

    def fail_execute(*args, **kwargs):
        raise AssertionError("executor must not be called by reports")

    monkeypatch.setattr(executors, "execute_job", fail_execute)
    original = reports._render_plot

    def fail_one(spec, *args, **kwargs):
        if spec.filename == "bandwidth_vs_outer_iteration.png":
            raise RuntimeError("forced plot failure")
        return original(spec, *args, **kwargs)

    monkeypatch.setattr(reports, "_render_plot", fail_one)

    artifacts = write_single_index_reports(tmp_path, dpi=40)

    failed = artifacts.loc[
        artifacts.path.str.endswith("bandwidth_vs_outer_iteration.png")
    ].iloc[0]
    later = artifacts.loc[
        artifacts.path.str.endswith("rho_vs_outer_iteration.png")
    ].iloc[0]
    assert failed["status"] == "error"
    assert "forced plot failure" in failed["error"]
    assert later["status"] == "created"
