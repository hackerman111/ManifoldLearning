from __future__ import annotations

import math

import numpy as np
import pandas as pd

import adp.evaluation.single_index.executors as executors
import adp.evaluation.single_index.reports as reports
from adp.evaluation.single_index.reports import (
    add_report_metrics,
    prepare_quantile_band,
    write_single_index_reports,
)
from adp.evaluation.single_index.schema import (
    ARTIFACT_COLUMNS,
    INNER_ITERATION_COLUMNS,
    LOCAL_DIAGNOSTIC_COLUMNS,
    OUTER_ITERATION_COLUMNS,
    RUN_SUMMARY_COLUMNS,
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
                    "schema_version": 1,
                    "series_id": "series-report-test",
                    "run_id": run_id,
                    "experiment": selector,
                    "seed": seed,
                    "diagnostic": seed < 2,
                    "d": 5 + 20 * (seed % 2),
                    "n": 50 + 50 * seed,
                    "n_over_d": 2.0 + 3.0 * (seed % 2),
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
                    "runtime_sec": 0.5 + selector_index * 0.1 + seed,
                    "peak_memory_mb": 100.0 + 10.0 * seed,
                    "status": "numerical_failure" if failed else "success",
                    "stop_reason": "numerical_exception" if failed else "tolerance",
                }
            )
            for outer_k in range(2):
                outer_rows.append(
                    {
                        "schema_version": 1,
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
                            "schema_version": 1,
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
                            "schema_version": 1,
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
                            "schema_version": 1,
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
        [{"schema_version": 1, "series_id": "series-report-test", "status": "complete"}],
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
