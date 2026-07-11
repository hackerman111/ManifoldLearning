import math

import numpy as np
import pandas as pd

import adp.evaluation.single_index.reports as reports
from adp.evaluation.single_index.reports import (
    build_single_index_summary,
    fit_scaling_exponents,
    paired_method_differences,
    select_worst_five,
    write_single_index_reports,
)
from adp.evaluation.single_index.schema import ARTIFACT_COLUMNS


def sample_runs():
    return pd.DataFrame(
        [
            {
                "run_id": "full-1",
                "scenario_id": "S01",
                "family": "S",
                "method": "full_adp",
                "repeat": 0,
                "data_seed": 10,
                "status": "success",
                "cosine_abs": 0.9,
                "algorithm_time_sec": 1.0,
                "full_run_time_sec": 1.2,
            },
            {
                "run_id": "full-2",
                "scenario_id": "S01",
                "family": "S",
                "method": "full_adp",
                "repeat": 1,
                "data_seed": 11,
                "status": "success",
                "cosine_abs": 0.7,
                "algorithm_time_sec": 2.0,
                "full_run_time_sec": 2.2,
            },
            {
                "run_id": "full-3",
                "scenario_id": "S01",
                "family": "S",
                "method": "full_adp",
                "repeat": 2,
                "data_seed": 12,
                "status": "failed",
                "cosine_abs": 0.1,
            },
            {
                "run_id": "full-4",
                "scenario_id": "S01",
                "family": "S",
                "method": "full_adp",
                "repeat": 3,
                "data_seed": 13,
                "status": "unavailable",
                "cosine_abs": np.nan,
            },
            {
                "run_id": "ols-1",
                "scenario_id": "S01",
                "family": "S",
                "method": "ols",
                "repeat": 0,
                "data_seed": 10,
                "status": "success",
                "cosine_abs": 0.7,
                "algorithm_time_sec": 0.1,
                "full_run_time_sec": 0.2,
            },
            {
                "run_id": "ols-2",
                "scenario_id": "S01",
                "family": "S",
                "method": "ols",
                "repeat": 1,
                "data_seed": 11,
                "status": "success",
                "cosine_abs": 0.6,
                "algorithm_time_sec": 0.1,
                "full_run_time_sec": 0.2,
            },
        ]
    )


def test_summary_keeps_failures_in_denominator_and_excludes_nan_quality():
    summary = build_single_index_summary(
        sample_runs(),
        bootstrap_resamples=200,
        random_state=17,
    )
    row = summary.query("scenario_id == 'S01' and method == 'full_adp'").iloc[0]

    assert row["total_count"] == 4
    assert row["success_count"] == 2
    assert row["failed_count"] == 1
    assert row["unavailable_count"] == 1
    assert row["success_rate"] == 0.5
    assert row["failure_rate"] == 0.25
    assert row["cosine_abs_count"] == 2
    assert row["cosine_abs_mean"] == 0.8
    assert row["cosine_abs_median"] == 0.8
    assert math.isclose(row["cosine_abs_iqr"], 0.1)
    assert math.isclose(row["cosine_abs_q05"], 0.71)
    assert math.isclose(row["cosine_abs_q95"], 0.89)
    assert row["success_ci95_low"] < 0.5 < row["success_ci95_high"]
    assert 0.7 <= row["cosine_abs_bootstrap_ci95_low"] <= 0.8
    assert 0.8 <= row["cosine_abs_bootstrap_ci95_high"] <= 0.9


def test_worst_five_returns_lowest_finite_successful_runs():
    runs = pd.DataFrame(
        {
            "run_id": [f"run-{index}" for index in range(8)],
            "scenario_id": ["S01"] * 8,
            "method": ["full_adp"] * 8,
            "status": ["success"] * 7 + ["failed"],
            "cosine_abs": [0.8, 0.2, 0.6, 0.1, 0.4, 0.3, 0.5, np.nan],
        }
    )

    worst = select_worst_five(runs)

    assert list(worst["run_id"]) == ["run-3", "run-1", "run-5", "run-4", "run-6"]


def test_scaling_fit_and_paired_differences_use_matching_data_seeds():
    scaling = pd.DataFrame(
        {
            "scenario_id": ["M01"] * 3 + ["M06"] * 3,
            "method": ["full_adp"] * 6,
            "status": ["success"] * 6,
            "data_n": [100, 200, 400, 100, 200, 400],
            "data_d": [5] * 6,
            "algorithm_n_centers": [10] * 6,
            "algorithm_n_directions": [4] * 6,
            "solver_outer_steps": [2] * 6,
            "solver_inner_steps": [3] * 6,
            "algorithm_time_sec": [1, 2, 4, 1, 1, 1],
            "full_run_rss_peak_delta_mib": [1, 1, 1, 2, 4, 8],
        }
    )

    fitted = fit_scaling_exponents(scaling)
    m01 = fitted.query("scenario_id == 'M01'").iloc[0]
    m06 = fitted.query("scenario_id == 'M06'").iloc[0]
    assert math.isclose(m01["exponent"], 1.0)
    assert m01["x_column"] == "data_n"
    assert m01["y_column"] == "algorithm_time_sec"
    assert math.isclose(m06["exponent"], 1.0)
    assert m06["y_column"] == "full_run_rss_peak_delta_mib"

    paired = paired_method_differences(sample_runs())
    row = paired.iloc[0]
    assert row["reference_method"] == "full_adp"
    assert row["comparison_method"] == "ols"
    assert row["pair_count"] == 2
    assert math.isclose(row["cosine_abs_delta_mean"], 0.15)


def test_report_publishes_numeric_csv_when_one_plot_fails(tmp_path, monkeypatch):
    runs = sample_runs().query("status == 'success'").copy()
    parameters = runs[
        ["run_id", "scenario_id", "family", "method", "repeat", "data_seed"]
    ].copy()
    parameters["data_n"] = [100, 200, 100, 200]
    parameters["data_d"] = 5
    parameters["algorithm_n_centers"] = 10
    parameters["algorithm_n_directions"] = 4
    parameters["solver_outer_steps"] = 2
    parameters["solver_inner_steps"] = 3
    runs.to_csv(tmp_path / "single_index_runs.csv", index=False)
    parameters.to_csv(tmp_path / "single_index_initial_parameters.csv", index=False)
    pd.DataFrame(columns=["run_id", "scenario_id", "method", "outer_k"]).to_csv(
        tmp_path / "single_index_iterations.csv", index=False
    )
    pd.DataFrame(
        columns=["run_id", "scenario_id", "method", "outer_k", "inner_k", "cg_k"]
    ).to_csv(tmp_path / "single_index_solver_iterations.csv", index=False)
    pd.DataFrame(columns=["run_id", "scenario_id", "method", "category"]).to_csv(
        tmp_path / "single_index_failures.csv", index=False
    )
    pd.DataFrame(columns=ARTIFACT_COLUMNS).to_csv(
        tmp_path / "single_index_artifacts.csv", index=False
    )

    original = reports._render_plot

    def fail_g02(plot_id, *args, **kwargs):
        if plot_id == "G02":
            raise RuntimeError("forced plot failure")
        return original(plot_id, *args, **kwargs)

    monkeypatch.setattr(reports, "_render_plot", fail_g02)

    saved = write_single_index_reports(
        tmp_path,
        bootstrap_resamples=50,
        random_state=9,
    )

    assert saved["summary"].exists()
    assert saved["scaling"].exists()
    assert saved["paired"].exists()
    assert not any(column.startswith("Unnamed") for column in pd.read_csv(saved["summary"]))
    artifacts = pd.read_csv(saved["artifacts"])
    summary_artifact = artifacts.query("name == 'summary'").iloc[0]
    failed_plot = artifacts.query("name == 'G02'").iloc[0]
    assert summary_artifact["status"] == "created"
    assert failed_plot["status"] == "error"
    assert "forced plot failure" in failed_plot["error"]
