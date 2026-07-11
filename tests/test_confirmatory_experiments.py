import subprocess
import sys

import pandas as pd

import experiments.adp_confirmatory_common as confirmatory_common
from experiments.adp_confirmatory_common import (
    ConfirmatoryConfig,
    RunJob,
    ScenarioSpec,
    build_scenarios,
    final_success_summary,
    make_adp_config,
    run_confirmatory_experiments,
    summarize_records,
)


def test_build_scenarios_supports_experiments_4_5_and_synthetic_6():
    config = ConfirmatoryConfig(
        d_values=(6,),
        n_over_d_values=(8,),
        corr_values=(0.0,),
        snr_values=(20.0,),
        link_values=("linear",),
        q_values=(0.3,),
        seeds=1,
    )

    scenarios_4 = build_scenarios(config, experiment="4")
    scenarios_5 = build_scenarios(config, experiment="5")
    scenarios_6 = build_scenarios(config, experiment="6")

    assert len(scenarios_4) == 1
    assert len(scenarios_5) == 1
    assert len(scenarios_6) == 1
    assert scenarios_4[0].n == 48
    assert scenarios_5[0].q == 0.3
    assert scenarios_6[0].experiment == "6"


def test_h0_inflation_is_passed_to_adp_config():
    config = ConfirmatoryConfig(
        d_values=(5,),
        n_over_d_values=(8,),
        corr_values=(0.0,),
        snr_values=(20.0,),
        link_values=("linear",),
        q_values=(0.3,),
        h0_inflation=1.25,
    )
    scenario = build_scenarios(config, experiment="4")[0]

    adp_config = make_adp_config(config, scenario, random_state=123, method="full_adp")

    assert adp_config.initial_bandwidth_inflation == 1.25


def test_confirmatory_runner_writes_normalized_csv_logs_and_plots(tmp_path):
    config = ConfirmatoryConfig(
        d_values=(5,),
        n_over_d_values=(8,),
        corr_values=(0.0,),
        snr_values=(20.0,),
        link_values=("linear",),
        q_values=(0.3,),
        seeds=1,
        outer_steps=2,
        inner_steps=2,
        n_directions=4,
        min_neighbors=4.0,
        center_fraction=0.4,
        methods=("full_adp", "step0_only"),
        experiments=("4", "5", "6"),
        bootstrap_reps=20,
    )

    saved = run_confirmatory_experiments(config, tmp_path, n_jobs=1)

    assert saved["runs"].exists()
    assert saved["iterations"].exists()
    assert saved["summary"].exists()
    assert saved["final_success"].exists()
    assert saved["series"].exists()
    assert saved["artifacts"].exists()
    assert saved["initial_parameters"].exists()
    assert saved["rho_plot"].exists()
    assert saved["h_plot"].exists()
    assert saved["mass_plot"].exists()
    assert saved["rho_cos_scatter_plot"].exists()
    assert saved["cos_plot"].exists()
    assert saved["success_plot"].exists()
    assert saved["failure_plot"].exists()
    assert saved["ablation_plot"].exists()
    assert saved["final_success_plot"].exists()

    records = pd.read_csv(saved["iterations"])
    expected = {
        "experiment",
        "seed",
        "scenario_id",
        "method",
        "outer_k",
        "h_k",
        "rho_k",
        "local_mass_mean",
        "local_mass_q05",
        "local_mass_min",
        "cos_beta_k",
        "cos_delta_from_k0",
        "success_08",
        "success_09",
        "failed",
        "run_id",
    }
    assert expected.issubset(records.columns)
    assert {"4", "5", "6"}.issubset(set(records["experiment"].astype(str)))
    assert records["cos_beta_k"].between(0.0, 1.0).all()

    runs = pd.read_csv(saved["runs"])
    resource_columns = {
        "algorithm_time_sec",
        "algorithm_rss_min_mib",
        "algorithm_rss_mean_mib",
        "algorithm_rss_max_mib",
        "full_run_time_sec",
        "full_run_rss_min_mib",
        "full_run_rss_mean_mib",
        "full_run_rss_max_mib",
        "result_persist_time_sec",
    }
    assert resource_columns.issubset(runs.columns)
    assert (runs["algorithm_time_sec"] > 0.0).all()
    assert (runs["full_run_time_sec"] >= runs["algorithm_time_sec"]).all()
    assert set(runs["run_id"]) == set(records["run_id"])

    series = pd.read_csv(saved["series"]).iloc[0]
    assert series["config_experiments"] == "4|5|6"
    assert series["actual_n_jobs"] == 1
    assert bool(series["synthetic_experiment_6_from_final_success_protocol"])
    assert not list(tmp_path.glob("*.json"))


def test_summary_records_exposes_tests_md_checks_for_rho_and_cos_growth():
    config = ConfirmatoryConfig(bootstrap_reps=20, base_seed=10, n_directions=4)
    records = pd.DataFrame(
        [
            {
                "experiment": "4",
                "scenario_id": "s4",
                "method": "full_adp",
                "seed": 0,
                "outer_k": 0,
                "rho_k": None,
                "local_mass_mean": 12.0,
                "local_mass_q05": 8.0,
                "cos_beta_k": 0.40,
                "cos_delta_from_k0": 0.0,
                "success_08": False,
                "success_09": False,
                "failed": False,
                "runtime_sec": 0.1,
            },
            {
                "experiment": "4",
                "scenario_id": "s4",
                "method": "full_adp",
                "seed": 0,
                "outer_k": 1,
                "rho_k": 0.55,
                "local_mass_mean": 11.0,
                "local_mass_q05": 8.5,
                "cos_beta_k": 0.52,
                "cos_delta_from_k0": 0.12,
                "success_08": False,
                "success_09": False,
                "failed": False,
                "runtime_sec": 0.1,
            },
            {
                "experiment": "5",
                "scenario_id": "s5",
                "method": "full_adp",
                "seed": 0,
                "outer_k": 0,
                "rho_k": None,
                "local_mass_mean": 12.0,
                "local_mass_q05": 8.0,
                "cos_beta_k": 0.50,
                "cos_delta_from_k0": 0.0,
                "success_08": False,
                "success_09": False,
                "failed": False,
                "runtime_sec": 0.1,
            },
            {
                "experiment": "5",
                "scenario_id": "s5",
                "method": "full_adp",
                "seed": 0,
                "outer_k": 2,
                "rho_k": 0.35,
                "local_mass_mean": 10.0,
                "local_mass_q05": 8.2,
                "cos_beta_k": 0.85,
                "cos_delta_from_k0": 0.35,
                "success_08": True,
                "success_09": False,
                "failed": False,
                "runtime_sec": 0.1,
            },
            {
                "experiment": "5",
                "scenario_id": "s5",
                "method": "full_adp",
                "seed": 1,
                "outer_k": 0,
                "rho_k": None,
                "local_mass_mean": 12.0,
                "local_mass_q05": 8.0,
                "cos_beta_k": 0.40,
                "cos_delta_from_k0": 0.0,
                "success_08": False,
                "success_09": False,
                "failed": False,
                "runtime_sec": 0.1,
            },
            {
                "experiment": "5",
                "scenario_id": "s5",
                "method": "full_adp",
                "seed": 1,
                "outer_k": 2,
                "rho_k": 0.30,
                "local_mass_mean": 11.0,
                "local_mass_q05": 8.4,
                "cos_beta_k": 0.82,
                "cos_delta_from_k0": 0.42,
                "success_08": True,
                "success_09": False,
                "failed": False,
                "runtime_sec": 0.1,
            },
        ]
    )

    summary = summarize_records(records, config)
    exp4 = summary[summary["experiment"].astype(str) == "4"].iloc[0]
    exp5 = summary[summary["experiment"].astype(str) == "5"].iloc[0]

    assert exp4["rho_in_range_rate"] == 1.0
    assert exp4["local_mass_q05_gate_rate"] == 1.0
    assert exp4["rho_median_trend_ok"] is True
    assert exp5["improvement_rate"] == 1.0
    assert exp5["growth_median_positive"] is True
    assert exp5["growth_ci95_low_positive"] is True
    assert exp5["growth_pass"] is True


def test_experiment_5_cli_runs_parallel_smoke(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "experiments/adp_experiment_5_cos_growth.py",
            "--out",
            str(tmp_path),
            "--d",
            "5",
            "--n-over-d",
            "8",
            "--corr",
            "0.0",
            "--snr",
            "20",
            "--links",
            "linear",
            "--q",
            "0.3",
            "--seeds",
            "1",
            "--outer-steps",
            "2",
            "--inner-steps",
            "2",
            "--n-directions",
            "4",
            "--min-neighbors",
            "4",
            "--center-fraction",
            "0.4",
            "--methods",
            "full_adp,step0_only",
            "--jobs",
            "2",
            "--bootstrap-reps",
            "20",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "experiment_5_cos_growth_iterations.csv" in result.stdout
    assert (tmp_path / "experiment_5_cos_growth_runs.csv").exists()
    assert (tmp_path / "experiment_5_cos_growth_iterations.csv").exists()
    assert (tmp_path / "experiment_5_cos_growth_summary.csv").exists()
    assert (tmp_path / "experiment_5_cos_growth_series.csv").exists()
    assert (tmp_path / "experiment_5_cos_growth_artifacts.csv").exists()


def test_confirmatory_runner_reports_tqdm_job_totals_and_postfix(monkeypatch, tmp_path):
    scenario = ScenarioSpec(
        experiment="5",
        scenario_id="scenario_a",
        scenario_index=0,
        d=5,
        n=40,
        n_over_d=8,
        corr=0.0,
        snr=20.0,
        link="linear",
        q=0.3,
    )
    jobs = [
        RunJob(experiment="5", scenario=scenario, seed_id=0, method="full_adp"),
        RunJob(experiment="5", scenario=scenario, seed_id=1, method="step0_only"),
    ]

    tqdm_calls = []
    postfixes = []

    class FakeTqdm:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
            self.kwargs = kwargs
            tqdm_calls.append(kwargs)

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **kwargs):
            postfixes.append(kwargs)

    def fake_run_job(job, config):
        return [
            {
                "experiment": job.experiment,
                "scenario_id": job.scenario.scenario_id,
                "method": job.method,
                "seed": job.seed_id,
                "outer_k": 0,
            }
        ]

    def fake_save_plots(records, summary, final_success, output_dir, *, output_prefix):
        path = output_dir / f"{output_prefix}_plot.png"
        path.write_text("plot")
        return {"plot": path}

    monkeypatch.setattr(confirmatory_common, "tqdm", FakeTqdm)
    monkeypatch.setattr(confirmatory_common, "build_jobs", lambda config: jobs)
    monkeypatch.setattr(confirmatory_common, "run_job", fake_run_job)
    monkeypatch.setattr(confirmatory_common, "summarize_records", lambda records, config: pd.DataFrame())
    monkeypatch.setattr(confirmatory_common, "final_success_summary", lambda records: pd.DataFrame())
    monkeypatch.setattr(confirmatory_common, "save_plots", fake_save_plots)

    run_confirmatory_experiments(
        ConfirmatoryConfig(experiments=("5",)),
        tmp_path,
        n_jobs=1,
        output_prefix="progress_test",
    )

    assert tqdm_calls[0]["total"] == len(jobs)
    assert tqdm_calls[0]["unit"] == "job"
    assert tqdm_calls[0]["desc"] == "progress_test sequential"
    assert postfixes == [
        {
            "experiment": "5",
            "scenario": "scenario_a",
            "seed": 0,
            "method": "full_adp",
            "refresh": True,
        },
        {
            "experiment": "5",
            "scenario": "scenario_a",
            "seed": 1,
            "method": "step0_only",
            "refresh": True,
        },
    ]


def test_confirmatory_runner_writes_line_progress_for_redirected_logs(monkeypatch, tmp_path, capsys):
    scenario = ScenarioSpec(
        experiment="4",
        scenario_id="scenario_log",
        scenario_index=0,
        d=5,
        n=40,
        n_over_d=8,
        corr=0.0,
        snr=20.0,
        link="linear",
        q=0.3,
    )
    jobs = [
        RunJob(experiment="4", scenario=scenario, seed_id=0, method="full_adp"),
        RunJob(experiment="4", scenario=scenario, seed_id=1, method="full_adp"),
    ]

    class QuietTqdm:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **kwargs):
            pass

    def fake_run_job(job, config):
        return [
            {
                "experiment": job.experiment,
                "scenario_id": job.scenario.scenario_id,
                "method": job.method,
                "seed": job.seed_id,
                "outer_k": 0,
            }
        ]

    def fake_save_plots(records, summary, final_success, output_dir, *, output_prefix):
        path = output_dir / f"{output_prefix}_plot.png"
        path.write_text("plot")
        return {"plot": path}

    monkeypatch.setattr(confirmatory_common, "tqdm", QuietTqdm)
    monkeypatch.setattr(confirmatory_common, "build_jobs", lambda config: jobs)
    monkeypatch.setattr(confirmatory_common, "run_job", fake_run_job)
    monkeypatch.setattr(confirmatory_common, "summarize_records", lambda records, config: pd.DataFrame())
    monkeypatch.setattr(confirmatory_common, "final_success_summary", lambda records: pd.DataFrame())
    monkeypatch.setattr(confirmatory_common, "save_plots", fake_save_plots)

    run_confirmatory_experiments(
        ConfirmatoryConfig(experiments=("4",), progress_log_every=1),
        tmp_path,
        n_jobs=1,
        output_prefix="progress_log_test",
    )

    captured = capsys.readouterr()

    assert "progress_log_test sequential: 1/2 jobs" in captured.err
    assert "progress_log_test sequential: 2/2 jobs" in captured.err
    assert "experiment=4" in captured.err
    assert "scenario=scenario_log" in captured.err
    assert "method=full_adp" in captured.err


def test_confirmatory_runner_writes_initial_parameters_before_jobs(monkeypatch, tmp_path):
    scenario = ScenarioSpec(
        experiment="5",
        scenario_id="scenario_initial_parameters",
        scenario_index=3,
        d=5,
        n=40,
        n_over_d=8,
        corr=0.3,
        snr=10.0,
        link="tanh",
        q=0.3,
    )
    jobs = [
        RunJob(experiment="5", scenario=scenario, seed_id=2, method="full_adp"),
        RunJob(experiment="5", scenario=scenario, seed_id=2, method="step0_only"),
        RunJob(experiment="5", scenario=scenario, seed_id=2, method="no_regularization"),
        RunJob(experiment="5", scenario=scenario, seed_id=2, method="fixed_h"),
        RunJob(experiment="5", scenario=scenario, seed_id=2, method="random_beta_init"),
    ]
    initial_parameters_path = tmp_path / "initial_parameters_test_initial_parameters.csv"

    def fake_run_job(job, config):
        assert initial_parameters_path.exists()
        return [
            {
                "experiment": job.experiment,
                "scenario_id": job.scenario.scenario_id,
                "method": job.method,
                "seed": job.seed_id,
                "outer_k": 0,
            }
        ]

    def fake_save_plots(records, summary, final_success, output_dir, *, output_prefix):
        path = output_dir / f"{output_prefix}_plot.png"
        path.write_text("plot")
        return {"plot": path}

    monkeypatch.setattr(confirmatory_common, "build_jobs", lambda config: jobs)
    monkeypatch.setattr(confirmatory_common, "run_job", fake_run_job)
    monkeypatch.setattr(confirmatory_common, "summarize_records", lambda records, config: pd.DataFrame())
    monkeypatch.setattr(confirmatory_common, "final_success_summary", lambda records: pd.DataFrame())
    monkeypatch.setattr(confirmatory_common, "save_plots", fake_save_plots)

    config = ConfirmatoryConfig(
        experiments=("5",),
        base_seed=101,
        n_directions=7,
        min_neighbors=5.0,
        center_fraction=0.25,
        lambda_rel=0.02,
        outer_steps=6,
        inner_steps=9,
        gamma_h=0.8,
    )
    saved = run_confirmatory_experiments(
        config,
        tmp_path,
        n_jobs=1,
        output_prefix="initial_parameters_test",
    )

    assert saved["initial_parameters"] == initial_parameters_path
    parameters = pd.read_csv(initial_parameters_path)
    assert set(parameters["schema_version"]) == {1}
    assert set(parameters["series_config_outer_steps"]) == {6}
    assert len(parameters) == len(jobs)

    tests_by_method = {
        row["method"]: row for _, row in parameters.iterrows()
    }
    full_adp = tests_by_method["full_adp"]
    assert full_adp["data_seed"] == 50_302_223
    assert full_adp["fit_seed"] == 50_302_324
    assert full_adp["scenario_scenario_id"] == "scenario_initial_parameters"
    assert full_adp["config_n_centers"] == 10
    assert full_adp["config_lambda_penalty"] == 0.1
    assert tests_by_method["step0_only"]["config_outer_steps"] == 1
    assert tests_by_method["no_regularization"]["config_lambda_penalty"] == 0.0
    assert tests_by_method["fixed_h"]["config_bandwidth_decay"] == 1.0
    assert tests_by_method["random_beta_init"]["initial_beta_seed"] == 50_302_421
    assert tests_by_method["random_beta_init"]["beta0_kind"] == "random_sparse"
    assert tests_by_method["random_beta_init"]["beta0_q"] == 1.0
    assert tests_by_method["random_beta_init"]["beta0_seed"] == 50_302_421


def test_experiments_4_5_6_have_separate_cli_files(tmp_path):
    scripts = [
        (
            "experiments/adp_experiment_4_rho.py",
            "experiment_4_rho",
            "4",
        ),
        (
            "experiments/adp_experiment_5_cos_growth.py",
            "experiment_5_cos_growth",
            "5",
        ),
        (
            "experiments/adp_experiment_6_final_success.py",
            "experiment_6_final_success",
            "6",
        ),
    ]

    for script, prefix, experiment in scripts:
        out_dir = tmp_path / prefix
        result = subprocess.run(
            [
                sys.executable,
                script,
                "--out",
                str(out_dir),
                "--d",
                "5",
                "--n-over-d",
                "8",
                "--corr",
                "0.0",
                "--snr",
                "20",
                "--links",
                "linear",
                "--q",
                "0.3",
                "--seeds",
                "1",
                "--outer-steps",
                "2",
                "--inner-steps",
                "2",
                "--n-directions",
                "4",
                "--min-neighbors",
                "4",
                "--center-fraction",
                "0.4",
                "--methods",
                "full_adp,step0_only",
                "--jobs",
                "1",
                "--bootstrap-reps",
                "20",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        assert f"{prefix}_iterations.csv" in result.stdout
        assert (out_dir / f"{prefix}_runs.csv").exists()
        assert (out_dir / f"{prefix}_iterations.csv").exists()
        assert (out_dir / f"{prefix}_summary.csv").exists()
        assert (out_dir / f"{prefix}_series.csv").exists()
        series = pd.read_csv(out_dir / f"{prefix}_series.csv").iloc[0]
        assert str(series["config_experiments"]) == experiment


def test_final_success_summary_uses_tests_md_thresholds():
    records = pd.DataFrame(
        [
            {
                "scenario_id": "s1",
                "method": "full_adp",
                "outer_k": 2,
                "cos_beta_k": 0.86,
                "local_mass_q05": 8.0,
                "failed": False,
                "n_directions": 4,
            },
            {
                "scenario_id": "s1",
                "method": "full_adp",
                "outer_k": 2,
                "cos_beta_k": 0.82,
                "local_mass_q05": 9.0,
                "failed": False,
                "n_directions": 4,
            },
        ]
    )

    summary = final_success_summary(records)

    row = summary.iloc[0]
    assert row["median_cos_ge_08"] is True
    assert row["success_08_rate_ge_08"] is True
    assert row["failure_rate_le_005"] is True
    assert row["local_mass_gate"] is True
    assert row["protocol_pass"] is True
