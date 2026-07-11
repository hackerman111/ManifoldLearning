from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from adp.common.experiment_log import stable_run_id
from adp.evaluation.metrics import direction_metrics
from adp.evaluation.single_index.baselines import (
    BaselineUnavailable,
    fit_baseline,
)
from adp.evaluation.single_index.correctness import run_correctness
from adp.evaluation.single_index.datasets import (
    DatasetUnavailable,
    generate_synthetic_data,
    load_cached_real_dataset,
)
from adp.evaluation.single_index.executors import RunOutcome, execute_job
from adp.evaluation.single_index.scenarios import scenario_registry, scenarios_for_profile
from adp.evaluation.single_index.types import (
    SeedBundle,
    SingleIndexJob,
    SingleIndexSeriesConfig,
)


def make_job(scenario, *, repeat=0, seeds=None, method=None):
    seeds = seeds or SeedBundle(data=11, beta=12, centers=13, directions=14, init=15)
    method = method or scenario.methods[0]
    return SingleIndexJob(
        scenario=scenario,
        method=method,
        repeat=repeat,
        seeds=seeds,
        run_id=stable_run_id(
            "single_index",
            scenario.scenario_id,
            method,
            repeat,
            config_fingerprint="executor-test",
        ),
    )


def test_synthetic_seed_components_are_reproducible_and_isolated():
    scenario = scenarios_for_profile("smoke")[1]
    job = make_job(scenario)
    same = make_job(scenario)
    changed_centers = make_job(
        scenario,
        seeds=replace(job.seeds, centers=job.seeds.centers + 1),
    )
    changed_directions = make_job(
        scenario,
        seeds=replace(job.seeds, directions=job.seeds.directions + 1),
    )

    first = generate_synthetic_data(job)
    repeated = generate_synthetic_data(same)
    new_centers = generate_synthetic_data(changed_centers)
    new_directions = generate_synthetic_data(changed_directions)

    np.testing.assert_array_equal(first.X, repeated.X)
    np.testing.assert_array_equal(first.y, repeated.y)
    np.testing.assert_array_equal(first.beta, repeated.beta)
    np.testing.assert_array_equal(first.centers, repeated.centers)
    np.testing.assert_array_equal(first.directions, repeated.directions)
    np.testing.assert_array_equal(first.X, new_centers.X)
    np.testing.assert_array_equal(first.beta, new_centers.beta)
    assert not np.array_equal(first.centers, new_centers.centers)
    np.testing.assert_array_equal(first.X, new_directions.X)
    np.testing.assert_array_equal(first.centers, new_directions.centers)
    assert not np.array_equal(first.directions, new_directions.directions)


def test_corr_creates_empirical_feature_covariance():
    base = scenarios_for_profile("smoke")[1]
    scenario = replace(
        base,
        data={**base.data, "n": 6000, "d": 6, "corr": 0.65, "noise": 0.0},
        algorithm={**base.algorithm, "n_centers": 12, "n_directions": 4},
    )

    data = generate_synthetic_data(make_job(scenario))
    correlation = np.corrcoef(data.X, rowvar=False)
    off_diagonal = correlation[np.triu_indices_from(correlation, k=1)]

    assert np.mean(off_diagonal) == pytest.approx(0.65, abs=0.04)
    assert np.min(off_diagonal) > 0.55


def test_linear_and_random_baselines_follow_adapter_contract():
    base = scenarios_for_profile("smoke")[1]
    scenario = replace(
        base,
        data={**base.data, "n": 300, "d": 5, "link": "linear", "noise": 0.0},
    )
    data = generate_synthetic_data(make_job(scenario))

    ols = fit_baseline("ols", data.X, data.y, seed=7)
    random_first = fit_baseline("random_direction", data.X, data.y, seed=7)
    random_second = fit_baseline("random_direction", data.X, data.y, seed=7)

    assert direction_metrics(ols, data.beta)["cosine_abs"] > 0.95
    np.testing.assert_array_equal(random_first, random_second)
    with pytest.raises(BaselineUnavailable, match="mave"):
        fit_baseline("mave", data.X, data.y, seed=7)


def test_cached_real_dataset_requires_explicit_local_file(tmp_path):
    with pytest.raises(DatasetUnavailable, match="D01"):
        load_cached_real_dataset("D01", tmp_path, allow_download=False)

    pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0],
            "x2": [2.0, 1.0, 0.0],
            "target": [0.5, 1.0, 1.5],
        }
    ).to_csv(tmp_path / "D01.csv", index=False)

    dataset = load_cached_real_dataset("D01", tmp_path, allow_download=False)

    assert dataset.X.shape == (3, 2)
    assert dataset.y.shape == (3,)
    assert dataset.sha256
    assert dataset.path.name == "D01.csv"


def test_every_correctness_scenario_has_a_finite_executor_result():
    correctness = [scenario for scenario in scenario_registry() if scenario.family == "C"]

    for scenario in correctness:
        outcome = run_correctness(make_job(scenario))
        assert isinstance(outcome, RunOutcome)
        assert outcome.stop_reason
        assert outcome.metrics["passed"] is True
        assert np.isfinite(outcome.metrics["primary_error"])
        assert outcome.metrics["primary_error"] >= 0.0


def test_execute_recovery_job_returns_metrics_iterations_and_algorithm_usage():
    scenario = scenarios_for_profile("smoke")[1]
    config = SingleIndexSeriesConfig(
        profile="smoke",
        base_seed=1,
        jobs=1,
        statistics_workers=1,
    )

    outcome = execute_job(make_job(scenario), config)

    assert isinstance(outcome, RunOutcome)
    assert 0.0 <= outcome.metrics["cosine_abs"] <= 1.0
    assert outcome.iterations
    assert outcome.algorithm_usage["algorithm_time_sec"] > 0.0
