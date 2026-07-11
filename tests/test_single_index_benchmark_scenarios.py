import math

import pytest

from adp.evaluation.single_index.scenarios import (
    PROFILE_IDS,
    scenario_registry,
    scenarios_for_profile,
)
from adp.evaluation.single_index.types import (
    SeedBundle,
    SingleIndexJob,
    SingleIndexScenario,
    SingleIndexSeriesConfig,
)


def test_registry_covers_every_protocol_scenario_once():
    scenarios = scenario_registry()
    identifiers = [scenario.scenario_id for scenario in scenarios]
    expected = {
        *(f"C{index:02d}" for index in range(1, 13)),
        *(f"S{index:02d}" for index in range(1, 7)),
        *(f"T{index:02d}" for index in range(1, 11)),
        *(f"R{index:02d}" for index in range(1, 16)),
        *(f"M{index:02d}" for index in range(1, 9)),
        *(f"I{index:02d}" for index in range(1, 5)),
        "B01",
        *(f"A{index:02d}" for index in range(1, 10)),
        *(f"D{index:02d}" for index in range(1, 5)),
    }

    assert len(identifiers) == len(set(identifiers))
    assert set(identifiers) == expected


def test_registry_values_and_executor_routing_are_valid():
    for scenario in scenario_registry():
        assert scenario.family == scenario.scenario_id[0]
        assert scenario.repeats > 0
        assert scenario.methods
        assert scenario.hypothesis
        for values in (scenario.data, scenario.algorithm, scenario.solver):
            for value in values.values():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    assert math.isfinite(value)
        if "n" in scenario.data:
            assert scenario.data["n"] > 0
        if "d" in scenario.data:
            assert scenario.data["d"] > 0
        if scenario.family == "C":
            assert scenario.executor == "correctness"
        elif scenario.family == "D":
            assert scenario.executor == "real_data"
        elif scenario.family == "M":
            assert scenario.executor == "scaling"
        else:
            assert scenario.executor == "recovery"


def test_profiles_are_nested_and_smoke_has_required_coverage():
    smoke = set(PROFILE_IDS["smoke"])
    minimal = set(PROFILE_IDS["minimal"])
    full = set(PROFILE_IDS["full"])

    assert smoke < minimal < full
    assert {"C01", "S01", "M01", "B01"}.issubset(smoke)
    assert {scenario.scenario_id for scenario in scenarios_for_profile("smoke")} == smoke
    assert scenarios_for_profile("publication")


def test_scenario_rejects_invalid_dimensions_and_nonfinite_values():
    with pytest.raises(ValueError, match="data.n must be positive"):
        SingleIndexScenario(
            scenario_id="S99",
            family="S",
            executor="recovery",
            hypothesis="invalid n",
            data={"n": 0, "d": 2},
            algorithm={"n_centers": 2, "n_directions": 2},
            solver={"outer_steps": 1},
            repeats=1,
            methods=("full_adp",),
        )

    with pytest.raises(ValueError, match="data.noise must be finite"):
        SingleIndexScenario(
            scenario_id="S99",
            family="S",
            executor="recovery",
            hypothesis="invalid noise",
            data={"n": 10, "d": 2, "noise": math.nan},
            algorithm={"n_centers": 2, "n_directions": 2},
            solver={"outer_steps": 1},
            repeats=1,
            methods=("full_adp",),
        )


def test_series_config_and_job_validate_parallelism_and_identity():
    with pytest.raises(ValueError, match="jobs must be positive"):
        SingleIndexSeriesConfig(profile="smoke", base_seed=1, jobs=0, statistics_workers=1)
    with pytest.raises(ValueError, match="statistics_workers must be positive"):
        SingleIndexSeriesConfig(profile="smoke", base_seed=1, jobs=1, statistics_workers=0)

    scenario = scenarios_for_profile("smoke")[0]
    seeds = SeedBundle(data=1, beta=2, centers=3, directions=4, init=5)
    job = SingleIndexJob(
        scenario=scenario,
        method=scenario.methods[0],
        repeat=0,
        seeds=seeds,
        run_id="run-test",
    )

    assert job.scenario == scenario
    assert job.seeds.directions == 4


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="unknown single-index profile"):
        scenarios_for_profile("missing")
