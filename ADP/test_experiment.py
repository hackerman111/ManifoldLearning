from pathlib import Path

import numpy as np
import pytest

from ADP import ADP_Config, ADP_Data, ADP_SolverResult, ADP_solver


def _unchanged_single(statistics, initial_index, **params):
    return ADP_SolverResult(
        index=np.asarray(initial_index),
        coefficients=np.ones(len(statistics.I)),
        diagnostics={"inner_iterations": 1, "beta_delta": 0.0},
    )


def _unchanged_multi(statistics, initial_index, **params):
    width = np.asarray(initial_index).shape[1]
    return ADP_SolverResult(
        index=np.asarray(initial_index),
        coefficients=np.ones((len(statistics.I), width)),
        diagnostics={
            "inner_iterations": 1,
            "beta_delta": 0.0,
            "eigenvalues": np.ones(width),
        },
    )


def test_python_file_exports_valid_experiment(tmp_path: Path):
    from ADP.experiment import load_experiment

    source = tmp_path / "experiment_file.py"
    source.write_text(
        """
from ADP import ADP_Config
from ADP.experiment import ADP_Experiment, ADP_ExperimentPoint, ADP_ExperimentVariant

experiment = ADP_Experiment(
    name="pair",
    mode="single",
    runs=2,
    seed=11,
    points=(ADP_ExperimentPoint("noise", 24, 3, 0.1, {"sigma_eps": 0.1}),),
    variants={"base": ADP_ExperimentVariant(ADP_Config(), "auto")},
)
""",
        encoding="utf-8",
    )

    experiment = load_experiment(source)

    assert experiment.name == "pair"
    assert experiment.points[0].metadata == {"sigma_eps": 0.1}
    assert tuple(experiment.variants) == ("base",)


def test_multi_varpro_is_rejected():
    from ADP.experiment import (
        ADP_Experiment,
        ADP_ExperimentPoint,
        ADP_ExperimentVariant,
        validate_experiment,
    )

    experiment = ADP_Experiment(
        name="bad",
        mode="multi",
        index_dim=2,
        points=(ADP_ExperimentPoint("p", 24, 4),),
        variants={"v": ADP_ExperimentVariant(ADP_Config(), "varpro")},
    )

    with pytest.raises(ValueError, match="varpro.*multi"):
        validate_experiment(experiment)


def test_default_multi_data_and_optional_truth():
    from ADP.experiment import (
        ADP_Experiment,
        ADP_ExperimentPoint,
        ADP_ExperimentVariant,
        make_data,
    )

    point = ADP_ExperimentPoint("p", 30, 5, 0.0)
    experiment = ADP_Experiment(
        name="multi",
        mode="multi",
        index_dim=2,
        points=(point,),
        variants={"v": ADP_ExperimentVariant(ADP_Config(), "lsmr")},
    )
    first = make_data(experiment, point, 13)
    second = make_data(experiment, point, 13)

    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.Y, second.Y)
    assert first.true_index.shape == (5, 2)
    np.testing.assert_allclose(
        first.true_index.T @ first.true_index,
        np.eye(2),
        atol=1e-12,
    )
    assert ADP_Data(first.X, first.Y, None).true_index is None


def test_points_must_be_a_tuple():
    from ADP.experiment import (
        ADP_Experiment,
        ADP_ExperimentPoint,
        ADP_ExperimentVariant,
        validate_experiment,
    )

    point = ADP_ExperimentPoint("p", 24, 3)
    experiment = ADP_Experiment(
        name="iterator",
        mode="single",
        points=iter((point,)),
        variants={"v": ADP_ExperimentVariant(ADP_Config())},
    )

    with pytest.raises(TypeError, match="points.*tuple"):
        validate_experiment(experiment)


@pytest.mark.parametrize(
    ("truth", "error", "message"),
    [
        (np.array([1.0 + 1.0j, 0.0, 0.0]), TypeError, "real numeric"),
        (
            np.full(3, np.finfo(float).max),
            ValueError,
            "norm.*finite.*nonzero",
        ),
    ],
)
def test_invalid_single_truth_is_rejected(truth, error, message):
    from ADP.experiment import (
        ADP_Experiment,
        ADP_ExperimentPoint,
        ADP_ExperimentVariant,
        validate_data,
    )

    point = ADP_ExperimentPoint("p", 6, 3)
    experiment = ADP_Experiment(
        name="truth",
        mode="single",
        points=(point,),
        variants={"v": ADP_ExperimentVariant(ADP_Config())},
    )
    data = ADP_Data(np.zeros((6, 3)), np.zeros(6), truth)

    with pytest.raises(error, match=message):
        validate_data(data, experiment, point)


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_fit_reports_each_outer_iteration(mode):
    from ADP import ADP_Config, ADP_multi_index, ADP_single_index

    rng = np.random.default_rng(5)
    X = rng.normal(size=(30, 4))
    Y = np.sin(X[:, 0])
    config = ADP_Config(
        seed=5,
        N_loc=6,
        N_lin=8,
        N_J=5,
        N_phi=3,
        h_min=1e6,
        index_init="random",
    )
    events = []
    if mode == "single":
        model = ADP_single_index(config, ADP_solver(_unchanged_single))
    else:
        model = ADP_multi_index(2, config, ADP_solver(_unchanged_multi))

    model.fit(X, Y, progress=events.append)

    assert len(events) == len(model.result_.trace) == 1
    assert events[0]["k"] == 0
    assert (("rho" in events[0]) if mode == "single" else ("alpha" in events[0]))
    assert model.coefficients_ is not None


def test_auto_solver_uses_current_single_model_default():
    from ADP.experiment import (
        ADP_Experiment,
        ADP_ExperimentPoint,
        ADP_ExperimentVariant,
    )
    from ADP.experiment_runner import _build_model
    from ADP.single_index.solvers.LSMR import solve as solve_lsmr

    experiment = ADP_Experiment(
        name="auto",
        mode="single",
        points=(ADP_ExperimentPoint("p", 24, 3),),
        variants={
            "v": ADP_ExperimentVariant(ADP_Config(), "auto", {"max_steps": 7})
        },
    )

    model = _build_model(experiment, experiment.variants["v"], seed=9)

    assert model.solver.method is solve_lsmr
    assert model.solver.settings == {"tol": 1e-8, "max_steps": 7}
    assert model.config.seed == 9


def test_auto_solver_uses_current_multi_model_default():
    from ADP.experiment import (
        ADP_Experiment,
        ADP_ExperimentPoint,
        ADP_ExperimentVariant,
    )
    from ADP.experiment_runner import _build_model
    from ADP.single_index.solvers.LSMR import solve as solve_lsmr

    experiment = ADP_Experiment(
        name="auto_multi",
        mode="multi",
        index_dim=2,
        points=(ADP_ExperimentPoint("p", 24, 4),),
        variants={
            "v": ADP_ExperimentVariant(ADP_Config(), "auto", {"max_steps": 7})
        },
    )

    model = _build_model(experiment, experiment.variants["v"], seed=9)

    assert model.solver.method is solve_lsmr
    assert model.solver.settings == {"tol": 1e-7 / 2, "max_steps": 7}
    assert model.config.seed == 9
