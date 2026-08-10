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


def _scaled_kernel(scale, value, *, offset=0.0):
    return scale * np.asarray(value) + offset


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


def _paired_experiment(data_factory=None):
    from ADP.experiment import (
        ADP_Experiment,
        ADP_ExperimentPoint,
        ADP_ExperimentVariant,
    )

    return ADP_Experiment(
        name="paired",
        mode="single",
        runs=2,
        seed=20,
        points=(
            ADP_ExperimentPoint("p1", 24, 3, 0.0, {"sigma_eps": 0.0}),
            ADP_ExperimentPoint("p2", 24, 3, 0.1, {"sigma_eps": 0.1}),
        ),
        variants={
            "A": ADP_ExperimentVariant(
                ADP_Config(N_J=4, N_phi=3, h_min=1e6)
            ),
            "B": ADP_ExperimentVariant(
                ADP_Config(N_J=4, N_phi=3, h_min=1e6)
            ),
        },
        data_factory=data_factory,
    )


def test_jobs_share_seed_sequence_and_alternate_variants():
    from ADP.experiment_runner import _build_jobs

    jobs = _build_jobs(_paired_experiment())

    assert [(job.point.name, job.seed, job.variant_name) for job in jobs] == [
        ("p1", 20, "A"),
        ("p1", 20, "B"),
        ("p1", 21, "B"),
        ("p1", 21, "A"),
        ("p2", 20, "A"),
        ("p2", 20, "B"),
        ("p2", 21, "B"),
        ("p2", 21, "A"),
    ]


def test_job_ids_encode_names_without_delimiter_collisions(tmp_path: Path):
    from ADP.experiment import ADP_ExperimentPoint, ADP_ExperimentVariant
    from ADP.experiment_runner import _Job, _SeriesStore

    variant = ADP_ExperimentVariant(ADP_Config())
    left = _Job(
        ADP_ExperimentPoint("p", 24, 3),
        0,
        20,
        0,
        "v__seed-20__B",
        variant,
    )
    right = _Job(
        ADP_ExperimentPoint("p__seed-20__v", 24, 3),
        0,
        20,
        1,
        "B",
        variant,
    )
    store = _SeriesStore(tmp_path)

    assert left.run_id != right.run_id
    assert store.commit_path(left) != store.commit_path(right)


def test_job_spec_serializes_partial_kernel():
    import json
    from dataclasses import replace
    from functools import partial

    from ADP.experiment import ADP_ExperimentVariant
    from ADP.experiment_runner import _build_jobs, _job_spec

    kernel = partial(_scaled_kernel, 2.0, offset=[0.25])
    variant = ADP_ExperimentVariant(ADP_Config(kernel=kernel))
    base = _paired_experiment()
    experiment = replace(
        base,
        runs=1,
        points=(base.points[0],),
        variants={"partial": variant},
    )

    spec = _job_spec(_build_jobs(experiment)[0], experiment)

    assert spec["requested_config"]["kernel"] == {
        "function": f"{_scaled_kernel.__module__}:{_scaled_kernel.__qualname__}",
        "args": [2.0],
        "keywords": {"offset": [0.25]},
    }
    json.dumps(spec, allow_nan=False)


def test_spec_value_rejects_callable_without_stable_name():
    from ADP.experiment_runner import _spec_value

    class CallableKernel:
        def __call__(self, value):
            return value

    with pytest.raises(TypeError, match="stable qualified name"):
        _spec_value(CallableKernel())


def test_spec_value_recursively_normalizes_numpy_json_values(tmp_path: Path):
    import json

    from ADP.experiment_runner import _atomic_json, _spec_value

    left = np.longdouble("1.25")
    right = np.nextafter(left, np.longdouble(np.inf), dtype=np.longdouble)
    left_spec = _spec_value(left)
    right_spec = _spec_value(right)

    assert left_spec == {
        "__numpy_scalar__": "longdouble",
        "dtype": str(left.dtype),
        "value": np.format_float_scientific(left, unique=True, trim="k"),
    }
    assert left_spec != right_spec
    normalized = _spec_value(np.array([left, right]))
    assert normalized == [left_spec, right_spec]
    json.dumps(normalized, allow_nan=False)
    for value, expected_type in (
        (np.bool_(True), bool),
        (np.int64(2), int),
        (np.float32(3.5), float),
        (np.str_("x"), str),
    ):
        assert type(_spec_value(value)) is expected_type

    finite_path = tmp_path / "finite.json"
    finite = {
        "adjacent": normalized,
        "large": _spec_value(np.longdouble("1e400")),
    }
    _atomic_json(finite_path, finite)
    assert json.loads(finite_path.read_text(encoding="utf-8")) == finite

    for value in (
        np.longdouble("nan"),
        np.longdouble("inf"),
        np.longdouble("-inf"),
    ):
        with pytest.raises(ValueError, match="longdouble must be finite"):
            _spec_value(value)


def test_series_store_round_trips_truthless_data(tmp_path: Path):
    from ADP.experiment_runner import _SeriesStore

    experiment = _paired_experiment()
    store = _SeriesStore.create(tmp_path, experiment)
    point = experiment.points[0]
    data = ADP_Data(np.ones((24, 3)), np.zeros(24), None)

    store.save_data(point, 20, data)
    restored = store.load_data(point, 20)

    np.testing.assert_array_equal(restored.X, data.X)
    np.testing.assert_array_equal(restored.Y, data.Y)
    assert restored.true_index is None


def test_series_store_requires_matching_resume_spec(tmp_path: Path):
    from dataclasses import replace

    from ADP.experiment_runner import _SeriesStore, _build_jobs

    experiment = _paired_experiment()
    job = _build_jobs(experiment)[0]
    store = _SeriesStore.create(tmp_path, experiment)

    assert not store.is_complete(job, experiment)
    store.commit(job, experiment, {"status": "success"})
    assert store.is_complete(job, experiment)
    store.commit(job, experiment, {"status": "ignored"})
    assert store.read_commits()[0]["status"] == "success"

    changed = replace(experiment, name="changed")
    with pytest.raises(ValueError, match="resume specification differs"):
        store.is_complete(job, changed)
    with pytest.raises(ValueError, match="resume specification differs"):
        store.commit(job, changed, {"status": "success"})
