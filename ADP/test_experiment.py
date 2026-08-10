import csv
from pathlib import Path

import numpy as np
import pytest

from ADP import ADP_Config, ADP_Data, ADP_SolverResult, ADP_solver


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


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


def test_runner_pairs_inputs_saves_failures_and_resumes(tmp_path: Path, monkeypatch):
    import json
    from dataclasses import replace

    import ADP.experiment_runner as runner

    calls = []
    factory_calls = []

    def data_factory(point, rng):
        factory_calls.append(point.name)
        X = rng.normal(size=(point.n, point.d))
        truth = np.eye(point.d)[0]
        return ADP_Data(X, X @ truth, truth)

    class FakeResult:
        def __init__(self, beta):
            self.beta_init = beta.copy()
            self.beta_final = beta.copy()
            self.trace = [
                {
                    "k": 0,
                    "h": 1.0,
                    "rho": 1.0,
                    "mean_mass": 5.0,
                    "beta": beta.copy(),
                }
            ]
            self.stop_reason = "h_min"

    class FakeModel:
        def __init__(self, seed, variant):
            self.config = ADP_Config(seed=seed)
            self.variant = variant
            self.solver = ADP_solver(_unchanged_single)

        def fit(self, X, Y, *, progress=None):
            calls.append((self.variant, self.config.seed, X.copy(), Y.copy()))
            if progress is not None:
                progress({"k": 0, "h": 1.0, "rho": 1.0})
            beta = np.eye(X.shape[1])[0]
            self.result_ = FakeResult(beta)
            self.beta_ = beta
            self.coefficients_ = np.ones(len(X))
            self.effective_parameters_ = {
                "N_J": 4,
                "N_phi": 3,
                "N_loc": 5,
                "N_lin": 6,
                "h_min": 1.0,
            }
            self.profile_ = {
                "total_time_seconds": 0.01,
                "peak_memory_bytes": 32,
                "stages": {
                    "solve": {"time_seconds": 0.004, "memory_bytes": 64}
                },
            }
            return self

    experiment = _paired_experiment(data_factory)
    monkeypatch.setattr(
        runner,
        "_build_model",
        lambda experiment, variant, seed: FakeModel(
            seed,
            next(
                name
                for name, value in experiment.variants.items()
                if value is variant
            ),
        ),
    )

    series_dir, failures = runner.run_experiment(
        experiment,
        tmp_path,
        save_models=False,
        show_progress=False,
    )
    runs = _read_csv(series_dir / "run_summary.csv")
    outer_rows = _read_csv(series_dir / "outer_iterations.csv")
    paired = _read_csv(series_dir / "paired_comparison.csv")
    summary = _read_csv(series_dir / "comparison_summary.csv")
    series = _read_csv(series_dir / "series.csv")

    assert len(runs) == 8
    assert len(outer_rows) == 8
    assert len(paired) == 4
    assert summary
    assert series[0]["status"] == "complete"
    assert {
        "variant",
        "cosine_abs",
        "projector_error",
        "fit_wall_time_sec",
        "algorithm_rss_max_mib",
        "stage_solve_time_sec",
        "stage_solve_memory_mib",
    } <= set(runs[0])
    assert {
        "A_variant",
        "B_variant",
        "A_seed",
        "B_seed",
        "A_data_artifact",
        "B_data_artifact",
        "delta_fit_wall_time_sec",
        "delta_stage_solve_time_sec",
    } <= set(paired[0])
    assert "winner" not in paired[0]
    for name in (
        "run_summary.csv",
        "outer_iterations.csv",
        "paired_comparison.csv",
        "comparison_summary.csv",
    ):
        text = (series_dir / name).read_text(encoding="utf-8").lower()
        assert ",nan" not in text
        assert ",inf" not in text
        assert ",-inf" not in text

    run_summary_path = series_dir / "run_summary.csv"
    complete_summary = run_summary_path.read_bytes()
    complete_tables = {
        path.name: path.read_bytes() for path in series_dir.glob("*.csv")
    }
    store = runner._SeriesStore(series_dir)
    first_job = runner._build_jobs(experiment)[0]
    extraneous = json.loads(
        store.commit_path(first_job).read_text(encoding="utf-8")
    )
    extraneous["run"]["status"] = "numerical_failure"
    runner._atomic_json(store.commit_dir / "zz-extraneous.json", extraneous)

    first_call_count = len(calls)
    resumed_dir, resumed_failures = runner.run_experiment(
        experiment,
        tmp_path,
        resume=series_dir,
        save_models=False,
        show_progress=False,
    )

    assert failures == resumed_failures == 0
    assert resumed_dir == series_dir
    assert run_summary_path.read_bytes() == complete_summary
    assert {
        path.name: path.read_bytes() for path in series_dir.glob("*.csv")
    } == complete_tables
    assert len(factory_calls) == 4
    assert len(calls) == first_call_count == 8
    for offset in range(0, len(calls), 2):
        assert calls[offset][1] == calls[offset + 1][1]
        np.testing.assert_array_equal(calls[offset][2], calls[offset + 1][2])
        np.testing.assert_array_equal(calls[offset][3], calls[offset + 1][3])
    assert not list((series_dir / "models").glob("*.npz"))

    changed_a = replace(
        experiment.variants["A"],
        config=replace(experiment.variants["A"].config, N_J=5),
    )
    changed = replace(
        experiment,
        variants={"A": changed_a, "B": experiment.variants["B"]},
    )
    with pytest.raises(ValueError, match="resume specification differs"):
        runner.run_experiment(
            changed,
            tmp_path,
            resume=series_dir,
            save_models=False,
            show_progress=False,
        )
    assert run_summary_path.read_bytes() == complete_summary

    pending_job = runner._build_jobs(experiment)[1]
    store.commit_path(pending_job).unlink()
    store.data_path(pending_job.point, pending_job.seed).unlink()
    calls_before_broken_resume = len(calls)
    factory_calls_before_broken_resume = len(factory_calls)
    summary_before_broken_resume = run_summary_path.read_bytes()

    with pytest.raises(FileNotFoundError, match="referenced data artifact"):
        runner.run_experiment(
            experiment,
            tmp_path,
            resume=series_dir,
            save_models=False,
            show_progress=False,
        )

    assert len(calls) == calls_before_broken_resume
    assert len(factory_calls) == factory_calls_before_broken_resume
    assert run_summary_path.read_bytes() == summary_before_broken_resume


def test_export_tables_keep_stable_empty_headers_and_reject_metadata_collisions(
    tmp_path: Path,
):
    from dataclasses import replace

    import ADP.experiment_runner as runner

    base = _paired_experiment()
    experiment = replace(
        base,
        runs=1,
        points=(base.points[0],),
        variants={"A": base.variants["A"]},
    )
    jobs = runner._build_jobs(experiment)
    store = runner._SeriesStore.create(tmp_path, experiment)
    run = runner._base_run(jobs[0], experiment, store.series_dir.name)
    run["stage_solve_time_sec"] = 0.01
    store.commit(jobs[0], experiment, {"run": run, "outer": []})

    runner._export_tables(store, experiment, jobs, status="partial")

    with (store.series_dir / "paired_comparison.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        paired = csv.DictReader(stream)
        assert list(paired) == []
        assert paired.fieldnames == list(runner.PAIR_COLUMNS)
        assert "winner" not in paired.fieldnames
    for name, expected in runner.DETAIL_HEADERS.items():
        with (store.series_dir / name).open(
            newline="", encoding="utf-8"
        ) as stream:
            detail = csv.DictReader(stream)
            assert list(detail) == []
            assert detail.fieldnames == list(expected)

    sanitized = store.series_dir / "sanitized.csv"
    runner._atomic_csv(
        sanitized,
        ("metric", "payload", "callable"),
        (
            {
                "metric": np.longdouble("nan"),
                "payload": {"bad": np.inf},
                "callable": _unchanged_single,
            },
        ),
    )
    row = _read_csv(sanitized)[0]
    assert row == {
        "metric": "",
        "payload": '{"bad":""}',
        "callable": (
            f"{_unchanged_single.__module__}:"
            f"{_unchanged_single.__qualname__}"
        ),
    }

    colliding_point = replace(base.points[0], metadata={"variant": "bad"})
    colliding = replace(experiment, points=(colliding_point,))
    colliding_store = runner._SeriesStore.create(tmp_path, colliding)
    with pytest.raises(ValueError, match="metadata.*variant.*collides"):
        runner._export_tables(
            colliding_store,
            colliding,
            runner._build_jobs(colliding),
            status="partial",
        )


def test_runner_commits_fit_failures_and_continues(tmp_path: Path, monkeypatch):
    import ADP.experiment_runner as runner

    calls = []

    class SometimesFails:
        def __init__(self, seed, variant):
            self.config = ADP_Config(seed=seed)
            self.variant = variant
            self.solver = ADP_solver(_unchanged_single)

        def fit(self, X, Y, *, progress=None):
            calls.append((self.variant, self.config.seed))
            if self.variant == "A":
                raise FloatingPointError("synthetic failure")
            beta = np.eye(X.shape[1])[0]
            self.result_ = type(
                "Result",
                (),
                {
                    "trace": [
                        {"k": 0, "h": 1.0, "rho": 1.0, "beta": beta}
                    ],
                    "stop_reason": "h_min",
                },
            )()
            self.beta_ = beta
            self.coefficients_ = np.ones(len(X))
            self.effective_parameters_ = {}
            self.profile_ = {
                "total_time_seconds": 0.01,
                "peak_memory_bytes": 32,
                "stages": {},
            }
            return self

    experiment = _paired_experiment()
    monkeypatch.setattr(
        runner,
        "_build_model",
        lambda experiment, variant, seed: SometimesFails(
            seed,
            next(
                name
                for name, value in experiment.variants.items()
                if value is variant
            ),
        ),
    )

    series_dir, failures = runner.run_experiment(
        experiment,
        tmp_path,
        save_models=False,
        show_progress=False,
    )

    assert len(calls) == 8
    assert failures == 4
    assert len(list((series_dir / "commits").glob("*.json"))) == 8
    paired = _read_csv(series_dir / "paired_comparison.csv")
    assert len(paired) == 4
    assert {row["A_status"] for row in paired} == {"numerical_failure"}
    assert {row["B_status"] for row in paired} == {"success"}
    for row in paired:
        assert row["delta_cosine_abs"] == ""
        assert row["delta_projector_error"] == ""
        assert row["delta_algorithm_time_sec"] == ""
        assert row["A_outer_iterations"] == ""
        assert row["B_outer_iterations"] == "1"
        assert row["delta_outer_iterations"] == ""
        for metric in (
            "fit_wall_time_sec",
            "algorithm_rss_peak_delta_mib",
        ):
            assert float(row[f"delta_{metric}"]) == pytest.approx(
                float(row[f"B_{metric}"]) - float(row[f"A_{metric}"])
            )
    failed_runs = [
        commit["run"]
        for commit in runner._SeriesStore(series_dir).read_commits()
        if commit["run"]["status"] == "numerical_failure"
    ]
    for run in failed_runs:
        assert run["series_id"] == series_dir.name
        assert run["experiment"] == experiment.name
        assert run["mode"] == experiment.mode
        assert run["index_dim"] == experiment.index_dim
        assert run["effective_config"]["seed"] == run["seed"]
        assert run["effective_seed"] == run["seed"]
        assert run["effective_N_J"] == 4
        assert run["error_traceback"]


def test_data_generation_failures_keep_provenance(tmp_path: Path):
    from dataclasses import replace

    import ADP.experiment_runner as runner

    def fail_factory(point, rng):
        raise ValueError("synthetic data failure")

    base = _paired_experiment(fail_factory)
    experiment = replace(base, runs=1, points=(base.points[0],))

    series_dir, failures = runner.run_experiment(
        experiment,
        tmp_path,
        save_models=False,
        show_progress=False,
    )

    commits = runner._SeriesStore(series_dir).read_commits()
    assert failures == len(commits) == 2
    paired = _read_csv(series_dir / "paired_comparison.csv")
    assert len(paired) == 1
    assert paired[0]["A_outer_iterations"] == ""
    assert paired[0]["B_outer_iterations"] == ""
    assert paired[0]["delta_outer_iterations"] == ""
    assert not any(
        row["metric"] == "outer_iterations"
        for row in _read_csv(series_dir / "comparison_summary.csv")
    )
    for commit in commits:
        run = commit["run"]
        assert run["series_id"] == series_dir.name
        assert run["experiment"] == experiment.name
        assert run["mode"] == experiment.mode
        assert run["index_dim"] == experiment.index_dim
        assert run["effective_config"]["seed"] == run["seed"]
        assert run["effective_seed"] == run["seed"]
        assert run["data_artifact"] == ""
        assert run["model_artifact"] == ""
        assert run["error_type"] == "ValueError"
        assert "synthetic data failure" in run["error_traceback"]


@pytest.mark.parametrize("missing", ["basis_", "eigenvalues_"])
def test_multi_result_requires_fitted_basis_and_eigenvalues(
    tmp_path: Path,
    monkeypatch,
    missing,
):
    import ADP.experiment_runner as runner
    from ADP.experiment import (
        ADP_Experiment,
        ADP_ExperimentPoint,
        ADP_ExperimentVariant,
    )

    point = ADP_ExperimentPoint("p", 12, 3, 0.0)
    experiment = ADP_Experiment(
        name="strict_multi",
        mode="multi",
        index_dim=2,
        points=(point,),
        variants={"v": ADP_ExperimentVariant(ADP_Config())},
    )
    job = runner._build_jobs(experiment)[0]
    store = runner._SeriesStore.create(tmp_path, experiment)
    basis = np.eye(point.d, experiment.index_dim)
    data = ADP_Data(np.ones((point.n, point.d)), np.zeros(point.n), basis)
    store.save_data(point, job.seed, data)

    class MalformedMulti:
        config = ADP_Config(seed=job.seed)
        solver = ADP_solver(_unchanged_multi)

        def fit(self, X, Y, *, progress=None):
            self.beta_ = basis
            if missing != "basis_":
                self.basis_ = basis
            if missing != "eigenvalues_":
                self.eigenvalues_ = np.ones(experiment.index_dim)
            self.coefficients_ = np.ones((len(X), experiment.index_dim))
            self.effective_parameters_ = {}
            self.profile_ = {
                "total_time_seconds": 0.01,
                "peak_memory_bytes": 32,
                "stages": {},
            }
            self.result_ = type(
                "Result",
                (),
                {
                    "trace": [
                        {
                            "k": 0,
                            "h": 1.0,
                            "alpha": 1.0,
                            "basis": basis,
                            "eigenvalues": np.ones(experiment.index_dim),
                        }
                    ],
                    "stop_reason": "h_min",
                },
            )()
            return self

    monkeypatch.setattr(
        runner,
        "_build_model",
        lambda experiment, variant, seed: MalformedMulti(),
    )

    outcome = runner._execute_job(
        store,
        experiment,
        job,
        data,
        False,
        None,
    )

    assert outcome["run"]["status"] == "numerical_failure"
    assert missing in outcome["run"]["error_message"]
    assert outcome["run"]["model_artifact"] == ""


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (
            RuntimeError("outer_steps exhausted before reaching h_min"),
            "nonconverged",
        ),
        (FloatingPointError("synthetic numerical failure"), "numerical_failure"),
    ],
)
def test_execute_job_classifies_fit_exceptions(
    tmp_path: Path,
    monkeypatch,
    error,
    status,
):
    from dataclasses import replace

    import ADP.experiment_runner as runner

    base = _paired_experiment()
    experiment = replace(
        base,
        runs=1,
        points=(base.points[0],),
        variants={"A": base.variants["A"]},
    )
    job = runner._build_jobs(experiment)[0]
    store = runner._SeriesStore.create(tmp_path, experiment)
    truth = np.eye(job.point.d)[0]
    data = ADP_Data(
        np.ones((job.point.n, job.point.d)),
        np.zeros(job.point.n),
        truth,
    )
    store.save_data(job.point, job.seed, data)

    class Raises:
        config = replace(job.variant.config, seed=job.seed)
        solver = ADP_solver(_unchanged_single)

        def fit(self, X, Y, *, progress=None):
            self.profile_ = {
                "total_time_seconds": 0.01,
                "peak_memory_bytes": 32,
                "stages": {},
            }
            raise error

    monkeypatch.setattr(
        runner,
        "_build_model",
        lambda experiment, variant, seed: Raises(),
    )

    outcome = runner._execute_job(
        store,
        experiment,
        job,
        data,
        False,
        None,
    )
    run = outcome["run"]

    assert run["status"] == status
    assert run["error_type"] == type(error).__name__
    assert run["error_message"] == str(error)
    assert type(error).__name__ in run["error_traceback"]
    assert str(error) in run["error_traceback"]
    assert run["algorithm_memory_samples"] >= 2


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_successful_job_saves_required_model_arrays(tmp_path: Path, monkeypatch, mode):
    from ADP.experiment import (
        ADP_Experiment,
        ADP_ExperimentPoint,
        ADP_ExperimentVariant,
    )
    import ADP.experiment_runner as runner

    point = ADP_ExperimentPoint("p", 12, 3, 0.0)
    index_dim = 1 if mode == "single" else 2
    experiment = ADP_Experiment(
        name=f"snapshot_{mode}",
        mode=mode,
        index_dim=index_dim,
        points=(point,),
        variants={"v": ADP_ExperimentVariant(ADP_Config())},
    )
    job = runner._build_jobs(experiment)[0]
    store = runner._SeriesStore.create(tmp_path, experiment)
    index = (
        np.eye(point.d)[0]
        if mode == "single"
        else np.eye(point.d, index_dim)
    )
    data = ADP_Data(
        np.ones((point.n, point.d)),
        np.zeros(point.n),
        index,
    )
    store.save_data(point, job.seed, data)

    class Successful:
        config = ADP_Config(seed=job.seed)
        solver = ADP_solver(
            _unchanged_single if mode == "single" else _unchanged_multi
        )

        def fit(self, X, Y, *, progress=None):
            self.beta_ = index
            if mode == "multi":
                self.basis_ = index
                self.eigenvalues_ = np.ones(index_dim)
            self.coefficients_ = np.ones(len(X))
            self.effective_parameters_ = {}
            trace = {
                "k": 0,
                "h": 1.0,
                "rho" if mode == "single" else "alpha": 1.0,
                "beta" if mode == "single" else "basis": index,
            }
            if mode == "multi":
                trace["eigenvalues"] = self.eigenvalues_
            self.result_ = type(
                "Result",
                (),
                {"trace": [trace], "stop_reason": "h_min"},
            )()
            self.profile_ = {
                "total_time_seconds": 0.01,
                "peak_memory_bytes": 32,
                "stages": {},
            }
            return self

    monkeypatch.setattr(
        runner,
        "_build_model",
        lambda experiment, variant, seed: Successful(),
    )

    outcome = runner._execute_job(
        store,
        experiment,
        job,
        data,
        True,
        None,
    )

    assert outcome["run"]["status"] == "success"
    path = store.series_dir / outcome["run"]["model_artifact"]
    with np.load(path, allow_pickle=False) as archive:
        expected = {"index", "coefficients"}
        if mode == "multi":
            expected.add("eigenvalues")
        assert set(archive.files) == expected
        np.testing.assert_array_equal(archive["index"], index)


def test_keyboard_interrupt_propagates_and_closes_contexts(
    tmp_path: Path,
    monkeypatch,
):
    from dataclasses import replace

    import ADP.experiment_runner as runner

    base = _paired_experiment()
    experiment = replace(
        base,
        runs=1,
        points=(base.points[0],),
        variants={"A": base.variants["A"]},
    )
    events = []
    bars = []

    class Bar:
        disable = True

        def __init__(self, **kwargs):
            self.closed = False
            bars.append(self)

        def set_postfix(self, **kwargs):
            pass

        def update(self, amount):
            pass

        def close(self):
            self.closed = True

    class ThreadContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            events.append("threadpool_exit")

    class Sampler:
        def __init__(self):
            self.samples = []

        def __enter__(self):
            self.samples.append(1.0)
            return self

        def __exit__(self, exc_type, exc, tb):
            self.samples.append(1.0)
            events.append("rss_exit")

    class Interrupted:
        config = ADP_Config(seed=experiment.seed)
        solver = ADP_solver(_unchanged_single)

        def fit(self, X, Y, *, progress=None):
            raise KeyboardInterrupt

    monkeypatch.setattr(runner, "tqdm", Bar)
    monkeypatch.setattr(
        runner,
        "threadpool_limits",
        lambda **kwargs: ThreadContext(),
    )
    monkeypatch.setattr(runner, "_RSSSampler", Sampler)
    monkeypatch.setattr(
        runner,
        "_build_model",
        lambda experiment, variant, seed: Interrupted(),
    )

    with pytest.raises(KeyboardInterrupt):
        runner.run_experiment(
            experiment,
            tmp_path,
            save_models=False,
            show_progress=False,
        )

    assert events == ["rss_exit", "threadpool_exit"]
    assert len(bars) == 2
    assert all(bar.closed for bar in bars)
    series_dir = next((tmp_path / experiment.name).iterdir())
    assert not list((series_dir / "commits").glob("*.json"))
    series = _read_csv(series_dir / "series.csv")
    assert series[0]["status"] == "partial"
    assert series[0]["completed_jobs"] == "0"
