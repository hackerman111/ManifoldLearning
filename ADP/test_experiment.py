import csv
from pathlib import Path

import numpy as np
import pytest

from ADP import ADP_Config, ADP_Data, ADP_SolverResult, ADP_solver


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_cli_builds_multi_manual_experiment():
    from ADP.cli import build_parser, experiment_from_args

    parser = build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "multi",
            "--index-dim",
            "2",
            "--solver",
            "lsmr",
            "--runs",
            "3",
        ]
    )

    experiment = experiment_from_args(args, parser)

    assert experiment.mode == "multi"
    assert experiment.index_dim == 2
    assert experiment.runs == 3
    assert experiment.variants["default"].solver == "lsmr"
    assert experiment.variants["default"].config.index_init == "pilot"
    assert experiment.variants["default"].config.lambda_penalty == 10000.0


def test_cli_enables_smart_weights_explicitly():
    from ADP.cli import build_parser, experiment_from_args

    parser = build_parser()

    legacy = experiment_from_args(parser.parse_args([]), parser)
    smart = experiment_from_args(
        parser.parse_args(["--smart-weights"]),
        parser,
    )

    assert legacy.variants["default"].config.smart_weights is False
    assert smart.variants["default"].config.smart_weights is True


def test_single_experiment_rejects_pilot_initialization():
    from ADP import ADP_Experiment, ADP_ExperimentPoint, ADP_ExperimentVariant
    from ADP.experiment import validate_experiment

    experiment = ADP_Experiment(
        name="single-pilot",
        mode="single",
        points=(ADP_ExperimentPoint("p", 24, 3),),
        variants={
            "v": ADP_ExperimentVariant(ADP_Config(index_init="pilot")),
        },
    )

    with pytest.raises(ValueError, match="pilot initialization is multi-index only"):
        validate_experiment(experiment)


def test_single_model_rejects_pilot_initialization():
    from ADP import ADP_single_index

    X = np.ones((24, 3))
    Y = np.zeros(24)
    model = ADP_single_index(config=ADP_Config(index_init="pilot"))

    with pytest.raises(ValueError, match="pilot initialization is multi-index only"):
        model.fit(X, Y)


def test_cli_dry_run_creates_nothing(tmp_path: Path, capsys):
    from ADP.cli import main

    source = tmp_path / "exp.py"
    source.write_text(
        """
from ADP import ADP_Config, ADP_Experiment, ADP_ExperimentPoint, ADP_ExperimentVariant

experiment = ADP_Experiment(
    name="dry",
    mode="single",
    runs=2,
    points=(ADP_ExperimentPoint("p", 24, 3),),
    variants={
        "A": ADP_ExperimentVariant(ADP_Config()),
        "B": ADP_ExperimentVariant(ADP_Config()),
    },
)
""",
        encoding="utf-8",
    )

    assert main(
        [
            "--experiment-file",
            str(source),
            "--output-dir",
            str(tmp_path),
            "--dry-run",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "variants: A, B" in output
    assert "total jobs: 4" in output
    assert not (tmp_path / "dry").exists()


def test_cli_reports_only_uses_archive_without_experiment(
    tmp_path: Path, monkeypatch, capsys
):
    from ADP.cli import main
    import ADP.experiment_reports as reports

    calls = []
    monkeypatch.setattr(
        reports,
        "write_reports",
        lambda path: calls.append(Path(path)) or Path(path) / "artifacts.csv",
    )

    assert main(["--reports-only", str(tmp_path)]) == 0
    assert calls == [tmp_path]
    assert f"отчёты обновлены: {tmp_path / 'artifacts.csv'}" in capsys.readouterr().out


def test_cli_experiment_file_ignores_manual_flags(tmp_path: Path, capsys):
    from ADP.cli import main

    source = tmp_path / "exp.py"
    source.write_text(
        """
from ADP import ADP_Config, ADP_Experiment, ADP_ExperimentPoint, ADP_ExperimentVariant
experiment = ADP_Experiment(
    name="file",
    mode="single",
    points=(ADP_ExperimentPoint("p", 24, 3),),
    variants={"v": ADP_ExperimentVariant(ADP_Config())},
)
""",
        encoding="utf-8",
    )

    assert main(
        [
            "--experiment-file",
            str(source),
            "--mode",
            "multi",
            "--index-dim",
            "99",
            "--solver",
            "varpro",
            "--runs",
            "0",
            "--n",
            "1",
            "--d",
            "0",
            "--dry-run",
        ]
    ) == 0
    assert "total jobs: 1" in capsys.readouterr().out


def test_cli_reports_only_rejects_conflicting_modes(tmp_path: Path):
    from ADP.cli import main

    with pytest.raises(SystemExit) as error:
        main(["--reports-only", str(tmp_path), "--dry-run"])

    assert error.value.code == 2


def test_cli_keyboard_interrupt_returns_130(monkeypatch):
    import ADP.cli as cli

    monkeypatch.setattr(
        cli,
        "run_experiment",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    assert cli.main([]) == 130


def test_cli_failures_return_one(tmp_path: Path, monkeypatch, capsys):
    import ADP.cli as cli

    series_dir = tmp_path / "series"
    monkeypatch.setattr(
        cli, "run_experiment", lambda *args, **kwargs: (series_dir, 2)
    )

    assert cli.main([]) == 1
    output = capsys.readouterr().out
    assert f"серия сохранена: {series_dir}" in output
    assert "ошибок: 2" in output


def test_cli_terminal_only_prints_readable_summary_and_creates_nothing(
    tmp_path: Path, monkeypatch, capsys
):
    import ADP.cli as cli

    output_dir = tmp_path / "must-not-exist"
    runs = [
        {
            "point": "manual",
            "variant": "default",
            "status": "success",
            "cosine_initial": 0.7,
            "cosine_abs": 0.9,
            "effective_N_J": 8,
            "algorithm_time_sec": 0.5,
            "algorithm_rss_max_mib": 20.0,
            "tracemalloc_peak_mib": 5.0,
            "outer_iterations": 2,
            "stop_reason": "h_min",
            "profile_stages": {
                "initialization": {
                    "time_seconds": 0.1,
                    "memory_bytes": 1 * 2**20,
                },
                "solver": {
                    "time_seconds": 0.3,
                    "memory_bytes": 3 * 2**20,
                },
            },
        },
        {
            "point": "manual",
            "variant": "default",
            "status": "nonconverged",
            "cosine_initial": 0.9,
            "cosine_abs": 0.8,
            "effective_N_J": 8,
            "algorithm_time_sec": 0.7,
            "algorithm_rss_max_mib": 22.0,
            "tracemalloc_peak_mib": 7.0,
            "outer_iterations": 4,
            "stop_reason": "outer_steps",
            "profile_stages": {
                "initialization": {
                    "time_seconds": 0.2,
                    "memory_bytes": 2 * 2**20,
                },
                "solver": {
                    "time_seconds": 0.4,
                    "memory_bytes": 4 * 2**20,
                },
            },
        },
    ]
    calls = []
    monkeypatch.setattr(
        cli,
        "run_experiment_terminal",
        lambda experiment, *, show_progress: calls.append(
            (experiment.name, show_progress, cli.sys.dont_write_bytecode)
        )
        or (runs, 1),
    )

    assert cli.main(
        [
            "--terminal-only",
            "--output-dir",
            str(output_dir),
            "--no-progress",
        ]
    ) == 1

    output = capsys.readouterr().out
    assert "Эксперимент: manual | режим: single | jobs: 2" in output
    assert "=== manual / default ===" in output
    assert "запусков: 2" in output
    assert "статусы: success=1; nonconverged=1; numerical_failure=0" in output
    assert "инициализация индекса: local" in output
    assert "косинус в начале, медиана: 0.800000" in output
    assert "косинус в конце, медиана: 0.850000" in output
    assert "количество центров N_J: 8" in output
    assert "количество итераций, медиана: 3" in output
    assert "причины остановки: h_min=1; outer_steps=1" in output
    assert "этап             время, с  доля времени  память, MiB  доля памяти" in output
    assert "инициализация" in output
    assert "солвер" in output
    assert output.index("инициализация") < output.index("солвер")
    assert (
        "итого, медиана: 0.600000 с; пик fit: 6.0000 MiB; "
        "RSS max: 22.0 MiB"
    ) in output
    assert calls == [("manual", False, True)]
    assert not output_dir.exists()


@pytest.mark.parametrize(
    "arguments",
    (
        ("--terminal-only", "--resume", "series"),
        ("--terminal-only", "--reports-only", "series"),
    ),
)
def test_cli_terminal_only_rejects_archive_modes(arguments):
    from ADP.cli import main

    with pytest.raises(SystemExit) as error:
        main(list(arguments))

    assert error.value.code == 2


def test_cli_terminal_only_experiment_file_writes_no_bytecode(
    tmp_path: Path, monkeypatch
):
    import ADP.cli as cli

    source = tmp_path / "experiment.py"
    source.write_text(
        """
from ADP import ADP_Config, ADP_Experiment, ADP_ExperimentPoint, ADP_ExperimentVariant
experiment = ADP_Experiment(
    name="memory",
    mode="single",
    points=(ADP_ExperimentPoint("p", 24, 3),),
    variants={"v": ADP_ExperimentVariant(ADP_Config())},
)
""",
        encoding="utf-8",
    )
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    monkeypatch.setattr(
        cli,
        "run_experiment_terminal",
        lambda *args, **kwargs: ([], 0),
    )

    assert cli.main(
        ["--terminal-only", "--experiment-file", str(source), "--no-progress"]
    ) == 0

    assert {path.relative_to(tmp_path) for path in tmp_path.rglob("*")} == before


def test_cli_terminal_only_kernel_import_writes_no_bytecode(tmp_path, monkeypatch):
    import ADP.cli as cli

    source = tmp_path / "terminal_only_kernel.py"
    source.write_text("def kernel(value):\n    return value\n", encoding="utf-8")
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.setattr(
        cli,
        "run_experiment_terminal",
        lambda *args, **kwargs: ([], 0),
    )

    assert cli.main(
        [
            "--terminal-only",
            "--kernel",
            "terminal_only_kernel:kernel",
            "--no-progress",
        ]
    ) == 0

    assert {path.relative_to(tmp_path) for path in tmp_path.rglob("*")} == before


def test_terminal_summary_uses_projector_error_for_multi(capsys):
    from ADP.cli import _print_terminal_summary, build_parser, experiment_from_args

    parser = build_parser()
    experiment = experiment_from_args(
        parser.parse_args(["--mode", "multi", "--index-dim", "2"]), parser
    )
    _print_terminal_summary(
        experiment,
        [
            {
                "point": "manual",
                "variant": "default",
                "status": "numerical_failure",
                "error_type": "ValueError",
                "error_message": "N_J must lie between ceil(n / N_loc) and n",
                "projector_error_initial": "",
                "projector_error": "",
                "effective_N_J": "",
                "algorithm_time_sec": "",
                "algorithm_rss_max_mib": "",
                "tracemalloc_peak_mib": "",
                "outer_iterations": "",
                "stop_reason": "",
                "profile_stages": {},
            }
        ],
    )

    output = capsys.readouterr().out
    assert "ошибка проектора в начале, медиана: —" in output
    assert "ошибка проектора в конце, медиана: —" in output
    assert "неуспешные запуски:" in output
    assert (
        "numerical_failure (1): ValueError: "
        "N_J must lie between ceil(n / N_loc) and n"
    ) in output
    assert "профиль: —" in output
    assert "—" in output


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


def test_terminal_runner_reuses_paired_data_without_store(monkeypatch):
    import ADP.experiment_runner as runner

    factory_calls = []
    data_ids = {}

    def data_factory(point, rng):
        factory_calls.append(point.name)
        X = rng.normal(size=(point.n, point.d))
        truth = np.eye(point.d)[0]
        return ADP_Data(X, X @ truth, truth)

    def execute(store, experiment, job, data, save_models, progress_callback):
        assert store is None
        assert not save_models
        data_ids.setdefault((job.point.name, job.seed), []).append(id(data))
        run = runner._base_run(job, experiment, "terminal")
        run.update(
            {
                "status": "success",
                "cosine_abs": 1.0,
                "fit_wall_time_sec": 0.01,
                "algorithm_rss_max_mib": 10.0,
                "outer_iterations": 1,
            }
        )
        return {"run": run, "outer": []}

    monkeypatch.setattr(
        runner._SeriesStore,
        "create",
        lambda *args, **kwargs: pytest.fail("terminal mode created a store"),
    )
    for name in ("_atomic_json", "_atomic_npz", "_atomic_csv"):
        monkeypatch.setattr(
            runner,
            name,
            lambda *args, _name=name, **kwargs: pytest.fail(
                f"terminal mode called {_name}"
            ),
        )
    monkeypatch.setattr(runner, "_execute_job", execute)

    runs, failures = runner.run_experiment_terminal(
        _paired_experiment(data_factory), show_progress=False
    )

    assert failures == 0
    assert len(runs) == 8
    assert len(factory_calls) == 4
    assert all(len(ids) == 2 and ids[0] == ids[1] for ids in data_ids.values())


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

    reduced = replace(experiment, runs=1)
    with pytest.raises(ValueError, match="resume specification differs"):
        runner.run_experiment(
            reduced,
            tmp_path,
            resume=series_dir,
            save_models=False,
            show_progress=False,
        )
    assert {
        path.name: path.read_bytes() for path in series_dir.glob("*.csv")
    } == complete_tables

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


def test_report_failure_surfaces_after_normal_runner_completion(
    tmp_path: Path,
    monkeypatch,
):
    from dataclasses import replace

    import ADP.experiment_reports as reports
    import ADP.experiment_runner as runner

    def fail_factory(point, rng):
        raise ValueError("synthetic data failure")

    def fail_reports(_series_dir):
        raise RuntimeError("report failure")

    base = _paired_experiment(fail_factory)
    experiment = replace(
        base,
        runs=1,
        points=(base.points[0],),
        variants={"A": base.variants["A"]},
    )
    monkeypatch.setattr(reports, "write_reports", fail_reports)

    with pytest.raises(RuntimeError, match="report failure"):
        runner.run_experiment(
            experiment,
            tmp_path,
            save_models=False,
            show_progress=False,
        )

    series_dir = next((tmp_path / experiment.name).iterdir())
    assert (series_dir / "run_summary.csv").is_file()


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
    initial_index = np.roll(index, 1, axis=0)
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
                {
                    "beta_init": initial_index,
                    "trace": [trace],
                    "stop_reason": "h_min",
                },
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
    if mode == "single":
        assert outcome["run"]["cosine_initial"] == 0.0
        assert outcome["run"]["cosine_abs"] == 1.0
    else:
        assert outcome["run"]["projector_error_initial"] > 0.0
        assert outcome["run"]["projector_error"] == 0.0
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
    import ADP.experiment_reports as reports

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

    def fail_reports(_series_dir):
        raise RuntimeError("report failure")

    monkeypatch.setattr(reports, "write_reports", fail_reports)

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


def test_report_aggregates_match_main_conventions(tmp_path):
    import pandas as pd

    from ADP.experiment_reports import _quantiles, _render_quantile, _wilson

    frame = pd.DataFrame(
        {
            "variant": ["A", "A", "A", "B", "B", "B"],
            "outer_k": [0, 0, 1, 0, 0, 1],
            "cosine_abs": [0.7, 0.9, 0.95, 0.6, 0.8, 0.85],
        }
    )

    summary = _quantiles(
        frame,
        "outer_k",
        "cosine_abs",
        groups=("variant",),
    )

    assert {"variant", "outer_k", "q05", "median", "q95"} <= set(summary)
    assert summary.loc[
        (summary["variant"] == "A") & (summary["outer_k"] == 0),
        "median",
    ].item() == pytest.approx(0.8)
    low, center, high = _wilson(8, 10)
    assert 0 <= low < center < high <= 1

    path = tmp_path / "quality.png"
    assert _render_quantile(
        frame,
        path,
        x="outer_k",
        y="cosine_abs",
        groups=("variant",),
        title="Качество по внешней итерации",
        xlabel="Внешняя итерация",
        ylabel="Абсолютный косинус",
    ) == path
    assert path.stat().st_size > 0


def test_report_renderer_closes_figure_on_invalid_scale(tmp_path):
    import matplotlib.pyplot as plt
    import pandas as pd

    from ADP.experiment_reports import _render_quantile

    frame = pd.DataFrame({"x": [1.0], "y": [1.0]})
    before = set(plt.get_fignums())
    try:
        with pytest.raises(ValueError, match="scale"):
            _render_quantile(
                frame,
                tmp_path / "invalid.png",
                x="x",
                y="y",
                title="t",
                xlabel="x",
                ylabel="y",
                xscale="invalid",
            )
        assert set(plt.get_fignums()) == before
    finally:
        for number in set(plt.get_fignums()) - before:
            plt.close(number)


def test_report_log_domain_skips_nonpositive_and_rejects_categories(tmp_path):
    import pandas as pd

    from ADP.experiment_reports import _render_quantile

    path = tmp_path / "nonpositive.png"
    assert _render_quantile(
        pd.DataFrame({"x": [-1.0, 0.0], "y": [1.0, 2.0]}),
        path,
        x="x",
        y="y",
        title="t",
        xlabel="x",
        ylabel="y",
        xscale="log",
    ) is False
    assert not path.exists()

    with pytest.raises(ValueError, match="numeric"):
        _render_quantile(
            pd.DataFrame({"x": ["left", "right"], "y": [1.0, 2.0]}),
            tmp_path / "categorical-log.png",
            x="x",
            y="y",
            title="t",
            xlabel="x",
            ylabel="y",
            xscale="log2",
        )


def test_report_categorical_groups_ignore_unused_levels():
    import pandas as pd

    from ADP.experiment_reports import _quantiles

    frame = pd.DataFrame(
        {
            "variant": pd.Categorical(["A"], categories=["A", "B"]),
            "outer_k": pd.Categorical([0], categories=[0, 1]),
            "cosine_abs": [0.8],
        }
    )

    summary = _quantiles(frame, "outer_k", "cosine_abs", ("variant",))

    assert len(summary) == 1
    assert summary["variant"].astype(str).tolist() == ["A"]
    assert summary["outer_k"].astype(int).tolist() == [0]


def test_report_heatmap_limits_follow_displayed_aggregates(tmp_path, monkeypatch):
    import matplotlib.axes
    import pandas as pd

    from ADP.experiment_reports import _render_heatmap

    limits = []
    original = matplotlib.axes.Axes.imshow

    def capture_limits(axis, *args, **kwargs):
        limits.append((kwargs.get("vmin"), kwargs.get("vmax")))
        return original(axis, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", capture_limits)
    assert _render_heatmap(
        pd.DataFrame(
            {
                "x": [1, 1, 1, 2],
                "row": [1, 1, 1, 1],
                "value": [1.0, 1.0, 100.0, 2.0],
            }
        ),
        tmp_path / "heatmap.png",
        x="x",
        y="row",
        value="value",
        title="t",
        xlabel="x",
        ylabel="row",
    )

    assert limits == [(1.0, 2.0)]


@pytest.mark.parametrize(
    ("name", "extra"),
    [
        ("_render_quantile", {}),
        ("_render_median_line", {}),
        ("_render_proportion", {"y": "success"}),
        ("_render_box", {}),
        ("_render_scatter", {}),
        ("_render_heatmap", {"y": "row", "value": "value"}),
        ("_render_stacked", {"y": "", "components": ("a", "b")}),
    ],
)
def test_report_renderers_smoke_and_close(tmp_path, name, extra):
    import matplotlib.pyplot as plt
    import pandas as pd

    import ADP.experiment_reports as reports

    frame = pd.DataFrame(
        {
            "variant": ["A", "A", "B", "B"],
            "x": [1, 2, 1, 2],
            "y": [0.2, 0.8, 0.4, 0.9],
            "success": [True, False, True, True],
            "row": [1, 1, 2, 2],
            "value": [0.2, 0.8, 0.4, 0.9],
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [2.0, 2.0, 1.0, 1.0],
        }
    )
    renderer = getattr(reports, name)
    options = {
        "x": "x",
        "y": "y",
        "groups": ("variant",),
        "title": "t",
        "xlabel": "x",
        "ylabel": "y",
        **extra,
    }
    before = set(plt.get_fignums())
    path = tmp_path / f"{name}.png"

    assert renderer(frame, path, **options) == path
    assert path.stat().st_size > 0
    empty_path = tmp_path / f"{name}-empty.png"
    assert renderer(frame.iloc[:0], empty_path, **options) is False
    assert not empty_path.exists()
    assert set(plt.get_fignums()) == before


@pytest.mark.parametrize(
    ("name", "extra"),
    [
        ("_render_heatmap", {"y": "row", "value": "value"}),
        ("_render_stacked", {"y": "", "components": ("a",)}),
    ],
)
def test_report_aggregate_token_is_validated(tmp_path, name, extra):
    import pandas as pd

    import ADP.experiment_reports as reports

    with pytest.raises(ValueError, match="aggregate"):
        getattr(reports, name)(
            pd.DataFrame({"x": [1], "row": [1], "value": [1.0], "a": [1.0]}),
            tmp_path / f"{name}.png",
            x="x",
            title="t",
            xlabel="x",
            ylabel="y",
            aggregate="invalid",
            **extra,
        )


def test_reports_create_single_and_multi_specific_plots(tmp_path: Path):
    import pandas as pd

    from ADP.experiment_reports import write_reports

    single = tmp_path / "single"
    single.mkdir()
    pd.DataFrame(
        [
            {
                "series_id": "s",
                "run_id": "r",
                "experiment": "p",
                "point": "p",
                "variant": "A",
                "mode": "single",
                "seed": 1,
                "d": 3,
                "n_over_d": 10,
                "sigma_eps": 0.1,
                "link": "sin",
                "status": "success",
                "cosine_abs": 0.9,
                "projector_error": 0.2,
                "algorithm_time_sec": 1.0,
                "algorithm_rss_max_mib": 20.0,
                "stage_initialization_time_sec": 0.1,
                "stage_directions_time_sec": 0.1,
                "stage_statistics_time_sec": 0.3,
                "stage_solver_time_sec": 0.4,
                "stage_update_time_sec": 0.1,
                "outer_iterations": 1,
            }
        ]
    ).to_csv(single / "run_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "series_id": "s",
                "run_id": "r",
                "experiment": "p",
                "point": "p",
                "variant": "A",
                "mode": "single",
                "seed": 1,
                "outer_k": 0,
                "h_k": 1.0,
                "rho_k": 0.8,
                "alpha_k": np.nan,
                "cosine_abs": 0.9,
                "projector_error": 0.2,
            }
        ]
    ).to_csv(single / "outer_iterations.csv", index=False)

    write_reports(single)

    assert (single / "plots/points/p/quality_vs_outer_iteration.png").is_file()
    assert (single / "plots/points/p/rho_vs_outer_iteration.png").is_file()
    assert (single / "plots/summary/quality_heatmap_d_nd_ratio.png").is_file()
    assert (single / "plots/summary/success_rate_vs_sigma_eps.png").is_file()
    assert (single / "plots/summary/quality_by_link_function.png").is_file()
    assert (single / "plots/summary/runtime_vs_dimension.png").is_file()
    assert (single / "plots/summary/runtime_breakdown.png").is_file()
    assert (single / "plots/summary/status_breakdown.png").is_file()

    multi = tmp_path / "multi"
    multi.mkdir()
    runs = pd.read_csv(single / "run_summary.csv").assign(
        mode="multi", cosine_abs=np.nan
    )
    outer = pd.read_csv(single / "outer_iterations.csv").assign(
        mode="multi", cosine_abs=np.nan, rho_k=np.nan, alpha_k=0.7
    )
    runs.to_csv(multi / "run_summary.csv", index=False)
    outer.to_csv(multi / "outer_iterations.csv", index=False)

    write_reports(multi)

    assert (
        multi / "plots/points/p/projector_error_vs_outer_iteration.png"
    ).is_file()
    assert (multi / "plots/points/p/alpha_vs_outer_iteration.png").is_file()
    assert not (multi / "plots/points/p/quality_vs_outer_iteration.png").exists()
    artifacts = pd.read_csv(multi / "artifacts.csv")
    assert {"created", "skipped"} <= set(artifacts["status"])

    truthless = tmp_path / "truthless"
    truthless.mkdir()
    runs.assign(cosine_abs=np.nan, projector_error=np.nan).to_csv(
        truthless / "run_summary.csv", index=False
    )
    outer.assign(cosine_abs=np.nan, projector_error=np.nan).to_csv(
        truthless / "outer_iterations.csv", index=False
    )

    write_reports(truthless)

    assert not (
        truthless / "plots/points/p/quality_vs_outer_iteration.png"
    ).exists()
    assert not (
        truthless / "plots/points/p/projector_error_vs_outer_iteration.png"
    ).exists()


def test_plot_manifest_covers_approved_main_families():
    from ADP.experiment_reports import PLOT_MANIFEST, _for_mode

    diagnostic = {
        "quality_vs_outer_iteration.png",
        "projector_error_vs_outer_iteration.png",
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
        "runtime_breakdown.png",
        "runtime_share_breakdown.png",
        "status_breakdown.png",
    }
    families = {
        ("quality_heatmap_d_nd_ratio.png", ("d", "n_over_d")),
        ("success_rate_heatmap.png", ("d", "n_over_d")),
        ("runtime_vs_dimension.png", ("d", "n_over_d")),
        ("memory_vs_dimension.png", ("d", "n_over_d")),
        ("iterations_heatmap_d_nd_ratio.png", ("d", "n_over_d")),
        ("quality_vs_sigma_eps.png", ("sigma_eps",)),
        ("success_rate_vs_sigma_eps.png", ("sigma_eps",)),
        ("runtime_vs_sigma_eps.png", ("sigma_eps",)),
        ("outer_iterations_vs_sigma_eps.png", ("sigma_eps",)),
        ("final_objective_vs_sigma_eps.png", ("sigma_eps",)),
        ("quality_vs_correlation.png", ("rho_corr",)),
        ("success_rate_vs_correlation.png", ("rho_corr",)),
        ("local_condition_vs_correlation.png", ("rho_corr",)),
        ("solver_iterations_vs_correlation.png", ("rho_corr",)),
        ("runtime_vs_correlation.png", ("rho_corr",)),
        ("singular_fraction_vs_correlation.png", ("rho_corr",)),
        ("quality_vs_sigma_x.png", ("sigma_x",)),
        ("h0_vs_sigma_x.png", ("sigma_x",)),
        ("final_bandwidth_vs_sigma_x.png", ("sigma_x",)),
        ("local_mass_vs_sigma_x.png", ("sigma_x",)),
        ("runtime_vs_sigma_x.png", ("sigma_x",)),
        ("bandwidth_ratio_vs_sigma_x.png", ("sigma_x",)),
        ("quality_by_link_function.png", ("link",)),
        ("success_rate_by_link_function.png", ("link",)),
        ("outer_iterations_by_link_function.png", ("link",)),
        ("objective_by_link_function.png", ("link",)),
        ("local_slopes_by_link_function.png", ("link",)),
        ("quality_by_x_distribution.png", ("x_distribution",)),
        ("quality_by_noise_distribution.png", ("noise_distribution",)),
        ("quality_by_heteroscedasticity.png", ("heteroscedastic",)),
        ("quality_vs_outlier_fraction.png", ("effective_outlier_fraction",)),
        ("failure_rate_vs_outliers.png", ("effective_outlier_fraction",)),
        ("quality_vs_model_misspecification.png", ("delta",)),
        ("objective_vs_model_misspecification.png", ("delta",)),
    }

    assert diagnostic == {
        spec.filename for spec in PLOT_MANIFEST if not spec.required_metadata
        and not spec.required_any_metadata
    }
    assert families <= {
        (spec.filename, spec.required_metadata) for spec in PLOT_MANIFEST
    }
    shared_distributions = {
        spec.filename: spec.required_any_metadata
        for spec in PLOT_MANIFEST
        if spec.filename
        in {"failure_rate_by_distribution.png", "runtime_by_distribution.png"}
    }
    assert shared_distributions == {
        "failure_rate_by_distribution.png": (
            "x_distribution",
            "noise_distribution",
        ),
        "runtime_by_distribution.png": (
            "x_distribution",
            "noise_distribution",
        ),
    }
    assert all("variant" in spec.groups for spec in PLOT_MANIFEST)
    assert next(
        spec.groups
        for spec in PLOT_MANIFEST
        if spec.filename == "bandwidth_vs_outer_iteration.png"
    ) == ("variant", "d", "n_over_d")
    multi_quality = _for_mode(
        next(
            spec
            for spec in PLOT_MANIFEST
            if spec.filename == "quality_vs_outer_iteration.png"
        ),
        "multi",
    )
    assert "Ошибка проектора" in multi_quality.title
    assert "Качество" not in multi_quality.title
    assert "correctness_rate.png" not in {
        spec.filename for spec in PLOT_MANIFEST
    }


def test_reports_document_missing_optional_tables(tmp_path, monkeypatch):
    import pandas as pd

    import ADP.experiment_reports as reports

    pd.DataFrame(
        [
            {
                "run_id": "r",
                "experiment": "p",
                "point": "p",
                "variant": "A",
                "mode": "single",
                "status": "success",
            }
        ]
    ).to_csv(tmp_path / "run_summary.csv", index=False)
    monkeypatch.setattr(
        reports,
        "PLOT_MANIFEST",
        (
            reports.PlotSpec(
                "missing.png",
                "outer",
                "quantile",
                "outer_k",
                "cosine_abs",
                "t",
                "x",
                "y",
                scope="point",
            ),
        ),
    )

    reports.write_reports(tmp_path)

    artifacts = pd.read_csv(tmp_path / "artifacts.csv")
    assert artifacts[["filename", "status"]].to_dict("records") == [
        {"filename": "missing.png", "status": "skipped"}
    ]


def test_reports_remove_stale_plots_and_isolate_errors(tmp_path, monkeypatch):
    import pandas as pd

    import ADP.experiment_reports as reports

    runs = pd.DataFrame(
        [
            {
                "run_id": "r",
                "experiment": "p",
                "point": "p",
                "variant": "A",
                "mode": "single",
                "status": "success",
                "d": 3,
                "cosine_abs": 0.9,
                "algorithm_time_sec": 1.0,
            }
        ]
    )
    runs.to_csv(tmp_path / "run_summary.csv", index=False)
    quality = reports.PlotSpec(
        "quality.png", "runs", "quantile", "d", "cosine_abs", "t", "x", "y"
    )
    monkeypatch.setattr(reports, "PLOT_MANIFEST", (quality,))
    reports.write_reports(tmp_path)
    target = tmp_path / "plots/summary/quality.png"
    assert target.is_file()

    runs.assign(cosine_abs=np.nan).to_csv(
        tmp_path / "run_summary.csv", index=False
    )
    reports.write_reports(tmp_path)
    assert not target.exists()

    runs.assign(mode="multi", projector_error=0.2).to_csv(
        tmp_path / "run_summary.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "run_id": "r",
                "outer_k": 0,
                "projector_error": 0.2,
            }
        ]
    ).to_csv(tmp_path / "outer_iterations.csv", index=False)
    native = reports.PlotSpec(
        "projector_error_vs_outer_iteration.png",
        "outer",
        "quantile",
        "outer_k",
        "projector_error",
        "t",
        "x",
        "y",
        scope="point",
    )
    adapted = reports.PlotSpec(
        "quality_vs_outer_iteration.png",
        "outer",
        "quantile",
        "outer_k",
        "cosine_abs",
        "t",
        "x",
        "y",
        scope="point",
    )
    runtime = reports.PlotSpec(
        "runtime.png",
        "runs",
        "median_line",
        "d",
        "algorithm_time_sec",
        "t",
        "x",
        "y",
    )
    monkeypatch.setattr(reports, "PLOT_MANIFEST", (native, adapted, runtime))
    calls = []

    def fail(frame, path, **kwargs):
        calls.append(1)
        path.mkdir(parents=True)
        raise RuntimeError("broken plot")

    monkeypatch.setattr(reports, "_render_quantile", fail)

    reports.write_reports(tmp_path)

    artifacts = pd.read_csv(tmp_path / "artifacts.csv")
    assert artifacts["status"].tolist() == ["error", "skipped", "created"]
    assert "RuntimeError: broken plot" in artifacts["error"].iloc[0]
    assert "cleanup" in artifacts["error"].iloc[0]
    assert len(calls) == 1


def test_reports_reject_point_path_traversal_without_touching_sentinel(
    tmp_path,
    monkeypatch,
):
    import pandas as pd

    import ADP.experiment_reports as reports

    series = tmp_path / "series"
    series.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "quality.png"
    sentinel.write_bytes(b"untouched")
    point = "../../../outside"
    pd.DataFrame(
        [
            {
                "run_id": "r",
                "experiment": "p",
                "point": point,
                "variant": "A",
                "mode": "single",
                "status": "success",
            }
        ]
    ).to_csv(series / "run_summary.csv", index=False)
    pd.DataFrame(
        [{"run_id": "r", "outer_k": 0, "cosine_abs": 0.9}]
    ).to_csv(series / "outer_iterations.csv", index=False)
    monkeypatch.setattr(
        reports,
        "PLOT_MANIFEST",
        (
            reports.PlotSpec(
                "quality.png",
                "outer",
                "quantile",
                "outer_k",
                "cosine_abs",
                "t",
                "x",
                "y",
                scope="point",
            ),
        ),
    )

    reports.write_reports(series)

    assert sentinel.read_bytes() == b"untouched"
    artifacts = pd.read_csv(series / "artifacts.csv")
    assert artifacts["status"].tolist() == ["error"]
    assert "path" in artifacts["error"].item().lower()

    symlinked = tmp_path / "symlinked"
    symlinked.mkdir()
    pd.DataFrame(
        [
            {
                "run_id": "r",
                "experiment": "p",
                "point": "p",
                "variant": "A",
                "mode": "single",
                "status": "success",
            }
        ]
    ).to_csv(symlinked / "run_summary.csv", index=False)
    pd.DataFrame(
        [{"run_id": "r", "outer_k": 0, "cosine_abs": 0.9}]
    ).to_csv(symlinked / "outer_iterations.csv", index=False)
    symlink_target = outside / "points" / "p"
    symlink_target.mkdir(parents=True)
    symlink_sentinel = symlink_target / "quality.png"
    symlink_sentinel.write_bytes(b"also untouched")
    (symlinked / "plots").symlink_to(outside, target_is_directory=True)

    reports.write_reports(symlinked)

    assert symlink_sentinel.read_bytes() == b"also untouched"
    assert pd.read_csv(symlinked / "artifacts.csv")["status"].tolist() == [
        "error"
    ]


def test_report_artifacts_include_derived_source_tables(tmp_path, monkeypatch):
    import pandas as pd

    import ADP.experiment_reports as reports

    pd.DataFrame(
        [
            {
                "run_id": "r",
                "experiment": "p",
                "point": "p",
                "variant": "A",
                "mode": "single",
                "status": "success",
                "d": 3,
                "rho_corr": 0.5,
                "sigma_x": 2.0,
            }
        ]
    ).to_csv(tmp_path / "run_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "run_id": "r",
                "outer_k": 0,
                "objective_after": 2.0,
                "h_k": 4.0,
            }
        ]
    ).to_csv(tmp_path / "outer_iterations.csv", index=False)
    pd.DataFrame(
        [
            {
                "run_id": "r",
                "outer_k": 0,
                "condition": 4.0,
                "local_mass": 3.0,
                "is_singular": 0.0,
            }
        ]
    ).to_csv(tmp_path / "local_diagnostics.csv", index=False)
    monkeypatch.setattr(
        reports,
        "PLOT_MANIFEST",
        (
            reports.PlotSpec(
                "objective.png",
                "runs",
                "quantile",
                "d",
                "objective",
                "t",
                "x",
                "y",
            ),
            reports.PlotSpec(
                "condition.png",
                "outer",
                "quantile",
                "rho_corr",
                "condition_median",
                "t",
                "x",
                "y",
            ),
            reports.PlotSpec(
                "h-initial.png",
                "runs",
                "quantile",
                "d",
                "h_initial",
                "t",
                "x",
                "y",
            ),
            reports.PlotSpec(
                "h-final.png",
                "runs",
                "quantile",
                "d",
                "h_final",
                "t",
                "x",
                "y",
            ),
            reports.PlotSpec(
                "bandwidth-ratio.png",
                "runs",
                "quantile",
                "d",
                "bandwidth_ratio",
                "t",
                "x",
                "y",
            ),
            reports.PlotSpec(
                "local-mass.png",
                "outer",
                "quantile",
                "rho_corr",
                "local_mass_mean",
                "t",
                "x",
                "y",
            ),
            reports.PlotSpec(
                "singular.png",
                "outer",
                "quantile",
                "rho_corr",
                "singular_fraction",
                "t",
                "x",
                "y",
            ),
        ),
    )
    monkeypatch.setattr(
        reports,
        "_render_quantile",
        lambda frame, path, **options: path,
    )

    reports.write_reports(tmp_path)

    sources = {
        row.filename: set(row.source_tables.split(","))
        for row in pd.read_csv(tmp_path / "artifacts.csv").itertuples()
    }
    assert sources == {
        "objective.png": {"run_summary.csv", "outer_iterations.csv"},
        "h-initial.png": {"run_summary.csv", "outer_iterations.csv"},
        "h-final.png": {"run_summary.csv", "outer_iterations.csv"},
        "bandwidth-ratio.png": {
            "run_summary.csv",
            "outer_iterations.csv",
        },
        "condition.png": {
            "outer_iterations.csv",
            "run_summary.csv",
            "local_diagnostics.csv",
        },
        "local-mass.png": {
            "outer_iterations.csv",
            "run_summary.csv",
            "local_diagnostics.csv",
        },
        "singular.png": {
            "outer_iterations.csv",
            "run_summary.csv",
            "local_diagnostics.csv",
        },
    }


def test_runtime_stack_skips_rows_with_partial_stage_timings(
    tmp_path,
    monkeypatch,
):
    import pandas as pd

    import ADP.experiment_reports as reports

    pd.DataFrame(
        [
            {
                "run_id": "r",
                "experiment": "p",
                "point": "p",
                "variant": "A",
                "mode": "single",
                "status": "success",
                "algorithm_time_sec": 1.0,
                "stage_a": 1.0,
                "stage_b": np.nan,
            }
        ]
    ).to_csv(tmp_path / "run_summary.csv", index=False)
    monkeypatch.setattr(
        reports,
        "PLOT_MANIFEST",
        (
            reports.PlotSpec(
                "runtime.png",
                "runs",
                "stacked",
                "experiment",
                "algorithm_time_sec",
                "t",
                "x",
                "y",
                components=("stage_a", "stage_b"),
            ),
        ),
    )

    reports.write_reports(tmp_path)

    artifacts = pd.read_csv(tmp_path / "artifacts.csv")
    assert artifacts["status"].tolist() == ["skipped"]
    assert not (tmp_path / "plots/summary/runtime.png").exists()


def test_stale_cleanup_error_is_isolated_from_later_plots(tmp_path, monkeypatch):
    import pandas as pd

    import ADP.experiment_reports as reports

    pd.DataFrame(
        [
            {
                "run_id": "r",
                "experiment": "p",
                "point": "p",
                "variant": "A",
                "mode": "single",
                "status": "success",
                "d": 3,
                "cosine_abs": np.nan,
                "algorithm_time_sec": 1.0,
            }
        ]
    ).to_csv(tmp_path / "run_summary.csv", index=False)
    monkeypatch.setattr(
        reports,
        "PLOT_MANIFEST",
        (
            reports.PlotSpec(
                "blocked.png",
                "runs",
                "quantile",
                "d",
                "cosine_abs",
                "t",
                "x",
                "y",
            ),
            reports.PlotSpec(
                "later.png",
                "runs",
                "median_line",
                "d",
                "algorithm_time_sec",
                "t",
                "x",
                "y",
            ),
        ),
    )
    blocked = tmp_path / "plots/summary/blocked.png"
    blocked.mkdir(parents=True)

    reports.write_reports(tmp_path)

    artifacts = pd.read_csv(tmp_path / "artifacts.csv")
    assert artifacts["status"].tolist() == ["error", "created"]
    assert "cleanup" in artifacts["error"].iloc[0]
    assert (tmp_path / "plots/summary/later.png").is_file()
