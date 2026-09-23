from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ADP import ADP_Config, ADP_multi_index, ADP_single_index, ADP_solver
from ADP.cli.main import _config, _run, _synthetic_data, build_parser
from ADP.engine.common import index_fit
from ADP.engine.common.logger import format_profile
from ADP.solver.CG import solve as solve_cg
from ADP.solver.HYBRID import solve as solve_hybrid
from ADP.solver.LSMR import HPAOResult, solve


@pytest.mark.parametrize(
    ("mode", "initialization"),
    [
        ("single", "local"),
        ("single", "local-cv"),
        ("single", "random"),
        ("multi", "local"),
        ("multi", "local-cv"),
        ("multi", "pilot"),
        ("multi", "random"),
    ],
)
@pytest.mark.parametrize("estimator", ["new", "legacy"])
@pytest.mark.parametrize("selection", ["best", "last"])
def test_models_match_cli(
    mode: str, initialization: str, estimator: str, selection: str
) -> None:
    """Публичные модели повторяют CLI при разных вариантах инициализации."""
    m = 1 if mode == "single" else 2
    args = build_parser().parse_args(
        [
            "--mode",
            mode,
            "--index-dim",
            str(m),
            "--n",
            "90",
            "--d",
            "4",
            "--N_J",
            "18",
            "--N_loc",
            "20",
            "--N_phi",
            "6",
            "--N_lin",
            "12",
            "--outer-steps",
            "2",
            "--index-init",
            initialization,
            "--estimator",
            estimator,
            "--select-step",
            selection,
            "--training-set",
            "exclude_centers",
            "--center-displacement",
            "0.2",
            "--fixed-directions",
            "--seed",
            "17",
        ]
    )
    data = _synthetic_data(90, 4, m, 0.05, 8)
    expected, _, _, metadata = _run(args, data=data)
    config = _config(args)
    model = ADP_single_index(config) if m == 1 else ADP_multi_index(m, config)
    snapshots = []
    assert model.fit(data[0], data[1], progress=snapshots.append) is model
    actual = model.beta_ if m == 1 else model.basis_.T
    np.testing.assert_array_equal(actual, expected)
    assert model.result_.stop_reason == metadata["stop_reason"]
    assert model.trace_ is model.result_.trace
    assert len(snapshots) == len(metadata["trace"])
    assert (
        model.effective_parameters_["selected_iteration"]
        == metadata["selected_iteration"]
    )
    assert model.effective_parameters_["training_size"] == 72
    for row, expected_row in zip(model.trace_, metadata["trace"], strict=True):
        assert row["h"] == expected_row["h"]
        assert row["factor"] == expected_row["factor"]
        assert row["err"] == expected_row["err"]
        assert row["solver"] == expected_row["solver"]
    if m > 1:
        np.testing.assert_allclose(model.basis_.T @ model.basis_, np.eye(m), atol=1e-12)
        np.testing.assert_array_equal(model.transform(data[0]), data[0] @ model.basis_)
        np.testing.assert_array_equal(
            model.eigenvalues_, metadata["selected_eigenvalues"]
        )


@pytest.mark.parametrize("m", [1, 2])
@pytest.mark.parametrize("estimator", ["new", "legacy"])
def test_solver_mass_and_selected_coefficients(m: int, estimator: str) -> None:
    """Mass и coefficients согласованы именно относительно выбранного индекса."""
    X, Y, _ = _synthetic_data(80, 4, m, 0.05, 3)
    calls = []

    def record(index, U, I, **settings):
        result = solve(index, U, I, **settings)
        calls.append((result, settings["mass"]))
        return result

    config = ADP_Config(N_J=16, N_phi=5, outer_steps=2, estimator=estimator)
    solver = ADP_solver(record, max_steps=3)
    model = (
        ADP_single_index(config, solver)
        if m == 1
        else ADP_multi_index(m, config, solver)
    )
    model.fit(X, Y)
    selected = model.effective_parameters_["selected_iteration"]
    raw, mass = calls[selected]
    assert (mass is not None) == (estimator == "new")
    if m == 1:
        expected = raw.coefficients[:, None] * raw.index
        actual = model.coefficients_[:, None] * model.beta_
    else:
        expected = raw.coefficients @ raw.index
        actual = model.coefficients_ @ model.basis_.T
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_best_step_retains_its_coefficients(monkeypatch: pytest.MonkeyPatch) -> None:
    """Best выбирает первый шаг; последний solver result не должен протечь наружу."""
    X, Y, _ = _synthetic_data(60, 4, 1, 0.0, 5)
    statistics = index_fit.calculate_statistics
    count = 0

    def increasing_error(*args, **kwargs):
        nonlocal count
        result = statistics(*args, **kwargs)
        count += 1
        return replace(result, S=np.full_like(result.S, count * 1e6))

    def different_index(index, U, I, **settings):
        basis = np.zeros_like(index)
        basis[count - 1] = 1
        return HPAOResult(basis, np.full(len(I), float(count)), {})

    monkeypatch.setattr(index_fit, "calculate_statistics", increasing_error)
    config = ADP_Config(N_J=12, N_phi=4, outer_steps=2, index_init="random", h_min=1e-6)
    model = ADP_single_index(config, ADP_solver(different_index)).fit(X, Y)
    assert len(model.trace_) == 2
    assert model.effective_parameters_["selected_iteration"] == 0
    np.testing.assert_allclose(abs(model.beta_), [1, 0, 0, 0])
    np.testing.assert_allclose(abs(model.coefficients_), np.ones(12))


@pytest.mark.parametrize("m", [1, 2])
def test_current_engine_rejects_missing_gpu_and_invalid_outputs(
    m: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    X, Y, _ = _synthetic_data(60, 4, m, 0.0, 2)
    config = ADP_Config(N_J=12, N_phi=4, outer_steps=1, gpu=True)
    model = ADP_single_index(config) if m == 1 else ADP_multi_index(m, config)

    def unavailable():
        raise RuntimeError("GPU execution requires an available CUDA device")

    monkeypatch.setattr("ADP.gpu.require_cupy", unavailable)
    with pytest.raises(RuntimeError, match="CUDA device"):
        model.fit(X, Y)
    assert not hasattr(model, "result_")

    def invalid(index, U, I, **settings):
        shape = (len(I),) if m == 1 else (len(I), m)
        return HPAOResult(index, np.full(shape, np.nan), {})

    model.config = replace(config, gpu=False)
    model.solver = ADP_solver(invalid)
    with pytest.raises(RuntimeError, match="invalid local coefficients"):
        model.fit(X, Y)


def test_progress_cannot_mutate_fit_and_list_inputs_work() -> None:
    X, Y, _ = _synthetic_data(60, 4, 1, 0.0, 2)
    config = ADP_Config(N_J=12, N_phi=4, outer_steps=1)

    def mutate(row):
        row["beta"][:] = np.nan
        row["solver"].clear()

    model = ADP_single_index(config).fit(X.tolist(), Y.tolist(), progress=mutate)
    assert np.all(np.isfinite(model.trace_[0]["beta"]))
    assert model.trace_[0]["solver"]
    assert model.result_.stop_reason == "outer_steps"
    assert model.n_features_in_ == 4
    assert "итого" in format_profile(model.profile_)


@pytest.mark.parametrize(
    ("mode", "method"), [("single", "cg"), ("multi", "cg"), ("multi", "hybrid")]
)
def test_models_accept_current_solver_variants(mode: str, method: str) -> None:
    m = 1 if mode == "single" else 2
    data = _synthetic_data(80, 4, m, 0.05, 4)
    args = build_parser().parse_args(
        [
            "--mode",
            mode,
            "--index-dim",
            str(m),
            "--n",
            "80",
            "--d",
            "4",
            "--N_J",
            "16",
            "--N_phi",
            "5",
            "--outer-steps",
            "1",
            "--solver",
            method,
            "--solver-max-steps",
            "2",
        ]
    )
    expected, _, _, _ = _run(args, data=data)
    solver = ADP_solver(
        solve_cg if method == "cg" else solve_hybrid, max_steps=2, tol=1e-6
    )
    config = _config(args)
    model = (
        ADP_single_index(config, solver)
        if m == 1
        else ADP_multi_index(m, config, solver)
    )
    model.fit(data[0], data[1])
    np.testing.assert_array_equal(model.beta_ if m == 1 else model.basis_.T, expected)


def test_empty_neighborhood_is_not_hidden_by_model() -> None:
    """Миграция сохраняет явную ошибку текущего estimator при пустом соседстве."""
    X, Y, _ = _synthetic_data(90, 4, 2, 0.05, 8)
    config = ADP_Config(
        N_J=18,
        N_phi=6,
        N_lin=12,
        outer_steps=2,
        index_init="random",
        training_set="exclude_centers",
        center_displacement=0.2,
        redraw_directions=False,
        seed=17,
    )
    with pytest.raises(ValueError, match="positive finite mass"):
        ADP_multi_index(2, config).fit(X, Y)
