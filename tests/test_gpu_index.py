"""CPU references and real-CUDA checks for the current single/multi engine."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ADP import ADP_Config, ADP_multi_index, ADP_single_index, ADP_solver
from ADP.cli.main import _synthetic_data
from ADP.engine.common.gpu_statistics import GPUStatistics, _neighbor_moments
from ADP.engine.common.statistic import calculate_statistics
from ADP.solver import LSMR
from ADP.solver.CG import solve as solve_cg
from ADP.solver.HYBRID import solve as solve_hybrid


@pytest.fixture(scope="module")
def cp():
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"CUDA unavailable: {error}")
    if not count:
        pytest.skip("CUDA device unavailable")
    return cp


def test_neighbor_formula_numpy_reference() -> None:
    rng = np.random.default_rng(13)
    X = rng.normal(size=(40, 6)) + 1e9
    Y = rng.normal(size=40)
    Phi = rng.normal(size=(5, 4, 6))
    indices = np.array([rng.choice(40, 6, replace=False) for _ in range(5)])
    W = np.zeros((5, 40))
    W[np.arange(5)[:, None], indices] = rng.uniform(0.1, 3, size=(5, 6))
    mass = W.sum(axis=1)
    x_bar = X.mean(axis=0)
    for normalized in (False, True):
        ref = calculate_statistics(X, Y, W, Phi, normalized=normalized)
        got = _neighbor_moments(
            X - x_bar,
            Y,
            indices,
            W[np.arange(5)[:, None], indices] / mass[:, None],
            Phi,
            mass,
            normalized,
            xp=np,
        )
        np.testing.assert_allclose(got.I, ref.I, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(got.U, ref.U, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("resident", [False, True])
def test_statistics_cuda_reference(cp, normalized, sparse, resident) -> None:
    rng = np.random.default_rng(15)
    X = rng.normal(size=(80, 7)) + 1e9
    X[:, -1] = X[:, 0] + rng.normal(scale=1e-6, size=80)
    Y = rng.normal(size=80) + 1e4
    Phi = rng.normal(size=(9, 5, 7))
    W = rng.uniform(0.1, 1, size=(9, 80))
    if sparse:
        W[:, 10:] = 0
    W *= np.geomspace(1e-5, 1e5, 9)[:, None]
    ref = calculate_statistics(X, Y, W, Phi, normalized=normalized)
    engine = GPUStatistics(X, Y, keep_on_device=resident)
    for batch in (1, 4, 32):
        got = engine.calculate(W, Phi, batch, normalized=normalized)
        for name in ("I", "U", "mass", "mean", "n_eff", "S"):
            value = getattr(got, name)
            if isinstance(value, cp.ndarray):
                value = cp.asnumpy(value)
            # Offset in Y amplifies cancellation in I; scale-aware absolute bound.
            scale = max(1.0, float(np.max(np.abs(getattr(ref, name)))))
            np.testing.assert_allclose(
                value, getattr(ref, name), rtol=2e-9, atol=2e-10 * scale
            )
        assert np.isfinite(got.eta).all()
    W[0] = 0
    with pytest.raises(ValueError, match="positive finite mass"):
        engine.calculate(W, Phi, normalized=normalized)


@pytest.mark.parametrize("m", [1, 2])
@pytest.mark.parametrize("ridge", [0.0, 1e-6, 1.0, 1e4])
def test_gpu_operator_and_correction(cp, m, ridge) -> None:
    rng = np.random.default_rng(17)
    U = rng.normal(size=(11, 6, 5))
    I = rng.normal(size=(11, 6))
    index = np.linalg.qr(rng.normal(size=(5, m)))[0].T
    if m == 1:
        index = index[0]
    mass = np.geomspace(0.1, 10, 11)
    coefficients, _ = LSMR._local_refit(I, U, index)
    operator = LSMR._linear_operator(U, coefficients, np.sqrt(mass), index.shape)
    gpu = LSMR._linear_operator(
        cp.asarray(U), cp.asarray(coefficients), cp.sqrt(cp.asarray(mass)), index.shape
    )
    x = rng.normal(size=index.size)
    y = rng.normal(size=I.size)
    np.testing.assert_allclose(
        cp.asnumpy(gpu @ cp.asarray(x)), operator @ x, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        cp.asnumpy(gpu.rmatvec(cp.asarray(y))),
        operator.rmatvec(y),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        float(cp.vdot(gpu @ cp.asarray(x), cp.asarray(y))),
        float(cp.vdot(cp.asarray(x), gpu.rmatvec(cp.asarray(y)))),
        atol=1e-11,
        rtol=1e-12,
    )
    dense = np.column_stack([operator @ e for e in np.eye(index.size)])
    residual = np.sqrt(mass)[:, None] * (I - LSMR._predict(U, index, coefficients))
    expected = np.linalg.lstsq(
        np.vstack([dense, np.sqrt(ridge) * np.eye(index.size)]),
        np.concatenate([residual.ravel(), np.zeros(index.size)]),
        rcond=None,
    )[0]
    step = LSMR._global_correction(
        cp.asarray(I),
        cp.asarray(U),
        cp.asarray(index),
        cp.asarray(coefficients),
        cp.asarray(mass),
        ridge,
        1e-8,
        200,
    )
    np.testing.assert_allclose(cp.asnumpy(step[0]), expected, atol=2e-9, rtol=2e-8)
    assert np.isfinite(step[3])
    assert step[2] <= 200


@pytest.mark.parametrize("m", [1, 2])
def test_gpu_solver_reference_and_failure(cp, m) -> None:
    rng = np.random.default_rng(42)
    U = rng.normal(size=(20, 6, 5))
    I = rng.normal(size=(20, 6))
    index = np.linalg.qr(rng.normal(size=(5, m)))[0].T
    if m == 1:
        index = index[0]
    mass = np.geomspace(0.1, 10, 20)
    ref = LSMR.solve(index, U, I, mass=mass, max_steps=3)
    got = LSMR.solve(index, cp.asarray(U), cp.asarray(I), mass=mass, max_steps=3)
    np.testing.assert_allclose(got.index, ref.index, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(
        got.diagnostics["loss_history"],
        ref.diagnostics["loss_history"],
        rtol=2e-9,
        atol=2e-10,
    )
    assert got.diagnostics["normal_residual_ratio"] <= 0.1
    assert got.diagnostics["backend"] == "cupy"
    with pytest.raises(ValueError, match="finite"):
        LSMR.solve(index, cp.full_like(cp.asarray(U), cp.nan), cp.asarray(I))
    with pytest.raises(RuntimeError, match="certificate"):
        LSMR.solve(
            index,
            cp.asarray(U),
            cp.asarray(I),
            lambda_prox=0,
            lsmr_maxiter=1,
            max_steps=1,
        )
    # Rank deficient local fits must remain finite and use the same rank cutoff.
    U[:, :, 1:] = 0
    c, ranks = LSMR._local_refit(cp.asarray(I), cp.asarray(U), cp.asarray(index))
    expected, expected_ranks = LSMR._local_refit(I, U, index)
    np.testing.assert_allclose(cp.asnumpy(c), expected, atol=1e-10, rtol=1e-10)
    np.testing.assert_array_equal(cp.asnumpy(ranks), expected_ranks)


@pytest.mark.parametrize("m", [1, 2])
@pytest.mark.parametrize("gpu_solver", [False, True])
@pytest.mark.parametrize("estimator", ["new", "legacy"])
def test_cuda_model_matches_cpu(cp, m, gpu_solver, estimator) -> None:
    X, Y, _ = _synthetic_data(100, 5, m, 0.03, 19)
    config = ADP_Config(
        N_J=20,
        N_phi=6,
        N_lin=15,
        outer_steps=2,
        select_step="last",
        estimator=estimator,
    )

    def make(cfg):
        return ADP_single_index(cfg) if m == 1 else ADP_multi_index(m, cfg)

    ref = make(config).fit(X, Y)
    got = make(replace(config, gpu=True, gpu_solver=gpu_solver)).fit(X, Y)
    a = ref.beta_ if m == 1 else ref.basis_.T
    b = got.beta_ if m == 1 else got.basis_.T
    np.testing.assert_allclose(
        np.atleast_2d(a).T @ np.atleast_2d(a),
        np.atleast_2d(b).T @ np.atleast_2d(b),
        rtol=1e-7,
        atol=1e-8,
    )
    assert got.effective_parameters_["gpu"]
    assert len(got.trace_) == len(ref.trace_)


@pytest.mark.parametrize("method", [solve_cg, solve_hybrid])
def test_cpu_solvers_accept_gpu_statistics(cp, method) -> None:
    X, Y, _ = _synthetic_data(80, 5, 2, 0.03, 21)
    config = ADP_Config(N_J=16, N_phi=6, outer_steps=1, gpu=True)
    model = ADP_multi_index(2, config, ADP_solver(method, max_steps=2)).fit(X, Y)
    assert np.isfinite(model.basis_).all()
    with pytest.raises(NotImplementedError, match="built-in LSMR"):
        ADP_multi_index(2, replace(config, gpu_solver=True), ADP_solver(method)).fit(
            X, Y
        )


def test_gpu_configuration_and_cli_boundaries() -> None:
    from ADP.cli.main import _config, _run, build_parser

    args = build_parser().parse_args(["--gpu", "--gpu-solver"])
    assert _config(args).gpu and _config(args).gpu_solver
    with pytest.raises(ValueError, match="requires gpu"):
        ADP_Config(gpu_solver=True)
    with pytest.raises(TypeError, match="boolean"):
        ADP_Config(gpu_solver="yes")
    args = build_parser().parse_args(["--mode", "manifold", "--gpu"])
    with pytest.raises(NotImplementedError, match="single and multi"):
        _run(args)
    args = build_parser().parse_args(
        [
            "--mode",
            "multi",
            "--index-dim",
            "2",
            "--gpu",
            "--gpu-solver",
            "--solver",
            "hybrid",
        ]
    )
    with pytest.raises(NotImplementedError, match="LSMR only"):
        _run(args)
