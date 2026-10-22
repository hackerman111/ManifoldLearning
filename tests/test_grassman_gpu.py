"""Frozen CPU references and real CUDA checks for the current Grassmann."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ADP import ADP_Config, ADP_multi_index, ADP_solver
from ADP.gpu import to_numpy
from ADP.solver import grassman as g


@pytest.fixture(scope="module")
def cp():
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"CUDA unavailable: {error}")
    if not count:
        pytest.skip("CUDA unavailable")
    return cp


def problem(m=5, scenario="smooth"):
    rng = np.random.default_rng(2210 + m)
    U = rng.normal(size=(24, 12, 13))
    I = rng.normal(size=(24, 12))
    B = np.linalg.qr(rng.normal(size=(13, m)))[0].T
    mass = np.geomspace(1e-4, 1e4, len(I))
    if scenario == "intrinsic":
        U[0] = 0
        U[1] = np.outer(rng.normal(size=12), rng.normal(size=13))
    elif scenario == "collinear":
        U[:, :, 1:] = U[:, :, :1] + 1e-12 * U[:, :, 1:]
    return B, U, I, mass


@pytest.mark.parametrize("m", [1, 2, 5])
@pytest.mark.parametrize("scenario", ["smooth", "intrinsic", "collinear"])
def test_cpu_frozen_reference(m, scenario):
    path = (
        Path(__file__).parents[1]
        / "experiments/grassman_gpu_2026_10_22/cpu_reference.py"
    )
    spec = importlib.util.spec_from_file_location("ADP.solver._gpu_reference", path)
    assert spec is not None and spec.loader is not None
    reference = importlib.util.module_from_spec(spec)
    # dataclass resolves the module while evaluating annotations.
    import sys

    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    B, U, I, mass = problem(m, scenario)
    expected = reference.solve(B, U, I, mass=mass, max_steps=4)
    got = g.solve(B, U, I, mass=mass, max_steps=4)
    np.testing.assert_array_equal(got.index, expected.index)
    np.testing.assert_array_equal(got.coefficients, expected.coefficients)
    assert got.diagnostics == expected.diagnostics


@pytest.mark.parametrize("m", [1, 2, 5])
@pytest.mark.parametrize("tau", [0.0, 1e-8, 0.3])
def test_cuda_profile_jacobian_and_gradient(cp, m, tau):
    B, U, I, mass = problem(m)
    Y = B.T
    V = np.linalg.qr(np.eye(13) - Y @ Y.T)[0][:, :2]
    M, W = U @ Y, U @ V
    ref = g._profile(M, I, mass, tau)
    state = g._profile(cp.asarray(M), cp.asarray(I), cp.asarray(mass), tau)
    np.testing.assert_allclose(
        to_numpy(state.coefficients), ref.coefficients, rtol=2e-11
    )
    np.testing.assert_array_equal(to_numpy(state.ranks), ref.ranks)
    assert state.smooth == ref.smooth
    assert state.value == pytest.approx(ref.value, rel=3e-13)
    jac, residual = g._core_jacobian(state, cp.asarray(W), cp.asarray(mass), tau)
    expected_jac, expected_r = g._core_jacobian(ref, W, mass, tau)
    np.testing.assert_allclose(to_numpy(jac), expected_jac, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(to_numpy(residual), expected_r, rtol=2e-11, atol=1e-11)
    gradient = g._gradient(cp.asarray(Y), cp.asarray(U), state, cp.asarray(mass))
    np.testing.assert_allclose(
        to_numpy(gradient), g._gradient(Y, U, ref, mass), rtol=2e-11, atol=1e-8
    )


def test_cuda_tall_profile_keeps_batched_small_svd(cp, monkeypatch):
    rng = np.random.default_rng(1722)
    M = rng.normal(size=(16, 48, 5))
    I = rng.normal(size=(16, 48))
    mass = np.geomspace(1e-3, 1e3, 16)
    ref = g._profile(M, I, mass, 0.1)
    shapes = []
    original = cp.linalg.svd

    def record(value, *args, **kwargs):
        shapes.append(value.shape)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(cp.linalg, "svd", record)
    got = g._profile(cp.asarray(M), cp.asarray(I), cp.asarray(mass), 0.1)
    np.testing.assert_allclose(to_numpy(got.coefficients), ref.coefficients, atol=1e-12)
    assert got.value == pytest.approx(ref.value, rel=1e-13)
    assert all(max(shape[-2:]) <= 32 for shape in shapes)


@pytest.mark.parametrize("shape", [(7, 48, 5), (7, 12, 50), (7, 4, 48)])
def test_cuda_qr_svd_wide_tall_and_rank_loss(cp, shape):
    rng = np.random.default_rng(227)
    A = rng.normal(size=shape)
    A[0, :, -1] = A[0, :, 0]
    left, singular, right = g._local_svd(cp.asarray(A))
    np.testing.assert_allclose(
        to_numpy((left * singular[:, None, :]) @ right), A, rtol=1e-11, atol=1e-11
    )
    expected = np.linalg.svd(A, compute_uv=False)
    np.testing.assert_allclose(to_numpy(singular), expected, rtol=1e-11, atol=1e-11)
    values = g._local_svd(cp.asarray(A), compute_uv=False)
    np.testing.assert_allclose(to_numpy(values), expected, rtol=1e-11, atol=1e-11)


def test_cuda_augmented_gn_compresses_before_lstsq(cp, monkeypatch):
    B, U, I, mass = problem()
    ref = g.solve(B, U, I, mass=mass, max_steps=1)
    shapes = []
    original = cp.linalg.lstsq

    def record(design, *args, **kwargs):
        shapes.append(design.shape)
        return original(design, *args, **kwargs)

    monkeypatch.setattr(cp.linalg, "lstsq", record)
    got = g.solve(B, cp.asarray(U), cp.asarray(I), mass=mass, max_steps=1)
    np.testing.assert_allclose(
        got.index.T @ got.index, ref.index.T @ ref.index, atol=1e-10
    )
    assert shapes and all(rows == columns for rows, columns in shapes)


@pytest.mark.parametrize("m", [1, 2, 5])
@pytest.mark.parametrize("scenario", ["smooth", "intrinsic", "collinear"])
def test_cuda_solver_same_objective_and_subspace(cp, m, scenario):
    B, U, I, mass = problem(m, scenario)
    options = {
        "mass": mass,
        "max_steps": 4,
        "local_ridge": 1e-4 if scenario == "collinear" else 0.0,
    }
    expected = g.solve(B, U, I, **options)
    got = g.solve(B, cp.asarray(U), cp.asarray(I), **options)
    assert isinstance(got.index, np.ndarray) and isinstance(
        got.coefficients, np.ndarray
    )
    np.testing.assert_allclose(
        got.index.T @ got.index, expected.index.T @ expected.index, atol=2e-8
    )
    np.testing.assert_allclose(
        got.diagnostics["profile_history"],
        expected.diagnostics["profile_history"],
        rtol=1e-9,
        atol=2e-7,
    )
    assert got.diagnostics["backend"] == "cupy"
    assert got.diagnostics["stop_reason"] == expected.diagnostics["stop_reason"]
    assert got.diagnostics["converged"] == expected.diagnostics["converged"]
    assert got.diagnostics["local_rank_loss"] == expected.diagnostics["local_rank_loss"]
    np.testing.assert_allclose(got.index @ got.index.T, np.eye(m), atol=1e-12)
    assert np.max(np.diff(got.diagnostics["profile_history"]), initial=0) <= 1e-8


def test_cuda_residency_workspace_ridge_and_validation(cp, monkeypatch):
    B, U, I, mass = problem()
    downloads = []
    original = cp.asnumpy

    def record(value, *args, **kwargs):
        downloads.append(value.shape)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(cp, "asnumpy", record)
    options = {"mass": mass, "local_ridge": 0.1, "workspace_bytes": 1, "max_steps": 3}
    got = g.solve(B, cp.asarray(U), cp.asarray(I), **options)
    ref = g.solve(B, U, I, **options)
    assert got.diagnostics["workspace_fallbacks"] > 0
    assert U.shape not in downloads and I.shape not in downloads
    np.testing.assert_allclose(
        got.index.T @ got.index, ref.index.T @ ref.index, atol=1e-8
    )
    with pytest.raises(ValueError, match="finite"):
        g.solve(B, cp.full_like(cp.asarray(U), cp.nan), cp.asarray(I))
    with pytest.raises(ValueError, match="multi-index"):
        g.solve(B[0], cp.asarray(U), cp.asarray(I))
    with pytest.raises(TypeError, match="real numeric"):
        g.solve(B, cp.asarray(U, dtype=complex), cp.asarray(I))
    zero = g.solve(B, cp.zeros_like(cp.asarray(U)), cp.asarray(I), mass=mass)
    assert not zero.diagnostics["converged"]
    assert zero.diagnostics["stop_reason"] == "rank_boundary_stationary"


def test_cuda_public_full_fit(cp):
    rng = np.random.default_rng(22)
    X = rng.normal(size=(100, 13))
    y = np.sin(X[:, 0]) + X[:, 1] ** 2 + X[:, 2:5].sum(axis=1)
    config = ADP_Config(N_J=30, N_phi=12, N_lin=30, N_loc=25, outer_steps=2, seed=5)
    ref = ADP_multi_index(5, config=config, solver=ADP_solver(g.solve, max_steps=3))
    got = ADP_multi_index(
        5,
        config=replace(config, gpu=True, gpu_solver=True),
        solver=ADP_solver(g.solve, max_steps=3),
    )
    ref.fit(X, y)
    got.fit(X, y)
    np.testing.assert_allclose(
        got.basis_ @ got.basis_.T, ref.basis_ @ ref.basis_.T, atol=1e-7
    )
    assert len(got.trace_) == len(ref.trace_) == 2
    assert all(d["backend"] == "cupy" for d in got.solver_diagnostics_)
