import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from ADP import ADP_Config, ADP_SolverResult, ADP_Statistics, ADP_solver


def _cuda_available():
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


CUDA_AVAILABLE = _cuda_available()


def test_gpu_config_cli_and_varpro_validation():
    from ADP.cli import build_parser, experiment_from_args

    parser = build_parser()
    default = experiment_from_args(parser.parse_args([]), parser)
    gpu = experiment_from_args(parser.parse_args(["--gpu"]), parser)

    assert default.variants["default"].config.gpu is False
    assert gpu.variants["default"].config.gpu is True
    with pytest.raises(SystemExit):
        experiment_from_args(
            parser.parse_args(["--gpu", "--solver", "varpro"]),
            parser,
        )


def test_gpu_request_has_no_cpu_fallback(monkeypatch):
    from ADP.gpu import require_cupy

    fake_cupy = SimpleNamespace(
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(getDeviceCount=lambda: 0),
        )
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    with pytest.raises(RuntimeError, match="CUDA device"):
        require_cupy()


def test_gpu_rejects_custom_solver_before_data_initialization():
    from ADP import ADP_single_index

    def custom_solver(statistics, beta, **kwargs):
        return ADP_SolverResult(beta, None, {})

    model = ADP_single_index(
        ADP_Config(gpu=True),
        ADP_solver(custom_solver),
    )
    with pytest.raises(ValueError, match="built-in LSMR"):
        model.fit([], [])


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA device is unavailable")
def test_gpu_statistics_match_cpu_for_local_and_dense_blocks():
    import cupy as cp

    from ADP import calculate_statistics, calculate_statistics_gpu

    rng = np.random.default_rng(12)
    n, d, J, P = 17, 4, 4, 3
    X = rng.normal(size=(n, d))
    Y = rng.normal(size=n)
    directions = rng.normal(size=(J, P, d))
    weights = rng.uniform(0.1, 1.0, size=(J, n))
    weights[:2] = 0.0
    weights[0, [1, 6, 9]] = [0.4, 1.0, 0.2]
    weights[1, [2, 8, 13]] = [0.7, 0.3, 0.9]

    cpu = calculate_statistics(X, Y, weights, directions, batch_size=2)
    gpu = calculate_statistics_gpu(
        X,
        Y,
        iter(((0, weights[:2]), (2, weights[2:]))),
        directions,
        batch_size=2,
    )

    assert isinstance(gpu.I, cp.ndarray)
    assert isinstance(gpu.U, cp.ndarray)
    assert isinstance(gpu.mass, np.ndarray)
    assert isinstance(gpu.mean, np.ndarray)
    assert isinstance(gpu.n_eff, np.ndarray)
    assert isinstance(gpu.eta, np.ndarray)
    np.testing.assert_allclose(cp.asnumpy(gpu.I), cpu.I, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(cp.asnumpy(gpu.U), cpu.U, rtol=1e-11, atol=1e-11)
    for name in ("mass", "mean", "n_eff", "eta"):
        np.testing.assert_allclose(gpu[name], cpu[name], rtol=1e-11, atol=1e-11)


def _statistics(I, U):
    J, P = I.shape
    d = U.shape[2]
    return ADP_Statistics(
        I=I,
        U=U,
        mass=np.ones(J),
        mean=np.zeros((J, d)),
        n_eff=np.ones(J),
        eta=np.zeros((J, P)),
    )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA device is unavailable")
def test_gpu_lsmr_matches_cpu_for_single_and_multi_index():
    import cupy as cp

    from ADP.single_index.solvers.LSMR import solve

    rng = np.random.default_rng(23)
    J, P, d = 9, 4, 6
    I = rng.normal(size=(J, P))
    U = rng.normal(size=(J, P, d))
    gpu_statistics = _statistics(cp.asarray(I), cp.asarray(U))
    cpu_statistics = _statistics(I, U)

    beta, _ = np.linalg.qr(rng.normal(size=(d, 2)), mode="reduced")
    settings = dict(lambda_penalty=3.0, local_ridge=1e-6, max_steps=2, tol=1e-8)

    cpu_single = solve(cpu_statistics, beta[:, 0], **settings)
    gpu_single = solve(gpu_statistics, beta[:, 0], **settings)
    np.testing.assert_allclose(
        abs(gpu_single.index @ cpu_single.index),
        1.0,
        rtol=1e-7,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        gpu_single.coefficients,
        cpu_single.coefficients,
        rtol=1e-6,
        atol=1e-7,
    )

    cpu_multi = solve(cpu_statistics, beta, **settings)
    gpu_multi = solve(gpu_statistics, beta, **settings)
    np.testing.assert_allclose(
        gpu_multi.index @ gpu_multi.index.T,
        cpu_multi.index @ cpu_multi.index.T,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        gpu_multi.diagnostics["eigenvalues"],
        cpu_multi.diagnostics["eigenvalues"],
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA device is unavailable")
def test_gpu_models_match_cpu_single_and_multi_index():
    from ADP import ADP_multi_index, ADP_single_index

    rng = np.random.default_rng(31)
    X = rng.normal(size=(40, 5))
    Y = np.sin(X[:, 0]) + 0.02 * rng.normal(size=len(X))
    config = ADP_Config(
        seed=4,
        N_loc=5,
        N_lin=8,
        N_J=8,
        N_phi=3,
        h_min=1e6,
        index_init="random",
        lambda_penalty=100.0,
    )

    cpu_single = ADP_single_index(config).fit(X, Y)
    gpu_single = ADP_single_index(replace(config, gpu=True)).fit(X, Y)
    np.testing.assert_allclose(
        abs(cpu_single.beta_ @ gpu_single.beta_),
        1.0,
        rtol=1e-6,
        atol=1e-6,
    )

    cpu_multi = ADP_multi_index(2, config).fit(X, Y)
    gpu_multi = ADP_multi_index(2, replace(config, gpu=True)).fit(X, Y)
    np.testing.assert_allclose(
        cpu_multi.basis_ @ cpu_multi.basis_.T,
        gpu_multi.basis_ @ gpu_multi.basis_.T,
        rtol=1e-5,
        atol=1e-5,
    )
