import numpy as np

from ADP import ADP_Config, ADP_SolverResult, ADP_solver
from ADP.ADP_Statistic import ADP_Statistics
from ADP.engine.calculus import (
    calculate_multi_weight,
    initialize_basis_local,
    initialize_basis_random,
)
from ADP.single_index.solvers.LSMR import solve


def _statistics(I, U):
    J = len(I)
    return ADP_Statistics(
        I=I,
        U=U,
        mass=np.ones(J),
        mean=np.zeros((J, U.shape[2])),
        n_eff=np.ones(J),
        eta=np.zeros_like(I),
    )


def test_multi_initializers_are_orthonormal():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 5))
    true_basis, _ = np.linalg.qr(rng.normal(size=(5, 2)))
    Y = np.sin(X @ true_basis[:, 0]) + np.square(X @ true_basis[:, 1])
    centers = X[:16]
    distance2 = np.maximum(
        np.square(centers).sum(1)[:, None]
        + np.square(X).sum(1)[None, :]
        - 2 * centers @ X.T,
        0,
    )

    local = initialize_basis_local(
        X,
        Y,
        centers,
        distance2,
        20,
        lambda q: np.maximum(1 - q, 0),
        1e-8,
        2,
    )
    random = initialize_basis_random(rng, 5, 2)

    assert local.shape == random.shape == (5, 2)
    np.testing.assert_allclose(local.T @ local, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(random.T @ random, np.eye(2), atol=1e-12)


def test_multi_weight_matches_explicit_low_rank_tensor():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(9, 5))
    centers = rng.normal(size=(4, 5))
    basis, _ = np.linalg.qr(rng.normal(size=(5, 2)))
    eigenvalues = np.array([1.7, 0.4])
    alpha, h = 0.35, 1.2
    kernel = lambda q: np.exp(-q)

    actual = np.vstack(
        [
            block
            for _, block in calculate_multi_weight(
                X,
                centers,
                basis,
                eigenvalues,
                h,
                alpha,
                kernel,
                block_size=2,
            )
        ]
    )
    projector = basis @ basis.T
    tensor2 = (
        alpha**2 * (np.eye(5) - projector)
        + basis @ np.diag(eigenvalues) @ basis.T
    ) / h**2
    differences = centers[:, None, :] - X[None, :, :]
    argument = np.einsum("jnd,de,jne->jn", differences, tensor2, differences)
    np.testing.assert_allclose(actual, kernel(argument), rtol=1e-12, atol=1e-12)


def test_lsmr_accepts_vector_and_matrix_indices():
    rng = np.random.default_rng(5)
    J, directions, d, m = 12, 5, 4, 2
    U = rng.normal(size=(J, directions, d))

    beta = rng.normal(size=d)
    beta /= np.linalg.norm(beta)
    slopes = rng.normal(size=J)
    single = solve(
        _statistics(slopes[:, None] * (U @ beta), U),
        beta,
        lambda_penalty=1e-3,
        local_ridge=1e-8,
        max_steps=2,
    )

    basis, _ = np.linalg.qr(rng.normal(size=(d, m)))
    coefficients = rng.normal(size=(J, m))
    I = np.einsum("jpd,dm,jm->jp", U, basis, coefficients)
    multi = solve(
        _statistics(I, U),
        basis,
        lambda_penalty=1e-3,
        local_ridge=1e-8,
        max_steps=2,
    )

    assert single.index.shape == (d,)
    assert multi.index.shape == (d, m)
    assert multi.coefficients.shape == (J, m)
    np.testing.assert_allclose(multi.index.T @ multi.index, np.eye(m), atol=1e-10)
    assert np.asarray(multi.diagnostics["eigenvalues"]).shape == (m,)


def test_multi_result_uses_projector_distance():
    from ADP.multi_index.ADP_multi_index import ADP_multi_index_result

    basis = np.eye(4)[:, :2]
    rotated = basis @ np.array([[0.0, -1.0], [1.0, 0.0]])
    result = ADP_multi_index_result(
        beta_init=basis,
        beta_final=rotated,
        beta_true=basis,
    )
    result.Calculate_dist()
    assert result.dist_init == 0.0
    assert result.dist_final < 1e-15


def test_multi_model_random_initialization_and_fit_contract():
    from ADP.multi_index.ADP_multi_index import ADP_multi_index

    rng = np.random.default_rng(6)
    X = rng.normal(size=(48, 4))
    Y = np.sin(X[:, 0]) + X[:, 1] ** 2

    def unchanged(statistics, initial_index, **params):
        return ADP_SolverResult(
            index=initial_index,
            coefficients=np.ones((len(statistics.I), 2)),
            diagnostics={"eigenvalues": np.ones(2)},
        )

    config = ADP_Config(
        seed=6,
        N_loc=12,
        N_lin=12,
        N_J=4,
        N_phi=3,
        h_min=1e6,
        index_init="random",
    )
    model = ADP_multi_index(
        2,
        config=config,
        solver=ADP_solver(unchanged),
    ).fit(X, Y)

    assert model.basis_.shape == (4, 2)
    np.testing.assert_allclose(model.basis_.T @ model.basis_, np.eye(2), atol=1e-12)
    assert model.transform(X).shape == (48, 2)
    assert model.result_.stop_reason == "h_min"


def test_multi_index_public_api():
    from ADP import ADP_multi_index, _weight_blocks_multi

    assert ADP_multi_index.__name__ == "ADP_multi_index"
    assert callable(_weight_blocks_multi)
