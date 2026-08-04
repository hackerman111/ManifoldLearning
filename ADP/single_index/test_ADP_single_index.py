import numpy as np

from ADP.single_index.ADP_single_index import ADP_single_index


def test_rho_and_localizing_tensor():
    model = object.__new__(ADP_single_index)
    model.X = np.array([[0.0, 0.0, 0.5], [0.0, 2.0, 0.0]])
    model.x_j = model.X[:, :1]
    model.beta = np.array([1.0, 0.0])
    model.d = 2
    model.h_k = 1.0
    model.N_loc = 2.5
    model.kernel = lambda argument: (argument <= 1).astype(float)

    model.Calculate_rho_k()
    assert np.isclose(model.rho_k, 0.5)

    model.Calculate_T_k()
    expected_square = model.rho_k**2 * np.eye(model.d) + np.outer(
        model.beta, model.beta
    )
    np.testing.assert_allclose(model.T_k.T @ model.T_k, expected_square)
    assert model.Calculate_weight(model.kernel).shape == (1, 3)
