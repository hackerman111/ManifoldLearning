import numpy as np
import pytest

from adp.data import Data
from adp.edr import EDR


def test_data_generates_reproducible_single_index_arrays():
    data_a = Data(
        n_samples=30,
        n_features=4,
        n_centers=6,
        noise_scale=0.0,
        link="sin",
        random_state=12,
    )
    data_b = Data(
        n_samples=30,
        n_features=4,
        n_centers=6,
        noise_scale=0.0,
        link="sin",
        random_state=12,
    )

    X_a = data_a.Generate_X()
    beta_a = data_a.Generate_beta()
    function_a = data_a.Generate_func()
    noise_a = data_a.Generate_Noise()
    Y_a = data_a.Generate_Y()
    centers_a = data_a.Generate_Centers()

    X_b = data_b.Generate_X()
    beta_b = data_b.Generate_beta()
    data_b.Generate_func()
    data_b.Generate_Noise()
    Y_b = data_b.Generate_Y()
    centers_b = data_b.Generate_Centers()

    assert X_a.shape == (30, 4)
    assert beta_a.shape == (4,)
    assert noise_a.shape == (30,)
    assert Y_a.shape == (30,)
    assert centers_a.shape == (6, 4)
    assert np.linalg.norm(beta_a) == pytest.approx(1.0)
    assert np.allclose(Y_a, function_a(X_a @ beta_a))

    assert np.allclose(X_a, X_b)
    assert np.allclose(beta_a, beta_b)
    assert np.allclose(Y_a, Y_b)
    assert np.allclose(centers_a, centers_b)


def test_data_generate_y_builds_missing_dependencies_once():
    data = Data(
        n_samples=25,
        n_features=3,
        n_centers=5,
        noise_scale=0.0,
        link="linear",
        random_state=4,
    )

    Y = data.Generate_Y()
    centers = data.Generate_Centers()

    assert data.X.shape == (25, 3)
    assert data.beta.shape == (3,)
    assert data.noise.shape == (25,)
    assert data.func is not None
    assert Y.shape == (25,)
    assert centers.shape == (5, 3)
    assert np.allclose(Y, data.X @ data.beta)


def test_data_rejects_unknown_link():
    data = Data(link="unknown")

    with pytest.raises(ValueError, match="Unknown link"):
        data.Generate_func()


def test_edr_generate_data_uses_data_generator_dimensions():
    model = EDR(
        n_samples=36,
        n_features=4,
        n_centers=7,
        n_directions=3,
        min_neighbors=3,
        random_state=9,
    )

    model.Generate_Data()

    assert model.X.shape == (36, 4)
    assert model.Y.shape == (36,)
    assert model.beta.shape == (4,)
    assert model.centers.shape == (7, 4)
    assert np.allclose(np.mean(model.X, axis=0), np.zeros(4))
    assert np.linalg.norm(model.beta) == pytest.approx(1.0)
