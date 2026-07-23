import sys
from types import SimpleNamespace

import numpy as np
import pytest


def _install_fake_cupy(monkeypatch) -> None:
    fake = SimpleNamespace(
        asarray=np.asarray,
        asnumpy=np.asarray,
        exp=np.exp,
        maximum=np.maximum,
        square=np.square,
        einsum=np.einsum,
        matmul=np.matmul,
        swapaxes=np.swapaxes,
        zeros=np.zeros,
        quantile=np.quantile,
        isfinite=np.isfinite,
        finfo=np.finfo,
    )
    monkeypatch.setitem(sys.modules, "cupy", fake)


def _statistics_inputs() -> dict[str, np.ndarray | str]:
    return {
        "X": np.array([[0.0, 1.0], [1.0, 0.0], [0.5, -0.5]]),
        "y": np.array([1.0, -1.0, 0.5]),
        "centers": np.array([[0.25, 0.25]]),
        "directions": np.array([[[1.0, 0.0], [0.0, 1.0]]]),
        "q": np.array([[0.25, 0.5, 1.25]]),
        "kernel": "epanechnikov",
    }


@pytest.mark.parametrize("bad_value", (np.nan, np.inf, -np.inf))
def test_cupy_kernel_rejects_nonfinite_q(monkeypatch, bad_value) -> None:
    from adp.backends.cupy_backend import CupyBackend

    _install_fake_cupy(monkeypatch)

    with pytest.raises(ValueError, match="q.*конечные"):
        CupyBackend("float32").kernel(
            np.array([0.0, bad_value], dtype=np.float32),
            "epanechnikov",
        )


@pytest.mark.parametrize("field", ("X", "y", "centers", "directions", "q"))
def test_cupy_statistics_reject_nonfinite_inputs(monkeypatch, field) -> None:
    from adp.backends.cupy_backend import CupyBackend

    _install_fake_cupy(monkeypatch)
    inputs = _statistics_inputs()
    invalid = np.asarray(inputs[field]).copy()
    invalid.reshape(-1)[0] = np.nan
    inputs[field] = invalid

    with pytest.raises(ValueError, match=rf"{field}.*конечные"):
        CupyBackend().random_projection_sums(**inputs)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("X_rank", "X.*двумерным"),
        ("y_rank", "y.*вектором длины n"),
        ("y_length", "y.*вектором длины n"),
        ("centers_dimension", "centers.*C x d"),
        ("directions_rank", "directions.*C x P x d"),
        ("directions_dimension", "directions.*C x P x d"),
        ("q_rank", "q.*C x n"),
        ("center_count", "одинаковое число центров"),
        ("observation_count", "q.*C x n"),
    ),
)
def test_cupy_statistics_reject_invalid_shapes(monkeypatch, case, message) -> None:
    from adp.backends.cupy_backend import CupyBackend

    _install_fake_cupy(monkeypatch)
    inputs = _statistics_inputs()
    if case == "X_rank":
        inputs["X"] = np.zeros(6)
    elif case == "y_rank":
        inputs["y"] = np.zeros((3, 1))
    elif case == "y_length":
        inputs["y"] = np.zeros(2)
    elif case == "centers_dimension":
        inputs["centers"] = np.zeros((1, 3))
    elif case == "directions_rank":
        inputs["directions"] = np.zeros((2, 2))
    elif case == "directions_dimension":
        inputs["directions"] = np.zeros((1, 2, 3))
    elif case == "q_rank":
        inputs["q"] = np.zeros(3)
    elif case == "center_count":
        inputs["centers"] = np.zeros((2, 2))
    elif case == "observation_count":
        inputs["q"] = np.zeros((1, 2))

    with pytest.raises(ValueError, match=message):
        CupyBackend().random_projection_sums(**inputs)


def test_cupy_statistics_reject_unknown_kernel(monkeypatch) -> None:
    from adp.backends.cupy_backend import CupyBackend

    _install_fake_cupy(monkeypatch)
    inputs = _statistics_inputs()
    inputs["kernel"] = "unknown"

    with pytest.raises(ValueError, match="kernel"):
        CupyBackend().random_projection_sums(**inputs)


def test_cupy_pairwise_quantities_are_stable_under_large_float32_offset(
    monkeypatch,
) -> None:
    from adp.backends.cupy_backend import CupyBackend

    _install_fake_cupy(monkeypatch)
    rng = np.random.default_rng(734)
    offset = np.float32(10_000.0)
    X = (rng.normal(size=(18, 5)).astype(np.float32) + offset).astype(np.float32)
    centers = (
        rng.normal(size=(7, 5)).astype(np.float32) + offset
    ).astype(np.float32)
    beta = rng.normal(size=5).astype(np.float32)
    beta /= np.linalg.norm(beta)

    differences = X[None, :, :] - centers[:, None, :]
    expected_norm2 = np.einsum("cnd,cnd->cn", differences, differences)
    expected_projection2 = np.square(
        np.einsum("cnd,d->cn", differences, beta)
    )

    backend = CupyBackend("float32")
    actual_norm2 = backend.to_numpy(backend.pairwise_norm2(X, centers))
    actual_projection2 = backend.to_numpy(
        backend.pairwise_projection2(X, centers, beta)
    )

    np.testing.assert_allclose(actual_norm2, expected_norm2, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        actual_projection2,
        expected_projection2,
        rtol=2e-6,
        atol=2e-6,
    )


def test_cupy_statistics_use_centered_differences_for_large_float32_offset(
    monkeypatch,
) -> None:
    from adp.backends.cupy_backend import CupyBackend

    _install_fake_cupy(monkeypatch)
    rng = np.random.default_rng(901)
    X = rng.normal(size=(16, 4)).astype(np.float32)
    y = rng.normal(size=16).astype(np.float32)
    centers = rng.normal(size=(5, 4)).astype(np.float32)
    directions = rng.normal(size=(5, 3, 4)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    q = rng.uniform(0.0, 1.4, size=(5, 16)).astype(np.float32)
    offset = np.float32(10_000.0)

    backend = CupyBackend("float32")
    shifted_X = X + offset
    shifted_centers = centers + offset
    shifted = backend.random_projection_sums(
        X=shifted_X,
        y=y,
        centers=shifted_centers,
        directions=directions,
        q=q,
        kernel="quartic",
    )

    weights = np.square(np.maximum(np.float32(1.0) - q, np.float32(0.0)))
    differences = shifted_X[None, :, :] - shifted_centers[:, None, :]
    projected = np.einsum("cnd,cpd->cnp", differences, directions)
    projected *= weights[:, :, None]
    expected = (
        np.einsum("cnp,n->cp", projected, y),
        projected.sum(axis=1),
        np.einsum("cnp,cnd->cpd", projected, differences),
        weights.sum(axis=1),
    )
    for shifted_part, expected_part in zip(shifted[:4], expected, strict=True):
        np.testing.assert_allclose(
            backend.to_numpy(shifted_part),
            expected_part,
            rtol=3e-5,
            atol=3e-5,
        )
