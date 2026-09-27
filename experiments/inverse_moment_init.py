"""Изолированный SIR+SAVE initializer; population scope описан в proof.md."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from scipy.linalg import qr, solve_triangular
from scipy.sparse.linalg import LinearOperator, eigsh

from ADP.engine.common.initialize import _orient_basis


def _slice_operator(Z: np.ndarray, Y: np.ndarray, slices: int) -> LinearOperator:
    """Сумма SIR и SAVE без хранения H ковариаций размера (d,d)."""
    n, d = Z.shape
    ordered = Z[np.argsort(Y, kind="stable")]
    groups = np.array_split(ordered, slices)
    means = np.array([group.mean(axis=0) for group in groups])
    probabilities = np.array([len(group) / n for group in groups])
    for group, mean in zip(groups, means, strict=True):
        group -= mean

    def action(vector: np.ndarray) -> np.ndarray:
        value = vector.reshape(d)
        result = means.T @ (probabilities * (means @ value))
        for group, probability in zip(groups, probabilities, strict=True):
            residual = value - group.T @ (group @ value) / len(group)
            result += probability * (
                residual - group.T @ (group @ residual) / len(group)
            )
        return result

    operator_type: Any = LinearOperator  # SciPy callable-constructor typing.
    return operator_type((d, d), matvec=action, rmatvec=action, dtype=np.float64)


def initialize_basis_inverse_moments(
    X: np.ndarray,
    Y: np.ndarray,
    index_dim: int,
    *,
    slices: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Вернуть (d,m) basis и eigen diagnostics для Gaussian SDR-гипотезы.

    ESTIMATOR: M=sum p_h*(mu_h mu_h.T+(I-C_h)^2). Thin pivoted QR
    нормирует X, triangular solve возвращает directions в исходные
    координаты. O(nd+d²) памяти; без normal matrix и explicit inverse.
    Ошибка при rank deficiency, малом eigengap или незавершённом eigensolve.
    """
    if np.iscomplexobj(X) or np.iscomplexobj(Y):
        raise TypeError("inverse moments require real inputs")
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.shape != (len(X),):
        raise ValueError("expected X=(n,d), Y=(n,)")
    n, d = X.shape
    for name, value in (("index_dim", index_dim), ("slices", slices), ("seed", seed)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
    if n <= d or not 1 <= index_dim < d - 1:
        raise ValueError("inverse moments require n>d and 1<=index_dim<d-1")
    if not 2 <= slices <= n // 2 or seed < 0:
        raise ValueError("require 2<=slices<=n//2 and nonnegative seed")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        raise ValueError("inverse moments require finite inputs")
    if np.all(Y[0] == Y):
        raise ValueError("constant response does not identify an index")

    # Вычитание опорной строки уменьшает cancellation при большом offset.
    centered = X - X[0]
    centered -= centered.mean(axis=0)
    whitened, triangular, permutation = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray],
        qr(centered, mode="economic", pivoting=True),
    )
    del centered
    singular = np.linalg.svd(triangular, compute_uv=False)
    cutoff = np.finfo(float).eps * max(n, d) * singular[0]
    if singular[-1] <= cutoff:
        raise RuntimeError("inverse-moment whitening requires full column rank")
    whitened *= np.sqrt(n)
    operator = _slice_operator(whitened, Y, slices)
    del whitened
    eigen_solve: Any = eigsh  # SciPy currently infers tol as int.
    values, vectors = eigen_solve(
        operator,
        k=index_dim + 1,
        which="LA",
        v0=np.random.default_rng(seed).normal(size=d),
        tol=1e-10,
        maxiter=1000,
    )
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    residual = float(
        np.linalg.norm(cast(np.ndarray, operator @ vectors) - vectors * values)
        / max(float(values[0]), np.finfo(float).tiny)
    )
    gap = float(values[index_dim - 1] - values[index_dim])
    if not np.all(np.isfinite(values)) or not np.isfinite(residual) or residual > 1e-8:
        raise RuntimeError("inverse-moment eigen residual failed")
    if gap <= 1e-8 * max(float(values[0]), np.finfo(float).tiny):
        raise RuntimeError("inverse-moment sample eigengap is not resolved")
    raw = np.empty((d, index_dim))
    raw[permutation] = solve_triangular(triangular, vectors[:, :index_dim])
    basis, _ = np.linalg.qr(raw, mode="reduced")
    if not np.all(np.isfinite(basis)):
        raise RuntimeError("inverse-moment back-transform is non-finite")
    return _orient_basis(basis), {
        "slices": slices,
        "eigenvalues": values.tolist(),
        "relative_eigen_residual": residual,
        "sample_eigengap": gap,
        "whitening_condition": float(singular[0] / singular[-1]),
    }
