"""Общие результаты, проверки и локальный refit CPU solver-а."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_LOCAL_REFIT_BYTES = 4 * 1024**2


@dataclass(frozen=True, slots=True)
class HPAOResult:
    """Результат внутреннего solver-а при фиксированных статистиках."""

    index: np.ndarray
    coefficients: np.ndarray
    diagnostics: dict[str, object]


def _validate_inputs(
    index, U, I, mass
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Проверить формы и конечность dense-статистик перед solver-ом."""
    for name, value in (("initial index", index), ("U", U), ("I", I), ("mass", mass)):
        if value is None:
            continue
        if hasattr(value, "__cuda_array_interface__"):
            raise NotImplementedError("Grassmann solver supports CPU arrays only")
        array = np.asarray(value)
        if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"{name} must be a real numeric array")

    index = np.asarray(index, dtype=float)
    U = np.asarray(U, dtype=float)
    I = np.asarray(I, dtype=float)
    if U.ndim != 3:
        raise ValueError("U must have shape (J, p, d)")
    if 0 in U.shape:
        raise ValueError("U dimensions J, p, and d must be positive")
    if I.shape != U.shape[:2]:
        raise ValueError("I must have shape (J, p) matching U")
    if index.ndim == 1:
        valid_index = index.shape == (U.shape[2],)
    elif index.ndim == 2:
        valid_index = index.shape[1] == U.shape[2] and 1 <= index.shape[0] <= U.shape[2]
    else:
        valid_index = False
    if not valid_index:
        raise ValueError("initial index must have shape (d,) or (m, d)")
    if not all(np.all(np.isfinite(array)) for array in (index, U, I)):
        raise ValueError("initial index, U, and I must contain only finite values")

    if mass is None:
        mass = np.ones(U.shape[0])
    else:
        mass_array = np.asarray(mass)
        if np.iscomplexobj(mass_array) or not np.issubdtype(
            mass_array.dtype, np.number
        ):
            raise TypeError("mass must be a real numeric array")
        mass = np.asarray(mass, dtype=float)
    if mass.shape != (U.shape[0],):
        raise ValueError("mass must have shape (J,)")
    if not np.all(np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError("mass must contain only finite positive values")
    return index, U, I, mass


def _normalize_index(index: np.ndarray) -> np.ndarray:
    """Нормировать single-index или ортонормировать строки multi-index."""
    if index.ndim == 1:
        norm = np.linalg.norm(index)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("initial index must have a nonzero finite norm")
        return index / norm
    if np.linalg.matrix_rank(index) < index.shape[0]:
        raise ValueError("initial multi-index matrix must have full row rank")
    basis, _ = np.linalg.qr(index.T, mode="reduced")
    return basis.T


def _local_refit(
    I: np.ndarray, U: np.ndarray, index: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Найти minimum-norm локальные коэффициенты без normal equations."""
    if index.ndim == 1:
        projected = U @ index  # (J, p)
        denominator = np.einsum("jp,jp->j", projected, projected, optimize=True)
        numerator = np.einsum("jp,jp->j", I, projected, optimize=True)
        coefficients = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
        return coefficients, (denominator > 0).astype(np.intp)

    J, P, _ = U.shape
    m = len(index)
    coefficients = np.empty((J, m))
    ranks = np.empty(J, dtype=np.intp)
    # NUMERICAL: minimum-norm SVD сохраняет cutoff исходного lstsq(rcond=None).
    # Пакеты убирают J вызовов Python; рабочие SVD-массивы ограничены 4 MiB
    # либо размером одного центра, если один центр превышает этот бюджет.
    block_size = max(1, _LOCAL_REFIT_BYTES // (8 * (3 * P * m + 3 * m * m)))
    rcond = np.finfo(float).eps * max(P, m)
    for start in range(0, J, block_size):
        stop = min(J, start + block_size)
        projected = U[start:stop] @ index.T
        left, singular, right = np.linalg.svd(projected, full_matrices=False)
        keep = singular > rcond * singular[:, :1]
        coordinates = (left.swapaxes(1, 2) @ I[start:stop, :, None]).squeeze(-1)
        np.divide(coordinates, singular, out=coordinates, where=keep)
        coordinates[~keep] = 0
        coefficients[start:stop] = (
            right.swapaxes(1, 2) @ coordinates[..., None]
        ).squeeze(-1)
        ranks[start:stop] = np.count_nonzero(keep, axis=1)
        # Вблизи потери ранга разные LAPACK-драйверы заметно расходятся
        # по слабым компонентам. Сохраняем исходный lstsq для таких центров.
        sensitive = singular[:, -1] <= math.sqrt(np.finfo(float).eps) * singular[:, 0]
        for local in np.flatnonzero(sensitive):
            j = start + local
            coefficients[j], _, ranks[j], _ = np.linalg.lstsq(
                projected[local], I[j], rcond=None
            )
    return coefficients, ranks


def _predict(U: np.ndarray, index: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Вычислить локальные fitted values для single- или multi-index."""
    if index.ndim == 1:
        return coefficients[:, None] * (U @ index)
    return (U @ (coefficients @ index)[..., None]).squeeze(-1)


def _loss(I, U, index, coefficients, mass) -> float:
    """Вычислить взвешенную половину квадратичной ошибки ``I - fitted``."""
    residual = I - _predict(U, index, coefficients)
    return 0.5 * float(np.einsum("j,jp,jp->", mass, residual, residual))
