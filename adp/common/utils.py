from __future__ import annotations

from typing import Callable

import numpy as np

from .types import KernelName


PAIRWISE_DIFFERENCE_BUFFER_BYTES = 64 * 1024 * 1024


def as_2d_float(
    value: np.ndarray | None,  # Исходное значение.
    name: str,  # Имя аргумента для сообщения об ошибке.
) -> np.ndarray:
    """Проверяет и приводит значение к двумерному float-массиву.

    Вход:
        value: объект, который должен стать матрицей.
        name: имя аргумента для ошибки.
    Выход:
        NumPy-массив размера n x d.
    """

    if value is None:
        raise ValueError(f"{name} не должен быть None")
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} должен быть двумерным массивом")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} должен содержать только конечные значения")
    return arr


def as_1d_float(
    value: np.ndarray,  # Исходное значение.
    name: str,  # Имя аргумента для сообщения об ошибке.
) -> np.ndarray:
    """Проверяет и приводит значение к одномерному float-массиву.

    Вход:
        value: объект, который должен стать вектором.
        name: имя аргумента для ошибки.
    Выход:
        NumPy-вектор длины n.
    """

    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} должен быть одномерным массивом")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} должен содержать только конечные значения")
    return arr


def unit_vector(
    value: np.ndarray,  # Исходный вектор.
) -> np.ndarray:
    """Нормирует вектор до единичной длины.

    Вход:
        value: исходный вектор.
    Выход:
        Нормированный одномерный массив.
    """

    arr = np.asarray(value).reshape(-1)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Нельзя нормировать вектор с неконечными значениями")
    scale = np.max(np.abs(arr), initial=arr.dtype.type(0.0))
    if scale == 0.0:
        raise ValueError("Нельзя нормировать нулевой вектор")
    scaled = arr / scale
    scaled_norm = arr.dtype.type(
        np.sqrt(np.sum(scaled * scaled, dtype=np.float64))
    )
    if not np.isfinite(scaled_norm) or scaled_norm <= 0.0:
        raise ValueError("Не удалось получить конечную норму вектора")
    normalized = scaled / scaled_norm
    if not np.all(np.isfinite(normalized)):
        raise ValueError("Не удалось получить конечный единичный вектор")
    return normalized


def normalize_rows(
    value: np.ndarray,  # Массив направлений.
) -> np.ndarray:
    """Нормирует последнюю ось массива.

    Вход:
        value: массив вида ... x d.
    Выход:
        Массив той же формы с единичной нормой по последней оси.
    """

    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Направления должны содержать только конечные значения")
    if arr.ndim == 0 or arr.shape[-1] == 0:
        raise ValueError("Направления должны иметь непустую последнюю ось")
    scales = np.max(np.abs(arr), axis=-1, keepdims=True)
    if np.any(scales == 0.0):
        raise ValueError("Нельзя нормировать нулевое направление")
    scaled = arr / scales
    scaled_norms = np.sqrt(
        np.sum(scaled * scaled, axis=-1, keepdims=True, dtype=np.float64)
    ).astype(arr.dtype, copy=False)
    if np.any(~np.isfinite(scaled_norms)) or np.any(scaled_norms <= 0.0):
        raise ValueError("Не удалось получить конечные нормы направлений")
    normalized = scaled / scaled_norms
    if not np.all(np.isfinite(normalized)):
        raise ValueError("Не удалось получить конечные единичные направления")
    return normalized


def stable_l2_norm(
    value: np.ndarray,  # Массив, евклидова норма которого нужна.
) -> float:
    """Считает L2-норму без переполнения промежуточных квадратов."""

    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Норма определена только для конечных значений")
    scale = float(np.max(np.abs(arr), initial=arr.dtype.type(0.0)))
    if scale == 0.0:
        return 0.0
    scaled = arr / arr.dtype.type(scale)
    return scale * float(np.sqrt(np.sum(scaled * scaled, dtype=np.float64)))


def pairwise_norm2(
    X: np.ndarray,  # Матрица наблюдений n x d.
    centers: np.ndarray,  # Матрица центров J x d.
) -> np.ndarray:
    """Считает квадраты расстояний от наблюдений до центров.

    Вход:
        X: матрица наблюдений n x d.
        centers: матрица центров J x d.
    Выход:
        Матрица J x n со значениями ||X_i - c_j||^2.
    """

    x, xcenters = _prepare_pairwise_inputs(X, centers)
    norm2 = np.empty((xcenters.shape[0], x.shape[0]), dtype=x.dtype)
    block_size = _pairwise_center_block_size(x)
    for start in range(0, xcenters.shape[0], block_size):
        stop = min(start + block_size, xcenters.shape[0])
        differences = np.subtract(
            x[None, :, :],
            xcenters[start:stop, None, :],
        )
        np.einsum(
            "cnd,cnd->cn",
            differences,
            differences,
            out=norm2[start:stop],
            optimize=True,
        )
    return norm2


def pairwise_projection2(
    X: np.ndarray,  # Матрица наблюдений n x d.
    centers: np.ndarray,  # Матрица центров J x d.
    beta: np.ndarray,  # Направление проекции.
) -> np.ndarray:
    """Считает квадраты проекций разностей на beta.

    Вход:
        X: матрица наблюдений n x d.
        centers: матрица центров J x d.
        beta: направление beta длины d.
    Выход:
        Матрица J x n со значениями <X_i - c_j, beta>^2.
    """

    x, xcenters = _prepare_pairwise_inputs(X, centers)
    xbeta = np.asarray(beta, dtype=x.dtype)
    if xbeta.ndim != 1 or xbeta.shape[0] != x.shape[1]:
        raise ValueError(f"beta должен иметь форму {(x.shape[1],)}")
    if not np.all(np.isfinite(xbeta)):
        raise ValueError("beta должен содержать только конечные значения")
    projection2 = np.empty((xcenters.shape[0], x.shape[0]), dtype=x.dtype)
    block_size = _pairwise_center_block_size(x)
    for start in range(0, xcenters.shape[0], block_size):
        stop = min(start + block_size, xcenters.shape[0])
        differences = np.subtract(
            x[None, :, :],
            xcenters[start:stop, None, :],
        )
        projected = differences @ xbeta
        np.square(projected, out=projection2[start:stop])
    return projection2


def _prepare_pairwise_inputs(
    X: np.ndarray,
    centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validates pairwise inputs while preserving float32 or float64."""

    x_input = np.asarray(X)
    centers_input = np.asarray(centers)
    dtype = np.result_type(x_input.dtype, centers_input.dtype)
    if not np.issubdtype(dtype, np.floating):
        dtype = np.dtype(np.float64)
    x = np.asarray(x_input, dtype=dtype)
    xcenters = np.asarray(centers_input, dtype=dtype)
    if x.ndim != 2 or xcenters.ndim != 2:
        raise ValueError("X и centers должны быть двумерными массивами")
    if x.shape[1] != xcenters.shape[1]:
        raise ValueError("X и centers должны иметь одинаковую размерность d")
    if not np.all(np.isfinite(x)):
        raise ValueError("X должен содержать только конечные значения")
    if not np.all(np.isfinite(xcenters)):
        raise ValueError("centers должен содержать только конечные значения")
    return x, xcenters


def _pairwise_center_block_size(X: np.ndarray) -> int:
    """Bounds the temporary C x n x d difference buffer."""

    bytes_per_center = max(1, X.shape[0] * X.shape[1] * X.dtype.itemsize)
    return max(1, PAIRWISE_DIFFERENCE_BUFFER_BYTES // bytes_per_center)


def kernel_np(
    q: np.ndarray,  # Значения квадратичной формы.
    name: KernelName,  # Имя ядра.
) -> np.ndarray:
    """Вычисляет веса ядра.

    Вход:
        q: массив значений неотрицательной квадратичной формы.
        name: epanechnikov, quartic или gaussian.
    Выход:
        Массив весов той же формы.
    """

    if (
        not isinstance(name, str)
        or name not in {"epanechnikov", "quartic", "gaussian"}
    ):
        raise ValueError(
            "kernel должен быть 'epanechnikov', 'quartic' или 'gaussian'"
        )
    if not np.all(np.isfinite(q)):
        raise ValueError("q должен содержать только конечные значения")
    if name == "gaussian":
        return np.exp(-0.5 * q)
    if name == "quartic":
        return np.maximum(1.0 - q, 0.0) ** 2
    return np.maximum(1.0 - q, 0.0)


def average_kernel_weight(
    q: np.ndarray,  # Матрица значений квадратичной формы J x n.
    name: KernelName,  # Имя ядра.
) -> float:
    """Возвращает среднюю массу локальных весов.

    Вход:
        q: матрица квадратичной формы J x n.
        name: имя ядра.
    Выход:
        Средняя сумма весов по центрам.
    """

    return float(kernel_np(q, name).sum(axis=1).mean())


def link_function(
    link: str | Callable[[np.ndarray], np.ndarray],  # Имя связи или вызываемый объект.
) -> tuple[Callable[[np.ndarray], np.ndarray], str]:
    """Возвращает функцию связи для генерации single-index данных.

    Вход:
        link: имя встроенной связи или callable.
    Выход:
        Пара (функция, имя функции).
    """

    if callable(link):
        return link, getattr(link, "__name__", "callable")
    if link == "linear":
        return linear_link, "linear"
    if link == "sin":
        return np.sin, "sin"
    if link == "quadratic":
        return quadratic_link, "quadratic"
    if link == "tanh":
        return np.tanh, "tanh"
    raise ValueError(
        "link должен быть callable или одним из: linear, sin, quadratic, tanh"
    )


def linear_link(
    z: np.ndarray,  # Одномерный индекс X beta.
) -> np.ndarray:
    """Возвращает линейную функцию связи.

    Вход:
        z: значения X beta.
    Выход:
        z без изменений.
    """

    return z


def quadratic_link(
    z: np.ndarray,  # Одномерный индекс X beta.
) -> np.ndarray:
    """Возвращает квадратичную функцию связи.

    Вход:
        z: значения X beta.
    Выход:
        z^2.
    """

    return z**2
