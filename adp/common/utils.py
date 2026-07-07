from __future__ import annotations

from typing import Callable

import numpy as np

from .types import KernelName


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

    arr = np.asarray(value, dtype=float).reshape(-1)
    norm = np.linalg.norm(arr)
    if norm < np.finfo(float).eps:
        raise ValueError("Нельзя нормировать нулевой вектор")
    return arr / norm


def normalize_rows(
    value: np.ndarray,  # Массив направлений.
) -> np.ndarray:
    """Нормирует последнюю ось массива.

    Вход:
        value: массив вида ... x d.
    Выход:
        Массив той же формы с единичной нормой по последней оси.
    """

    norms = np.linalg.norm(value, axis=-1, keepdims=True)
    norms = np.maximum(norms, np.finfo(float).eps)
    return value / norms


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

    x_sq = np.einsum("ij,ij->i", X, X)
    center_sq = np.einsum("ij,ij->i", centers, centers)
    norm2 = center_sq[:, None] + x_sq[None, :] - 2.0 * (centers @ X.T)
    np.maximum(norm2, 0.0, out=norm2)
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

    x_proj = X @ beta
    center_proj = centers @ beta
    return np.square(x_proj[None, :] - center_proj[:, None])


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

    if name == "gaussian":
        return np.exp(-0.5 * q)
    if name == "quartic":
        return np.maximum(1.0 - q * q, 0.0)
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
    raise ValueError("link должен быть callable или одним из: linear, sin, quadratic, tanh")


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
