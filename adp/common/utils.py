from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import linalg

from .types import KernelName


def as_2d_float(value: np.ndarray | None, name: str) -> np.ndarray:
    """Проверяет и приводит значение к двумерному float-массиву.

    Вход:
        value: исходное значение.
        name: имя аргумента для текста ошибки.
    Выход:
        Двумерный NumPy-массив dtype=float.
    """

    if value is None:
        raise ValueError(f"{name} не должен быть None")
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} должен быть двумерным массивом")
    return arr


def as_1d_float(value: np.ndarray, name: str) -> np.ndarray:
    """Проверяет и приводит значение к одномерному float-массиву.

    Вход:
        value: исходное значение.
        name: имя аргумента для текста ошибки.
    Выход:
        Одномерный NumPy-массив dtype=float.
    """

    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} должен быть одномерным массивом")
    return arr


def unit_vector(value: np.ndarray) -> np.ndarray:
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


def normalize_rows(value: np.ndarray) -> np.ndarray:
    """Нормирует последнюю ось массива.

    Вход:
        value: массив направлений.
    Выход:
        Массив с единичной нормой вдоль последней оси.
    """

    norms = np.linalg.norm(value, axis=-1, keepdims=True)
    norms = np.maximum(norms, np.finfo(float).eps)
    return value / norms


def pairwise_norm2(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Считает попарные квадраты расстояний до центров.

    Вход:
        X: матрица наблюдений n x d.
        centers: матрица центров J x d.
    Выход:
        Матрица размера J x n с ||X_i - c_j||^2.
    """

    diff = X[None, :, :] - centers[:, None, :]
    return np.einsum("jnd,jnd->jn", diff, diff)


def pairwise_projection2(X: np.ndarray, centers: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Считает квадраты проекций разностей на beta.

    Вход:
        X: матрица наблюдений n x d.
        centers: матрица центров J x d.
        beta: направление проекции.
    Выход:
        Матрица размера J x n с ((X_i - c_j), beta)^2.
    """

    diff = X[None, :, :] - centers[:, None, :]
    return np.square(np.einsum("jnd,d->jn", diff, beta))


def kernel_np(q: np.ndarray, name: KernelName) -> np.ndarray:
    """Вычисляет NumPy-веса выбранного ядра.

    Вход:
        q: квадратичная форма расстояний.
        name: имя ядра.
    Выход:
        Массив весов ядра.
    """

    if name == "gaussian":
        return np.exp(-0.5 * q)
    if name == "quartic":
        return np.maximum(1.0 - q * q, 0.0)
    return np.maximum(1.0 - q, 0.0)


def average_kernel_weight(q: np.ndarray, name: KernelName) -> float:
    """Возвращает среднюю локальную массу ядра.

    Вход:
        q: матрица значений квадратичной формы.
        name: имя ядра.
    Выход:
        Средняя сумма весов по центрам.
    """

    return float(kernel_np(q, name).sum(axis=1).mean())


def safe_solve(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Решает линейную систему с fallback на least squares.

    Вход:
        lhs: левая матрица системы.
        rhs: правая часть системы.
    Выход:
        Вектор решения.
    """

    try:
        return linalg.solve(lhs, rhs, assume_a="pos")
    except Exception:
        return linalg.lstsq(lhs, rhs)[0]


def link_function(link: str | Callable[[np.ndarray], np.ndarray]) -> tuple[Callable[[np.ndarray], np.ndarray], str]:
    """Возвращает функцию связи для генерации данных.

    Вход:
        link: имя встроенной связи или callable.
    Выход:
        Кортеж функции связи и её имени.
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


def linear_link(z: np.ndarray) -> np.ndarray:
    """Возвращает линейную функцию связи.

    Вход:
        z: одномерный индекс X beta.
    Выход:
        Значение z без изменений.
    """

    return z


def quadratic_link(z: np.ndarray) -> np.ndarray:
    """Возвращает квадратичную функцию связи.

    Вход:
        z: одномерный индекс X beta.
    Выход:
        Значение z^2.
    """

    return z**2
