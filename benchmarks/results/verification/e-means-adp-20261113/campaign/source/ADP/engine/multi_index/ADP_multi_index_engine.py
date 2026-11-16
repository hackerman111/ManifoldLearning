"""Ортонормирование и sign-invariant метрики multi-index подпространств."""

# ruff: noqa: RUF002

from __future__ import annotations

import numpy as np

from ...core.multi.ADP_multi_index_utils import require_full_rank, validate_basis


def orthonormal_basis(value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Ортонормировать multi-index базис и канонизировать знаки столбцов.

    Сначала проверяется форма и конечность, затем выполняется reduced QR.
    Диагональ ``R`` проверяет полный ранг, а знак каждой колонки фиксируется
    по координате максимального модуля. Вход и выход имеют форму ``(d, m)``.
    """
    basis = validate_basis(value, shape)
    basis, triangular = np.linalg.qr(basis, mode="reduced")
    require_full_rank(triangular)
    columns = np.arange(basis.shape[1])
    signs = np.sign(basis[np.argmax(np.abs(basis), axis=0), columns])
    basis *= np.where(signs == 0, 1.0, signs)
    return basis


def subspace_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Вычислить нормированное sign/basis-invariant расстояние подпространств.

    Для обеих матриц строятся ортогональные проекторы, после чего берётся
    Frobenius-норма их разности и делится на ``sqrt(2m)``. Так поворот базиса
    внутри одного и того же подпространства не меняет результат.
    """
    left_projector, shape = projectors_for_distance(left, "left")
    right_projector, _ = projectors_for_distance(right, "right", shape)
    return float(
        np.linalg.norm(left_projector - right_projector, ord="fro")
        / np.sqrt(2.0 * shape[1])
    )


def projectors_for_distance(
    value: np.ndarray,
    name: str,
    expected_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Построить проектор для проверки расстояния между базисами.

    Функция является тонкой типизированной обёрткой над ``projector``: она
    проверяет ожидаемую форму и возвращает пару ``(P, shape)``.
    """
    return projector(value, name, expected_shape)


def projector(
    value: np.ndarray,
    name: str,
    expected_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Построить ортогональный проектор ``P=Q Q.T`` из полного базиса.

    Базис валидируется и раскладывается SVD; левые сингулярные векторы задают
    одно и то же подпространство независимо от его внутреннего базиса. Вход
    имеет форму ``(d, m)``, выходной проектор — ``(d, d)``.
    """
    basis = validate_basis(value, expected_shape, name=name, require_rank=True)
    left, _, _ = np.linalg.svd(basis, full_matrices=False)
    return left @ left.T, basis.shape
