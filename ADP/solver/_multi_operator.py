# ruff: noqa: RUF002

from __future__ import annotations

import numpy as np


def forward(U: np.ndarray, matrix: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Выполнить exact multi-index forward action ``U_j (L_j B)``.

    ``U`` приходит из локальных ADP-статистик формы ``(J,P,d)``,
    ``matrix`` — basis ``(m,d)``, coefficients — ``(J,m)``; на выходе
    получается prediction ``(J,P)`` для глобального solver-а. Сначала
    вычисляется ``coefficients @ matrix`` формы ``(J,d)``, поэтому тензор
    ``(J,P,m)`` не материализуется.
    """
    local = coefficients @ matrix  # (J, d)
    return (U @ local[..., None]).squeeze(-1)  # (J, P)


def adjoint(U: np.ndarray, data: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Выполнить exact adjoint к forward и вернуть gradient ``(m,d)``.

    ``data=(J,P)`` поступает из residual solver-а; сначала каждая строка
    ``U_j.T @ data_j`` поднимается в ``d``-мерное пространство, затем суммы
    взвешиваются coefficients и возвращаются в координаты basis.
    """
    pulled = (U.swapaxes(1, 2) @ data[..., None]).squeeze(-1)  # (J, d)
    return coefficients.T @ pulled
