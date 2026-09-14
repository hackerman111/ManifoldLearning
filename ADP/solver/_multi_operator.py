from __future__ import annotations

import numpy as np


def forward(U: np.ndarray, matrix: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """EXACT: ``U_j B.T L_j``; временная память ``(J,d)``, без ``(J,P,m)``."""
    local = coefficients @ matrix  # (J, d)
    return (U @ local[..., None]).squeeze(-1)  # (J, P)


def adjoint(U: np.ndarray, data: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """EXACT: сумма ``L_j (U_j.T data_j).T``, результат ``(m,d)``."""
    pulled = (U.swapaxes(1, 2) @ data[..., None]).squeeze(-1)  # (J, d)
    return coefficients.T @ pulled
