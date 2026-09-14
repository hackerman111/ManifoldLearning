from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from ..core.ADP_Statistic import ADP_Statistics
from . import utils


def calculate_statistics(
    X: np.ndarray,
    Y: np.ndarray,
    weights: np.ndarray | Iterator[tuple[int, np.ndarray]],
    directions: np.ndarray,
    batch_size: int = 32,
    *,
    normalized: bool = False,
) -> ADP_Statistics:
    """Вычислить устойчивые локальные ADP-статистики.

    При ``normalized=True`` величины ``I`` и ``U`` делятся на локальную
    массу; саму массу затем следует передать внешним весом в решатель.
    """
    utils._check_batch_size(batch_size)
    if not isinstance(normalized, bool):
        raise TypeError("normalized must be boolean")
    X, Y, Phi = utils._prepare_data(X, Y, directions)
    J, P = Phi.shape[:2]
    x_bar = X.mean(axis=0)
    Xc = X - x_bar

    I = np.empty((J, P))
    U = np.empty((J, P, X.shape[1]))
    mass = np.empty(J)
    mean = np.empty((J, X.shape[1]))
    n_eff = np.empty(J)
    eta = np.empty((J, P))
    S = np.empty(J)

    expected_start = 0
    for start, W in utils._weight_blocks(weights, J, X.shape[0], batch_size):
        utils._check_weight_block_order(start, expected_start, J)

        W, mass_block = utils._prepare_weight_block(W, X.shape[0])
        stop = start + W.shape[0]

        batch = slice(start, stop)
        Phib = Phi[batch]
        # ponytail: 25% is an empirical dense/local crossover; benchmark-based
        # dispatch is only needed if substantially different kernels are added.
        if 4 * int(np.count_nonzero(W, axis=1).max()) <= X.shape[0]:
            block_values = _local_block(
                Xc,
                Y,
                W,
                Phib,
                mass_block,
                normalized,
            )
        else:
            block_values = _dense_block(
                Xc,
                Y,
                W / mass_block[:, None],
                Phib,
                mass_block,
                normalized,
            )

        I[batch], U[batch], Mcb, n_eff[batch], eta[batch], S[batch] = block_values
        mass[batch] = mass_block
        mean[batch] = Mcb + x_bar
        expected_start = stop

    if expected_start != J:
        raise ValueError("weight blocks do not cover all directions")

    return ADP_Statistics(
        I=I,
        U=U,
        mass=mass,
        mean=mean,
        n_eff=n_eff,
        eta=eta,
        S=S,
    )


def _dense_block(
    Xc: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    Phi: np.ndarray,
    mass: np.ndarray,
    normalized: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = A @ Xc
    y_bar = A @ Y
    # EXACT: один GEMM для всех (B, P) строк вместо B отдельных GEMM.
    Q = (Phi.reshape(-1, Xc.shape[1]) @ Xc.T).reshape(*Phi.shape[:2], len(Xc))
    Q -= Phi @ mean[..., None]
    residual = (Q @ A[..., None]).squeeze(-1)
    eta = _normalized_residual(A, Q, residual)
    # После вычисления eta буфер Q используется под H.
    Q -= residual[..., None]
    Q *= A[:, None, :]
    H = Q
    summed = H.sum(axis=2)
    I = H @ Y - summed * y_bar[:, None]
    U = (H.reshape(-1, len(Xc)) @ Xc).reshape(Phi.shape)
    U -= summed[..., None] * mean[:, None, :]
    if not normalized:
        I *= mass[:, None]
        U *= mass[:, None, None]
    n_eff = 1.0 / np.square(A).sum(axis=1)
    return I, U, mean, n_eff, eta, y_bar


def _local_block(
    Xc: np.ndarray,
    Y: np.ndarray,
    W: np.ndarray,
    Phi: np.ndarray,
    mass: np.ndarray,
    normalized: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """EXACT: только ненулевые соседи; рабочая память O(k_max*(d+P)).

    W: (B,n), Phi: (B,P,d). Центрируем до проекции, как в прежней
    локальной ветке. Пустые строки отклонены на границе calculate_statistics.
    """
    B, P, d = Phi.shape
    I = np.empty((B, P))
    U = np.empty_like(Phi)
    mean = np.empty((B, d))
    n_eff = np.empty(B)
    eta = np.empty((B, P))
    y_bar = np.empty(B)
    for j in range(B):
        indices = np.flatnonzero(W[j])
        A = W[j, indices] / mass[j]
        local_X = Xc[indices]
        local_Y = Y[indices]
        mean[j] = A @ local_X
        y_bar[j] = A @ local_Y
        Q = Phi[j] @ (local_X - mean[j]).T
        residual = Q @ A
        denominator = np.abs(Q) @ A
        np.divide(np.abs(residual), denominator, out=eta[j], where=denominator != 0)
        eta[j, denominator == 0] = 0
        Q -= residual[:, None]
        Q *= A
        summed = Q.sum(axis=1)
        I[j] = Q @ local_Y - summed * y_bar[j]
        U[j] = Q @ local_X - summed[:, None] * mean[j]
        n_eff[j] = 1.0 / (A @ A)
    if not normalized:
        I *= mass[:, None]
        U *= mass[:, None, None]
    return I, U, mean, n_eff, eta, y_bar


def _normalized_residual(A, Q, residual):
    denominator = (np.abs(Q) @ A[..., None]).squeeze(-1)
    return np.divide(
        np.abs(residual),
        denominator,
        out=np.zeros_like(residual),
        where=denominator != 0,
    )
