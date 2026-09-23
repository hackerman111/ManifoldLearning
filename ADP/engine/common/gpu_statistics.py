"""NUMERICAL: GPU-моменты того же finite-sketch estimator, только float64."""

# ruff: noqa: RUF002, RUF003

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from ...core.ADP_Statistic import ADP_Statistics
from ...gpu import require_cupy
from . import utils
from .statistic import _dense_moments, _DenseMoments, _normalized_residual

# Ограничение явных временных массивов; CUDA/BLAS workspace сюда не входит.
_WORKING_BYTES = 64 * 1024**2


def _neighbor_moments(
    Xc: Any,
    Y: Any,
    indices: Any,
    A: Any,
    Phi: Any,
    mass: Any,
    normalized: bool,
    *,
    xp: Any,
) -> _DenseMoments:
    """Пакет (B,k,d) только по ненулевым соседям; нет (J,n,d)."""
    local_X = Xc[indices]
    local_Y = Y[indices]
    mean = (A[:, None, :] @ local_X).squeeze(1)
    y_bar = (A * local_Y).sum(axis=1)
    Q = Phi @ (local_X - mean[:, None, :]).swapaxes(1, 2)
    residual = (Q @ A[..., None]).squeeze(-1)
    eta = _normalized_residual(A, Q, residual, xp=xp)
    Q -= residual[..., None]
    Q *= A[:, None, :]
    summed = Q.sum(axis=2)
    I = (Q @ local_Y[..., None]).squeeze(-1) - summed * y_bar[:, None]
    U = Q @ local_X - summed[..., None] * mean[:, None, :]
    if not normalized:
        I *= mass[:, None]
        U *= mass[:, None, None]
    return _DenseMoments(I, U, mean, 1.0 / xp.square(A).sum(axis=1), eta, y_bar)


class GPUStatistics:
    """Владелец Xc/Y на GPU на время одного fit; веса приходят блоками с CPU.

    Формулы совпадают с statistic.py. В statistics-only режиме I/U скачиваются
    по блокам: полный U занимает память только CPU. В resident-режиме I/U
    остаются на GPU до окончания solver, host получает только диагностики.
    Пакеты ограничены batch_size и оценкой рабочих массивов в 64 MiB.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, *, keep_on_device: bool) -> None:
        self.cp = require_cupy()
        X, Y = utils._prepare_xy(X, Y, require_overdetermined=False)
        # Общий CPU reference-centering сохраняет исходный offset и seed protocol.
        self.x_bar = X.mean(axis=0)
        self.Xc = self.cp.asarray(X - self.x_bar)
        self.Y = self.cp.asarray(Y)
        self.keep_on_device = keep_on_device
        self.cp.cuda.get_current_stream().synchronize()

    def calculate(
        self,
        weights: np.ndarray | Iterator[tuple[int, np.ndarray]],
        directions: np.ndarray,
        batch_size: int = 32,
        *,
        normalized: bool = False,
    ) -> ADP_Statistics:
        """Вернуть I=(J,P), U=(J,P,d); отклонить пустые/невалидные веса."""
        utils._check_batch_size(batch_size)
        utils.require(
            isinstance(normalized, bool), "normalized must be boolean", TypeError
        )
        Phi = utils._finite_real_array(directions, "directions")
        utils.require(
            Phi.ndim == 3 and 0 not in Phi.shape, "directions must have shape (J,P,d)"
        )
        utils.require(
            Phi.shape[2] == self.Xc.shape[1], "directions and X dimensions do not match"
        )
        J, P, d = Phi.shape
        cp = self.cp
        output = cp if self.keep_on_device else np
        I = output.empty((J, P), dtype=float)
        U = output.empty((J, P, d), dtype=float)
        mass = np.empty(J)
        mean = np.empty((J, d))
        n_eff = np.empty(J)
        eta = np.empty((J, P))
        S = np.empty(J)
        expected_start = 0
        n = len(self.Xc)
        for start, W in utils._weight_blocks(weights, J, n, batch_size):
            utils._check_weight_block_order(start, expected_start, J)
            W, mass_block = utils._prepare_weight_block(W, n)
            width = int(np.count_nonzero(W, axis=1).max())
            sparse = 4 * width <= n
            # Не допускаем B*k*d padding сверх бюджета даже при большом batch_size.
            row_bytes = (
                8 * (3 * width * (d + P + 1) + 3 * P * d)
                if sparse
                else 8 * (n * (3 * P + 2) + 3 * P * d)
            )
            block_size = min(batch_size, max(1, _WORKING_BYTES // row_bytes))
            for offset in range(0, len(W), block_size):
                end = min(len(W), offset + block_size)
                batch = slice(start + offset, start + end)
                local_mass = mass_block[offset:end]
                phib = cp.asarray(Phi[batch])
                mb = cp.asarray(local_mass)
                if sparse:
                    # Упаковываем exact support на CPU: не копируем плотный W на GPU.
                    wb = W[offset:end]
                    rows, columns = np.nonzero(wb)
                    counts = np.bincount(rows, minlength=len(wb))
                    slots = np.arange(len(rows)) - np.repeat(
                        np.cumsum(counts) - counts, counts
                    )
                    indices = np.zeros((len(wb), width), dtype=np.int64)
                    A = np.zeros((len(wb), width))
                    indices[rows, slots] = columns
                    A[rows, slots] = wb[rows, columns] / local_mass[rows]
                    moments = _neighbor_moments(
                        self.Xc,
                        self.Y,
                        cp.asarray(indices),
                        cp.asarray(A),
                        phib,
                        mb,
                        normalized,
                        xp=cp,
                    )
                else:
                    moments = _dense_moments(
                        self.Xc,
                        self.Y,
                        cp.asarray(W[offset:end] / local_mass[:, None]),
                        phib,
                        mb,
                        normalized,
                        include_eta=True,
                        xp=cp,
                    )
                if not bool(cp.all(cp.isfinite(moments.I))) or not bool(
                    cp.all(cp.isfinite(moments.U))
                ):
                    raise RuntimeError("GPU statistics produced non-finite I/U")
                I[batch] = moments.I if self.keep_on_device else cp.asnumpy(moments.I)
                U[batch] = moments.U if self.keep_on_device else cp.asnumpy(moments.U)
                mean[batch] = cp.asnumpy(moments.mean) + self.x_bar
                n_eff[batch] = cp.asnumpy(moments.n_eff)
                eta[batch] = cp.asnumpy(moments.eta)
                S[batch] = cp.asnumpy(moments.y_bar)
                mass[batch] = local_mass
            expected_start = start + len(W)
        utils.require(expected_start == J, "weight blocks do not cover all directions")
        cp.cuda.get_current_stream().synchronize()
        return ADP_Statistics(I=I, U=U, mass=mass, mean=mean, n_eff=n_eff, eta=eta, S=S)
