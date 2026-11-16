"""Общие dense/sparse и CPU/GPU локальные статистики ADP.

Функции принимают ``X=(n,d)``, ``Y=(n,)``, направления ``Phi=(J,P,d)`` и
плотные либо sparse-блоки весов. Они потоково строят ``I=(J,P)``,
``U=(J,P,d)``, локальные массы и effective sample size без полной копии
дополнительных тензоров для всех центров.
"""

# ruff: noqa: RUF002

from collections.abc import Iterator

import numpy as np

from ...core.ADP_Statistic import ADP_Statistics
from ...gpu import require_cupy
from . import utils
from .box_kernel import (
    SparseNeighborhoodBlock,
    SparseStatisticsCache,
)


def _weight_blocks(weights, J, n, batch_size):
    """Нормализовать dense/iterator weights к потоку ``(start, block)``.

    Sparse blocks сохраняют собственный offset, а плотные веса разбиваются
    общим utility-потоком. Это внутренний адаптер статистического engine.
    """
    if isinstance(weights, Iterator):
        for value in weights:
            if isinstance(value, SparseNeighborhoodBlock):
                yield value.start, value
            else:
                yield value
        return
    yield from utils._weight_blocks(weights, J, n, batch_size)


def _statistics_cache(value):
    """Проверить или создать cache для sparse statistics."""
    if value is None:
        return SparseStatisticsCache()
    utils.require(
        isinstance(value, SparseStatisticsCache),
        "sparse_cache must be SparseStatisticsCache",
        TypeError,
    )
    return value


def calculate_statistics(
    X,
    Y,
    weights,
    directions,
    batch_size=32,
    *,
    sparse_cache=None,
) -> ADP_Statistics:
    """Вычислить локальные ADP-моменты из dense или sparse weight blocks.

    Для центра ``j`` и направления ``p`` функция строит центрированный
    weighted residual, затем возвращает ``I[j,p]`` и ``U[j,p,:]`` как его
    ковариации с ``Y`` и ``X``. Входы имеют формы ``X=(n,d)``, ``Y=(n,)``,
    ``Phi=(J,P,d)``; результат направляется в solver как ``I=(J,P)`` и
    ``U=(J,P,d)``. Локальная масса сохраняется отдельно для внешнего веса.
    """
    utils._check_batch_size(batch_size)
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

    cache = _statistics_cache(sparse_cache)
    cache.begin_call()
    expected_start = 0
    mass_block = np.empty(0)
    for start, W in _weight_blocks(weights, J, X.shape[0], batch_size):
        utils._check_weight_block_order(start, expected_start, J)
        if isinstance(W, SparseNeighborhoodBlock):
            stop = start + W.rows
        else:
            W, mass_block = utils._prepare_weight_block(W, X.shape[0])
            stop = start + W.shape[0]
        batch = slice(start, stop)
        Phib = Phi[batch]
        if isinstance(W, SparseNeighborhoodBlock):
            mass_block = W.mass
            block_values = _sparse_block(
                Xc,
                Y,
                W,
                Phib,
                mass_block,
                cache,
            )
        else:
            max_nonzero = int(np.count_nonzero(W, axis=1).max())
            # ponytail: 25% is an empirical dense/local crossover; benchmark-based
            # dispatch is only needed if substantially different kernels are added.
            if 4 * max_nonzero <= X.shape[0]:
                block_values = _local_block(
                    Xc,
                    Y,
                    W,
                    Phib,
                    mass_block,
                    max_nonzero,
                )
            else:
                A = W / mass_block[:, None]
                block_values = _dense_block(Xc, Y, A, Phib, mass_block)

        I[batch], U[batch], Mcb, n_eff[batch], eta[batch] = block_values
        mass[batch] = mass_block
        mean[batch] = Mcb + x_bar
        expected_start = stop

    cache.finish_call()

    utils.require(expected_start == J, "weight blocks do not cover all directions")

    return ADP_Statistics(
        I=I,
        U=U,
        mass=mass,
        mean=mean,
        n_eff=n_eff,
        eta=eta,
    )


def calculate_statistics_gpu(
    X,
    Y,
    weights,
    directions,
    batch_size=32,
    *,
    sparse_cache=None,
) -> ADP_Statistics:
    """Вычислить те же ``I/U`` на CUDA backend через CuPy.

    Формула и формы совпадают с CPU-веткой; преобразование в CuPy происходит
    на входе, поэтому GPU-результат выходит из функции готовым для GPU solver.
    """
    cp = require_cupy()
    utils._check_batch_size(batch_size)
    X, Y, Phi = utils._prepare_data(X, Y, directions)
    J, P = Phi.shape[:2]
    X_gpu = cp.asarray(X)
    Y_gpu = cp.asarray(Y)
    Phi_gpu = cp.asarray(Phi)
    x_bar = X_gpu.mean(axis=0)
    Xc = X_gpu - x_bar

    I = cp.empty((J, P))
    U = cp.empty((J, P, X.shape[1]))
    mass = np.empty(J)
    mean = cp.empty((J, X.shape[1]))
    n_eff = cp.empty(J)
    eta = cp.empty((J, P))

    cache = _statistics_cache(sparse_cache)
    cache.begin_call()
    expected_start = 0
    for start, W in _weight_blocks(weights, J, X.shape[0], batch_size):
        utils._check_weight_block_order(start, expected_start, J)
        if isinstance(W, SparseNeighborhoodBlock):
            mass_block = W.mass
            stop = start + W.rows
        else:
            W, mass_block = utils._prepare_weight_block(W, X.shape[0])
            stop = start + W.shape[0]
        batch = slice(start, stop)
        Phib = Phi_gpu[batch]
        if isinstance(W, SparseNeighborhoodBlock):
            block_values = _sparse_block(
                Xc,
                Y_gpu,
                W,
                Phib,
                mass_block,
                cache,
                xp=cp,
            )
        else:
            W_gpu = cp.asarray(W)
            mass_gpu = cp.asarray(mass_block)
            max_nonzero = int(np.count_nonzero(W, axis=1).max())
            if 4 * max_nonzero <= X.shape[0]:
                block_values = _local_block(
                    Xc,
                    Y_gpu,
                    W_gpu,
                    Phib,
                    mass_gpu,
                    max_nonzero,
                    xp=cp,
                )
            else:
                A = W_gpu / mass_gpu[:, None]
                block_values = _dense_block(
                    Xc,
                    Y_gpu,
                    A,
                    Phib,
                    mass_gpu,
                    xp=cp,
                )

        I[batch], U[batch], Mcb, n_eff[batch], eta[batch] = block_values
        mass[batch] = mass_block
        mean[batch] = Mcb + x_bar
        expected_start = stop

    cache.finish_call()

    utils.require(expected_start == J, "weight blocks do not cover all directions")

    return ADP_Statistics(
        I=I,
        U=U,
        mass=mass,
        mean=cp.asnumpy(mean),
        n_eff=cp.asnumpy(n_eff),
        eta=cp.asnumpy(eta),
    )


def _dense_block(Xc, Y, A, Phi, mass, *, xp=np):
    """Посчитать локальные моменты для плотного блока нормированных весов.

    Через центрированные projections ``Q`` строятся ADP-вектор ``I``, оператор
    ``U``, локальные means, effective sample size и residual diagnostic. ``xp``
    позволяет использовать NumPy или CuPy; результаты имеют формы ``(B,P)``
    и ``(B,P,d)``.
    """
    mean = A @ Xc
    y_bar = A @ Y
    Q = Phi @ Xc.T - Phi @ mean[..., None]
    residual = (Q @ A[..., None]).squeeze(-1)
    eta = _normalized_residual(A, Q, residual, xp=xp)
    H = (Q - residual[..., None]) * A[:, None, :]
    summed = H.sum(axis=2)
    I = mass[:, None] * (H @ Y - summed * y_bar[:, None])
    U = mass[:, None, None] * (H @ Xc - summed[..., None] * mean[:, None, :])
    n_eff = 1.0 / xp.square(A).sum(axis=1)
    return I, U, mean, n_eff, eta


def _local_block(Xc, Y, W, Phi, mass, width, *, xp=np):
    """Посчитать статистики, используя только ненулевых соседей блока.

    Ненулевые элементы ``W`` упаковываются в прямоугольные neighbor arrays
    ширины ``width`` и передаются в ``_local_values``; это экономит память при
    compact support.
    """
    rows, columns = xp.nonzero(W)
    counts = xp.bincount(rows, minlength=len(W))
    offsets = xp.repeat(xp.cumsum(counts) - counts, counts)
    slots = xp.arange(len(rows)) - offsets

    indices = xp.zeros((len(W), width), dtype=xp.intp)
    local_weights = xp.zeros((len(W), width), dtype=W.dtype)
    indices[rows, slots] = columns
    local_weights[rows, slots] = W[rows, columns]

    return _local_values(Xc, Y, indices, local_weights, Phi, mass, xp=xp)


def _sparse_block(Xc, Y, block, Phi, mass, cache, *, xp=np):
    """Посчитать статистики из ``SparseNeighborhoodBlock`` и его cache."""
    utils.require(
        block.n_observations == len(Xc),
        "sparse block and X dimensions do not match",
    )
    cache.observe(block)
    indices, local_weights = block.padded()
    return _local_values(
        Xc,
        Y,
        xp.asarray(indices),
        xp.asarray(local_weights),
        Phi,
        xp.asarray(mass),
        xp=xp,
    )


def _local_values(Xc, Y, indices, local_weights, Phi, mass, *, xp=np):
    """Выполнить weighted local-centering и собрать ADP-моменты.

    ``indices`` и ``local_weights`` задают соседей формы ``(B,k)``. После
    нормировки весов считаются локальные mean, response mean, centered
    projections, ``I``, ``U``, ``n_eff`` и ``eta`` без плотной матрицы весов.
    """
    A = local_weights / mass[:, None]
    local_X = Xc[indices]
    local_Y = Y[indices]
    mean = xp.einsum("bm,bmd->bd", A, local_X, optimize=True)
    y_bar = xp.einsum("bm,bm->b", A, local_Y, optimize=True)
    Q = Phi @ xp.swapaxes(local_X - mean[:, None, :], 1, 2)
    residual = (Q @ A[..., None]).squeeze(-1)
    eta = _normalized_residual(A, Q, residual, xp=xp)
    H = (Q - residual[..., None]) * A[:, None, :]
    summed = H.sum(axis=2)
    I = mass[:, None] * ((H @ local_Y[..., None]).squeeze(-1) - summed * y_bar[:, None])
    U = mass[:, None, None] * (H @ local_X - summed[..., None] * mean[:, None, :])
    n_eff = 1.0 / xp.square(A).sum(axis=1)
    return I, U, mean, n_eff, eta


def _normalized_residual(A, Q, residual, *, xp=np):
    """Нормировать weighted projection residual с защитой нулевого знаменателя."""
    denominator = (xp.abs(Q) @ A[..., None]).squeeze(-1)
    nonzero = denominator != 0
    return xp.where(
        nonzero,
        xp.abs(residual) / xp.where(nonzero, denominator, 1.0),
        0.0,
    )
