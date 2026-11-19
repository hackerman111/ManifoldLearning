"""Ядро LOO1-WLS-ADP: CSR-окрестности, векторизованный 2-DGEMM и streaming."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from ...core.ADP_Config import epanechnikov
from ...core.loo1_wls import _require_positive_int, _require_real
from . import utils
from .initialize import (
    _centered_ridge_gradient,
    _principal_gradient_basis_with_spectrum,
)
from .utils import _prepare_multi_localization
from .weights import _projected_multi_components

# Ограничение памяти (<= 16 MiB) для компактного support в streaming-поиске.
_STREAM_ACTIVE_EDGE_CAP = 1_000_000


@dataclass(frozen=True, slots=True)
class LOO1Neighborhoods:
    """Плоское CSR-представление активных локальных окрестностей LOO1-WLS."""

    n: int
    total_centers: int
    active_centers: np.ndarray
    indptr: np.ndarray
    counts: np.ndarray
    col_idx: np.ndarray
    a: np.ndarray
    yc: np.ndarray
    mass: np.ndarray
    full_mass: np.ndarray
    full_S: np.ndarray


def _iter_weight_blocks(
    weights: np.ndarray | Iterable[tuple[int, np.ndarray]],
    total_centers: int,
    n: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Унифицировать потоковый итератор весов и плотную матрицу (J, n)."""
    if isinstance(weights, Iterator):
        yield from weights
        return
    if (
        isinstance(weights, (list, tuple))
        and len(weights) > 0
        and isinstance(weights[0], tuple)
        and len(weights[0]) == 2
    ):
        yield from weights
        return
    dense = utils._finite_real_array(weights, "weights")
    if dense.shape != (total_centers, n):
        raise ValueError("weights must have shape (total_centers, n)")
    if np.any(dense < 0.0):
        raise ValueError("weights must be nonnegative")
    yield 0, dense


def build_loo1_neighborhoods(
    X: np.ndarray,
    Y: np.ndarray,
    weights: np.ndarray | Iterable[tuple[int, np.ndarray]],
    *,
    min_neighbors: int,
    total_centers: int,
) -> LOO1Neighborhoods:
    """Собрать CSR-окрестности активных центров из потоковых блоков весов.

    Входы: ``X`` формы ``(n, d)``, ``Y`` формы ``(n,)``, итератор блоков
    ``(start, W_block)`` (или плотная матрица ``(J, n)``), порог числа
    ненулевых соседей ``min_neighbors`` и полное число центров ``total_centers``.
    Исключает неактивные центры (``count < min_neighbors`` или нулевая масса),
    сохраняя ``full_mass`` и ``full_S`` для всех центров. При отсутствии
    активных окрестностей поднимает ``RuntimeError``.
    """
    X, Y = utils._prepare_xy(X, Y, require_overdetermined=False)
    min_neighbors = _require_positive_int("min_neighbors", min_neighbors)
    total_centers = _require_positive_int("total_centers", total_centers)
    n = X.shape[0]

    # NUMERICAL: глобальный сдвиг Y уменьшает cancellation в локальном центрировании.
    y_mean = float(np.mean(Y))
    y_shifted = Y - y_mean

    full_mass = np.empty(total_centers, dtype=float)
    full_S = np.empty(total_centers, dtype=float)

    active_parts: list[np.ndarray] = []
    counts_parts: list[np.ndarray] = []
    mass_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    a_parts: list[np.ndarray] = []
    yc_parts: list[np.ndarray] = []

    expected_start = 0
    for start, block in _iter_weight_blocks(weights, total_centers, n):
        utils._check_weight_block_order(start, expected_start, total_centers)
        W = utils._finite_real_array(block, "weights")
        if W.ndim != 2 or W.shape[0] == 0 or W.shape[1] != n:
            raise ValueError("weight blocks must have shape (B, n)")
        if np.any(W < 0.0):
            raise ValueError("weights must be nonnegative")
        batch = W.shape[0]
        stop = expected_start + batch
        if stop > total_centers:
            raise ValueError("weight blocks exceed total_centers")

        mass_block = W.sum(axis=1)
        if not np.all(np.isfinite(mass_block)):
            raise ValueError("weight row masses must be finite")
        wy_shifted = W @ y_shifted
        y_bar_shifted = np.divide(
            wy_shifted,
            mass_block,
            out=np.zeros(batch, dtype=float),
            where=mass_block > 0.0,
        )
        full_mass[expected_start:stop] = mass_block
        full_S[expected_start:stop] = y_mean + y_bar_shifted

        row_idx, col_idx = np.nonzero(W > 0.0)
        counts_block = np.bincount(row_idx, minlength=batch)
        active_mask = (mass_block > 0.0) & (counts_block >= min_neighbors)
        if np.any(active_mask):
            active_local = np.flatnonzero(active_mask)
            keep_edges = active_mask[row_idx]
            r_keep = row_idx[keep_edges]
            c_keep = col_idx[keep_edges]
            a_keep = W[r_keep, c_keep] / mass_block[r_keep]
            yc_keep = y_shifted[c_keep] - y_bar_shifted[r_keep]

            active_parts.append(expected_start + active_local)
            counts_parts.append(counts_block[active_local])
            mass_parts.append(mass_block[active_local])
            col_parts.append(c_keep.astype(np.intp, copy=False))
            a_parts.append(a_keep)
            yc_parts.append(yc_keep)

        expected_start = stop

    if expected_start != total_centers:
        raise ValueError("weight blocks did not cover all total_centers")
    if not active_parts:
        raise RuntimeError("no active neighborhoods satisfy min_neighbors")

    active_centers = np.concatenate(active_parts).astype(np.intp, copy=False)
    counts = np.concatenate(counts_parts).astype(np.intp, copy=False)
    mass = np.concatenate(mass_parts)
    col_idx_all = np.concatenate(col_parts).astype(np.intp, copy=False)
    a_all = np.concatenate(a_parts)
    yc_all = np.concatenate(yc_parts)

    indptr = np.empty(len(counts) + 1, dtype=np.intp)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])

    return LOO1Neighborhoods(
        n=n,
        total_centers=total_centers,
        active_centers=active_centers,
        indptr=indptr,
        counts=counts,
        col_idx=col_idx_all,
        a=a_all,
        yc=yc_all,
        mass=mass,
        full_mass=full_mass,
        full_S=full_S,
    )


def _sym_outer_reduceat(
    left: np.ndarray,
    right: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Собрать симметричные ``(B, m, m)`` через 1D ``reduceat`` без ``(E, m, m)``."""
    batch = len(offsets)
    m = left.shape[1]
    out = np.empty((batch, m, m), dtype=float)
    for row in range(m):
        left_r = left[:, row]
        for col in range(row, m):
            acc = np.add.reduceat(left_r * right[:, col], offsets)
            out[:, row, col] = acc
            if col != row:
                out[:, col, row] = acc
    return out


def _sym_batched_matvec(
    sym_matrices: np.ndarray,
    vectors: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Умножить симметричные ``(B, m, m)`` на ``(E, m)`` без тензора ``(E, m, m)``."""
    m = vectors.shape[1]
    out = np.zeros_like(vectors)
    for row in range(m):
        diag = np.repeat(sym_matrices[:, row, row], counts)
        out[:, row] += diag * vectors[:, row]
        for col in range(row + 1, m):
            off = np.repeat(sym_matrices[:, row, col], counts)
            out[:, row] += off * vectors[:, col]
            out[:, col] += off * vectors[:, row]
    return out


def _evaluate_loo1_wls_validated(
    Y: np.ndarray,
    X: np.ndarray,
    neighborhoods: LOO1Neighborhoods,
    *,
    ridge: float,
    leverage_cap: float,
    smooth_cap_eps: float,
    center_chunk: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    """Внутреннее ядро 2-DGEMM оценки для уже проверенных массивов и параметров."""
    n, _ = X.shape
    _, m = Y.shape
    j_active = len(neighborhoods.active_centers)
    if j_active == 0:
        raise RuntimeError("neighborhoods contains no active centers")

    # DGEMM #1: проецирование данных один раз на всю оценку, O(n * d * m).
    U_all = X @ Y  # (n, m)
    Gamma = np.zeros((n, m), dtype=float)
    active_coeffs = np.empty((j_active, m), dtype=float)

    indptr = neighborhoods.indptr
    counts = neighborhoods.counts
    col_idx = neighborhoods.col_idx
    a_all = neighborhoods.a
    yc_all = neighborhoods.yc
    mass_all = neighborhoods.mass

    eye_m = np.eye(m, dtype=float)
    diag_idx = np.arange(m)
    obj = 0.0
    max_leverage = 0.0
    clipped_count = 0

    for c0 in range(0, j_active, center_chunk):
        c1 = min(j_active, c0 + center_chunk)
        batch = c1 - c0
        e0 = int(indptr[c0])
        e1 = int(indptr[c1])
        offsets = indptr[c0:c1] - e0
        counts_c = counts[c0:c1]
        col_c = col_idx[e0:e1]
        a_c = a_all[e0:e1]
        yc_c = yc_all[e0:e1]
        mass_c = mass_all[c0:c1]

        U_c = U_all[col_c]  # (E_c, m)
        u_bar = np.add.reduceat(a_c[:, None] * U_c, offsets, axis=0)  # (B, m)
        Z = U_c - np.repeat(u_bar, counts_c, axis=0)  # (E_c, m)

        aZ = a_c[:, None] * Z  # (E_c, m)
        b = np.add.reduceat(aZ * yc_c[:, None], offsets, axis=0)  # (B, m)
        G = _sym_outer_reduceat(aZ, Z, offsets)  # (B, m, m)
        if ridge > 0.0:
            G[:, diag_idx, diag_idx] += ridge

        # Решаем G [c | G^{-1}] = [b | I_m] одной батчевой факторизацией без inv.
        rhs = np.empty((batch, m, m + 1), dtype=float)
        rhs[:, :, 0] = b
        rhs[:, :, 1:] = eye_m
        try:
            sol = np.linalg.solve(G, rhs)
        except np.linalg.LinAlgError as error:
            raise RuntimeError(
                "singular local Gram matrix in LOO1-WLS evaluation; increase ridge"
            ) from error

        c = sol[:, :, 0]  # (B, m)
        G_inv = 0.5 * (sol[:, :, 1:] + sol[:, :, 1:].swapaxes(-1, -2))  # (B, m, m)
        active_coeffs[c0:c1] = c

        c_rep = np.repeat(c, counts_c, axis=0)  # (E_c, m)
        H = _sym_batched_matvec(G_inv, Z, counts_c)  # (E_c, m): G_j^{-1} z_{ji}
        q = np.einsum("ei,ei->e", Z, H)  # (E_c,)
        lev = a_c * (1.0 + q)  # полная диагональ hat-матрицы (включая intercept)

        r = yc_c - np.einsum("ei,ei->e", Z, c_rep)  # (E_c,)
        max_leverage = max(max_leverage, float(np.max(lev)))
        clipped_count += int(np.count_nonzero(lev >= leverage_cap))

        if smooth_cap_eps == 0.0:
            ell = np.minimum(lev, leverage_cap)
            d_ell = np.where(lev < leverage_cap, 1.0, 0.0)
        else:
            arg = (leverage_cap - lev) / smooth_cap_eps
            ell = leverage_cap - smooth_cap_eps * np.logaddexp(0.0, arg)
            # Overflow-free sigmoid: exp получает только неположительный аргумент.
            pos = arg >= 0.0
            d_ell = np.empty_like(arg)
            d_ell[pos] = 1.0 / (1.0 + np.exp(-arg[pos]))
            exp_neg = np.exp(arg[~pos])
            d_ell[~pos] = exp_neg / (1.0 + exp_neg)

        inv_one_minus_ell = 1.0 / (1.0 - ell)
        loss_edge = a_c * np.square(r) * inv_one_minus_ell
        loss_center = np.add.reduceat(loss_edge, offsets, axis=0)
        obj += float(np.dot(mass_c, loss_center))

        # Точный adjoint dF_j / dZ_{ji} по остаткам r, коэффициентам c и leverage ell.
        u = 2.0 * a_c * r * inv_one_minus_ell  # (E_c,)
        v = loss_edge * inv_one_minus_ell * d_ell  # (E_c,)
        p = np.add.reduceat(u[:, None] * Z, offsets, axis=0)  # (B, m)
        lam = (G_inv @ p[:, :, None]).squeeze(-1)  # (B, m)
        lam_rep = np.repeat(lam, counts_c, axis=0)  # (E_c, m)

        avH = (a_c * v)[:, None] * H  # (E_c, m)
        M = _sym_outer_reduceat(avH, H, offsets)  # (B, m, m)
        Mz = _sym_batched_matvec(M, Z, counts_c)  # (E_c, m)

        z_dot_lam = np.einsum("ei,ei->e", Z, lam_rep)
        dZ = (
            (-u + a_c * z_dot_lam)[:, None] * c_rep
            - (a_c * r)[:, None] * lam_rep
            + 2.0 * avH
            - 2.0 * a_c[:, None] * Mz
        )

        # Adjoint локального центрирования z_{ji} = U_i - sum_k a_{jk} U_k.
        sum_dZ = np.add.reduceat(dZ, offsets, axis=0)  # (B, m)
        m_rep = np.repeat(mass_c, counts_c)
        Gamma_loc = m_rep[:, None] * (
            dZ - a_c[:, None] * np.repeat(sum_dZ, counts_c, axis=0)
        )
        for dim in range(m):
            Gamma[:, dim] += np.bincount(col_c, weights=Gamma_loc[:, dim], minlength=n)

    # DGEMM #2: перенос градиента в исходное пространство R^{d x m}.
    grad_E = X.T @ Gamma  # (d, m)

    if (
        not np.isfinite(obj)
        or not np.all(np.isfinite(grad_E))
        or not np.all(np.isfinite(active_coeffs))
    ):
        raise RuntimeError("LOO1-WLS evaluation produced non-finite values")

    nnz = len(col_idx)
    info: dict[str, float | int] = {
        "max_leverage": max_leverage,
        "clipped_fraction": float(clipped_count / nnz) if nnz > 0 else 0.0,
        "active_centers": int(j_active),
        "nnz_weights": int(nnz),
    }
    return float(obj), grad_E, active_coeffs, mass_all.copy(), info


def evaluate_loo1_wls(
    Y: np.ndarray,
    X: np.ndarray,
    neighborhoods: LOO1Neighborhoods,
    *,
    ridge: float = 1e-4,
    leverage_cap: float = 0.85,
    smooth_cap_eps: float = 0.0,
    center_chunk: int = 512,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    """Вычислить LOO1-WLS функционал и точный евклидов градиент за 2 DGEMM.

    Входы: ``Y`` формы ``(d, m)``, ``X`` формы ``(n, d)``, ``neighborhoods``.
    Инвариант: функционал O(m)-инвариантен; рабочая память ограничена
    ``O(center_chunk * avg_k * m + n * m)``, без тензоров ``(J, n, d)`` и
    без явного ``np.linalg.inv``. Возвращает ``(obj, grad_E, active_coeffs,
    active_mass, info)``.
    """
    Y = utils._finite_real_array(Y, "Y")
    X = utils._finite_real_array(X, "X")
    if X.ndim != 2 or 0 in X.shape:
        raise ValueError("X must have non-empty shape (n, d)")
    if Y.ndim != 2 or 0 in Y.shape or Y.shape[0] != X.shape[1]:
        raise ValueError("Y must have non-empty shape (d, m) matching X")
    if X.shape[0] != neighborhoods.n:
        raise ValueError("X row count must match neighborhoods.n")

    ridge = _require_real("ridge", ridge)
    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative")
    leverage_cap = _require_real("leverage_cap", leverage_cap)
    if not 0.0 < leverage_cap < 1.0:
        raise ValueError("leverage_cap must lie strictly in (0, 1)")
    smooth_cap_eps = _require_real("smooth_cap_eps", smooth_cap_eps)
    if smooth_cap_eps < 0.0:
        raise ValueError("smooth_cap_eps must be nonnegative")
    center_chunk = _require_positive_int("center_chunk", center_chunk)

    return _evaluate_loo1_wls_validated(
        Y,
        X,
        neighborhoods,
        ridge=ridge,
        leverage_cap=leverage_cap,
        smooth_cap_eps=smooth_cap_eps,
        center_chunk=center_chunk,
    )


def subcritical_eigenvalues(
    coefficients: np.ndarray,
    mass: np.ndarray,
    *,
    exponent: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Вычислить канонический поворот ``vt`` и сублинейный спектр ``(s_a/s_1)^beta``.

    При ``0 < exponent < 1`` (по умолчанию ``0.5``) точка 0 отталкивает, что
    устраняет квадратичный коллапс слабых направлений. Порог ``1e-12`` страхует
    отношение сингулярных чисел от обнуления; вырожденные нулевые коэффициенты
    (``s_1 == 0`` или ``J < m``) отклоняются через ``RuntimeError``.
    """
    coefficients = utils._finite_real_array(coefficients, "coefficients")
    mass = utils._finite_real_array(mass, "mass")
    exponent = _require_real("exponent", exponent)
    if exponent < 0.0:
        raise ValueError("exponent must be nonnegative")
    if coefficients.ndim != 2 or 0 in coefficients.shape:
        raise ValueError("coefficients must have non-empty shape (J, m)")
    if mass.shape != (coefficients.shape[0],):
        raise ValueError("mass must have shape (J,)")
    if np.any(mass < 0.0) or float(np.sum(mass)) <= 0.0:
        raise ValueError("mass must be nonnegative with positive total mass")

    weighted = np.sqrt(mass)[:, None] * coefficients
    _, singular_values, vt = np.linalg.svd(weighted, full_matrices=False)
    m = coefficients.shape[1]
    if (
        len(singular_values) < m
        or not np.isfinite(singular_values[0])
        or singular_values[0] <= 0.0
    ):
        raise RuntimeError("local coefficients do not identify a multi-index")

    if exponent == 0.0:
        eigenvalues = np.ones(m, dtype=float)
    else:
        ratios = np.maximum(singular_values / singular_values[0], 1e-12)
        eigenvalues = np.power(ratios, exponent)
        eigenvalues[0] = 1.0
    return vt, eigenvalues


def pairwise_distance2_blocks(
    X: np.ndarray,
    centers: np.ndarray,
    block_size: int = 128,
) -> Iterator[tuple[int, np.ndarray]]:
    """Потоково вычислять блоки ``||x_i - c_j||^2`` при памяти ``O(B * n)``.

    Формула и прямое уточнение близких к нулю элементов точно повторяют
    ``pairwise_distance2`` из ``calculus.py`` без аллокации матрицы ``(J, n)``.
    """
    X, centers = utils._prepare_pairwise(X, centers)
    utils._check_batch_size(block_size)

    origin = X[0]
    X_shifted = X - origin
    C_shifted = centers - origin
    x_norm2 = np.square(X_shifted).sum(axis=1)
    c_norm2 = np.square(C_shifted).sum(axis=1)
    scale = max(1.0, float(np.max(c_norm2)), float(np.max(x_norm2)))
    recompute_threshold = 64.0 * np.finfo(float).eps * scale

    for start in range(0, len(centers), block_size):
        stop = min(len(centers), start + block_size)
        C_block = C_shifted[start:stop]
        d2_block = (
            c_norm2[start:stop, None] + x_norm2[None, :] - 2.0 * (C_block @ X_shifted.T)
        )
        np.maximum(d2_block, 0.0, out=d2_block)
        rows, columns = np.nonzero(d2_block <= recompute_threshold)
        for row, column in zip(rows, columns, strict=True):
            delta = C_block[row] - X_shifted[column]
            d2_block[row, column] = float(np.dot(delta, delta))
        yield start, d2_block


def _calculate_multi_weight_streaming(
    X: np.ndarray,
    centers: np.ndarray,
    basis: np.ndarray,
    eigenvalues: np.ndarray,
    h: float,
    alpha: float,
    kernel: Callable[[np.ndarray], np.ndarray],
    *,
    block_size: int = 128,
    tensor: str = "orthogonal",
) -> Iterator[tuple[int, np.ndarray]]:
    """Потоковые multi-index веса, повторяющие self-distance уточнение dense."""
    X, centers, basis, eigenvalues = _prepare_multi_localization(
        X, centers, basis, eigenvalues, h, alpha, kernel
    )
    utils._check_batch_size(block_size)
    utils.require(
        tensor in {"orthogonal", "full"}, "tensor must be 'orthogonal' or 'full'"
    )

    origin = X[0]
    projected_X = (X - origin) @ basis
    projected_centers = (centers - origin) @ basis
    inverse_h2 = 1.0 / (h * h)
    alpha2 = alpha * alpha

    for start, d2_block in pairwise_distance2_blocks(X, centers, block_size=block_size):
        stop = start + len(d2_block)
        orthogonal2, principal2 = _projected_multi_components(
            projected_X,
            projected_centers[start:stop],
            eigenvalues,
            d2_block,
        )
        residual2 = orthogonal2 if tensor == "orthogonal" else d2_block
        yield start, kernel((alpha2 * residual2 + principal2) * inverse_h2)


def search_bandwidth_streaming(
    X: np.ndarray,
    centers: np.ndarray,
    target: float,
    kernel: Callable[[np.ndarray], np.ndarray],
    *,
    lower: float,
    block_size: int = 128,
) -> float:
    """Потоковый аналог ``search_bandwidth`` без плотной матрицы ``(J, n)``."""
    X, centers = utils._prepare_pairwise(X, centers)
    utils._check_batch_size(block_size)
    target = _require_real("target", target)
    lower = _require_real("lower", lower)
    utils.require(target > 0.0, "target must be finite and positive")
    utils.require(lower > 0.0, "lower must be finite and positive")
    utils.require(callable(kernel), "kernel must be callable", TypeError)
    utils.require(
        not (kernel is epanechnikov and target > len(X)),
        "target mass cannot exceed n for the Epanechnikov kernel",
    )

    n_centers = len(centers)
    total_target = target * n_centers
    low = float(lower)

    origin = X[0]
    x_rad = float(np.sqrt(np.max(np.square(X - origin).sum(axis=1))))
    c_rad = float(np.sqrt(np.max(np.square(centers - origin).sum(axis=1))))
    upper_est = max(2.0 * low, x_rad + c_rad, 1.0)

    # Проход 1: одновременно проверяем lower, max(d2) и сужаем интервал.
    h_grid = np.geomspace(low, upper_est, 32)
    h_grid[0] = low
    masses = np.zeros(len(h_grid), dtype=float)
    max_d2 = 0.0
    for _, d2_block in pairwise_distance2_blocks(X, centers, block_size=block_size):
        block_max = float(np.max(d2_block))
        if block_max > max_d2:
            max_d2 = block_max
        for idx, h_val in enumerate(h_grid):
            masses[idx] += float(np.sum(kernel(d2_block / h_val**2)))

    if masses[0] / n_centers >= target:
        return low

    high: float | None = None
    if masses[-1] / n_centers >= target:
        first_ok = int(np.argmax(masses / n_centers >= target))
        low = float(h_grid[first_ok - 1])
        high = float(h_grid[first_ok])
    else:
        probe_low = float(h_grid[-1])
        for batch_start in range(1, 101, 16):
            batch_end = min(101, batch_start + 16)
            probes = np.array(
                [upper_est * (2.0**k) for k in range(batch_start, batch_end)],
                dtype=float,
            )
            probe_masses = np.zeros(len(probes), dtype=float)
            for _, d2_block in pairwise_distance2_blocks(
                X, centers, block_size=block_size
            ):
                for idx, h_val in enumerate(probes):
                    probe_masses[idx] += float(np.sum(kernel(d2_block / h_val**2)))
            if probe_masses[-1] / n_centers >= target:
                first_ok = int(np.argmax(probe_masses / n_centers >= target))
                low = probe_low if first_ok == 0 else float(probes[first_ok - 1])
                high = float(probes[first_ok])
                break
            probe_low = float(probes[-1])

    if high is None:
        raise RuntimeError("could not bracket a feasible bandwidth")

    eps = np.finfo(float).eps
    if kernel is epanechnikov:
        collected: list[np.ndarray] = []
        collected_count = 0
        overflowed = False
        h_bracket_high = high
        high2 = h_bracket_high * h_bracket_high
        for _, d2_block in pairwise_distance2_blocks(X, centers, block_size=block_size):
            active_block = d2_block[d2_block < high2]
            collected_count += active_block.size
            if collected_count > _STREAM_ACTIVE_EDGE_CAP:
                collected.clear()
                overflowed = True
                break
            if active_block.size:
                collected.append(active_block)

        if not overflowed:
            active = (
                np.concatenate(collected) if collected else np.empty(0, dtype=float)
            )
            # Точная 60-шаговая бисекция search_bandwidth на собранном support
            # (все пары d2 < h^2 при h <= h_bracket_high уже содержатся в active).
            low_b = float(lower)
            high_b = max(2.0 * low_b, math.sqrt(max_d2), 1.0)
            for _ in range(100):
                if (
                    high_b >= h_bracket_high
                    or float(epanechnikov(active / high_b**2).sum()) / n_centers
                    >= target
                ):
                    break
                high_b *= 2.0
            for _ in range(60):
                middle = (low_b + high_b) / 2.0
                if middle >= h_bracket_high:
                    high_b = middle
                    continue
                argument = active / middle**2
                if float(epanechnikov(argument).sum()) / n_centers >= target:
                    high_b = middle
                    support = argument < 1.0
                    if np.count_nonzero(support) <= active.size // 2:
                        active = active[support]
                else:
                    low_b = middle
            return float(high_b)

    # Общий потоковый 16-ричный поиск (4 бита на проход) при памяти O(B * n).
    for _ in range(16):
        if high - low <= 2.0 * eps * max(1.0, high):
            break
        interior = np.linspace(low, high, 17)[1:-1]
        interior_masses = np.zeros(len(interior), dtype=float)
        for _, d2_block in pairwise_distance2_blocks(X, centers, block_size=block_size):
            for idx, h_val in enumerate(interior):
                interior_masses[idx] += float(
                    np.sum(kernel(d2_block / (h_val * h_val)))
                )
        feasible = interior_masses >= total_target
        if np.any(feasible):
            first_ok = int(np.argmax(feasible))
            if first_ok > 0:
                low = float(interior[first_ok - 1])
            high = float(interior[first_ok])
        else:
            low = float(interior[-1])
    return float(high)


def _compact_alpha_bracket(
    blocks: list[tuple[np.ndarray, np.ndarray]],
    inverse_h2: float,
    total_target: float,
    low: float,
    high: float,
    tolerance: float,
) -> float:
    """Точная диадическая бисекция alpha на [low, high] по собранному support."""
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        alpha2 = middle * middle
        mass = 0.0
        for residual, principal in blocks:
            mass += float(
                epanechnikov((alpha2 * residual + principal) * inverse_h2).sum()
            )
        if mass >= total_target:
            low = middle
            for idx, (residual, principal) in enumerate(blocks):
                support = (alpha2 * residual + principal) * inverse_h2 < 1.0
                blocks[idx] = (residual[support], principal[support])
        else:
            high = middle
    return float(low)


def calculate_alpha_k_streaming(
    X: np.ndarray,
    centers: np.ndarray,
    basis: np.ndarray,
    eigenvalues: np.ndarray,
    h_k: float,
    N_loc: int,
    kernel: Callable[[np.ndarray], np.ndarray],
    *,
    tensor: str = "orthogonal",
    block_size: int = 128,
) -> float | None:
    """Потоковый аналог ``calculate_alpha_k`` на точной диадической сетке."""
    X, centers, basis, eigenvalues = _prepare_multi_localization(
        X, centers, basis, eigenvalues, h_k, 1.0, kernel
    )
    utils.require(np.isfinite(N_loc) and N_loc > 0, "N_loc must be finite and positive")
    utils.require(
        tensor in {"orthogonal", "full"}, "tensor must be 'orthogonal' or 'full'"
    )
    utils._check_batch_size(block_size)

    n_centers = len(centers)
    total_target = float(N_loc) * n_centers
    inverse_h2 = 1.0 / (h_k * h_k)
    tolerance = float(np.sqrt(np.finfo(float).eps))

    origin = X[0]
    projected_X = (X - origin) @ basis
    projected_centers = (centers - origin) @ basis

    # Проход 1: 16 равных диадических подинтервалов [0, 1] (4 шага бисекции).
    alpha_grid = np.linspace(0.0, 1.0, 17)
    alpha2_grid = np.square(alpha_grid)
    masses = np.zeros(len(alpha_grid), dtype=float)
    for start, d2_block in pairwise_distance2_blocks(X, centers, block_size=block_size):
        stop = start + len(d2_block)
        orthogonal2, principal2 = _projected_multi_components(
            projected_X,
            projected_centers[start:stop],
            eigenvalues,
            d2_block,
        )
        residual2 = orthogonal2 if tensor == "orthogonal" else d2_block
        for idx, alpha2 in enumerate(alpha2_grid):
            masses[idx] += float(
                np.sum(kernel((alpha2 * residual2 + principal2) * inverse_h2))
            )

    if masses[-1] >= total_target:
        return 1.0
    if masses[0] < total_target:
        return None

    last_ok = int(np.flatnonzero(masses >= total_target)[-1])
    low = float(alpha_grid[last_ok])
    high = float(alpha_grid[last_ok + 1])

    while high - low > tolerance:
        if kernel is epanechnikov:
            collected: list[tuple[np.ndarray, np.ndarray]] = []
            collected_count = 0
            overflowed = False
            low2 = low * low
            for start, d2_block in pairwise_distance2_blocks(
                X, centers, block_size=block_size
            ):
                stop = start + len(d2_block)
                orthogonal2, principal2 = _projected_multi_components(
                    projected_X,
                    projected_centers[start:stop],
                    eigenvalues,
                    d2_block,
                )
                residual2 = orthogonal2 if tensor == "orthogonal" else d2_block
                support = (low2 * residual2 + principal2) * inverse_h2 < 1.0
                count = int(np.count_nonzero(support))
                collected_count += count
                if collected_count > _STREAM_ACTIVE_EDGE_CAP:
                    collected.clear()
                    overflowed = True
                    break
                if count > 0:
                    collected.append((residual2[support], principal2[support]))
            if not overflowed:
                return _compact_alpha_bracket(
                    collected, inverse_h2, total_target, low, high, tolerance
                )

        # Диадическое разбиение (степень двойки) побитово повторяет calculate_alpha_k.
        remaining_bits = max(1, math.ceil(math.log2((high - low) / tolerance)))
        step_bits = min(4, remaining_bits)
        n_sub = 1 << step_bits
        interior = np.linspace(low, high, n_sub + 1)[1:-1]
        interior_a2 = np.square(interior)
        interior_masses = np.zeros(len(interior), dtype=float)
        for start, d2_block in pairwise_distance2_blocks(
            X, centers, block_size=block_size
        ):
            stop = start + len(d2_block)
            orthogonal2, principal2 = _projected_multi_components(
                projected_X,
                projected_centers[start:stop],
                eigenvalues,
                d2_block,
            )
            residual2 = orthogonal2 if tensor == "orthogonal" else d2_block
            for idx, alpha2 in enumerate(interior_a2):
                interior_masses[idx] += float(
                    np.sum(kernel((alpha2 * residual2 + principal2) * inverse_h2))
                )
        feasible_idx = np.flatnonzero(interior_masses >= total_target)
        if len(feasible_idx) == 0:
            high = float(interior[0])
        elif feasible_idx[-1] == len(interior) - 1:
            low = float(interior[-1])
        else:
            last_ok = int(feasible_idx[-1])
            low = float(interior[last_ok])
            high = float(interior[last_ok + 1])

    return float(low)


def initialize_basis_local_streaming(
    X: np.ndarray,
    Y: np.ndarray,
    centers: np.ndarray,
    N_lin: int,
    kernel: Callable[[np.ndarray], np.ndarray],
    local_ridge: float,
    index_dim: int,
    *,
    mass_weighted: bool = False,
    ridge_selection: str = "fixed",
    block_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Потоковый аналог ``initialize_basis_local_with_spectrum`` без ``(J, n)``."""
    X, Y = utils._prepare_xy(X, Y, require_overdetermined=False)
    X, centers = utils._prepare_pairwise(X, centers)
    utils._check_batch_size(block_size)
    utils.require(
        np.isfinite(local_ridge) and local_ridge >= 0,
        "local_ridge must be finite and nonnegative",
    )
    utils.require(
        not isinstance(index_dim, bool) and isinstance(index_dim, (int, np.integer)),
        "index_dim must be an integer",
        TypeError,
    )
    utils.require(
        1 <= index_dim <= X.shape[1],
        "index_dim must lie between 1 and d",
    )
    utils.require(
        isinstance(mass_weighted, bool), "mass_weighted must be boolean", TypeError
    )
    utils.require(
        ridge_selection in {"fixed", "loo"},
        "ridge_selection must be 'fixed' or 'loo'",
    )

    h_lin = search_bandwidth_streaming(
        X,
        centers,
        N_lin,
        kernel,
        lower=float(np.finfo(float).eps),
        block_size=block_size,
    )
    n, d = X.shape
    local_mass = np.empty(len(centers), dtype=float)
    gradients = np.empty((len(centers), d), dtype=float)
    rcond = float(np.finfo(float).eps) * max(n + d, d + 1)
    ridge_rows = None

    for start, d2_block in pairwise_distance2_blocks(X, centers, block_size=block_size):
        for local_idx in range(len(d2_block)):
            j = start + local_idx
            center = centers[j]
            weights = np.asarray(kernel(d2_block[local_idx : local_idx + 1] / h_lin**2))
            utils.require(
                weights.shape == (1, n),
                "initialization kernel must preserve the input shape",
            )
            weights = weights[0]
            utils.require(
                np.all(np.isfinite(weights)) and not np.any(weights < 0),
                "initialization weights must be finite and nonnegative",
            )
            local_mass[j] = float(weights.sum())
            utils.require(
                np.isfinite(local_mass[j]) and local_mass[j] > 0.0,
                "local initialization contains an empty neighborhood",
                RuntimeError,
            )
            support = weights != 0
            local_X = X[support]
            local_Y = Y[support]
            root_weight = np.sqrt(weights[support])
            gradient = None
            if len(local_X) < d or ridge_selection == "loo":
                gradient = _centered_ridge_gradient(
                    local_X - center,
                    local_Y,
                    root_weight,
                    local_ridge,
                    rcond,
                    ridge_selection=ridge_selection,
                )
            if gradient is not None:
                gradients[j] = gradient
                continue
            if ridge_rows is None:
                ridge_rows = np.zeros((d, d + 1), dtype=float)
                ridge_rows[:, 1:] = np.sqrt(local_ridge) * np.eye(d)
            design = np.column_stack((np.ones(len(local_X)), local_X - center))
            augmented_design = np.vstack((design * root_weight[:, None], ridge_rows))
            augmented_Y = np.concatenate((local_Y * root_weight, np.zeros(d)))
            gradients[j] = np.linalg.lstsq(
                augmented_design,
                augmented_Y,
                rcond=rcond,
            )[0][1:]

    return _principal_gradient_basis_with_spectrum(
        gradients,
        index_dim,
        mass=local_mass if mass_weighted else None,
    )
