"""Rank-constrained multi-index steps from ``SVD.tex`` and ``SVD_corr.tex``."""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve, solve_triangular
from scipy.sparse.linalg import LinearOperator, lsmr

from ._multi_operator import forward
from .LSMR import HPAOResult, _local_refit, _loss, _normalize_index, _validate_inputs

_QR_CACHE_CONDITION_LIMIT = 1e6
_DIRECT_WORKSPACE_BYTES = 16 * 1024**2
# One-thread probes kept a clear direct-solve margin through d=128, while the
# user-shaped d=400 case was at or beyond the crossover.
_DIRECT_DIMENSION_LIMIT = 128


class _FlatU:
    """Хранить один contiguous view и считать эквивалентные U-проходы."""

    __slots__ = ("copied", "flat", "passes", "shape", "tensor")

    def __init__(self, U: np.ndarray) -> None:
        self.shape = U.shape
        self.tensor = np.ascontiguousarray(U)
        self.copied = not np.shares_memory(self.tensor, U)
        self.flat = self.tensor.reshape(-1, U.shape[2])
        self.passes = 0

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        self.passes += 1
        return self.flat @ vector

    def rmatvec(self, vector: np.ndarray) -> np.ndarray:
        self.passes += 1
        return self.flat.T @ vector

    def matmat(self, matrix: np.ndarray) -> np.ndarray:
        self.passes += matrix.shape[1]
        return self.flat @ matrix

    def batched_rmatvec(self, data: np.ndarray) -> np.ndarray:
        self.passes += 1
        return (self.tensor.swapaxes(1, 2) @ data[..., None]).squeeze(-1)

    def normal_equations(
        self, rows: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Form A.T A and A.T target, counting d+1 GEMV-equivalent passes."""
        d = self.flat.shape[1]
        self.passes += d + 1
        normal = np.zeros((d, d))
        rhs = np.zeros(d)
        chunk_rows = max(1, _DIRECT_WORKSPACE_BYTES // (self.flat.itemsize * d))
        for start in range(0, len(rows), chunk_rows):
            stop = min(start + chunk_rows, len(rows))
            weighted = rows[start:stop, None] * self.flat[start:stop]
            normal += weighted.T @ weighted
            rhs += weighted.T @ target[start:stop]
        return normal, rhs


def solve_fixed_coefficients(
    P: np.ndarray,
    U: np.ndarray,
    I: np.ndarray,
    g: np.ndarray,
    mass: np.ndarray,
    *,
    rank: int,
    lambda_penalty: float,
    inner_tol: float = 1e-6,
    inner_maxiter: int = 20,
    rank_tol: float = 1e-8,
    lsmr_maxiter: int | None = None,
    warm_start: bool = True,
    adaptive_krylov: bool = False,
    direct_max_dimension: int | None = _DIRECT_DIMENSION_LIMIT,
    low_rank_target: str = "matrix",
) -> tuple[np.ndarray, dict[str, object]]:
    """Greedy fixed-g solve with rank bound on B or its correction B-P.

    Returns the raw B, not an orthonormal EDR basis. ``matrix`` preserves
    the historical rank-deficient B; ``correction`` returns P+Delta.
    Positive-ridge problems up to ``direct_max_dimension`` use a certified
    Cholesky solve; the remaining v subproblems use shifted-damp LSMR.
    """
    P, U, I, mass = _validate_inputs(P, U, I, mass)
    B, diagnostics, _, _ = _solve_fixed_coefficients_validated(
        P,
        U,
        I,
        g,
        mass,
        rank=rank,
        lambda_penalty=lambda_penalty,
        inner_tol=inner_tol,
        inner_maxiter=inner_maxiter,
        rank_tol=rank_tol,
        lsmr_maxiter=lsmr_maxiter,
        warm_start=warm_start,
        adaptive_krylov=adaptive_krylov,
        direct_max_dimension=direct_max_dimension,
        low_rank_target=low_rank_target,
    )
    return B, diagnostics


def _solve_fixed_coefficients_validated(
    P: np.ndarray,
    U: np.ndarray,
    I: np.ndarray,
    g: np.ndarray,
    mass: np.ndarray,
    *,
    rank: int,
    lambda_penalty: float,
    inner_tol: float,
    inner_maxiter: int,
    rank_tol: float,
    lsmr_maxiter: int | None,
    warm_start: bool,
    adaptive_krylov: bool,
    direct_max_dimension: int | None,
    low_rank_target: str,
) -> tuple[np.ndarray, dict[str, object], np.ndarray, np.ndarray]:
    """Внутренний fixed-g solve для уже проверенных P/U/I/mass."""
    if P.ndim != 2:
        raise ValueError("P must have shape (m, d)")
    m, d = P.shape
    if not np.allclose(P @ P.T, np.eye(m), rtol=1e-8, atol=1e-10):
        raise ValueError("P must have orthonormal rows")
    raw_g = np.asarray(g)
    if np.iscomplexobj(raw_g) or not np.issubdtype(raw_g.dtype, np.number):
        raise TypeError("g must be a real numeric array")
    g = np.asarray(g, dtype=float)
    if g.shape != (len(U), m) or not np.all(np.isfinite(g)):
        raise ValueError("g must have finite shape (J, m)")
    for name, value in (("rank", rank), ("inner_maxiter", inner_maxiter)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
    if not isinstance(low_rank_target, str) or low_rank_target not in {
        "matrix",
        "correction",
    }:
        raise ValueError("low_rank_target must be 'matrix' or 'correction'")
    min_rank = 0 if low_rank_target == "correction" else 1
    if not min_rank <= rank < m or inner_maxiter < 1:
        raise ValueError(f"require {min_rank} <= rank < m and inner_maxiter >= 1")
    if lsmr_maxiter is not None and (
        isinstance(lsmr_maxiter, bool)
        or not isinstance(lsmr_maxiter, (int, np.integer))
        or lsmr_maxiter < 1
    ):
        raise ValueError("lsmr_maxiter must be a positive integer or None")
    if direct_max_dimension is not None and (
        isinstance(direct_max_dimension, bool)
        or not isinstance(direct_max_dimension, (int, np.integer))
        or direct_max_dimension < 1
    ):
        raise ValueError("direct_max_dimension must be a positive integer or None")
    if not np.isfinite(lambda_penalty) or lambda_penalty < 0:
        raise ValueError("lambda_penalty must be finite and nonnegative")
    if not np.isfinite(inner_tol) or inner_tol <= 0:
        raise ValueError("inner_tol must be finite and positive")
    if not np.isfinite(rank_tol) or rank_tol < 0:
        raise ValueError("rank_tol must be finite and nonnegative")
    for name, value in (
        ("warm_start", warm_start),
        ("adaptive_krylov", adaptive_krylov),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a boolean")

    A = np.empty((m, 0))
    V = np.empty((d, 0))
    singular = np.empty(0)
    W = np.empty((*U.shape[:2], 0))
    flat_U = _FlatU(U)
    # A,V,s describe B in matrix mode and Delta=B-P in correction mode.
    base_residual = I - forward(U, P, g) if low_rank_target == "correction" else I
    if low_rank_target == "correction":
        flat_U.passes += 1  # initial U_j P.T g_j pass
    prior = np.zeros_like(P) if low_rank_target == "correction" else P
    root_mass = np.sqrt(mass)
    row_root_mass = np.repeat(root_mass, U.shape[1])
    weighted_base = row_root_mass * base_residual.ravel()
    identity_m = np.eye(m)
    prior_norm2 = float(np.sum(prior**2))
    objective = float(
        np.sum(mass[:, None] * base_residual**2) + lambda_penalty * prior_norm2
    )
    if not np.isfinite(objective):
        raise ValueError("initial rank objective must be finite")
    history = [objective]
    gains: list[float] = []
    certificates: list[float] = []
    lsmr_estimates: list[float] = []
    lsmr_iterations: list[int] = []
    lsmr_stops: list[int] = []
    lsmr_tolerances: list[float] = []
    lsmr_refinements = 0
    inner_iterations: list[int] = []
    inner_converged: list[bool] = []
    factor_cache_fallbacks = 0
    factor_condition_max = 1.0
    direct_solves = 0
    direct_fallbacks = 0
    use_direct = (
        lambda_penalty > 0
        and direct_max_dimension is not None
        and d <= direct_max_dimension
    )
    stop_reason = "rank_limit"

    for k in range(1, rank + 1):
        residual = base_residual - np.einsum(
            "jpk,jk->jp", W, (g @ A) * singular, optimize=True
        )
        target = row_root_mass * residual.ravel()
        pulled = flat_U.batched_rmatvec(residual)
        proximity = prior - (A * singular) @ V.T
        Q = g.T @ (mass[:, None] * pulled) + lambda_penalty * proximity
        initial = Q - (Q @ V) @ V.T
        gram = initial @ initial.T
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        leading = float(eigenvalues[-1])
        if leading <= 0:
            stop_reason = "zero_gradient"
            break
        v = initial.T @ eigenvectors[:, -1]
        v /= math.sqrt(leading)
        a = np.zeros(m)
        v_raw: np.ndarray | None = None
        warm_v: np.ndarray | None = None
        certificate_args: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        converged = False
        iterations = 0
        last_change = math.inf
        for _ in range(inner_maxiter):
            iterations += 1
            Uv = flat_U.matvec(v).reshape(U.shape[:2])
            weights = mass * np.einsum("jp,jp->j", Uv, Uv)
            G = g.T @ (weights[:, None] * g) + lambda_penalty * identity_m
            qv = Q @ v
            candidate_a = _solve_small(G, qv, lambda_penalty)
            norm_a = np.linalg.norm(candidate_a)
            if norm_a == 0 or not np.isfinite(norm_a):
                break
            a = candidate_a / norm_a
            previous_v = v
            krylov_tol = (
                float(np.clip(0.1 * last_change, 1e-10, 1e-4))
                if adaptive_krylov
                else min(1e-10, inner_tol * 0.1)
            )
            if use_direct:
                try:
                    v, v_raw, certificate_args = _v_step_direct(
                        flat_U,
                        target,
                        proximity,
                        g @ a,
                        a,
                        root_mass,
                        lambda_penalty,
                    )
                    direct_solves += 1
                except LinAlgError:
                    direct_fallbacks += 1
                    use_direct = False
            if not use_direct:
                v, v_raw, estimate, iterations_v, stop_v, certificate_args = _v_step(
                    flat_U,
                    target,
                    proximity,
                    g @ a,
                    a,
                    root_mass,
                    lambda_penalty,
                    inner_tol,
                    lsmr_maxiter,
                    x0=warm_v if warm_start else None,
                    krylov_tol=krylov_tol,
                )
                lsmr_estimates.append(estimate)
                lsmr_iterations.append(iterations_v)
                lsmr_stops.append(stop_v)
                lsmr_tolerances.append(krylov_tol)
            warm_v = v_raw
            alignment_change = max(0.0, 1.0 - abs(float(v @ previous_v)))
            last_change = math.sqrt(2.0 * alignment_change)
            if alignment_change < inner_tol:
                converged = True
                break
        if not np.any(a):
            stop_reason = "zero_gradient"
            break
        if v_raw is None or certificate_args is None:
            raise RuntimeError("truncated-SVD v solve did not produce a direction")
        certificate = _v_certificate(
            flat_U,
            v_raw,
            *certificate_args,
            ridge=lambda_penalty,
        )
        certificate_limit = max(1e-7, 10 * inner_tol)
        if certificate > certificate_limit and (
            adaptive_krylov or warm_start or use_direct
        ):
            direct_failed = use_direct
            strict_tol = min(1e-10, inner_tol * 0.1)
            v, v_raw, estimate, iterations_v, stop_v, certificate_args = _v_step(
                flat_U,
                target,
                proximity,
                g @ a,
                a,
                root_mass,
                lambda_penalty,
                inner_tol,
                lsmr_maxiter,
                x0=v_raw,
                krylov_tol=strict_tol,
            )
            lsmr_refinements += 1
            if direct_failed:
                direct_fallbacks += 1
                use_direct = False
            lsmr_estimates.append(estimate)
            lsmr_iterations.append(iterations_v)
            lsmr_stops.append(stop_v)
            lsmr_tolerances.append(strict_tol)
            certificate = _v_certificate(
                flat_U,
                v_raw,
                *certificate_args,
                ridge=lambda_penalty,
            )
        certificates.append(certificate)
        if certificate > certificate_limit:
            raise RuntimeError("truncated-SVD v solve failed its normal-residual check")
        # Finish with the exact a minimizer for the last v, even at the cap.
        Uv = flat_U.matvec(v).reshape(U.shape[:2])
        weights = mass * np.einsum("jp,jp->j", Uv, Uv)
        G = g.T @ (weights[:, None] * g) + lambda_penalty * identity_m
        candidate_a = _solve_small(G, Q @ v, lambda_penalty)
        norm_a = np.linalg.norm(candidate_a)
        if norm_a == 0 or not np.isfinite(norm_a):
            stop_reason = "zero_gradient"
            break
        a = candidate_a / norm_a
        inner_iterations.append(iterations)
        inner_converged.append(converged)
        alpha = g @ a
        denominator = float(np.sum(mass * alpha**2 * np.sum(Uv**2, axis=1)))
        denominator += lambda_penalty
        numerator = float(a @ Q @ v)
        if denominator <= 0 or not np.isfinite(denominator):
            stop_reason = "zero_curvature"
            break
        gain = numerator**2 / denominator
        if gain / max(1.0, objective) < rank_tol:
            stop_reason = "rank_tolerance"
            break

        # Compact QR/SVD of the k-column factors; no d-by-d matrix is formed.
        left = np.column_stack((A * singular, (numerator / denominator) * a))
        right = np.column_stack((V, v))
        ql, rl = np.linalg.qr(left, mode="reduced")
        qr, rr = np.linalg.qr(right, mode="reduced")
        small_a, values, small_vt = np.linalg.svd(rl @ rr.T, full_matrices=False)
        cutoff = np.finfo(float).eps * max(m, d) * values[0]
        if values[k - 1] <= cutoff:
            stop_reason = "no_rank_growth"
            break
        candidate_A = ql @ small_a[:, :k]
        candidate_V = qr @ small_vt[:k].T
        rr_condition = float(np.linalg.cond(rr))
        factor_condition_max = max(factor_condition_max, rr_condition)
        if rr_condition > _QR_CACHE_CONDITION_LIMIT:
            projected = flat_U.matmat(candidate_V).reshape(*U.shape[:2], k)
            factor_cache_fallbacks += 1
        else:
            transform = solve_triangular(rr, small_vt[:k].T)
            Wcat = np.concatenate((W, Uv[..., None]), axis=2)
            projected = (Wcat.reshape(-1, k) @ transform).reshape(*U.shape[:2], k)
        design = projected * (g @ candidate_A)[:, None, :]
        flat_design = design.reshape(-1, k)
        weighted_design = row_root_mass[:, None] * flat_design
        M = weighted_design.T @ weighted_design
        M += lambda_penalty * np.eye(k)
        b = weighted_design.T @ weighted_base
        if low_rank_target == "matrix":
            b += lambda_penalty * np.einsum(
                "mk,mk->k", candidate_A, P @ candidate_V, optimize=True
            )
        candidate_s = _solve_small(M, b, lambda_penalty)
        candidate_objective = _objective_from_factors(
            candidate_A,
            candidate_s,
            candidate_V,
            projected,
            prior,
            base_residual,
            g,
            mass,
            lambda_penalty,
            prior_norm2,
        )
        if not np.isfinite(candidate_objective):
            raise RuntimeError("conditional singular-value solve is non-finite")
        if candidate_objective > objective + 64 * np.finfo(float).eps * max(
            1, objective
        ):
            raise RuntimeError("conditional singular-value solve increased objective")
        A, V, singular, W, objective = (
            candidate_A,
            candidate_V,
            candidate_s,
            projected,
            candidate_objective,
        )
        history.append(objective)
        gains.append(gain)
        if _factor_rank(singular, m, d) < k:
            stop_reason = "no_rank_growth"
            break

    diagnostics = {
        "low_rank_target": low_rank_target,
        "effective_rank": _factor_rank(singular, m, d),
        "rank_stop_reason": stop_reason,
        "rank_objective_history": tuple(history),
        "v_normal_residual_max": max(certificates, default=0.0),
        "lsmr_normal_residual_estimate_max": max(lsmr_estimates, default=0.0),
        "lsmr_iterations": tuple(lsmr_iterations),
        "lsmr_iterations_total": sum(lsmr_iterations),
        "lsmr_stops": tuple(lsmr_stops),
        "lsmr_tolerances": tuple(lsmr_tolerances),
        "lsmr_refinements": lsmr_refinements,
        "inner_iterations": tuple(inner_iterations),
        "inner_converged": tuple(inner_converged),
        "rank_scales": tuple(float(x) for x in singular),
        "rank_gain_history": tuple(gains),
        "u_vector_passes": flat_U.passes,
        "u_flat_copy": flat_U.copied,
        "factor_cache_fallbacks": factor_cache_fallbacks,
        "factor_condition_max": factor_condition_max,
        "direct_solves": direct_solves,
        "direct_fallbacks": direct_fallbacks,
        "direct_max_dimension": direct_max_dimension,
        "warm_start": bool(warm_start),
        "adaptive_krylov": bool(adaptive_krylov),
    }
    low_rank = (A * singular) @ V.T
    if low_rank_target == "correction":
        correction_norm = float(np.linalg.norm(singular))
        diagnostics.update(
            correction_rank=diagnostics["effective_rank"],
            correction_singular_values=tuple(
                float(x) for x in sorted(np.abs(singular), reverse=True)
            ),
            correction_frobenius_norm=correction_norm,
            relative_correction_norm=correction_norm / math.sqrt(m),
        )
        return P + low_rank, diagnostics, V, singular
    return low_rank, diagnostics, V, singular


def _v_step_direct(
    flat_U: _FlatU,
    target: np.ndarray,
    proximity: np.ndarray,
    alpha: np.ndarray,
    a: np.ndarray,
    root_mass: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Solve the positive-ridge v subproblem through its small normal system."""
    if ridge <= 0:
        raise ValueError("direct v solve requires positive ridge")
    d = flat_U.shape[2]
    rows = np.repeat(root_mass * alpha, flat_U.shape[1])
    z = proximity.T @ a
    normal, rhs = flat_U.normal_equations(rows, target)
    normal.flat[:: d + 1] += ridge
    rhs += ridge * z
    solution = cho_solve(
        cho_factor(normal, lower=True, check_finite=False),
        rhs,
        check_finite=False,
    )
    if not np.all(np.isfinite(solution)):
        raise LinAlgError("direct truncated-SVD v solve returned non-finite values")
    norm = np.linalg.norm(solution)
    if norm == 0:
        raise LinAlgError("direct truncated-SVD v solve returned a zero direction")
    return solution / norm, solution, (rows, target, z)


def _v_step(
    flat_U,
    target,
    proximity,
    alpha,
    a,
    root_mass,
    ridge,
    tol,
    maxiter,
    x0=None,
    krylov_tol=None,
):
    """Решить shifted ridge v-подзадачу одним плоским LSMR-оператором."""
    d = flat_U.shape[2]
    rows = np.repeat(root_mass * alpha, flat_U.shape[1])
    operator = _v_operator(flat_U, rows)
    z = proximity.T @ a if ridge > 0 else np.zeros(d)
    if krylov_tol is None:
        krylov_tol = min(1e-10, tol * 0.1)
    if x0 is not None and ridge > 0:
        # scipy.lsmr interprets x0 as the center of its damp term. Solve an
        # augmented correction so a warm start cannot change the ridge target.
        root_ridge = math.sqrt(ridge)
        augmented = _v_augmented_operator(flat_U, rows, root_ridge)
        correction_target = np.concatenate(
            (target - operator @ x0, root_ridge * (z - x0))
        )
        result = lsmr(
            augmented,
            correction_target,
            atol=krylov_tol,
            btol=krylov_tol,
            maxiter=maxiter or max(50, 5 * d),
        )
        solution = x0 + result[0]
    else:
        shifted_target = target - operator @ z
        result = lsmr(
            operator,
            shifted_target,
            damp=math.sqrt(ridge),
            atol=krylov_tol,
            btol=krylov_tol,
            maxiter=maxiter or max(50, 5 * d),
        )
        solution = z + result[0]
    if not np.all(np.isfinite(solution)):
        raise RuntimeError("truncated-SVD v solve returned non-finite values")
    norm = np.linalg.norm(solution)
    if norm == 0:
        raise RuntimeError("truncated-SVD v solve returned a zero direction")
    return (
        solution / norm,
        solution,
        float(result[4]),
        int(result[2]),
        int(result[1]),
        (rows, target, z),
    )


def _v_operator(flat_U: _FlatU, rows: np.ndarray) -> LinearOperator:
    """Плоский design и точный adjoint для фиксированного левого фактора."""

    def matvec(x):
        return rows * flat_U.matvec(x)

    def rmatvec(y):
        return flat_U.rmatvec(rows * y)

    return LinearOperator(
        flat_U.flat.shape, matvec=matvec, rmatvec=rmatvec, dtype=float
    )


def _v_augmented_operator(
    flat_U: _FlatU, rows: np.ndarray, root_ridge: float
) -> LinearOperator:
    """Augmented correction operator for an objective-preserving warm start."""
    operator = _v_operator(flat_U, rows)
    n, d = operator.shape

    def matvec(x):
        return np.concatenate((operator @ x, root_ridge * x))

    def rmatvec(y):
        return operator.rmatvec(y[:n]) + root_ridge * y[n:]

    return LinearOperator((n + d, d), matvec=matvec, rmatvec=rmatvec, dtype=float)


def _v_certificate(flat_U, solution, rows, target, z, *, ridge):
    """Явно проверить gradient исходной v-подзадачи после inner-loop."""
    operator = _v_operator(flat_U, rows)
    residual = operator @ solution - target
    gradient = operator.rmatvec(residual) + ridge * (solution - z)
    normal_rhs = operator.rmatvec(target) + ridge * z
    return float(np.linalg.norm(gradient) / max(1.0, np.linalg.norm(normal_rhs)))


def _solve_small(matrix: np.ndarray, rhs: np.ndarray, ridge: float) -> np.ndarray:
    if ridge == 0:
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    try:
        return cho_solve(cho_factor(matrix, lower=True, check_finite=False), rhs)
    except LinAlgError:
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def _objective(B, P, U, I, g, mass, ridge):
    residual = I - forward(U, B, g)
    return float(np.sum(mass[:, None] * residual**2) + ridge * np.sum((B - P) ** 2))


def _objective_from_factors(
    A, singular, V, W, prior, base_residual, g, mass, ridge, prior_norm2
):
    """Objective for base+low_rank, with ridge target ``prior``."""
    residual = base_residual - np.einsum(
        "jpk,jk->jp", W, (g @ A) * singular, optimize=True
    )
    penalty = prior_norm2 + singular @ singular
    if prior_norm2:
        penalty -= 2 * np.einsum("mk,mk,k->", A, prior @ V, singular, optimize=True)
    return float(np.sum(mass[:, None] * residual**2) + ridge * penalty)


def _factor_rank(singular: np.ndarray, m: int, d: int) -> int:
    if not len(singular):
        return 0
    values = np.abs(singular)
    threshold = np.finfo(float).eps * max(m, d) * np.max(values)
    return int(np.count_nonzero(values > threshold))


def _complete_basis(V: np.ndarray, singular: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Keep recovered directions, then complete from the prior m-space."""
    rank = _factor_rank(singular, *P.shape)
    if rank == 0:
        return P.copy()
    order = np.argsort(np.abs(singular))[::-1][:rank]
    active = V[:, order].T
    residual = P - (P @ active.T) @ active
    _, values, complement = np.linalg.svd(residual, full_matrices=False)
    if values[P.shape[0] - rank - 1] <= np.finfo(float).eps * max(P.shape) * values[0]:
        raise RuntimeError("prior basis cannot complete the rank-constrained step")
    return np.vstack((active, complement[: P.shape[0] - rank]))


def solve(
    index_init: np.ndarray,
    U: np.ndarray,
    I: np.ndarray,
    *,
    mass: np.ndarray | None = None,
    lambda_prox: float = 1.0,
    rank: int,
    inner_tol: float = 1e-6,
    inner_maxiter: int = 20,
    rank_tol: float = 1e-8,
    lsmr_maxiter: int | None = None,
    warm_start: bool = True,
    adaptive_krylov: bool = False,
    direct_max_dimension: int | None = _DIRECT_DIMENSION_LIMIT,
    low_rank_target: str = "matrix",
) -> HPAOResult:
    """Opt-in ADP solver with rank bound on B or on its correction B-P.

    ``matrix`` retains the historical low-rank B and prior completion;
    ``correction`` orthonormalizes the full updated P+Delta.
    """
    if hasattr(U, "__cuda_array_interface__"):
        raise NotImplementedError("truncated-SVD solver requires CPU statistics")
    P, U, I, mass = _validate_inputs(index_init, U, I, mass)
    if P.ndim != 2:
        raise ValueError("truncated-SVD solver requires a multi-index matrix")
    P = _normalize_index(P)
    g, _ = _local_refit(I, U, P)
    raw, diagnostics, V, singular = _solve_fixed_coefficients_validated(
        P,
        U,
        I,
        g,
        mass,
        rank=rank,
        lambda_penalty=lambda_prox,
        inner_tol=inner_tol,
        inner_maxiter=inner_maxiter,
        rank_tol=rank_tol,
        lsmr_maxiter=lsmr_maxiter,
        warm_start=warm_start,
        adaptive_krylov=adaptive_krylov,
        direct_max_dimension=direct_max_dimension,
        low_rank_target=low_rank_target,
    )
    if low_rank_target == "correction":
        if diagnostics["correction_rank"] == 0:
            basis = P.copy()
        else:
            q, r = np.linalg.qr(raw.T, mode="reduced")
            values = np.linalg.svd(r, compute_uv=False)
            if values[-1] <= np.finfo(float).eps * max(raw.shape) * values[0]:
                raise RuntimeError("updated correction basis lost row rank")
            basis = q.T
        completion = "updated_basis_qr"
    else:
        basis = _complete_basis(V, singular, P)
        completion = "prior_projected"
    coefficients, local_ranks = _local_refit(I, U, basis)
    diagnostics.update(
        linear_solver="truncated_svd",
        loss=_loss(I, U, basis, coefficients, mass),
        local_rank_loss=int(np.count_nonzero(local_ranks < len(P))),
        completion=completion,
    )
    return HPAOResult(basis, coefficients, diagnostics)
