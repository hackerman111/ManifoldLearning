"""Точный профиль Schur, HVP и малые предобусловленные trust-шаги."""

# ruff: noqa: RUF002

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..engine.common.utils import _finite_real_array


class SchurCore:
    """Phi(polar(Y+V K A.T)); K=(q,a), A=(m,a), UY=M, UV=W.

    Неактивные коэффициенты исключаются augmented SVD без p*p Pi.
    Gram-матрицы только малые; tau=0 требует постоянного полного ранга.
    """

    def __init__(
        self,
        M: np.ndarray,
        W: np.ndarray,
        I: np.ndarray,
        mass: np.ndarray,
        tau: float,
        A: np.ndarray | None = None,
    ) -> None:
        M, W, I, mass = (
            _finite_real_array(x, name)
            for x, name in ((M, "M"), (W, "W"), (I, "I"), (mass, "mass"))
        )
        if M.ndim != 3 or W.ndim != 3 or 0 in M.shape or 0 in W.shape:
            raise ValueError("M and W must have non-empty shape (J,p,k)")
        if M.shape[:2] != W.shape[:2] or I.shape != M.shape[:2]:
            raise ValueError("M, W and I shapes must agree")
        J, _, m = M.shape
        if mass.shape != (J,) or np.any(mass <= 0):
            raise ValueError("mass must contain J positive values")
        if not np.isfinite(tau) or tau < 0:
            raise ValueError("tau must be finite and nonnegative")
        A = np.eye(m) if A is None else _finite_real_array(A, "A")
        if A.ndim != 2 or A.shape[0] != m or not 1 <= A.shape[1] <= m:
            raise ValueError("A must have shape (m,a), 1 <= a <= m")
        a, q = A.shape[1], W.shape[2]
        if not np.allclose(A.T @ A, np.eye(a), atol=1e-12, rtol=1e-12):
            raise ValueError("A must have orthonormal columns")
        self.shape, self.mass, self.tau = (q, a), mass, tau
        self._last_K = None
        self._cache = None
        D = M @ A
        packed = np.concatenate((D, W, I[..., None]), axis=2)
        if a < m:
            complement = np.linalg.qr(A, mode="complete")[0][:, a:]
            C = M @ complement
            if tau:
                C = np.concatenate(
                    (
                        C,
                        np.broadcast_to(
                            np.sqrt(tau) * np.eye(m - a), (J, m - a, m - a)
                        ),
                    ),
                    axis=1,
                )
                packed = np.concatenate(
                    (packed, np.zeros((J, m - a, a + q + 1))), axis=1
                )
            left, singular, _ = np.linalg.svd(C, full_matrices=False)
            if np.any(singular[:, -1] <= np.sqrt(np.finfo(float).eps) * singular[:, 0]):
                raise ValueError(
                    "Schur inactive block requires well-conditioned full rank"
                )
            packed -= left @ (left.swapaxes(1, 2) @ packed)
        gram = packed.swapaxes(1, 2) @ packed
        self.k = gram[:, -1, -1]
        self.alpha, self.beta = gram[:, :a, -1], gram[:, a : a + q, -1]
        self.E, self.F = gram[:, :a, :a], gram[:, :a, a : a + q]
        self.T = gram[:, a : a + q, a : a + q] + tau * np.eye(q)

    def _state(self, K: np.ndarray) -> tuple[np.ndarray, ...]:
        K = _finite_real_array(K, "K")
        if K.shape != self.shape:
            raise ValueError(f"K must have shape {self.shape}")
        if self._last_K is not None and np.array_equal(K, self._last_K):
            assert self._cache is not None
            return self._cache
        B = self.F.swapaxes(1, 2) + self.T @ K
        FK = self.F @ K
        S = self.E + FK + FK.swapaxes(1, 2) + K.T @ self.T @ K
        S += self.tau * np.eye(self.shape[1])
        # Schur complements can lose relative precision near a rank boundary.
        values = np.linalg.eigvalsh(S)
        if np.any(values[:, 0] <= np.sqrt(np.finfo(float).eps) * values[:, -1]):
            raise ValueError("Schur profile requires well-conditioned full rank")
        h = self.alpha + (K.T @ self.beta[..., None]).squeeze(-1)
        g = np.linalg.solve(S, h[..., None]).squeeze(-1)
        rho = self.beta - (B @ g[..., None]).squeeze(-1)
        if not np.all(np.isfinite(g)):
            raise RuntimeError("non-finite Schur coefficients")
        self._last_K, self._cache = K.copy(), (S, h, g, rho, B)
        return self._cache

    def evaluate(self, K: np.ndarray) -> tuple[float, np.ndarray]:
        _, h, g, rho, _ = self._state(K)
        value = float(self.mass @ (self.k - np.sum(h * g, axis=1)))
        grad = -2 * np.einsum("j,jq,ja->qa", self.mass, rho, g)
        if not np.isfinite(value) or not np.all(np.isfinite(grad)):
            raise RuntimeError("non-finite Schur profile")
        return value, grad

    def hvp(self, K: np.ndarray, E: np.ndarray) -> np.ndarray:
        S, _, g, rho, B = self._state(K)
        E = _finite_real_array(E, "E")
        if E.shape != self.shape:
            raise ValueError(f"E must have shape {self.shape}")
        Eg = E @ g[..., None]
        rhs = E.T @ rho[..., None] - B.swapaxes(1, 2) @ Eg
        dg = np.linalg.solve(S, rhs).squeeze(-1)
        drho = -(self.T @ Eg).squeeze(-1) - (B @ dg[..., None]).squeeze(-1)
        return -2 * (
            np.einsum("j,jq,ja->qa", self.mass, drho, g)
            + np.einsum("j,jq,ja->qa", self.mass, rho, dg)
        )

    def precondition(self, K: np.ndarray, damping: float) -> Callable:
        """SPD-блоки 2 sum mass*g_l²*T + damping*I; cross terms лишь в HVP."""
        if not np.isfinite(damping) or damping <= 0:
            raise ValueError("damping must be finite and positive")
        _, _, g, _, _ = self._state(K)
        blocks = 2 * np.einsum("j,ja,jqr->aqr", self.mass, g**2, self.T)
        blocks += damping * np.eye(self.shape[0])
        factors = np.linalg.cholesky(blocks)

        def apply(rhs: np.ndarray) -> np.ndarray:
            z = np.linalg.solve(factors, rhs.T[..., None])
            return np.linalg.solve(factors.swapaxes(1, 2), z).squeeze(-1).T

        return apply


def metric_direction(
    G: np.ndarray,
    coefficients: np.ndarray,
    mass: np.ndarray,
    rank: int,
    metric_floor: float = 1e-6,
) -> np.ndarray:
    """Метрический rank-q спуск с обоими правыми whitening-множителями."""
    G, coefficients, mass = (
        _finite_real_array(x, name)
        for x, name in ((G, "G"), (coefficients, "coefficients"), (mass, "mass"))
    )
    if G.ndim != 2 or coefficients.ndim != 2 or coefficients.shape[1] != G.shape[1]:
        raise ValueError("G and coefficients dimensions must agree")
    if mass.shape != (len(coefficients),) or np.any(mass <= 0):
        raise ValueError("mass must contain positive values")
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)):
        raise TypeError("rank must be an integer")
    if not 1 <= rank <= min(G.shape):
        raise ValueError("rank is out of range")
    if not np.isfinite(metric_floor) or not 0 < metric_floor <= 1:
        raise ValueError("metric_floor must be in (0,1]")
    R = coefficients.T @ (mass[:, None] * coefficients)
    values, vectors = np.linalg.eigh(R)
    values = np.maximum(
        values, metric_floor * max(float(values[-1]), np.finfo(float).tiny)
    )
    whitening = (vectors * (1 / np.sqrt(values))) @ vectors.T
    left, s, right = np.linalg.svd(G @ whitening, full_matrices=False)
    norm = np.linalg.norm(s[:rank])
    if norm == 0:
        return np.zeros_like(G)
    return -((left[:, :rank] * (s[:rank] / norm)) @ right[:rank]) @ whitening


def trust_step(
    gradient: np.ndarray,
    hvp: Callable,
    precondition: Callable,
    radius: float,
    *,
    maxiter: int | None = None,
    tol: float = 1e-6,
) -> tuple[np.ndarray, int, bool]:
    """Preconditioned Steihaug CG; граница в исходной Frobenius-норме."""
    gradient = _finite_real_array(gradient, "gradient")
    if not np.isfinite(radius) or radius <= 0 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("radius and tol must be finite and positive")
    if maxiter is None:
        maxiter = max(1, 2 * gradient.size)
    if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)):
        raise TypeError("maxiter must be an integer")
    if maxiter <= 0:
        raise ValueError("maxiter must be positive")
    x, r = np.zeros_like(gradient), gradient.copy()
    target = tol * np.linalg.norm(r)
    if np.linalg.norm(r) == 0:
        return x, 0, False
    z = precondition(r)
    rz = float(np.sum(r * z))
    if not np.isfinite(rz) or rz <= 0:
        raise ValueError("preconditioner must be positive definite")
    p = -z

    def boundary() -> np.ndarray:
        pp, xp = float(np.sum(p * p)), float(np.sum(x * p))
        remaining = max(0.0, radius**2 - float(np.sum(x * x)))
        root = np.sqrt(xp**2 + pp * remaining)
        alpha = remaining / (root + xp) if xp > 0 else (root - xp) / pp
        return x + alpha * p

    for iteration in range(1, maxiter + 1):
        Hp = _finite_real_array(hvp(p), "HVP")
        if Hp.shape != gradient.shape:
            raise ValueError("HVP must match gradient shape")
        curvature = float(np.sum(p * Hp))
        if curvature <= 0:
            return boundary(), iteration, True
        alpha = rz / curvature
        if np.linalg.norm(x + alpha * p) >= radius:
            return boundary(), iteration, False
        x += alpha * p
        r += alpha * Hp
        if np.linalg.norm(r) <= target:
            return x, iteration, False
        z = precondition(r)
        new_rz = float(np.sum(r * z))
        if not np.isfinite(new_rz) or new_rz <= 0:
            raise ValueError("preconditioner must be positive definite")
        p = -z + (new_rz / rz) * p
        rz = new_rz
    return x, maxiter, False
