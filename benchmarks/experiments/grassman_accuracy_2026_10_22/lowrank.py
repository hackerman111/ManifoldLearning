"""Joint additive corrections; experimental proposals for the live profile.

Y is (d,m), Delta=A C is horizontal and rank<=k before polar retraction.
The frozen-g quadratic only proposes a step: the original all-center profile
and gradient decide acceptance and stationarity. No dense ambient Hessian.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr

from ADP.solver import grassman as base
from ADP.solver.grassman_exp import horizontal, polar


class FrozenCorrection(LinearOperator):
    """L -> sqrt(mass) U horizontal(L) (right g), scratch O(Jpk+dk)."""

    def __init__(
        self,
        Y: np.ndarray,
        U: np.ndarray,
        g: np.ndarray,
        mass: np.ndarray,
        right: np.ndarray | None = None,
    ) -> None:
        if not U.flags.c_contiguous:
            raise ValueError("U must be C-contiguous; avoid an implicit full-U copy")
        self.Y, self.U, self.g = Y, U, g
        self.root = np.sqrt(mass)[:, None]
        self.J, self.p, self.d = U.shape
        self.m = Y.shape[1]
        self.right = np.eye(self.m) if right is None else right
        self.k = len(self.right)
        self.h = g @ self.right.T
        super().__init__(
            dtype=np.dtype(float), shape=(self.J * self.p, self.d * self.k)
        )

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        L = horizontal(self.Y, x.reshape(self.d, self.k))
        action = base._project(self.U, L)
        return (self.root * np.einsum("jpk,jk->jp", action, self.h)).ravel()

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        dual = self.root * x.reshape(self.J, self.p)
        result = np.zeros((self.d, self.k))
        chunk = max(1, (4 * 1024**2) // (8 * self.p * self.k))
        for start in range(0, self.J, chunk):
            stop = min(self.J, start + chunk)
            data = dual[start:stop, :, None] * self.h[start:stop, None, :]
            result += self.U[start:stop].reshape(-1, self.d).T @ data.reshape(
                -1, self.k
            )
        return horizontal(self.Y, result).ravel()


class _AugmentedCorrection(LinearOperator):
    def __init__(self, operator, damping):
        self.operator, self.root = operator, np.sqrt(damping)
        self.rows, columns = operator.shape
        super().__init__(dtype=np.dtype(float), shape=(self.rows + columns, columns))

    def _matvec(self, x):
        return np.concatenate((self.operator.matvec(x), self.root * x))

    def _rmatvec(self, x):
        return self.operator.rmatvec(x[: self.rows]) + self.root * x[self.rows :]


def _basis(Y, X, limit):
    Z = horizontal(Y, X)
    left, singular, _ = np.linalg.svd(Z, full_matrices=False)
    if not len(singular) or singular[0] == 0:
        return left[:, :0]
    keep = int(
        np.count_nonzero(singular > np.finfo(float).eps * max(Z.shape) * singular[0])
    )
    count = min(limit, keep, Y.shape[0] - Y.shape[1])
    # Remove roundoff in the vertical component before using an identity ridge.
    return np.linalg.qr(horizontal(Y, left[:, :count]), mode="reduced")[0]


def coarse_correction(
    Y: np.ndarray,
    U: np.ndarray,
    g: np.ndarray,
    residual: np.ndarray,
    mass: np.ndarray,
    V: np.ndarray,
    damping: float,
    workspace_bytes: int = 16 * 1024**2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Exact augmented LS over all k*m coordinates in Delta=V C.

    V must be orthonormal and horizontal. Returning zero on a workspace guard
    lets the caller use the unchanged baseline; no silent dimension truncation.
    """
    if not U.flags.c_contiguous:
        raise ValueError("U must be C-contiguous; avoid an implicit full-U copy")
    if not np.isfinite(damping) or damping < 0:
        raise ValueError("damping must be finite and nonnegative")
    J, p, _ = U.shape
    k, m = V.shape[1], g.shape[1]
    coordinates, rows = k * m, J * p
    estimate = 8 * (4 * rows * coordinates + coordinates**2 + J * p * k)
    if not coordinates or estimate > workspace_bytes:
        return np.zeros_like(Y), {"stop": "workspace", "coordinates": coordinates}
    W = base._project(U, V)
    design = np.empty((rows + coordinates, coordinates))
    design[:rows] = (
        np.sqrt(mass)[:, None, None, None] * W[..., None] * g[:, None, None, :]
    ).reshape(rows, coordinates)
    design[rows:] = np.sqrt(damping) * np.eye(coordinates)
    rhs = np.concatenate(
        ((np.sqrt(mass)[:, None] * residual).ravel(), np.zeros(coordinates))
    )
    C = np.linalg.lstsq(design, rhs, rcond=None)[0].reshape(k, m)
    Z = V @ C
    objective = float(np.linalg.norm(design @ C.ravel() - rhs) ** 2)
    return Z, {"stop": "coarse", "coordinates": coordinates, "objective": objective}


def _relative_damping(U, g, mass, V, damping):
    W = base._project(U, V)
    energy = np.einsum(
        "j,j,j->", mass, np.sum(g * g, axis=1), np.sum(W * W, axis=(1, 2))
    )
    return damping * float(energy) / (V.shape[1] * g.shape[1])


def correction(
    Y: np.ndarray,
    U: np.ndarray,
    state: base._Profile,
    mass: np.ndarray,
    G: np.ndarray,
    mode: str,
    rank_k: int | None = None,
    damping: float = 0.05,
    rounds: int = 2,
    lsmr_steps: int = 3,
    workspace_bytes: int = 16 * 1024**2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """SVD coarse block, LoRA joint factors, or normal-residual enrichment.

    All share ||sqrt(w)(r-U Delta g)||^2 + mu||Delta||^2, centered at
    Delta=0. LoRA updates both factors, keeping their gauge orthonormal so
    the ridge remains on the product, rather than penalizing each factor.
    """
    if not U.flags.c_contiguous:
        raise ValueError("U must be C-contiguous; avoid an implicit full-U copy")
    if mode not in {"svd", "lora", "enriched", "greedy"}:
        raise ValueError(f"unknown low-rank mode: {mode}")
    if damping <= 0 or not np.isfinite(damping):
        raise ValueError("damping must be finite and positive")
    k = Y.shape[1] if rank_k is None else rank_k
    if k < 1:
        raise ValueError("rank_k must be positive")
    V = _basis(Y, G, k)
    if V.shape[1] == 0:
        return np.zeros_like(Y), {"stop": "empty", "rank": 0}
    mu = _relative_damping(U, state.coefficients, mass, V, damping)
    if not np.isfinite(mu) or mu <= 0:
        return np.zeros_like(Y), {"stop": "damping", "rank": 0}
    g, residual = state.coefficients, state.residual
    if mode == "greedy":
        from ADP.solver.SVD import solve_fixed_coefficients

        # Literal pre-existing sequential rank-one SVD correction control.
        raw, stats = solve_fixed_coefficients(
            Y.T,
            U,
            (state.M @ g[..., None]).squeeze(-1) + residual,
            g,
            mass,
            rank=min(k, Y.shape[1] - 1),
            lambda_penalty=mu,
            low_rank_target="correction",
            rank_one_search="gradient",
            direct_max_dimension=None,
        )
        Z = horizontal(Y, raw.T - Y)
        return Z, {
            "stop": stats["rank_stop_reason"],
            "rank": min(k, Y.shape[1] - 1),
            "damping": mu,
            "lsmr": (),
        }
    Z, info = coarse_correction(Y, U, g, residual, mass, V, mu, workspace_bytes)
    info.update(rank=V.shape[1], damping=mu, lsmr=())
    if info["stop"] == "workspace" or mode == "svd":
        return Z, info
    rhs = (np.sqrt(mass)[:, None] * residual).ravel()
    if mode == "enriched":
        operator = FrozenCorrection(Y, U, g, mass)
        # Full normal residual identifies what the first coarse block missed.
        R = operator.rmatvec(rhs - operator.matvec(Z.ravel())).reshape(Y.shape) - mu * Z
        extra = R - V @ (V.T @ R)
        Q = _basis(Y, np.concatenate((V, _basis(Y, extra, k)), axis=1), 2 * k)
        candidate, trial = coarse_correction(
            Y, U, g, residual, mass, Q, mu, workspace_bytes
        )
        if trial.get("objective", np.inf) <= info["objective"]:
            Z, info = candidate, {**info, **trial, "rank": Q.shape[1]}
        return Z, info
    stops = []
    for _ in range(rounds):
        _, singular, right = np.linalg.svd(Z, full_matrices=False)
        keep = min(
            k,
            int(
                np.count_nonzero(
                    singular > np.finfo(float).eps * max(Z.shape) * singular[0]
                )
            ),
        )
        if not keep:
            break
        right = right[:keep]
        operator = FrozenCorrection(Y, U, g, mass, right)
        columns = operator.shape[1]
        augmented = _AugmentedCorrection(operator, mu)
        target = np.concatenate((rhs, np.zeros(columns)))
        result = cast(Callable[..., tuple], lsmr)(
            augmented,
            target,
            atol=1e-8,
            btol=1e-8,
            maxiter=lsmr_steps,
            x0=(Z @ right.T).ravel(),
        )
        stops.append((int(result[1]), int(result[2]), float(result[4])))
        if not np.all(np.isfinite(result[0])):
            break
        V = _basis(Y, result[0].reshape(Y.shape[0], keep), k)
        candidate, trial = coarse_correction(
            Y, U, g, residual, mass, V, mu, workspace_bytes
        )
        if trial.get("objective", np.inf) > info["objective"]:
            break
        Z, info = candidate, {**info, **trial, "rank": V.shape[1]}
    info["lsmr"] = tuple(stops)
    return Z, info


def step(Y, U, I, state, mass, G, tau, max_angle, workspace_bytes, mode, rank_k):
    """True-profile Armijo after a horizontal additive correction proposal."""
    if tau:
        return None, {"stop": "ridge_guard"}, 0
    Z, info = correction(
        Y, U, state, mass, G, mode, rank_k, workspace_bytes=workspace_bytes
    )
    info["plane_dimension"] = info.pop("rank", 0)
    if not np.all(np.isfinite(Z)):
        return None, {**info, "stop": "nonfinite", "rank": 0}, 0
    info["rank"] = int(np.linalg.matrix_rank(Z))
    slope = float(np.sum(G * Z))
    length = float(np.linalg.norm(Z, ord=2))
    if not np.isfinite(length) or not np.isfinite(slope) or slope >= 0 or length == 0:
        return None, info, 0
    factor = min(1.0, np.tan(max_angle) / length)
    Z *= factor
    slope *= factor
    T = base._project(U, Z)
    for backtrack in range(16):
        alpha = 0.5**backtrack
        Yn, Mn = polar(Y, state.M, Z, T, alpha)
        trial = base._profile(Mn, I, mass, tau)
        if trial.value <= state.value + 1e-4 * alpha * slope:
            return (Yn, trial), info, backtrack + 1
    return None, info, 16
