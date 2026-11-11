"""Direct full-moment ridge profile; opt-in CPU experiment, no ADP defaults."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.optimize import minimize

from ADP.core.ADP_Statistic import ADP_Statistics
from ADP.core.solver_common import HPAOResult
from ADP.engine.common import utils
from experiments.coupled_penalties import Profile


@dataclass
class Problem:
    covariance: np.ndarray
    cross: np.ndarray
    pi: np.ndarray
    tau: float
    energy: float
    scale: float
    info: dict


def calculate_statistics(X, Y, weights, directions, batch_size=32, *, normalized=False):
    """Consume weight blocks once; U=C and I=c are always normalized moments.

    This hook is exclusively for solve() below, not a projected ADP solver.
    Mass, mean and S retain the outer-loop meaning for either normalization.
    """
    utils._check_batch_size(batch_size)
    if not isinstance(normalized, bool):
        raise TypeError("normalized must be boolean")
    X, Y, directions = utils._prepare_data(X, Y, directions)
    j, d = len(directions), X.shape[1]
    covariance, cross = np.empty((j, d, d)), np.empty((j, d))
    mass, mean, n_eff, response = (
        np.empty(j),
        np.empty((j, d)),
        np.empty(j),
        np.empty(j),
    )
    expected = 0
    for start, block in utils._weight_blocks(weights, j, len(X), batch_size):
        utils._check_weight_block_order(start, expected, j)
        block, block_mass = utils._prepare_weight_block(block, len(X))
        stop = start + len(block)
        if stop > j:
            raise ValueError("too many weight rows")
        for k, row in enumerate(block, start):
            support = row > 0
            a = row[support] / block_mass[k - start]
            local_x, local_y = X[support], Y[support]
            mean[k], response[k] = a @ local_x, a @ local_y
            z, dy = local_x - mean[k], local_y - response[k]
            covariance[k] = z.T @ (a[:, None] * z)
            cross[k] = z.T @ (a * dy)
            n_eff[k] = 1 / (a @ a)
        mass[start:stop] = block_mass
        expected = stop
    if expected != j:
        raise ValueError("weight blocks do not cover all centers")
    return ADP_Statistics(
        cross, covariance, mass, mean, n_eff, np.empty((j, 0)), response
    )


def build(stats, basis):
    """Preserve the reference spectral mask; certify the common full-rank case."""
    covariance, cross = np.asarray(stats.U), np.asarray(stats.I)
    j, d, m = len(stats.mass), *basis.shape
    if covariance.shape != (j, d, d) or cross.shape != (j, d):
        raise ValueError("direct moments require U:(J,d,d), I:(J,d)")
    if (
        not np.isfinite(covariance).all()
        or not np.isfinite(cross).all()
        or not np.isfinite(stats.mass).all()
        or np.any(stats.mass <= 0)
    ):
        raise ValueError("moments must be finite with positive mass")
    if not np.allclose(covariance, covariance.swapaxes(1, 2), atol=1e-13, rtol=1e-12):
        raise ValueError("covariance must be symmetric")
    energies, incoming_energies, ranks = (
        np.empty(j),
        np.empty(j),
        np.empty(j, dtype=int),
    )
    fallbacks, copied, discarded = 0, False, 0.0
    identity = np.eye(d)
    for k, (c, h) in enumerate(zip(covariance, cross, strict=True)):
        upper = max(np.linalg.norm(c, ord=np.inf), np.finfo(float).tiny)
        # Keep ill-conditioned full-rank windows on the reference eigensolver too.
        delta = 1e-7 * upper
        try:
            if delta == 0:
                raise np.linalg.LinAlgError("zero covariance")
            shifted = c - delta * identity
            factor = cholesky(shifted, lower=True, check_finite=False)
            defect = np.linalg.norm(shifted - factor @ factor.T, ord=np.inf)
            if defect >= 0.25 * delta:
                raise np.linalg.LinAlgError("uncertified spectral margin")
            solved = cho_solve(cho_factor(c, check_finite=False), h, check_finite=False)
            energy = float(h @ solved)
            residual = np.linalg.norm(c @ solved - h) / max(np.linalg.norm(h), 1e-300)
            if not np.isfinite(energy) or energy < 0 or residual > 1e-10:
                raise np.linalg.LinAlgError("unreliable covariance solve")
            energies[k], ranks[k] = energy, d
            incoming_energies[k] = np.sum(basis * (c @ basis))
        except np.linalg.LinAlgError:
            values, vectors = np.linalg.eigh(c)
            if values[0] < -1e-10 * max(values[-1], 1):
                raise ValueError("covariance must be positive semidefinite") from None
            keep = values > 1e-12 * max(values[-1], np.finfo(float).tiny)
            projected = vectors.T @ h
            roots = np.sqrt(np.maximum(values, 0)) * keep
            inverse = np.divide(1.0, roots, out=np.zeros_like(roots), where=keep)
            target = vectors @ (inverse * projected)
            energies[k] = target @ target
            # Preserve the reference ridge even when B is numerically in ker(C).
            root = (vectors * roots) @ vectors.T
            incoming_energies[k] = np.sum((root @ basis) ** 2)
            ranks[k] = np.count_nonzero(keep)
            discarded += float(np.sum(projected[~keep] ** 2))
            if not copied:
                covariance, cross = covariance.copy(), cross.copy()
                copied = True
            covariance[k] = (vectors[:, keep] * values[keep]) @ vectors[:, keep].T
            cross[k] = vectors[:, keep] @ projected[keep]
            fallbacks += 1
    pi = stats.mass / stats.mass.sum()
    scale = float(np.median(incoming_energies) / m)
    if scale <= 0 and (np.any(covariance) or np.any(cross)):
        raise ValueError("zero median ridge scale cannot guarantee SPD")
    energy = float(pi @ energies)
    return Problem(
        covariance,
        cross,
        pi,
        0.01 * scale,
        energy,
        max(1.0, energy),
        {
            "rank_min": int(ranks.min()),
            "rank_max": int(ranks.max()),
            "rank_histogram": np.bincount(ranks, minlength=d + 1).tolist(),
            "spectral_fallbacks": fallbacks,
            "covariance_cutoff_relative": 1e-12,
            "discarded_cross_relative": float(
                np.sqrt(discarded) / max(np.linalg.norm(stats.I), 1e-300)
            ),
        },
    )


def profile(basis, problem):
    """One C*B contraction, small independent ridge solves, envelope gradient."""
    p = problem
    design = p.covariance @ basis
    blocks = basis.T @ design + p.tau * np.eye(basis.shape[1])
    rhs = p.cross @ basis
    if not np.any(blocks):
        coefficients = np.zeros_like(rhs)
    else:
        coefficients = np.linalg.solve(blocks, rhs[..., None]).squeeze(-1)
    residual = (design @ coefficients[..., None]).squeeze(-1) - p.cross
    gradient = 2 * residual.T @ (p.pi[:, None] * coefficients)
    value = float(p.energy - np.sum(p.pi[:, None] * rhs * coefficients))
    ridge = float(p.tau * np.sum(p.pi[:, None] * coefficients**2))
    normal = (blocks @ coefficients[..., None]).squeeze(-1) - rhs
    normal_residual = float(np.linalg.norm(normal) / max(np.linalg.norm(rhs), 1e-300))
    return Profile(
        value,
        gradient,
        coefficients,
        {"data": value - ridge, "ridge": ridge},
        0,
        normal_residual,
    )


def solve(index, stats, *, max_steps=20, tol=1e-6):
    """The reference QR chart and L-BFGS settings with the direct profile."""
    started = time.perf_counter()
    original = np.asarray(index, dtype=float).T
    d, m = original.shape
    if not 1 <= m < d:
        raise ValueError("index dimension must lie between 1 and d-1")
    if not np.allclose(original.T @ original, np.eye(m), atol=1e-10):
        raise ValueError("index must have orthonormal rows")
    p = build(stats, original)
    full = np.linalg.qr(original, mode="complete")[0]
    reference, complement = full[:, :m], full[:, m:]
    shape = (d - m, m)
    preparation = time.perf_counter() - started
    started = time.perf_counter()
    cache, history, residuals = {}, [], []

    def evaluate(flat):
        if "point" not in cache or not np.array_equal(cache["point"], flat):
            b, r = np.linalg.qr(
                reference + complement @ flat.reshape(shape), mode="reduced"
            )
            state = profile(b, p)
            horizontal = state.gradient - b @ (b.T @ state.gradient)
            gradient = complement.T @ np.linalg.solve(r, horizontal.T).T
            cache.update(point=flat.copy(), basis=b, state=state, gradient=gradient)
            residuals.append(state.normal_residual)
        return cache["state"].value / p.scale, cache["gradient"].ravel() / p.scale

    initial = np.zeros(np.prod(shape))
    history.append(evaluate(initial)[0] * p.scale)

    def accepted(flat):
        value = evaluate(flat)[0] * p.scale
        if value > history[-1] + 1e-10 * max(1.0, history[-1]):
            raise RuntimeError("chart accepted an increasing objective")
        history.append(value)

    result = minimize(
        evaluate,
        initial,
        jac=True,
        method="L-BFGS-B",
        callback=accepted,
        options={"maxiter": max_steps, "maxls": 20, "ftol": 1e-12, "gtol": 1e-8},
    )
    evaluate(result.x)
    b, state = cache["basis"], cache["state"]
    norm = float(np.linalg.norm(state.gradient - b @ (b.T @ state.gradient)) / p.scale)
    return HPAOResult(
        b.T,
        state.coefficients,
        {
            "method": "direct_full_ridge_chart_lbfgs",
            "profile_objective": state.value,
            "loss": state.value / 2,
            "profile_history": history,
            "iterations": int(result.nit),
            "profile_evaluations": len(residuals),
            "converged": norm <= tol,
            "riemannian_gradient": norm,
            "stop_reason": "stationary" if norm <= tol else str(result.message),
            "scipy_success": bool(result.success),
            "components": state.components,
            "normal_residual_max": max(residuals),
            "penalty": {
                "arm": "direct_full",
                "local_ridge": p.tau,
                "prepare_seconds": preparation,
                "solve_seconds": time.perf_counter() - started,
                "full_moments": p.info,
            },
        },
    )
