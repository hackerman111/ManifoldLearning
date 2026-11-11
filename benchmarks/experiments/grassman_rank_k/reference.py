"""Независимый multistart reference: direct lstsq, Riemannian CG, QR retract.

Численный ориентир сверху, без гарантии глобального минимума. Numerical
kernels production solver не импортируются; tau=0 protocol фиксирован.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np


def direct_profile(
    Y: np.ndarray, U: np.ndarray, I: np.ndarray, mass: np.ndarray
) -> tuple[float, np.ndarray]:
    """F и horizontal envelope gradient, независимые локальные lstsq."""
    M = U @ Y
    coefficients = np.stack(
        [
            np.linalg.lstsq(local, target, rcond=None)[0]
            for local, target in zip(M, I, strict=True)
        ]
    )
    residual = I - np.einsum("jpm,jm->jp", M, coefficients)
    f = float(np.sum(mass[:, None] * residual**2))
    adjoint = np.einsum("jpd,jp->jd", U, residual, optimize=True)
    G = -2 * (adjoint.T @ (mass[:, None] * coefficients))
    return f, G - Y @ (Y.T @ G)


def _cg(
    Y: np.ndarray,
    U: np.ndarray,
    I: np.ndarray,
    mass: np.ndarray,
    max_steps: int,
    tol: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    f, g = direct_profile(Y, U, I, mass)
    initial = f
    direction = -g
    scale = max(1.0, float(np.sum(mass[:, None] * I**2)))
    step = 1 / max(1.0, float(np.linalg.norm(g)))
    evaluations = 1
    accepted = 0
    stop = "max_steps"
    for _ in range(max_steps):
        if np.linalg.norm(g) <= tol * scale:
            stop = "stationary"
            break
        slope = float(np.sum(g * direction))
        if slope >= -0.01 * float(np.sum(g**2)):
            direction = -g
            slope = -float(np.sum(g**2))
        trial_step = min(step * 2, 0.5 / max(np.linalg.norm(direction, ord=2), 1e-30))
        candidate = None
        for _backtrack in range(45):
            Yn = np.linalg.qr(Y + trial_step * direction, mode="reduced")[0]
            # QR signs affect transported coordinates even though F is invariant.
            signs = np.where(np.diag(Y.T @ Yn) < 0, -1.0, 1.0)
            Yn *= signs
            fn, gn = direct_profile(Yn, U, I, mass)
            evaluations += 1
            if fn <= f + 1e-4 * trial_step * slope:
                candidate = Yn, fn, gn
                break
            trial_step *= 0.5
        if candidate is None:
            stop = "line_search_failed"
            break
        Yn, fn, gn = candidate
        old_g = g - Yn @ (Yn.T @ g)
        old_direction = direction - Yn @ (Yn.T @ direction)
        beta = max(
            0.0, float(np.sum(gn * (gn - old_g))) / max(float(np.sum(g**2)), 1e-300)
        )
        direction = -gn + beta * old_direction
        Y, f, g = Yn, fn, gn
        step = trial_step
        accepted += 1
    norm = float(np.linalg.norm(g))
    return Y, {
        "objective": f,
        "initial_objective": initial,
        "gradient_norm": norm,
        "normalized_gradient": norm / scale,
        "converged": norm <= tol * scale,
        "iterations": accepted,
        "profile_evaluations": evaluations,
        "stop_reason": stop,
        "orthogonality_error": float(np.linalg.norm(Y.T @ Y - np.eye(Y.shape[1]))),
    }


def reference(
    snapshot: dict[str, np.ndarray],
    max_steps: int = 1000,
    tol: float = 1e-9,
) -> dict[str, Any]:
    """Три старта: Y0, planted truth, независимый фиксированный random frame."""
    U, I, mass = (snapshot[key] for key in ("U", "I", "mass"))
    initial, truth = snapshot["index_init"].T, snapshot["truth"]
    rng = np.random.default_rng(816159)
    random = np.linalg.qr(rng.normal(size=initial.shape))[0]
    started = perf_counter()
    starts = []
    solutions = []
    for name, Y in (("initial", initial), ("truth", truth), ("random", random)):
        basis, row = _cg(Y.copy(), U, I, mass, max_steps, tol)
        row["start"] = name
        starts.append(row)
        solutions.append(basis)
    best = min(range(len(starts)), key=lambda k: starts[k]["objective"])
    Y = solutions[best]
    # Frobenius overlap identity; no dense ambient projector.
    error = float(np.sqrt(max(0.0, 1 - np.linalg.norm(Y.T @ truth) ** 2 / Y.shape[1])))
    return {
        **starts[best],
        "starts": starts,
        "best_start": starts[best]["start"],
        "wall_sec": perf_counter() - started,
        "projector_error": error,
        "reference_kind": "independent multistart Riemannian CG, numerical upper bound",
    }
