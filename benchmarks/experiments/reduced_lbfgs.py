"""Experimental guarded Riemannian L-BFGS for fixed multi-index statistics."""

from __future__ import annotations

import math

import numpy as np

from ADP.gpu import array_module
from ADP.solver.LSMR import (
    HPAOResult,
    _index_distance,
    _normalize_index,
    _stationarity,
    _validate_inputs,
)

from .multi_solver_derivation import ReducedReference, _retract, evaluate_reduced


def _horizontal(B: np.ndarray, value: np.ndarray) -> np.ndarray:
    return value - (value @ B.T) @ B


def _admissible(value: ReducedReference, pattern: np.ndarray) -> bool:
    return (
        np.array_equal(value.ranks, pattern)
        and value.value_defect <= 1e-10 * max(1.0, value.objective)
        and value.gradient_defect <= 1e-8
    )


def _direction(
    B: np.ndarray,
    gradient: np.ndarray,
    history: list[tuple[np.ndarray, np.ndarray, float]],
) -> tuple[np.ndarray, bool]:
    q = gradient.copy()
    alphas = []
    for s, y, rho in reversed(history):
        alpha = rho * float(np.sum(s * q))
        alphas.append(alpha)
        q -= alpha * y
    if history:
        s, y, _ = history[-1]
        scale = np.clip(np.sum(s * y) / np.sum(y * y), 1e-4, 1e4)
    else:
        scale = 1.0
    result = scale * q
    for (s, y, rho), alpha in zip(history, reversed(alphas), strict=True):
        beta = rho * float(np.sum(y * result))
        result += s * (alpha - beta)
    direction = _horizontal(B, -result)
    norm = float(np.linalg.norm(gradient))
    descent = float(np.sum(gradient * direction))
    if (
        not np.all(np.isfinite(direction))
        or descent > -1e-4 * norm * norm
        or np.linalg.norm(direction) > 1e4 * norm
    ):
        return -gradient, True
    return direction, False


def _transport_history(
    history: list[tuple[np.ndarray, np.ndarray, float]], B: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    transported = []
    for s, y, _ in history:
        s, y = _horizontal(B, s), _horizontal(B, y)
        sy = float(np.sum(s * y))
        if sy > 1e-10 * np.linalg.norm(s) * np.linalg.norm(y):
            transported.append((s, y, 1.0 / sy))
    return transported


def solve(
    index_init: np.ndarray,
    U: np.ndarray,
    I: np.ndarray,
    *,
    mass: np.ndarray | None = None,
    lambda_prox: float = 1.0,
    max_steps: int = 80,
    tol: float = 1e-6,
    history_size: int = 5,
    max_backtracks: int = 30,
) -> HPAOResult:
    """Optimize the current truncated local-refit objective on Grassmann.

    This selection-stage candidate is CPU-only and fails outside the proof
    domain. ``lambda_prox`` is validated for interface compatibility but is
    absent from the statistical objective and the L-BFGS update.
    """
    if array_module(U) is not np:
        raise NotImplementedError("experimental reduced solver requires CPU arrays")
    index, U, I, mass = _validate_inputs(index_init, U, I, mass)
    if index.ndim != 2:
        raise ValueError("reduced solver requires a multi-index matrix")
    if not np.isfinite(lambda_prox) or lambda_prox < 0:
        raise ValueError("lambda_prox must be finite and nonnegative")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    for name, value, minimum in (
        ("max_steps", max_steps, 1),
        ("history_size", history_size, 0),
        ("max_backtracks", max_backtracks, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

    B = _normalize_index(index)
    current = evaluate_reduced(B, U, I, mass)
    pattern = current.ranks.copy()
    if not _admissible(current, pattern):
        raise RuntimeError("initial reduced objective is outside the proof domain")
    gradient = _horizontal(B, current.gradient)
    history: list[tuple[np.ndarray, np.ndarray, float]] = []
    loss_history = [current.objective]
    gradient_history = [float(np.linalg.norm(gradient))]
    rank_history = [current.rank_loss]
    rejected_armijo = rejected_guard = fallbacks = 0
    evaluations = 1
    accepted = consecutive = 0
    relative_change = aligned_step = math.inf
    global_score, local_score, orthogonality = _stationarity(
        I, U, B, current.coefficients, mass, current.objective
    )

    for _ in range(max_steps):
        if np.linalg.norm(gradient) == 0:
            break
        direction, fallback = _direction(B, gradient, history)
        fallbacks += int(fallback)
        derivative = float(np.sum(gradient * direction))
        old_B, old_gradient, old_loss = B, gradient, current.objective
        step = 1.0
        for _ in range(max_backtracks):
            candidate_B = _retract(B, step * direction)
            evaluations += 1
            try:
                candidate = evaluate_reduced(candidate_B, U, I, mass)
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                rejected_guard += 1
                step *= 0.5
                continue
            if not _admissible(candidate, pattern):
                rejected_guard += 1
                step *= 0.5
                continue
            if candidate.objective <= old_loss + 1e-4 * step * derivative:
                break
            rejected_armijo += 1
            step *= 0.5
        else:
            raise RuntimeError("reduced L-BFGS line search exhausted its budget")

        B, current = candidate_B, candidate
        gradient = _horizontal(B, current.gradient)
        history = _transport_history(history, B)
        secant_s = _horizontal(B, step * direction)
        secant_y = gradient - _horizontal(B, old_gradient)
        sy = float(np.sum(secant_s * secant_y))
        if sy > 1e-10 * np.linalg.norm(secant_s) * np.linalg.norm(secant_y):
            history.append((secant_s, secant_y, 1.0 / sy))
            history = history[-history_size:] if history_size else []
        accepted += 1
        loss_history.append(current.objective)
        gradient_history.append(float(np.linalg.norm(gradient)))
        rank_history.append(current.rank_loss)
        relative_change = abs(old_loss - current.objective) / max(1.0, old_loss)
        aligned_step = _index_distance(B, old_B)
        global_score, local_score, orthogonality = _stationarity(
            I, U, B, current.coefficients, mass, current.objective
        )
        certified = (
            relative_change < tol
            and aligned_step < tol
            and max(global_score, local_score, orthogonality) < tol
        )
        consecutive = consecutive + 1 if certified else 0
        if consecutive == 2:
            break

    diagnostics = {
        "solver": "reduced-lbfgs",
        "accepted_steps": accepted,
        "converged": consecutive == 2,
        "stop_reason": (
            "converged"
            if consecutive == 2
            else "stationary_initial"
            if accepted == 0
            else "max_steps"
        ),
        "loss": current.objective,
        "loss_history": tuple(loss_history),
        "gradient_history": tuple(gradient_history),
        "local_rank_loss": tuple(rank_history),
        "riemannian_gradient": global_score,
        "local_gradient": local_score,
        "orthogonality": orthogonality,
        "relative_loss_change": relative_change,
        "aligned_step": aligned_step,
        "consecutive_certified_steps": consecutive,
        "certificate_failures": tuple(
            name
            for name, failed in (
                ("relative_loss_change", relative_change >= tol),
                ("aligned_step", aligned_step >= tol),
                ("riemannian_gradient", global_score >= tol),
                ("local_gradient", local_score >= tol),
                ("orthogonality", orthogonality >= tol),
                ("consecutive_steps", consecutive < 2),
            )
            if failed
        ),
        "value_defect": current.value_defect,
        "gradient_defect": current.gradient_defect,
        "objective_evaluations": evaluations,
        "armijo_rejections": rejected_armijo,
        "guard_rejections": rejected_guard,
        "descent_fallbacks": fallbacks,
        "history_size": len(history),
    }
    return HPAOResult(B, current.coefficients.copy(), diagnostics)
