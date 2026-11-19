"""R^{-1}-предобусловленный Grassmann-решатель для LOO1-WLS-ADP."""

from __future__ import annotations

import numpy as np
from scipy import linalg as scipy_linalg

from ..core.loo1_wls import LOO1WLSConfig, _require_positive_int, _require_real
from ..engine.common import utils
from ..engine.common.loo1_wls import (
    LOO1Neighborhoods,
    _evaluate_loo1_wls_validated,
)
from .LSMR import HPAOResult


def _polar_retract(matrix: np.ndarray) -> np.ndarray:
    """Полярная ретракция на многообразие Штифеля/Грассмана через тонкий SVD."""
    left, _, right_t = np.linalg.svd(matrix, full_matrices=False)
    return left @ right_t


def _precondition_operator(
    coefficients: np.ndarray,
    mass: np.ndarray,
    preconditioner_reg: float,
) -> tuple[np.ndarray, float]:
    """Построить (R + tau I_m)^{-1} при гарантии kappa_2 <= 1 + 1 / reg.

    R = sum_j N_j c_j c_j^T, tau = reg * max(1.0, tr(R_+)). Использование
    tr(R_+) в tau строго ограничивает число обусловленности сверху величиной
    ``1 + 1 / preconditioner_reg`` даже при вырожденном спектре R.
    """
    R = (coefficients * mass[:, None]).T @ coefficients
    R = 0.5 * (R + R.T)
    vals, vecs = scipy_linalg.eigh(R)
    pos_vals = np.maximum(vals, 0.0)
    tau = preconditioner_reg * max(1.0, float(np.sum(pos_vals)))
    inv_R = (vecs / (pos_vals + tau)) @ vecs.T
    cond_bound = float((pos_vals[-1] + tau) / (pos_vals[0] + tau))
    return inv_R, cond_bound


def precondition_gradient(
    grad_riemannian: np.ndarray,
    coefficients: np.ndarray,
    mass: np.ndarray,
    *,
    preconditioner_reg: float = 1e-3,
) -> tuple[np.ndarray, float]:
    """Применить регуляризованный правый предобусловливатель (R + tau I)^{-1}.

    Входы: горизонтальный градиент ``grad_riemannian`` формы ``(d, m)``,
    активные коэффициенты ``coefficients`` формы ``(J_act, m)``, веса ``mass``
    формы ``(J_act,)``. Возвращает ``(direction, cond_bound)``.
    """
    grad_riemannian = utils._finite_real_array(grad_riemannian, "grad_riemannian")
    coefficients = utils._finite_real_array(coefficients, "coefficients")
    mass = utils._finite_real_array(mass, "mass")
    preconditioner_reg = _require_real("preconditioner_reg", preconditioner_reg)
    if preconditioner_reg <= 0.0:
        raise ValueError("preconditioner_reg must be positive")
    if grad_riemannian.ndim != 2 or 0 in grad_riemannian.shape:
        raise ValueError("grad_riemannian must have non-empty shape (d, m)")
    if (
        coefficients.ndim != 2
        or coefficients.shape[0] == 0
        or coefficients.shape[1] != grad_riemannian.shape[1]
    ):
        raise ValueError("coefficients must have shape (J_act, m) matching gradient")
    if mass.shape != (coefficients.shape[0],):
        raise ValueError("mass must have shape (J_act,)")
    if np.any(mass < 0.0):
        raise ValueError("mass must be nonnegative")

    inv_R, cond_bound = _precondition_operator(coefficients, mass, preconditioner_reg)
    return grad_riemannian @ inv_R, cond_bound


def solve(
    initial_index: np.ndarray,
    X: np.ndarray,
    neighborhoods: LOO1Neighborhoods,
    *,
    config: LOO1WLSConfig | None = None,
    max_steps: int | None = None,
    tol: float | None = None,
) -> HPAOResult:
    """Минимизировать LOO1-WLS функционал на многообразии Грассмана.

    Вход ``initial_index`` задаётся строчно-ортонормированной матрицей ``(m, d)``
    в общем контракте multi-index решателей; выход ``HPAOResult.index`` имеет
    ту же форму ``(m, d)``, ``HPAOResult.coefficients`` — ``(total_centers, m)``
    (нули на неактивных центрах).
    """
    if not isinstance(neighborhoods, LOO1Neighborhoods):
        raise TypeError("neighborhoods must be LOO1Neighborhoods")
    if config is None:
        cfg = LOO1WLSConfig()
    elif isinstance(config, LOO1WLSConfig):
        cfg = config
    else:
        raise TypeError("config must be LOO1WLSConfig or None")

    eff_max_steps = (
        cfg.max_steps
        if max_steps is None
        else _require_positive_int("max_steps", max_steps)
    )
    if tol is None:
        eff_tol = cfg.tol
    else:
        eff_tol = _require_real("tol", tol)
        if eff_tol <= 0.0:
            raise ValueError("tol must be positive")

    X = utils._finite_real_array(X, "X")
    if X.ndim != 2 or 0 in X.shape:
        raise ValueError("X must have non-empty shape (n, d)")
    if X.shape[0] != neighborhoods.n:
        raise ValueError("X row count must match neighborhoods.n")

    B = utils._finite_real_array(initial_index, "initial_index")
    _, d = X.shape
    if B.ndim != 2 or B.shape[1] != d or not 1 <= B.shape[0] <= d:
        raise ValueError("initial_index must have shape (m, d) with 1 <= m <= d")
    m = B.shape[0]
    if np.linalg.matrix_rank(B) < m:
        raise ValueError("initial_index must have full row rank")

    # Ортонормируем столбцы Y = B^T формы (d, m).
    Y, _ = np.linalg.qr(B.T, mode="reduced")

    obj, grad_E, active_coeffs, active_mass, info = _evaluate_loo1_wls_validated(
        Y,
        X,
        neighborhoods,
        ridge=cfg.ridge,
        leverage_cap=cfg.leverage_cap,
        smooth_cap_eps=cfg.smooth_cap_eps,
        center_chunk=cfg.center_chunk,
    )
    grad_R = grad_E - Y @ (Y.T @ grad_E)
    grad_norm = float(np.linalg.norm(grad_R))
    initial_obj = obj
    scale = max(1.0, initial_obj)

    loss_history: list[float] = [obj]
    evaluations = 1
    accepted_steps = 0
    precond_fallbacks = 0
    max_cond_bound = 1.0
    converged = False
    stop_reason = "max_steps"

    prev_Y: np.ndarray | None = None
    prev_grad_R: np.ndarray | None = None

    for _ in range(eff_max_steps):
        if grad_norm <= eff_tol * scale or m == d:
            converged = True
            stop_reason = "stationary"
            break

        inv_R, cond_bound = _precondition_operator(
            active_coeffs, active_mass, cfg.preconditioner_reg
        )
        max_cond_bound = max(max_cond_bound, cond_bound)
        direction = grad_R @ inv_R
        # Повторная проекция убирает численный дрейф из горизонтального пространства.
        direction -= Y @ (Y.T @ direction)
        slope = float(np.sum(grad_R * direction))
        used_precond = True
        if not np.isfinite(slope) or slope <= 0.0:
            direction = grad_R
            slope = grad_norm * grad_norm
            used_precond = False
            precond_fallbacks += 1

        d_norm = float(np.linalg.norm(direction))
        if not np.isfinite(d_norm) or d_norm <= 0.0:
            converged = grad_norm <= eff_tol * scale
            stop_reason = "stationary" if converged else "zero_direction"
            break

        max_t = 1.0 / d_norm
        t0 = min(1.0, max_t)
        if prev_Y is not None and prev_grad_R is not None:
            s_tan = Y - prev_Y
            s_tan -= Y @ (Y.T @ s_tan)
            y_tan = grad_R - (prev_grad_R - Y @ (Y.T @ prev_grad_R))
            sy = float(np.sum(s_tan * y_tan))
            y_pre = y_tan @ inv_R if used_precond else y_tan
            ypy = float(np.sum(y_tan * y_pre))
            if sy > 1e-15 and ypy > 1e-15:
                t_bb = sy / ypy
                if np.isfinite(t_bb):
                    t0 = min(max(t_bb, 1e-6), max_t, 1e6)

        accepted = False
        attempts: list[tuple[np.ndarray, float, float]] = [(direction, slope, t0)]
        if used_precond:
            attempts.append(
                (grad_R, grad_norm * grad_norm, min(1.0, 1.0 / max(1e-12, grad_norm)))
            )

        for attempt_idx, (cand_dir, cand_slope, cand_t0) in enumerate(attempts):
            if attempt_idx > 0:
                precond_fallbacks += 1
            for ls in range(cfg.max_line_search):
                step = cand_t0 * (0.5**ls)
                Y_trial = _polar_retract(Y - step * cand_dir)
                t_obj, t_grad_E, t_coeffs, t_mass, t_info = (
                    _evaluate_loo1_wls_validated(
                        Y_trial,
                        X,
                        neighborhoods,
                        ridge=cfg.ridge,
                        leverage_cap=cfg.leverage_cap,
                        smooth_cap_eps=cfg.smooth_cap_eps,
                        center_chunk=cfg.center_chunk,
                    )
                )
                evaluations += 1
                if t_obj <= obj - cfg.armijo_c1 * step * cand_slope:
                    prev_Y = Y
                    prev_grad_R = grad_R
                    Y = Y_trial
                    obj = t_obj
                    grad_E = t_grad_E
                    active_coeffs = t_coeffs
                    active_mass = t_mass
                    info = t_info
                    grad_R = grad_E - Y @ (Y.T @ grad_E)
                    grad_norm = float(np.linalg.norm(grad_R))
                    accepted_steps += 1
                    loss_history.append(obj)
                    accepted = True
                    break
            if accepted:
                break

        if not accepted:
            stop_reason = "line_search_failed"
            break

    if grad_norm <= eff_tol * scale or m == d:
        converged = True
        stop_reason = "stationary"

    full_coefficients = np.zeros((neighborhoods.total_centers, m), dtype=float)
    full_coefficients[neighborhoods.active_centers] = active_coeffs

    diagnostics: dict[str, object] = {
        "linear_solver": "loo1_wls",
        "method": "preconditioned_grassmann",
        "loss": float(obj),
        "final_objective": float(obj),
        "initial_objective": float(initial_obj),
        "loss_history": tuple(loss_history),
        "converged": converged,
        "stop_reason": stop_reason,
        "accepted_steps": accepted_steps,
        "iterations": accepted_steps,
        "evaluations": evaluations,
        "riemannian_gradient": grad_norm / scale,
        "gradient_norm": grad_norm,
        "orthogonality_error": float(np.linalg.norm(Y.T @ Y - np.eye(m))),
        "max_preconditioner_cond": max_cond_bound,
        "preconditioner_fallbacks": precond_fallbacks,
        "max_leverage": info["max_leverage"],
        "clipped_fraction": info["clipped_fraction"],
        "active_centers": info["active_centers"],
        "nnz_weights": info["nnz_weights"],
    }
    return HPAOResult(
        index=Y.T.copy(),
        coefficients=full_coefficients,
        diagnostics=diagnostics,
    )
