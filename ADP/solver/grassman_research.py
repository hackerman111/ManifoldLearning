"""Opt-in Grassmann: Schur GN/Newton, метрический спуск и recycling.

Сохраняет finite-sketch профиль grassman.py; bounded optimizer APPROXIMATE.
Newton точен внутри малого q*m чарта, не полного касательного пространства.
"""

# ruff: noqa: RUF002

from __future__ import annotations

import numpy as np

from ..engine.common.utils import _finite_real_array
from . import grassman as base
from ._grassman_schur import SchurCore, metric_direction, trust_step
from .LSMR import HPAOResult, _normalize_index
from .statistic_operator import DenseOperator, FactorizedOperator


def _newton_core(
    core: SchurCore,
    limit: float,
    damping: float,
    steps: int,
    precondition: bool,
) -> tuple[np.ndarray, int, int]:
    """Trust-region в фиксированном чарте; indefinite HVP сохраняется."""
    K = np.zeros(core.shape)
    value, grad = core.evaluate(K)
    radius, iterations, negative = limit, 0, 0
    for _ in range(steps):
        if np.linalg.norm(grad) <= 1e-12 * max(1.0, abs(value)):
            break
        apply = core.precondition(K, damping) if precondition else lambda x: x

        def apply_hvp(E, current_K=K):
            return core.hvp(current_K, E)

        delta, count, indefinite = trust_step(grad, apply_hvp, apply, radius, tol=1e-5)
        iterations += count
        negative += int(indefinite)
        trial_K = K + delta
        length = np.linalg.norm(trial_K, ord=2)
        if length > limit:
            trial_K *= limit / length
        delta = trial_K - K
        predicted = -float(
            np.sum(grad * delta) + 0.5 * np.sum(delta * core.hvp(K, delta))
        )
        if predicted <= 0:
            radius *= 0.25
            continue
        try:
            trial_value, trial_grad = core.evaluate(trial_K)
        except ValueError:
            radius *= 0.25
            continue
        ratio = (value - trial_value) / predicted
        if ratio > 0.1:
            K, value, grad = trial_K, trial_value, trial_grad
        if ratio < 0.25:
            radius *= 0.25
        elif ratio > 0.75 and np.linalg.norm(delta) >= 0.9 * radius:
            radius = min(limit, 2 * radius)
    return K, iterations, negative


def solve(
    index_init: np.ndarray,
    U: np.ndarray | DenseOperator | FactorizedOperator,
    I: np.ndarray,
    *,
    mass: np.ndarray | None = None,
    lambda_prox: float = 0.05,
    max_steps: int = 5,
    tol: float = 1e-6,
    rank: int | None = None,
    local_ridge: float = 0.0,
    max_angle: float = 0.5,
    method: str = "schur_gn",
    precondition: bool = True,
    recycle_steps: int = 0,
    core_steps: int = 4,
    metric_floor: float = 1e-6,
    workspace_bytes: int = 16 * 1024**2,
) -> HPAOResult:
    """Решить frozen задачу; последний сертификат всегда по полному U.

    Recycling ограничен одной фиксированной статистикой и оболочкой <=2m.
    Положительный local_ridge меняет estimator; damping — только шаг.
    Прежний ADP_solver custom callback принимает этот entry point с dense U.
    CX передаётся явно через factorized_statistics, не материализуя U.
    """
    if hasattr(U, "__cuda_array_interface__"):
        raise NotImplementedError("research Grassmann solver requires CPU statistics")
    operator = (
        U if isinstance(U, (DenseOperator, FactorizedOperator)) else DenseOperator(U)
    )
    B, I = _finite_real_array(index_init, "index_init"), _finite_real_array(I, "I")
    J, p, d = operator.shape
    if B.ndim != 2 or B.shape[1] != d or not 1 <= B.shape[0] <= d:
        raise ValueError("initial index must be a multi-index matrix (m,d)")
    m = B.shape[0]
    if I.shape != (J, p):
        raise ValueError("I must have shape (J,p) matching U")
    mass = np.ones(J) if mass is None else _finite_real_array(mass, "mass")
    if mass.shape != (J,) or np.any(mass <= 0):
        raise ValueError("mass must contain J finite positive values")
    if method not in ("schur_gn", "schur_newton", "metric"):
        raise ValueError("method must be schur_gn, schur_newton or metric")
    for name, value in (
        ("max_steps", max_steps),
        ("core_steps", core_steps),
        ("recycle_steps", recycle_steps),
        ("workspace_bytes", workspace_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if value < (0 if name == "recycle_steps" else 1):
            raise ValueError(f"{name} is out of range")
    if not isinstance(precondition, bool):
        raise TypeError("precondition must be boolean")
    for name, value in (
        ("lambda_prox", lambda_prox),
        ("tol", tol),
        ("max_angle", max_angle),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if max_angle >= np.pi / 2:
        raise ValueError("max_angle must be less than pi/2")
    if not np.isfinite(local_ridge) or local_ridge < 0:
        raise ValueError("local_ridge must be finite and nonnegative")
    if not np.isfinite(metric_floor) or not 0 < metric_floor <= 1:
        raise ValueError("metric_floor must be in (0,1]")
    if rank is not None and (
        isinstance(rank, bool)
        or not isinstance(rank, (int, np.integer))
        or not 1 <= rank <= min(m, d - m)
    ):
        raise ValueError("rank must be in [1,min(m,d-m)]")
    Y = _normalize_index(B).T
    projects, adjoints, full_gradients = 0, 0, 0

    def project(V: np.ndarray) -> np.ndarray:
        nonlocal projects
        projects += 1
        result = operator.project(V)
        if not np.all(np.isfinite(result)):
            raise RuntimeError("non-finite operator projection")
        return result

    def gradient(
        state: base._Profile, shell: np.ndarray | None, images: np.ndarray | None
    ):
        nonlocal adjoints, full_gradients
        packed = (
            -2
            * mass[:, None, None]
            * state.residual[:, :, None]
            * state.coefficients[:, None]
        )
        if shell is None:
            adjoints += 1
            full_gradients += 1
            G = operator.adjoint(packed)
        else:
            assert images is not None
            G = shell @ (images.reshape(-1, shell.shape[1]).T @ packed.reshape(-1, m))
        if not np.all(np.isfinite(G)):
            raise RuntimeError("non-finite operator gradient")
        return G - Y @ (Y.T @ G)

    state = base._profile(project(Y), I, mass, local_ridge)
    history = [state.value]
    scale = max(1.0, float(np.sum(mass[:, None] * I**2)))
    limit = float(np.tan(max_angle))
    shell = images = None
    age = 0
    accepted = recycled = newton_iterations = negatives = rank_guards = (
        memory_guards
    ) = 0
    evaluations = gn_solves = 0
    stop = "max_steps"
    for _ in range(max_steps):
        reuse = shell is not None and age < recycle_steps
        G = gradient(state, shell if reuse else None, images if reuse else None)
        if reuse and np.linalg.norm(G) <= tol * scale:
            G = gradient(state, None, None)
            reuse = False
        if not state.smooth:
            rank_guards += 1
        if np.linalg.norm(G) <= tol * scale or m == d:
            stop = (
                "stationary" if state.smooth or m == d else "rank_boundary_stationary"
            )
            break
        left, singular, _ = np.linalg.svd(G, full_matrices=False)
        available = min(
            m,
            d - m,
            int(
                np.count_nonzero(
                    singular > np.finfo(float).eps * max(d, m) * singular[0]
                )
            ),
        )
        q = min(rank or available, available)
        if q == 0:
            stop = "no_descent_direction"
            break
        current = method if state.smooth else "metric"
        estimate = 8 * (4 * J * (p + m) * q * m + J * (m + q + 1) ** 2)
        if current != "metric" and estimate > workspace_bytes:
            current = "metric"
            memory_guards += 1
        Z = (
            metric_direction(G, state.coefficients, mass, q, metric_floor)
            if current == "metric"
            else None
        )
        V = (
            left[:, :q]
            if Z is None
            else np.linalg.svd(Z, full_matrices=False)[0][:, :q]
        )
        V = np.linalg.qr(V - Y @ (Y.T @ V), mode="reduced")[0]
        if reuse:
            assert images is not None and shell is not None
            W = images @ (shell.T @ V)
            age += 1
        else:
            W = project(V)
            shell = np.column_stack((Y, V))
            images = np.concatenate((state.M, W), axis=2)
            age = 0
        reduced = V.T @ G
        core = None
        if current != "metric":
            try:
                core = SchurCore(state.M, W, I, mass, local_ridge)
                core.evaluate(np.zeros((q, m)))
            except ValueError:
                current = "metric"
                core = None
                rank_guards += 1
        if current == "schur_newton":
            assert core is not None
            K, count, negative = _newton_core(
                core, limit, lambda_prox, core_steps, precondition
            )
            newton_iterations += count
            negatives += negative
        elif current == "schur_gn":
            jac, residual = base._core_jacobian(state, W, mass, local_ridge)
            design = np.vstack((jac, np.sqrt(lambda_prox) * np.eye(q * m)))
            rhs = np.concatenate((-residual, np.zeros(q * m)))
            K = np.linalg.lstsq(design, rhs, rcond=None)[0].reshape(q, m)
            gn_solves += 1
        else:
            # При rank-guard сохраняем уже спроецированную оболочку V.
            direction = metric_direction(
                reduced, state.coefficients, mass, q, metric_floor
            )
            K = direction
        length = np.linalg.norm(K, ord=2)
        if length > limit:
            K *= limit / length
        slope = float(np.sum(reduced * K))
        if slope >= 0 or length == 0:
            K = -reduced
            K *= limit / max(float(np.linalg.norm(K, ord=2)), np.finfo(float).tiny)
            slope = float(np.sum(reduced * K))
        candidate = None
        for backtrack in range(20):
            step = K * 0.5**backtrack
            threshold = state.value + 1e-4 * slope * 0.5**backtrack
            if core is not None:
                try:
                    value, _ = core.evaluate(step)
                    evaluations += 1
                    if value > threshold:
                        continue
                except ValueError:
                    continue
            Yn, Mn = base._polar(Y, state.M, V, W, step)
            trial = base._profile(Mn, I, mass, local_ridge)
            evaluations += 1
            if trial.value <= threshold:
                candidate = Yn, trial
                break
        if candidate is None:
            stop = "line_search_failed"
            break
        Y, state = candidate
        accepted += 1
        recycled += int(reuse)
        history.append(state.value)
    # Накопленные caches не используются для финальной сертификации.
    state = base._profile(project(Y), I, mass, local_ridge)
    norm = float(np.linalg.norm(gradient(state, None, None)))
    converged = bool(norm <= tol * scale and (state.smooth or m == d))
    if converged:
        stop = "stationary"
    diagnostics: dict[str, object] = {
        "linear_solver": "grassman_research",
        "method": method,
        "profile_objective": state.value,
        "profile_history": tuple(history),
        "loss": 0.5 * float(np.sum(mass[:, None] * state.residual**2)),
        "loss_history": tuple(v / 2 for v in history),
        "gradient_norm": norm,
        "riemannian_gradient": norm / scale,
        "converged": converged,
        "stop_reason": stop,
        "accepted_steps": accepted,
        "iterations": accepted,
        "orthogonality_error": float(np.linalg.norm(Y.T @ Y - np.eye(m))),
        "stationarity_applicable": state.smooth or m == d,
        "local_rank_loss": int(np.count_nonzero(state.ranks < m)),
        "operator_projects": projects,
        "operator_adjoints": adjoints,
        "full_gradient_evaluations": full_gradients,
        "recycled_steps": recycled,
        "newton_iterations": newton_iterations,
        "negative_curvature_steps": negatives,
        "rank_guard_fallbacks": rank_guards,
        "workspace_fallbacks": memory_guards,
        "gn_solves": gn_solves,
        "profile_evaluations": evaluations,
        "precondition": precondition,
        "local_ridge": local_ridge,
    }
    return HPAOResult(Y.T, state.coefficients, diagnostics)
