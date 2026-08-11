import math

import numpy as np

from ...ADP_Solver import ADP_SolverResult
from ...ADP_Statistic import ADP_Statistics


def solve(
    statistics: ADP_Statistics,
    beta: np.ndarray,
    *,
    lambda_penalty: float,
    local_ridge: float,
    max_steps: int = 500,
    tol: float = 1e-6,
    cautious_c: float = 1e-4,
    wolfe_c1: float = 1e-4,
    wolfe_c2: float = 0.9,
    line_search_steps: int = 50,
    zoom_steps: int = 60,
    max_angle: float = math.pi / 2.0,
    denominator_floor: float = 1e-14,
    stall_patience: int = 3,
) -> ADP_SolverResult:
    """Solve the single-index ADP block by VarPro + full Riemannian BFGS.

    The optimized fixed objective on the unit sphere is

        1/2 sum_j ||I_j - f_j U_j beta||^2
        + local_ridge / 2 sum_j f_j^2
        + lambda_penalty / 2 ||beta - beta_ref||^2,

    where beta_ref is the normalized input ``beta`` and every f_j is eliminated
    exactly for the current beta. Setting both penalties to zero recovers the
    unregularized profiled ADP objective, provided all ||U_j beta|| are nonzero.

    ``max_steps`` is the number of Riemannian BFGS iterations. The method stores
    a dense d x d inverse-Hessian approximation, hence O(d^2) extra memory.

    Near a well-conditioned minimizer the true remaining decrease in the
    objective eventually drops below what float64 can represent (the gradient
    is still informative there, but ``phi(alpha)`` stops changing to machine
    precision). Without a stopping rule for that regime, both line searches
    keep re-probing a flat, noise-dominated objective and can burn their full
    evaluation budget every remaining iteration for zero benefit. ``stall_patience``
    bounds how many consecutive iterations are allowed to make no
    numerically meaningful progress (judged on the objective value, which is
    what the line search actually controls) before the solver gives up and
    reports ``diagnostics["stall_detected"] = True`` rather than spinning
    through the rest of ``max_steps``. This never masks a real failure to
    reach ``tol``: ``diagnostics["converged"]`` still reflects the actual
    gradient-norm test.
    """
    _validate_settings(
        max_steps=max_steps,
        tol=tol,
        lambda_penalty=lambda_penalty,
        local_ridge=local_ridge,
        cautious_c=cautious_c,
        wolfe_c1=wolfe_c1,
        wolfe_c2=wolfe_c2,
        line_search_steps=line_search_steps,
        zoom_steps=zoom_steps,
        max_angle=max_angle,
        denominator_floor=denominator_floor,
        stall_patience=stall_patience,
    )

    I, U_flat, beta = _prepare_problem(statistics, beta)
    beta_ref = beta.copy()
    d = beta.size

    inverse_hessian = np.eye(d, dtype=float)
    value, grad, slopes, min_denominator = _value_grad(
        I,
        U_flat,
        beta,
        beta_ref,
        lambda_penalty=lambda_penalty,
        local_ridge=local_ridge,
        denominator_floor=denominator_floor,
    )

    if not np.isfinite(value) or grad is None:
        raise RuntimeError(
            "invalid VarPro objective at the initial index: "
            f"min_j ||U_j beta||^2 + local_ridge = {min_denominator!r} does not "
            f"clear denominator_floor = {denominator_floor!r}; the starting "
            "beta is (numerically) orthogonal to the row space of some U_j, "
            "so f_j*(beta) is unbounded there. Start from a different beta, "
            "increase local_ridge, or raise denominator_floor if that "
            "degeneracy is expected for this dataset."
        )

    initial_grad_norm = float(np.linalg.norm(grad))
    grad_scale = max(1.0, initial_grad_norm)
    beta_delta = 0.0
    updates = 0
    skipped_updates = 0
    hessian_resets = 0
    line_search_evaluations = 1
    line_search_fallbacks = 0
    converged = initial_grad_norm <= tol * grad_scale
    accepted_steps = 0
    stall_streak = 0
    stall_detected = False

    for _step in range(max_steps):
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm <= tol * grad_scale:
            converged = True
            break

        direction = -(inverse_hessian @ grad)
        direction = _project(beta, direction)
        directional_derivative = float(np.dot(grad, direction))

        # A positive-definite tangent inverse Hessian must give descent. If
        # roundoff or a bad update destroys this property, restart from I.
        descent_floor = 1e-14 * max(1.0, grad_norm * grad_norm)
        if (
            not np.all(np.isfinite(direction))
            or not np.isfinite(directional_derivative)
            or directional_derivative >= -descent_floor
        ):
            inverse_hessian = np.eye(d, dtype=float)
            direction = -grad
            directional_derivative = -grad_norm * grad_norm
            hessian_resets += 1

        direction_norm = float(np.linalg.norm(direction))
        if not np.isfinite(direction_norm) or direction_norm == 0.0:
            converged = grad_norm <= tol * grad_scale
            break

        def objective_grad(candidate: np.ndarray):
            return _value_grad(
                I,
                U_flat,
                candidate,
                beta_ref,
                lambda_penalty=lambda_penalty,
                local_ridge=local_ridge,
                denominator_floor=denominator_floor,
            )

        search = _strong_wolfe_line_search(
            beta,
            value,
            grad,
            direction,
            objective_grad,
            c1=wolfe_c1,
            c2=wolfe_c2,
            max_steps=line_search_steps,
            zoom_steps=zoom_steps,
            max_angle=max_angle,
        )
        line_search_evaluations += search[5]

        if search[0] is None:
            # Strong Wolfe can fail because the admissible geodesic interval is
            # bounded or because a nearly singular unregularized profile is
            # encountered. Fall back to monotone Armijo rather than accepting
            # an uncontrolled step.
            search = _armijo_line_search(
                beta,
                value,
                grad,
                direction,
                objective_grad,
                c1=wolfe_c1,
                max_steps=line_search_steps + zoom_steps,
                max_angle=max_angle,
            )
            line_search_evaluations += search[5]
            line_search_fallbacks += 1

        alpha, beta_new, value_new, grad_new, slopes_new, _, min_den_new = search
        if alpha is None:
            break

        step = alpha * direction
        theta = float(np.linalg.norm(step))
        if theta == 0.0 or not np.isfinite(theta):
            break

        # Both line searches only guarantee phi(alpha) <= phi(0) + c1 alpha
        # phi'(0); very close to a minimizer that inequality can be satisfied
        # by a candidate whose objective value is bit-identical to `value` in
        # float64, i.e. a step that is "accepted" but moves nowhere in any
        # representable sense. Counting these lets the loop recognize that it
        # has exhausted the precision the objective can offer, instead of
        # repeatedly re-running full line searches against a flat, noise-only
        # landscape until max_steps is exhausted.
        if (value - value_new) <= _progress_floor(value):
            stall_streak += 1
        else:
            stall_streak = 0

        unit_direction = direction / direction_norm

        # Transport H_k, g_k and the accepted step into T_{beta_new} S^{d-1}.
        transported_hessian = _transport_inverse_hessian(
            inverse_hessian,
            beta,
            unit_direction,
            theta,
        )
        transported_grad = _parallel_transport(
            beta,
            unit_direction,
            theta,
            grad,
        )
        transported_step = _parallel_transport(
            beta,
            unit_direction,
            theta,
            step,
        )

        y = _project(beta_new, grad_new - transported_grad)
        s = _project(beta_new, transported_step)
        sy = float(np.dot(s, y))
        ss = float(np.dot(s, s))

        # Cautious Riemannian BFGS update. The threshold tends to zero with
        # ||grad||, preserving the local BFGS behavior while skipping unsafe
        # curvature pairs in the nonconvex region.
        if (
            ss > 0.0
            and np.isfinite(sy)
            and sy > 0.0
            and sy / ss >= cautious_c * grad_norm
        ):
            inverse_hessian = _inverse_bfgs_update(
                transported_hessian,
                s,
                y,
            )
            updates += 1
        else:
            inverse_hessian = transported_hessian
            skipped_updates += 1

        # Force the ambient representation back to a tangent operator. This is
        # an O(d^2) rank-structured correction, not a dense P H P product.
        inverse_hessian = _clean_inverse_hessian(inverse_hessian, beta_new)

        beta_delta = min(
            float(np.linalg.norm(beta_new - beta)),
            float(np.linalg.norm(beta_new + beta)),
        )
        beta = beta_new
        value = value_new
        grad = grad_new
        slopes = slopes_new
        min_denominator = min_den_new
        accepted_steps += 1

        if stall_streak >= stall_patience:
            stall_detected = True
            break

    final_grad_norm = float(np.linalg.norm(grad))
    if final_grad_norm <= tol * grad_scale:
        converged = True

    # Without the directional prior the profiled problem is sign-invariant.
    # Align the returned representative with the input index for compatibility
    # with the existing solver interface.
    if lambda_penalty == 0.0 and np.dot(beta, beta_ref) < 0.0:
        beta = -beta
        slopes = -slopes

    return ADP_SolverResult(
        index=beta,
        coefficients=slopes,
        diagnostics={
            "inner_iterations": accepted_steps,
            "varpro_iterations": accepted_steps,
            "objective": float(value),
            "grad_norm": final_grad_norm,
            "initial_grad_norm": initial_grad_norm,
            "beta_delta": float(beta_delta),
            "bfgs_updates": updates,
            "bfgs_skipped_updates": skipped_updates,
            "bfgs_resets": hessian_resets,
            "line_search_evaluations": line_search_evaluations,
            "line_search_fallbacks": line_search_fallbacks,
            "min_profile_denominator": float(min_denominator),
            "converged": converged,
            "stall_detected": stall_detected,
        },
    )


def _progress_floor(value: float) -> float:
    """Smallest objective decrease that is not just float64 rounding noise."""
    return 16.0 * np.finfo(float).eps * max(1.0, abs(value))


def _validate_settings(
    *,
    max_steps,
    tol,
    lambda_penalty,
    local_ridge,
    cautious_c,
    wolfe_c1,
    wolfe_c2,
    line_search_steps,
    zoom_steps,
    max_angle,
    denominator_floor,
    stall_patience,
):
    for name, value in (
        ("max_steps", max_steps),
        ("line_search_steps", line_search_steps),
        ("zoom_steps", zoom_steps),
        ("stall_patience", stall_patience),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be positive")

    for name, value in (
        ("tol", tol),
        ("max_angle", max_angle),
        ("denominator_floor", denominator_floor),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    for name, value in (
        ("lambda_penalty", lambda_penalty),
        ("local_ridge", local_ridge),
        ("cautious_c", cautious_c),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")

    if not (0.0 < wolfe_c1 < wolfe_c2 < 1.0):
        raise ValueError("Wolfe constants must satisfy 0 < c1 < c2 < 1")
    if max_angle >= math.pi:
        raise ValueError("max_angle must be smaller than pi")


def _prepare_problem(statistics: ADP_Statistics, beta: np.ndarray):
    # Force float64 regardless of what statistics happens to carry. The
    # stall/convergence tests below are calibrated against float64 epsilon;
    # silently running the reductions in float32 would move the achievable
    # noise floor by ~9 orders of magnitude without changing anything else,
    # which is exactly the kind of mixed-precision behavior that belongs to
    # an explicit, separately validated mode rather than an implicit dtype
    # pass-through.
    I = np.asarray(statistics.I, dtype=float)
    U = np.asarray(statistics.U, dtype=float)
    beta = np.asarray(beta, dtype=float).copy()

    if I.ndim != 2:
        raise ValueError("statistics.I must have shape (J, P)")
    if U.ndim != 3:
        raise ValueError("statistics.U must have shape (J, P, d)")
    if U.shape[:2] != I.shape:
        raise ValueError("statistics.I and statistics.U dimensions do not match")
    if beta.shape != (U.shape[2],):
        raise ValueError("beta must have shape (d,)")
    if not np.all(np.isfinite(I)) or not np.all(np.isfinite(U)):
        raise ValueError("statistics must be finite")
    if not np.all(np.isfinite(beta)):
        raise ValueError("beta must be finite")

    norm = float(np.linalg.norm(beta))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("beta must have a finite nonzero norm")
    beta /= norm

    # Every downstream evaluation only ever needs U through the two
    # contractions folded into _value_grad; flattening once, up front, turns
    # both into a single GEMM/GEMV against a plain (J*P, d) matrix instead of
    # repeating a 3-tensor contraction on every one of the (possibly
    # thousands of) line-search trials taken over the run.
    J, P, d = U.shape
    U_flat = np.ascontiguousarray(U).reshape(J * P, d)
    return I, U_flat, beta


def _value_grad(
    I: np.ndarray,
    U_flat: np.ndarray,
    beta: np.ndarray,
    beta_ref: np.ndarray,
    *,
    lambda_penalty: float,
    local_ridge: float,
    denominator_floor: float,
):
    """Evaluate the VarPro objective and its Riemannian gradient at ``beta``.

    ``U_flat`` is ``statistics.U`` reshaped to ``(J * P, d)`` (see
    ``_prepare_problem``). Every evaluation here is called many times per
    outer BFGS step (once per line-search trial), so both O(JPd)
    contractions are written as a single GEMV against ``U_flat`` rather than
    an einsum over the 3-index tensor: a batched (J, P, d) x (d,) contraction
    and a (J, P, d) x (J, P) contraction do not reliably lower to one BLAS
    call the way a plain (JP, d) matrix does, and profiling this at the
    project's own target scale (J=2000, P=32, d=1000) showed the einsum form
    measurably slower for no numerical benefit.
    """
    J, P = I.shape
    projected = (U_flat @ beta).reshape(J, P)
    q = np.einsum("jp,jp->j", projected, projected, optimize=True)
    denominator = q + local_ridge
    min_denominator = float(np.min(denominator))

    if not np.all(np.isfinite(denominator)) or min_denominator <= denominator_floor:
        return math.inf, None, None, min_denominator

    c = np.einsum("jp,jp->j", I, projected, optimize=True)
    slopes = c / denominator
    residual = I - slopes[:, None] * projected

    value = 0.5 * float(np.sum(residual * residual))
    if local_ridge:
        value += 0.5 * local_ridge * float(np.dot(slopes, slopes))

    difference = beta - beta_ref
    if lambda_penalty:
        value += 0.5 * lambda_penalty * float(np.dot(difference, difference))

    weighted = (slopes[:, None] * residual).reshape(J * P)
    grad = -(U_flat.T @ weighted)
    if lambda_penalty:
        grad = grad + lambda_penalty * difference

    grad = _project(beta, grad)
    if not np.isfinite(value) or not np.all(np.isfinite(grad)):
        return math.inf, None, None, min_denominator
    return value, grad, slopes, min_denominator


def _project(beta: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return vector - beta * np.dot(beta, vector)


def _exp_map(beta: np.ndarray, direction: np.ndarray, alpha: float):
    norm = float(np.linalg.norm(direction))
    if norm == 0.0 or alpha == 0.0:
        return beta.copy(), None, 0.0

    unit = direction / norm
    theta = alpha * norm
    candidate = math.cos(theta) * beta + math.sin(theta) * unit

    # The formula is norm-preserving in exact arithmetic. Normalize only as a
    # finite-precision cleanup so downstream projections remain well scaled.
    candidate_norm = float(np.linalg.norm(candidate))
    if not np.isfinite(candidate_norm) or candidate_norm == 0.0:
        raise RuntimeError("exponential map produced an invalid index")
    candidate /= candidate_norm
    return candidate, unit, theta


def _parallel_transport(
    beta: np.ndarray,
    unit_direction: np.ndarray,
    theta: float,
    vector: np.ndarray,
) -> np.ndarray:
    coefficient = float(np.dot(unit_direction, vector))
    transported = vector + coefficient * (
        -math.sin(theta) * beta + (math.cos(theta) - 1.0) * unit_direction
    )
    return transported


def _transport_inverse_hessian(
    inverse_hessian: np.ndarray,
    beta: np.ndarray,
    unit_direction: np.ndarray,
    theta: float,
) -> np.ndarray:
    """Compute Q H Q^T for sphere parallel transport in O(d^2)."""
    c = math.cos(theta)
    s = math.sin(theta)
    V = np.column_stack((beta, unit_direction))
    A = np.array(((c - 1.0, -s), (s, c - 1.0)), dtype=float)

    vh = V.T @ inverse_hessian
    qh = inverse_hessian + V @ (A @ vh)
    transported = qh + (qh @ V) @ A.T @ V.T
    return 0.5 * (transported + transported.T)


def _inverse_bfgs_update(
    inverse_hessian: np.ndarray,
    s: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    sy = float(np.dot(s, y))
    hy = inverse_hessian @ y
    yhy = float(np.dot(y, hy))

    updated = (
        inverse_hessian
        - (np.outer(hy, s) + np.outer(s, hy)) / sy
        + ((sy + yhy) / (sy * sy)) * np.outer(s, s)
    )
    return 0.5 * (updated + updated.T)


def _clean_inverse_hessian(
    inverse_hessian: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Apply P H P + beta beta^T without dense matrix-matrix products."""
    hb = inverse_hessian @ beta
    bhb = float(np.dot(beta, hb))
    normal = np.outer(beta, beta)
    cleaned = (
        inverse_hessian
        - np.outer(beta, hb)
        - np.outer(hb, beta)
        + bhb * normal
        + normal
    )
    return 0.5 * (cleaned + cleaned.T)


def _directional_derivative(
    beta: np.ndarray,
    direction: np.ndarray,
    alpha: float,
    grad: np.ndarray,
) -> float:
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm == 0.0:
        return 0.0
    unit = direction / direction_norm
    theta = alpha * direction_norm
    transported_direction = _parallel_transport(
        beta,
        unit,
        theta,
        direction,
    )
    return float(np.dot(grad, transported_direction))


def _strong_wolfe_line_search(
    beta,
    value,
    grad,
    direction,
    objective_grad,
    *,
    c1,
    c2,
    max_steps,
    zoom_steps,
    max_angle,
):
    phi0 = float(value)
    derphi0 = float(np.dot(grad, direction))
    direction_norm = float(np.linalg.norm(direction))
    alpha_max = max_angle / direction_norm
    alpha = min(1.0, alpha_max)
    alpha_prev = 0.0
    phi_prev = phi0
    evaluations = 0

    if derphi0 >= 0.0 or alpha <= 0.0:
        return (None, None, None, None, None, evaluations, math.inf)

    for step_index in range(max_steps):
        candidate, _, _ = _exp_map(beta, direction, alpha)
        phi, candidate_grad, slopes, min_den = objective_grad(candidate)
        evaluations += 1

        if (
            not np.isfinite(phi)
            or candidate_grad is None
            or phi > phi0 + c1 * alpha * derphi0
            or (step_index > 0 and phi >= phi_prev)
        ):
            return _zoom(
                beta,
                value,
                grad,
                direction,
                objective_grad,
                alpha_prev,
                alpha,
                phi_prev,
                c1=c1,
                c2=c2,
                max_steps=zoom_steps,
                evaluations=evaluations,
            )

        derphi = _directional_derivative(
            beta,
            direction,
            alpha,
            candidate_grad,
        )
        if abs(derphi) <= -c2 * derphi0:
            return (
                alpha,
                candidate,
                phi,
                candidate_grad,
                slopes,
                evaluations,
                min_den,
            )

        if derphi >= 0.0:
            return _zoom(
                beta,
                value,
                grad,
                direction,
                objective_grad,
                alpha,
                alpha_prev,
                phi,
                c1=c1,
                c2=c2,
                max_steps=zoom_steps,
                evaluations=evaluations,
            )

        alpha_prev = alpha
        phi_prev = phi
        if alpha >= alpha_max:
            break
        alpha = min(2.0 * alpha, alpha_max)

    return (None, None, None, None, None, evaluations, math.inf)


def _zoom(
    beta,
    value,
    grad,
    direction,
    objective_grad,
    alpha_lo,
    alpha_hi,
    phi_lo,
    *,
    c1,
    c2,
    max_steps,
    evaluations,
):
    phi0 = float(value)
    derphi0 = float(np.dot(grad, direction))

    for _ in range(max_steps):
        alpha = 0.5 * (alpha_lo + alpha_hi)
        if alpha <= min(alpha_lo, alpha_hi) or alpha >= max(alpha_lo, alpha_hi):
            # The bracket has collapsed to float64 resolution: alpha_lo and
            # alpha_hi are adjacent representable numbers (or equal), so the
            # midpoint rounds back to an endpoint and bisection can no longer
            # narrow the interval. Continuing would just re-evaluate the same
            # one or two candidates for the rest of the budget.
            break
        candidate, _, _ = _exp_map(beta, direction, alpha)
        phi, candidate_grad, slopes, min_den = objective_grad(candidate)
        evaluations += 1

        if (
            not np.isfinite(phi)
            or candidate_grad is None
            or phi > phi0 + c1 * alpha * derphi0
            or phi >= phi_lo
        ):
            alpha_hi = alpha
            continue

        derphi = _directional_derivative(
            beta,
            direction,
            alpha,
            candidate_grad,
        )
        if abs(derphi) <= -c2 * derphi0:
            return (
                alpha,
                candidate,
                phi,
                candidate_grad,
                slopes,
                evaluations,
                min_den,
            )

        if derphi * (alpha_hi - alpha_lo) >= 0.0:
            alpha_hi = alpha_lo

        alpha_lo = alpha
        phi_lo = phi

    return (None, None, None, None, None, evaluations, math.inf)


def _armijo_line_search(
    beta,
    value,
    grad,
    direction,
    objective_grad,
    *,
    c1,
    max_steps,
    max_angle,
):
    derphi0 = float(np.dot(grad, direction))
    direction_norm = float(np.linalg.norm(direction))
    alpha = min(1.0, max_angle / direction_norm)
    evaluations = 0

    for _ in range(max_steps):
        candidate, _, _ = _exp_map(beta, direction, alpha)
        phi, candidate_grad, slopes, min_den = objective_grad(candidate)
        evaluations += 1
        if (
            np.isfinite(phi)
            and candidate_grad is not None
            and phi <= value + c1 * alpha * derphi0
        ):
            return (
                alpha,
                candidate,
                phi,
                candidate_grad,
                slopes,
                evaluations,
                min_den,
            )
        alpha *= 0.5

    return (None, None, None, None, None, evaluations, math.inf)
