"""Ordinary Grassmann metrics from anisotropic_full_moments.pdf; opt-in only.

All arms use the existing full-ridge profile, the same polar retraction and
Armijo search. Damping changes the metric, never the statistical ridge.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator, cg

from ADP.core.solver_common import HPAOResult
from experiments import full_ridge_batched as backend
from experiments.coupled_penalties import Profile

ARMS = ("euclidean", "left", "two_sided", "schur", "varpro", "kernel")
calculate_statistics = backend.calculate_statistics


def horizontal(basis, vector):
    return vector - basis @ (basis.T @ vector)


def polar(matrix):
    gram = matrix.T @ matrix
    values, vectors = np.linalg.eigh((gram + gram.T) / 2)
    if not np.isfinite(values).all() or values[0] <= 0:
        raise ValueError("polar retraction requires full column rank")
    return matrix @ ((vectors * values**-0.5) @ vectors.T)


def dual_metric(tensor):
    """Regularized (T.T T)^-1, with fixed relative floor and unit mean scale."""
    tensor = np.asarray(tensor, dtype=float)
    if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
        raise ValueError("kernel tensor must be square")
    if not np.isfinite(tensor).all() or not np.any(tensor):
        raise ValueError("kernel tensor must be finite and nonzero")
    scaled = tensor / np.max(np.abs(tensor))
    values, vectors = np.linalg.eigh(scaled.T @ scaled)
    inverse = 1 / np.maximum(values, 1e-8 * values[-1])
    inverse /= inverse.mean()
    return (vectors * inverse) @ vectors.T


def localization_tensor(basis, alpha, eigenvalues, tensor):
    """Square root of the actual quadratic used by calculate_multi_weight."""
    basis, eigenvalues = np.asarray(basis), np.asarray(eigenvalues)
    if basis.ndim != 2 or eigenvalues.shape != (basis.shape[1],):
        raise ValueError("invalid localization basis/eigenvalue shapes")
    if tensor not in ("orthogonal", "full"):
        raise ValueError("unknown ADP localization tensor")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("localization alpha must lie in [0,1]")
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues < 0):
        raise ValueError("localization eigenvalues must be finite and nonnegative")
    if not np.isfinite(basis).all() or not np.allclose(
        basis.T @ basis, np.eye(basis.shape[1]), atol=1e-10
    ):
        raise ValueError("localization basis must be orthonormal")
    residual = np.eye(len(basis))
    if tensor == "orthogonal":
        residual -= basis @ basis.T
    quadratic = alpha**2 * residual + (basis * eigenvalues) @ basis.T
    values, vectors = np.linalg.eigh((quadratic + quadratic.T) / 2)
    return (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T


class TangentGeometry:
    """Half-Hessian, Schur GN and exact variable-projection GN operators.

    Coordinates flatten K in C order for Delta=V K. Zero windows have zero
    contribution; nonzero windows require a regular coefficient design at tau=0.
    """

    def __init__(self, basis, problem, state):
        self.basis, self.problem, self.state = basis, problem, state
        d, m = basis.shape
        self.complement = np.linalg.qr(basis, mode="complete")[0][:, m:]
        self.shape = (d - m, m)
        self.design = problem.covariance @ basis
        self.system = basis.T @ self.design + problem.tau * np.eye(m)
        if problem.tau == 0:
            active = np.any(problem.covariance != 0, axis=(1, 2))
            spectrum = np.linalg.eigvalsh(self.system[active])
            if len(spectrum) and np.any(spectrum[:, 0] <= 1e-12 * spectrum[:, -1]):
                raise ValueError("nonregular coefficient rank at zero ridge")
            self.system = self.system.copy()
            self.system[~active] = np.eye(m)
        self.residual = (self.design @ state.coefficients[..., None]).squeeze(
            -1
        ) - problem.cross

    def apply(self, delta, kind):
        p, g = self.problem, self.state.coefficients
        if kind not in ("schur", "varpro", "hessian"):
            raise ValueError(kind)
        z = g @ delta.T
        cz = (p.covariance @ z[..., None]).squeeze(-1)
        t = cz @ self.basis
        q = self.residual @ delta
        rhs = t + q if kind == "hessian" else t
        u = np.linalg.solve(self.system, rhs[..., None]).squeeze(-1)
        local = cz - (self.design @ u[..., None]).squeeze(-1)
        if kind == "hessian":
            local += p.tau * z
        result = local.T @ (p.pi[:, None] * g)
        if kind == "varpro":
            v = np.linalg.solve(self.system, q[..., None]).squeeze(-1)
            result += self.residual.T @ (p.pi[:, None] * v)
        elif kind == "hessian":
            result -= self.residual.T @ (p.pi[:, None] * u)
        return horizontal(self.basis, result)

    def matrix(self, kind):
        v, p, g = self.complement, self.problem, self.state.coefficients
        q, m = self.shape
        if kind == "hessian":
            columns = [
                (v.T @ self.apply(v @ e.reshape(self.shape), kind)).ravel()
                for e in np.eye(q * m)
            ]
            result = np.array(columns).T
        elif kind in ("schur", "varpro"):
            cv = p.covariance @ v
            t = self.basis.T @ cv
            schur = v.T @ cv - t.swapaxes(1, 2) @ np.linalg.solve(self.system, t)
            gram = g[:, :, None] * g[:, None, :]
            result = np.einsum("j,jab,jkl->akbl", p.pi, schur, gram, optimize=True)
            result = result.reshape(q * m, q * m)
            if kind == "varpro":
                r = self.residual @ v
                mapping = np.einsum("ja,kl->jkal", r, np.eye(m)).reshape(-1, m, q * m)
                solved = np.linalg.solve(self.system, mapping)
                result += np.einsum(
                    "j,jak,jal->kl", p.pi, mapping, solved, optimize=True
                )
        else:
            raise ValueError(kind)
        return (result + result.T) / 2

    def solve(self, rhs, kind, *, damping, method="dense"):
        if not np.isfinite(damping) or damping <= 0:
            raise ValueError("metric damping must be positive")
        v, shape = self.complement, self.shape
        packed = (v.T @ rhs).ravel()
        size = len(packed)
        iterations = 0

        def apply(flat):
            return (
                v.T @ self.apply(v @ flat.reshape(shape), kind)
            ).ravel() + damping * flat

        if method == "dense":
            matrix = self.matrix(kind) + damping * np.eye(size)
            solved = cho_solve(
                cho_factor(matrix, check_finite=False), packed, check_finite=False
            )
            residual = matrix @ solved - packed
        elif method == "cg":

            def counted(_):
                nonlocal iterations
                iterations += 1

            solved, status = cg(
                LinearOperator((size, size), matvec=apply, dtype=float),
                packed,
                rtol=1e-10,
                atol=0.0,
                maxiter=max(50, 2 * size),
                callback=counted,
            )
            if status != 0:
                raise RuntimeError(f"metric CG did not converge: {status}")
            residual = apply(solved) - packed
        else:
            raise ValueError(method)
        relative = float(np.linalg.norm(residual) / max(np.linalg.norm(packed), 1e-300))
        if not np.isfinite(relative) or relative > 1e-8:
            raise RuntimeError("unreliable tangent metric solve")
        return v @ solved.reshape(shape), {
            "relative_residual": relative,
            "cg_iterations": iterations,
        }


class Metric:
    """Fixed feature factor, dynamic coefficient Gram, and fixed inner damping."""

    def __init__(self, basis, problem, state, arm, *, metric_tensor=None):
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm, self.problem = arm, problem
        d, m = basis.shape
        average = np.einsum("j,jab->ab", problem.pi, problem.covariance)
        feature_scale = max(float(np.trace(average) / d), 1e-12)
        gram = state.coefficients.T @ (problem.pi[:, None] * state.coefficients)
        index_scale = max(float(np.trace(gram) / m), 1e-12)
        self.right_damping = 0.01 * index_scale
        self.damping = 0.01 * max(feature_scale * index_scale, 1e-12)
        if arm == "kernel":
            if metric_tensor is None:
                raise ValueError("kernel metric requires the current ADP tensor")
            self.feature = dual_metric(metric_tensor)
            if self.feature.shape != (d, d):
                raise ValueError("kernel tensor dimension does not match the basis")
        else:
            self.feature = average + 0.1 * feature_scale * np.eye(d)
        self.factor = (
            cho_factor(self.feature, check_finite=False)
            if arm in ("left", "two_sided", "kernel")
            else None
        )

    def right(self, state):
        g = state.coefficients
        return g.T @ (self.problem.pi[:, None] * g) + self.right_damping * np.eye(
            g.shape[1]
        )

    def direction(self, basis, state):
        gradient = horizontal(basis, state.gradient)
        info = {"relative_residual": 0.0, "cg_iterations": 0}
        if self.arm == "euclidean":
            result = gradient
        elif self.arm in ("left", "two_sided", "kernel"):
            result = horizontal(
                basis, cho_solve(self.factor, gradient, check_finite=False)
            )
            if self.arm == "two_sided":
                result = np.linalg.solve(self.right(state), result.T).T
        else:
            geometry = TangentGeometry(basis, self.problem, state)
            result, info = geometry.solve(gradient, self.arm, damping=self.damping)
        return -result, info


def invariant_profile(basis, problem):
    """Physical Euclidean ridge for a full-rank generalized-Stiefel representative."""
    z, p = np.asarray(basis), problem
    if not np.isfinite(z).all() or np.linalg.matrix_rank(z) != z.shape[1]:
        raise ValueError("representative must be finite and full column rank")
    design = p.covariance @ z + p.tau * z
    blocks, rhs = z.T @ design, p.cross @ z
    coefficients = np.linalg.solve(blocks, rhs[..., None]).squeeze(-1)
    residual = (design @ coefficients[..., None]).squeeze(-1) - p.cross
    value = float(p.energy - np.sum(p.pi[:, None] * rhs * coefficients))
    gradient = 2 * residual.T @ (p.pi[:, None] * coefficients)
    normal = (blocks @ coefficients[..., None]).squeeze(-1) - rhs
    error = float(np.linalg.norm(normal) / max(np.linalg.norm(rhs), 1e-300))
    return Profile(value, gradient, coefficients, {"physical_ridge": True}, 0, error)


def solve(index, stats, arm, *, max_steps=20, tol=1e-6, metric_tensor=None):
    """Common polar-retracted gradient descent with standard-gradient stopping."""
    if arm not in ARMS or not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("invalid metric or iteration budget")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tolerance must be positive")
    started = time.perf_counter()
    basis = np.asarray(index, dtype=float).T.copy()
    if basis.ndim != 2:
        raise ValueError("index must be a row basis")
    d, m = basis.shape
    if (
        not 1 <= m < d
        or not np.isfinite(basis).all()
        or not np.allclose(basis.T @ basis, np.eye(m), atol=1e-10)
    ):
        raise ValueError("index must have orthonormal rows and 1<=m<d")
    p = backend.build(stats, basis)
    state = backend.profile(basis, p)
    metric = Metric(basis, p, state, arm, metric_tensor=metric_tensor)
    preparation = time.perf_counter() - started
    started = time.perf_counter()
    history, gradient_history, steps = [state.value], [], []
    residuals = [state.normal_residual]
    evaluations, metric_seconds, search_seconds = 1, 0.0, 0.0
    stop = "iteration_budget"
    metric_residual = 0.0
    for _ in range(max_steps):
        gradient = horizontal(basis, state.gradient)
        norm = float(np.linalg.norm(gradient) / p.scale)
        gradient_history.append(norm)
        if norm <= tol:
            stop = "stationary"
            break
        tick = time.perf_counter()
        direction, info = metric.direction(basis, state)
        metric_seconds += time.perf_counter() - tick
        metric_residual = max(metric_residual, info["relative_residual"])
        slope = float(np.sum(gradient * direction))
        if not np.isfinite(direction).all() or not np.isfinite(slope) or slope >= 0:
            raise RuntimeError("metric failed the strict descent check")
        step = min(1.0, 1 / max(float(np.linalg.norm(direction)), 1e-300))
        tick = time.perf_counter()
        for _trial in range(20):
            trial = polar(basis + step * direction)
            proposed = backend.profile(trial, p)
            evaluations += 1
            residuals.append(proposed.normal_residual)
            if (
                np.isfinite(proposed.value)
                and proposed.value <= state.value + 1e-4 * step * slope
            ):
                basis, state = trial, proposed
                steps.append(step)
                history.append(state.value)
                break
            step *= 0.5
        else:
            stop = "line_search_failed"
        search_seconds += time.perf_counter() - tick
        if stop == "line_search_failed":
            break
    norm = float(np.linalg.norm(horizontal(basis, state.gradient)) / p.scale)
    if norm <= tol:
        stop = "stationary"
    return HPAOResult(
        basis.T,
        state.coefficients,
        {
            "method": "polar_armijo_metric_gradient",
            "metric": arm,
            "profile_objective": state.value,
            "loss": state.value / 2,
            "profile_history": history,
            "gradient_history": gradient_history,
            "accepted_steps": steps,
            "iterations": len(steps),
            "profile_evaluations": evaluations,
            "converged": norm <= tol,
            "riemannian_gradient": norm,
            "stop_reason": stop,
            "components": state.components,
            "normal_residual_max": max(residuals),
            "metric_residual_max": metric_residual,
            "penalty": {
                "arm": arm,
                "local_ridge": p.tau,
                "prepare_seconds": preparation,
                "solve_seconds": time.perf_counter() - started,
                "metric_seconds": metric_seconds,
                "line_search_seconds": search_seconds,
                "metric_damping": metric.damping,
                "right_damping": metric.right_damping,
                "full_moments": p.info,
            },
        },
    )
