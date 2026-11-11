"""Shared polynomial regression on a Grassmann subspace; experimental only.

Unlike independent local linear fits, one nonlinear link uses every observation.
The Schur Gauss--Newton step eliminates polynomial-coefficient increments; it
is not the exact residual-Jacobian variable-projection GN away from zero residual.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial
from itertools import combinations_with_replacement

import numpy as np
from scipy.linalg import cho_factor, cho_solve, qr, solve_triangular, svdvals


def powers(m, degree):
    rows = []
    for order in range(degree + 1):
        for indices in combinations_with_replacement(range(m), order):
            rows.append(np.bincount(indices, minlength=m))
    return np.asarray(rows, dtype=int)


def features(z, exponents):
    """Complete normalized probabilists' Hermite basis in specified order."""
    n, m = z.shape
    degree = int(exponents.max())
    table = np.empty((degree + 1, n, m))
    table[0] = 1
    if degree:
        table[1] = z
    for k in range(1, degree):
        table[k + 1] = (z * table[k] - np.sqrt(k) * table[k - 1]) / np.sqrt(k + 1)
    design = np.ones((n, len(exponents)))
    for j in range(m):
        design *= table[exponents[:, j], :, j].T
    return design


def polar(matrix):
    u, _, vt = np.linalg.svd(matrix, full_matrices=False)
    return u @ vt


@dataclass
class State:
    value: float
    coefficients: np.ndarray
    prediction: np.ndarray
    residual: np.ndarray
    link_gradient: np.ndarray
    gradient: np.ndarray
    design_q: np.ndarray
    normal_residual: float
    design_condition: float


class PolynomialRankError(ValueError):
    """A coefficient profile is numerically nonregular at this subspace."""


class PolynomialProfile:
    def __init__(self, x, y, degree):
        self.x, self.y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if self.x.ndim != 2 or self.y.shape != (len(self.x),):
            raise ValueError("expected X:(n,d), y:(n,)")
        if not np.isfinite(self.x).all() or not np.isfinite(self.y).all():
            raise ValueError("inputs must be finite")
        if not isinstance(degree, (int, np.integer)) or not 1 <= degree <= 5:
            raise ValueError("degree must be an integer between 1 and 5")
        self.degree, self._powers = degree, None

    def __call__(self, basis):
        basis = np.asarray(basis, dtype=float)
        if (
            basis.ndim != 2
            or basis.shape[0] != self.x.shape[1]
            or not 1 <= basis.shape[1] < basis.shape[0]
            or not np.isfinite(basis).all()
            or not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-10)
        ):
            raise ValueError("basis must have finite orthonormal columns with 1<=m<d")
        m = basis.shape[1]
        if self._powers is None or self._powers.shape[1] != m:
            self._powers = powers(m, self.degree)
            self._derivatives = []
            lookup = {tuple(row): i for i, row in enumerate(self._powers)}
            for j in range(m):
                source = np.flatnonzero(self._powers[:, j])
                lowered = self._powers[source].copy()
                lowered[:, j] -= 1
                target = np.array([lookup[tuple(row)] for row in lowered])
                self._derivatives.append((source, target))
        if len(self.x) <= len(self._powers):
            raise ValueError("more observations than polynomial coefficients required")
        design = features(self.x @ basis, self._powers)
        if not np.isfinite(design).all():
            raise ValueError("polynomial features must be finite")
        q, r = qr(design, mode="economic", check_finite=False)
        singular = svdvals(r, check_finite=False)
        if singular[-1] <= 1e-10 * singular[0]:
            raise PolynomialRankError("polynomial coefficient design is rank deficient")
        coefficients = solve_triangular(r, q.T @ self.y, check_finite=False)
        prediction = design @ coefficients
        residual = prediction - self.y
        derivative_coefficients = np.zeros((len(self._powers), m))
        for j, (source, target) in enumerate(self._derivatives):
            derivative_coefficients[target, j] = (
                np.sqrt(self._powers[source, j]) * coefficients[source]
            )
        link_gradient = design @ derivative_coefficients
        ambient = 2 * self.x.T @ (residual[:, None] * link_gradient) / len(self.x)
        gradient = ambient - basis @ (basis.T @ ambient)
        normal_residual = float(
            np.linalg.norm(design.T @ residual)
            / max(np.linalg.norm(design) * np.linalg.norm(self.y), 1e-300)
        )
        if not np.isfinite(normal_residual) or normal_residual > 1e-10:
            raise RuntimeError("unreliable polynomial coefficient solve")
        return State(
            float(residual @ residual / len(self.x)),
            coefficients,
            prediction,
            residual,
            link_gradient,
            gradient,
            q,
            normal_residual,
            float(singular[0] / singular[-1]),
        )


def schur_system(complement_x, state):
    """Eliminate coefficient increments in the joint nonlinear LS model."""
    n = len(complement_x)
    jacobian = np.einsum("na,nk->nak", complement_x, state.link_gradient).reshape(n, -1)
    jacobian -= state.design_q @ (state.design_q.T @ jacobian)
    hessian = jacobian.T @ jacobian / n
    score = complement_x.T @ (state.residual[:, None] * state.link_gradient) / n
    return hessian, score


@dataclass
class Result:
    basis: np.ndarray
    coefficients: np.ndarray
    diagnostics: dict


def refine(x, y, basis, *, degree=3, max_steps=60, tol=1e-7, geometry="schur"):
    """Refine a data-derived start; neither truth nor generator metadata is used."""
    if geometry not in ("schur", "euclidean"):
        raise ValueError("unknown geometry")
    if not isinstance(max_steps, (int, np.integer)) or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    started = time.perf_counter()
    problem = PolynomialProfile(x, y, degree)
    basis = np.asarray(basis, dtype=float).copy()
    state = problem(basis)
    history, normal_residuals = [state.value], [state.normal_residual]
    evaluations, reason = 1, "iteration_limit"
    d, m = basis.shape
    if len(x) - len(state.coefficients) < m * (d - m):
        raise ValueError(
            "insufficient residual degrees of freedom for tangent information"
        )
    gradient_norm = float(np.linalg.norm(state.gradient))
    for _ in range(max_steps):
        gram = state.link_gradient.T @ state.link_gradient / len(x)
        eigenvalues = np.linalg.eigvalsh(gram)
        gradient_rank_ok = eigenvalues[0] > 1e-10 * max(float(eigenvalues[-1]), 1e-300)
        if not gradient_rank_ok:
            reason = "gradient_rank_deficient"
            break
        gradient_norm = float(np.linalg.norm(state.gradient))
        if gradient_norm <= tol:
            reason = "stationary"
            break
        complement = np.linalg.qr(basis, mode="complete")[0][:, m:]
        hessian, score = schur_system(problem.x @ complement, state)
        tangent_spectrum = np.linalg.eigvalsh(hessian)
        if tangent_spectrum[0] <= 1e-12 * max(float(tangent_spectrum[-1]), 1e-300):
            reason = "tangent_rank_deficient"
            break
        scale = max(float(np.trace(hessian) / len(hessian)), 1e-12)
        if geometry == "schur":
            damped = hessian + 1e-4 * scale * np.eye(len(hessian))
            flat = -cho_solve(cho_factor(damped), score.ravel())
        else:
            flat = -score.ravel() / max(float(np.linalg.eigvalsh(hessian)[-1]), 1e-12)
        delta = complement @ flat.reshape(d - m, m)
        delta /= max(1.0, np.linalg.norm(delta, 2) / 0.5)
        slope = float(np.sum(state.gradient * delta))
        if not np.isfinite(slope) or slope >= 0:
            reason = "no_descent"
            break
        step, accepted = 1.0, False
        for _ in range(20):
            candidate = polar(basis + step * delta)
            evaluations += 1
            try:
                trial = problem(candidate)
            except PolynomialRankError:
                step *= 0.5
                continue
            normal_residuals.append(trial.normal_residual)
            if trial.value <= state.value + 1e-4 * step * slope:
                basis, state, accepted = candidate, trial, True
                history.append(state.value)
                break
            step *= 0.5
        if not accepted:
            reason = "line_search_failed"
            break
    gradient_norm = float(np.linalg.norm(state.gradient))
    eigenvalues = np.linalg.eigvalsh(
        state.link_gradient.T @ state.link_gradient / len(x)
    )
    gradient_rank_ok = bool(
        eigenvalues[0] > 1e-10 * max(float(eigenvalues[-1]), 1e-300)
    )
    complement = np.linalg.qr(basis, mode="complete")[0][:, m:]
    final_hessian, _ = schur_system(problem.x @ complement, state)
    tangent_spectrum = np.linalg.eigvalsh(final_hessian)
    tangent_rank_ok = bool(
        tangent_spectrum[0] > 1e-12 * max(float(tangent_spectrum[-1]), 1e-300)
    )
    if not gradient_rank_ok:
        reason = "gradient_rank_deficient"
    elif not tangent_rank_ok:
        reason = "tangent_rank_deficient"
    elif gradient_norm <= tol:
        reason = "stationary"
    return Result(
        basis,
        state.coefficients,
        {
            "degree": degree,
            "geometry": geometry,
            "history": history,
            "iterations": len(history) - 1,
            "evaluations": evaluations,
            "stop_reason": reason,
            "converged": reason == "stationary",
            "gradient_norm": gradient_norm,
            "gradient_rank_ok": gradient_rank_ok,
            "tangent_rank_ok": tangent_rank_ok,
            "tangent_eigenvalues": tangent_spectrum.tolist(),
            "empirical_gradient_eigenvalues": eigenvalues.tolist(),
            "design_condition": state.design_condition,
            "normal_residual_max": max(normal_residuals),
            "refine_seconds": time.perf_counter() - started,
        },
    )


def initialize(x, y, m, *, seed=42):
    """The existing two-sided ADP fit, with the established study settings."""
    from ADP import ADP_Config
    from ADP.engine.common.index_fit import fit_index
    from experiments.anisotropic_metrics import calculate_statistics, solve
    from experiments.two_sided_meanmave import TimeOnlyProfiler

    n, d = np.shape(x)
    config = ADP_Config(
        seed=seed,
        N_lin=max(50, 2 * d),
        N_J=min(n, max(300, (n + 19) // 20)),
        N_phi=max(m + 1, min(d, 12)),
        N_loc=20,
        index_init="local",
        select_step="best",
    )
    return fit_index(
        x,
        y,
        config=config,
        mode="multi",
        index_dim=m,
        solve=partial(solve, arm="two_sided"),
        statistics_function=calculate_statistics,
        profiler=TimeOnlyProfiler(),
    )


def fit(x, y, m, *, seed=42, degree=3, max_steps=60):
    """Complete experimental estimator, including its ADP initialization cost."""
    started = time.perf_counter()
    initial = initialize(x, y, m, seed=seed)
    result = refine(x, y, initial.index.T, degree=degree, max_steps=max_steps)
    result.diagnostics.update(
        fit_seconds=time.perf_counter() - started,
        initialization_basis=initial.index.T.tolist(),
        initial_adp_basis=initial.initial_index.T.tolist(),
    )
    return result
