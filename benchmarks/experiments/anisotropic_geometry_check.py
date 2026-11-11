"""Reproduce algebra/finite-difference checks for the anisotropic EDR note.

This is a formula audit, not a recovery experiment or a replacement solver.
Run: python -m experiments.anisotropic_geometry_check
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from experiments.full_ridge_fast import Problem
from experiments.full_ridge_fast import profile as direct_profile


def polar(matrix):
    values, vectors = np.linalg.eigh(matrix.T @ matrix)
    return matrix @ (vectors * values**-0.5) @ vectors.T


def root(matrix):
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T


def local(basis, covariance, target, tau):
    design = root(covariance) @ basis
    system = design.T @ design + tau * np.eye(basis.shape[1])
    coefficients = np.linalg.solve(system, design.T @ target)
    residual = target - design @ coefficients
    value = residual @ residual + tau * (coefficients @ coefficients)
    normal = covariance @ basis @ coefficients - root(covariance) @ target
    return value, coefficients, normal, system


def run():
    rng = np.random.default_rng(20261026)
    checks = {}

    def close(name, actual, expected, *, rtol=2e-7, atol=2e-9):
        actual, expected = np.asarray(actual), np.asarray(expected)
        error = float(np.linalg.norm(actual - expected))
        scale = max(float(np.linalg.norm(expected)), 1.0)
        assert error <= atol + rtol * scale, (name, error, scale)
        checks[name] = {"absolute_error": error, "scaled_error": error / scale}

    d, m, windows = 7, 3, 8
    basis = np.linalg.qr(rng.normal(size=(d, m)))[0]
    complement = np.linalg.qr(basis, mode="complete")[0][:, m:]
    delta = complement @ rng.normal(size=(d - m, m))
    delta /= np.linalg.norm(delta)
    raw = rng.normal(size=(windows, d, d))
    covariance = raw @ raw.swapaxes(1, 2) + np.eye(d)
    targets = rng.normal(size=(windows, d))
    pi = rng.uniform(0.5, 2, size=windows)
    pi /= pi.sum()
    tau = 0.4

    def evaluate(b):
        return sum(
            w * local(b, c, y, tau)[0] for w, c, y in zip(pi, covariance, targets)
        )

    gradient = np.zeros_like(basis)
    hessian, schur_gn, varpro_gn = 0.0, 0.0, 0.0
    residual_derivatives = []
    for weight, c, y in zip(pi, covariance, targets):
        _, g, residual, system = local(basis, c, y, tau)
        gradient += 2 * weight * np.outer(residual, g)
        z = delta @ g
        q = delta.T @ residual
        t = basis.T @ c @ z
        k = t + q
        hessian += (
            2 * weight * (z @ c @ z + tau * (z @ z) - k @ np.linalg.solve(system, k))
        )
        s = c - c @ basis @ np.linalg.solve(system, basis.T @ c)
        schur_gn += weight * (z @ s @ z)
        varpro_gn += weight * (z @ s @ z + q @ np.linalg.solve(system, q))
        step = 1e-6
        residuals = []
        for sign in (-1, 1):
            b = polar(basis + sign * step * delta)
            _, gj, _, _ = local(b, c, y, tau)
            residuals.append(np.r_[y - root(c) @ b @ gj, -np.sqrt(tau) * gj])
        residual_derivatives.append(
            np.sqrt(weight) * (residuals[1] - residuals[0]) / (2 * step)
        )

    step = 1e-6
    first = (
        evaluate(polar(basis + step * delta)) - evaluate(polar(basis - step * delta))
    ) / (2 * step)
    close("profile_gradient", first, np.sum(gradient * delta))
    step = 2e-4
    second = (
        evaluate(polar(basis + step * delta))
        - 2 * evaluate(basis)
        + evaluate(polar(basis - step * delta))
    ) / step**2
    close("exact_ridge_grassmann_hessian", second, hessian, rtol=3e-6)
    close(
        "exact_varpro_gauss_newton", sum(x @ x for x in residual_derivatives), varpro_gn
    )
    checks["distinct_curvatures"] = {
        "half_hessian": hessian / 2,
        "schur_gn": schur_gn,
        "varpro_gn": varpro_gn,
    }
    assert abs(hessian / 2 - schur_gn) > 1e-3
    assert abs(varpro_gn - schur_gn) > 1e-3

    cross = np.array([root(c) @ y for c, y in zip(covariance, targets)])
    problem = Problem(
        covariance, cross, pi, tau, float(np.sum(pi[:, None] * targets**2)), 1.0, {}
    )
    state = direct_profile(basis, problem)
    close("live_profile_value", state.value, evaluate(basis))
    close("live_profile_gradient", state.gradient, gradient)

    transform = rng.normal(size=(m, m)) + 3 * np.eye(m)
    new_basis = basis @ transform
    invariant_value = 0.0
    for weight, c, y in zip(pi, covariance, targets):
        u = root(c)
        g = np.linalg.solve(
            new_basis.T @ (c + tau * np.eye(d)) @ new_basis, new_basis.T @ u @ y
        )
        invariant_value += weight * (
            np.linalg.norm(y - u @ new_basis @ g) ** 2
            + tau * np.linalg.norm(new_basis @ g) ** 2
        )
    close("ridge_invariant_extension", invariant_value, evaluate(basis))

    raw = rng.normal(size=(d, d))
    metric = raw @ raw.T + np.eye(d)
    sm = metric - metric @ basis @ np.linalg.solve(
        basis.T @ metric @ basis, basis.T @ metric
    )
    raw = rng.normal(size=(m, m))
    right = raw @ raw.T + np.eye(m)
    projector = np.eye(d) - basis @ basis.T
    fast_gradient = (
        projector @ np.linalg.solve(metric, projector @ gradient) @ np.linalg.inv(right)
    )
    slow_gradient = (
        complement
        @ np.linalg.solve(complement.T @ sm @ complement, complement.T @ gradient)
        @ np.linalg.inv(right)
    )
    close("separable_inverse_without_complement", fast_gradient, slow_gradient)
    close(
        "metric_duality",
        np.trace(fast_gradient.T @ sm @ delta @ right),
        np.sum(gradient * delta),
    )
    descent = -float(np.sum(gradient * fast_gradient))
    assert descent < 0
    checks["descent_derivative"] = descent

    h = basis.T @ metric @ basis
    values, vectors = np.linalg.eigh(h)
    horizontal_lift = (
        (np.eye(d) - basis @ np.linalg.solve(h, basis.T @ metric))
        @ delta
        @ (vectors * values**-0.5)
        @ vectors.T
    )
    close(
        "generalized_stiefel_pullback",
        np.trace(horizontal_lift.T @ metric @ horizontal_lift),
        np.trace(delta.T @ sm @ delta @ np.linalg.inv(h)),
    )
    generalized_basis = basis @ (vectors * values**-0.5) @ vectors.T
    ambient = rng.normal(size=(d, m))
    symmetric = (generalized_basis.T @ ambient + ambient.T @ generalized_basis) / 2
    riemannian = np.linalg.solve(metric, ambient) - generalized_basis @ symmetric
    close(
        "generalized_stiefel_gradient_tangent",
        generalized_basis.T @ metric @ riemannian
        + riemannian.T @ metric @ generalized_basis,
        np.zeros((m, m)),
    )
    candidate = generalized_basis - 0.1 * riemannian
    values, vectors = np.linalg.eigh(candidate.T @ metric @ candidate)
    retracted = candidate @ (vectors * values**-0.5) @ vectors.T
    close("generalized_stiefel_retraction", retracted.T @ metric @ retracted, np.eye(m))

    # Rank-deficient windows can jointly identify all tangent directions.
    tangent_matrix = np.zeros(((d - m) * m,) * 2)
    ideal_hessian = 0.0
    ideal_windows = []
    for weight in pi:
        factor = rng.normal(size=(d, m + 2))
        c = factor @ factor.T
        a = rng.normal(size=m)
        y = root(c) @ basis @ a
        system = basis.T @ c @ basis
        s = c - c @ basis @ np.linalg.solve(system, basis.T @ c)
        tangent_matrix += weight * np.kron(
            np.outer(a, a), complement.T @ s @ complement
        )
        ideal_hessian += 2 * weight * (delta @ a) @ s @ (delta @ a)
        ideal_windows.append((weight, c, y))

    def ideal_value(b):
        return sum(w * local(b, c, y, 0.0)[0] for w, c, y in ideal_windows)

    step = 1e-4
    second = (
        ideal_value(polar(basis + step * delta))
        - 2 * ideal_value(basis)
        + ideal_value(polar(basis - step * delta))
    ) / step**2
    close("zero_residual_singular_hessian", second, ideal_hessian, rtol=3e-6)
    eigenvalues = np.linalg.eigvalsh(tangent_matrix)
    assert eigenvalues[0] > 1e-6
    checks["joint_singular_identification"] = {
        "window_rank": m + 2,
        "ambient_dimension": d,
        "curvature_min": float(eigenvalues[0]),
        "curvature_max": float(eigenvalues[-1]),
    }

    # With common covariance, the exact full form equals the separated form.
    feature = complement.T @ sm @ complement
    ideal_coefficients = rng.normal(size=(windows, m))
    gram = np.einsum("j,ja,jb->ab", pi, ideal_coefficients, ideal_coefficients)
    full = sum(
        w * np.kron(np.outer(a, a), feature) for w, a in zip(pi, ideal_coefficients)
    )
    separated = np.kron(gram, feature)
    close("common_covariance_curvature", full, separated)
    close(
        "common_covariance_condition_one",
        eigh(2 * full, separated, eigvals_only=True),
        2 * np.ones(len(full)),
    )
    alpha, beta = 0.4, 2.7
    scaled_full = sum(
        w * scale * np.kron(np.outer(a, a), feature)
        for w, scale, a in zip(
            pi, np.linspace(alpha, beta, windows), ideal_coefficients
        )
    )
    spectrum = eigh(2 * scaled_full, separated, eigvals_only=True)
    assert spectrum[0] >= 2 * alpha - 1e-10 and spectrum[-1] <= 2 * beta + 1e-10
    checks["separated_condition_bound"] = {
        "observed_interval": [float(spectrum[0]), float(spectrum[-1])],
        "proved_interval": [2 * alpha, 2 * beta],
    }

    c = np.array([[2.0, 1.0], [1.0, 2.0]])
    truth = np.array([1.0, 0.0])
    optimum = np.linalg.solve(c + np.eye(2), c @ truth)
    optimum /= np.linalg.norm(optimum)
    close("ridge_bias_direction", optimum, np.array([5.0, 1.0]) / np.sqrt(26))
    close("ridge_bias_edr", abs(optimum[1]), 1 / np.sqrt(26))

    # Covariance-matched regularization preserves the entire objective up to scale.
    b = polar(rng.normal(size=(d, m)))
    value0, covariance_penalty = 0.0, 0.0
    for w, c, y in zip(pi, covariance, targets):
        design = root(c) @ b
        g0 = np.linalg.solve(design.T @ design, design.T @ y)
        g = g0 / (1 + tau)
        value0 += w * np.linalg.norm(y - design @ g0) ** 2
        covariance_penalty += w * (
            np.linalg.norm(y - design @ g) ** 2 + tau * g @ b.T @ c @ b @ g
        )
    close(
        "covariance_penalty_affine_identity",
        covariance_penalty,
        (value0 + tau * np.sum(pi[:, None] * targets**2)) / (1 + tau),
    )

    b = np.array([[1.0], [0.0]])
    rank_one_schurs = []
    for vector in (np.array([1.0, 1.0]), np.array([1.0, -1.0])):
        c = np.outer(vector, vector)
        rank_one_schurs.append(c - c @ b @ np.linalg.solve(b.T @ c @ b, b.T @ c))
    close("positive_average_no_identification", rank_one_schurs, np.zeros((2, 2, 2)))
    metric2 = np.diag([100.0, 1.0])
    v = np.ones(2)
    close("euclidean_angle", abs(v[1]) / np.linalg.norm(v), 1 / np.sqrt(2))
    close(
        "anisotropic_angle",
        np.sqrt(
            1
            - (truth @ metric2 @ v) ** 2
            / ((truth @ metric2 @ truth) * (v @ metric2 @ v))
        ),
        1 / np.sqrt(101),
    )

    return {
        "seed": 20261026,
        "scope": "algebra and small deterministic finite differences; no ADP recovery fits",
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/verification/anisotropic-geometry-20261026/checks.json"),
    )
    args = parser.parse_args()
    result = run()
    paths = [
        Path(__file__),
        Path("benchmarks/experiments/full_ridge_fast.py"),
        Path("benchmarks/experiments/full_ridge_batched.py"),
    ]
    result["source_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{len(result['checks'])} formula checks passed; {args.output}")


if __name__ == "__main__":
    main()
