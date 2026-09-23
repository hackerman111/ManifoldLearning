"""Small auditable candidates on captured manifold and HPAO subproblems."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse import linalg as sparse_linalg

from ADP.core.manifold import ADP_Manifold_utils as model_utils
from ADP.engine.manifol_engine import optimisation
from ADP.engine.manifol_engine.utils import orient_rows
from ADP.solver import LSMR


def _time(function, repeats: int = 1000) -> float:
    for _ in range(50):
        function()
    start = perf_counter()
    for _ in range(repeats):
        function()
    return (perf_counter() - start) / repeats


def penalty_one(B, projectors, weights):
    rows = projectors[:, 0, :]
    projected = (weights * (rows @ B[0])) @ rows
    return B - projected[None, :]


def penalty_reference(B, projectors, weights):
    coordinates = B @ projectors.swapaxes(1, 2)
    coordinates *= weights[:, None, None]
    projected = coordinates.transpose(1, 0, 2).reshape(len(B), -1) @ projectors.reshape(
        -1, B.shape[1]
    )
    return B - projected


def recover_one(B, slopes, gamma, target):
    M = float(np.dot(gamma, np.square(slopes[:, 0])))
    tolerance = 256.0 * np.finfo(float).eps * max(1.0, abs(M))
    model_utils.require_slope_matrix(np.asarray([M]), tolerance, target)
    singular = np.asarray([math.sqrt(max(0.0, M)) * np.linalg.norm(B)])
    model_utils.require_identified(singular, B.shape, 1, target)
    projector = orient_rows(B / np.linalg.norm(B))
    spectrum = np.ones(1)
    orthogonality = np.linalg.norm(projector @ projector.T - np.eye(1), ord="fro")
    model_utils.require_projector(orthogonality, spectrum, target)
    return projector, spectrum


def recover_reference(B, slopes, gamma, target):
    M = np.einsum("j,ja,jb->ab", gamma, slopes, slopes, optimize=True)
    values, vectors = np.linalg.eigh(M)
    tolerance = 256.0 * np.finfo(float).eps * max(1.0, float(np.linalg.norm(M, ord=2)))
    model_utils.require_slope_matrix(values, tolerance, target)
    np.maximum(values, 0.0, out=values)
    root = (vectors * np.sqrt(values)[None, :]) @ vectors.T
    factor = root @ B
    _, singular_values, right_vectors = np.linalg.svd(factor, full_matrices=False)
    model_utils.require_identified(singular_values, factor.shape, B.shape[0], target)
    projector = orient_rows(right_vectors.copy())
    spectrum = np.square(singular_values)
    spectrum /= spectrum[0]
    orthogonality = np.linalg.norm(
        projector @ projector.T - np.eye(B.shape[0]), ord="fro"
    )
    model_utils.require_projector(orthogonality, spectrum, target)
    return projector, spectrum


def hpao_operator_weighted(U, coefficients, sqrt_mass, index_shape):
    weighted = (
        coefficients * sqrt_mass
        if len(index_shape) == 1
        else coefficients * sqrt_mass[:, None]
    )
    rows = U.shape[0] * U.shape[1]

    def forward(vector):
        if len(index_shape) == 1:
            return (weighted[:, None] * (U @ vector)).ravel()
        local = weighted @ vector.reshape(index_shape)
        return (U @ local[..., None]).ravel()

    def adjoint(vector):
        data = vector.reshape(U.shape[:2])
        pulled = (U.swapaxes(1, 2) @ data[..., None]).squeeze(-1)
        if len(index_shape) == 1:
            return weighted @ pulled
        return (weighted.T @ pulled).ravel()

    return sparse_linalg.LinearOperator(
        (rows, math.prod(index_shape)), matvec=forward, rmatvec=adjoint, dtype=float
    )


def hpao_operator_reference(U, coefficients, sqrt_mass, index_shape):
    rows = U.shape[0] * U.shape[1]

    def forward(vector):
        if len(index_shape) == 1:
            data = coefficients[:, None] * (U @ vector)
        else:
            data = LSMR.multi_forward(U, vector.reshape(index_shape), coefficients)
        return (sqrt_mass[:, None] * data).ravel()

    def adjoint(vector):
        data = vector.reshape(U.shape[:2])
        if len(index_shape) == 1:
            return np.einsum(
                "j,j,jpd,jp->d", sqrt_mass, coefficients, U, data, optimize=True
            )
        return LSMR.multi_adjoint(U, sqrt_mass[:, None] * data, coefficients).ravel()

    return sparse_linalg.LinearOperator(
        (rows, math.prod(index_shape)), matvec=forward, rmatvec=adjoint, dtype=float
    )


def hpao_scaled_ridge(operator, U, coefficients, mass, ridge):
    """Exact change δ=S z in [A; sqrt(ridge) I] δ=[r;0]."""
    if coefficients.ndim == 1:
        diagonal = (mass * np.square(coefficients)) @ np.sum(np.square(U), axis=1)
    else:
        diagonal = (mass[:, None] * np.square(coefficients)).T @ np.sum(
            np.square(U), axis=1
        )
    scale = 1.0 / np.sqrt(diagonal.ravel() + ridge)
    rows, columns = operator.shape
    root = math.sqrt(ridge)

    def forward(z):
        delta = scale * z
        return np.concatenate((operator @ delta, root * delta))

    def adjoint(v):
        return scale * (operator.rmatvec(v[:rows]) + root * v[rows:])

    transformed = sparse_linalg.LinearOperator(
        (rows + columns, columns), matvec=forward, rmatvec=adjoint, dtype=float
    )
    return transformed, scale


def manifold(path: Path) -> dict:
    with np.load(path) as data:
        U, mass, weights = (data[k] for k in ("U", "mass", "weights"))
        projectors, slopes, B = (data[k] for k in ("source_projectors", "slopes", "B"))
        gamma = mass * weights
        normalized = weights / weights.sum()
        old_penalty = penalty_reference(B, projectors, normalized)
        new_penalty = penalty_one(B, projectors, normalized)
        old_projector, old_spectrum = recover_reference(B, slopes, gamma, 0)
        new_projector, new_spectrum = recover_one(B, slopes, gamma, 0)
        np.testing.assert_allclose(new_penalty, old_penalty, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(
            new_projector.T @ new_projector,
            old_projector.T @ old_projector,
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(new_spectrum, old_spectrum, rtol=0, atol=0)
        return {
            "penalty_max_error": float(np.max(np.abs(new_penalty - old_penalty))),
            "projector_distance": float(
                np.linalg.norm(
                    new_projector.T @ new_projector - old_projector.T @ old_projector
                )
            ),
            "penalty_old_us": 1e6
            * _time(lambda: penalty_reference(B, projectors, normalized)),
            "penalty_new_us": 1e6
            * _time(lambda: optimisation.penalty_action(B, projectors, normalized)),
            "recover_old_us": 1e6
            * _time(lambda: recover_reference(B, slopes, gamma, 0)),
            "recover_new_us": 1e6
            * _time(lambda: optimisation.recover_projector(B, slopes, gamma, 0)),
            "shape": list(U.shape),
        }


def hpao(path: Path) -> dict:
    with np.load(path) as data:
        U, I, index = (data[k] for k in ("U", "I", "index"))
        coefficients, mass = data["coefficients"], data["mass"]
        ridge, correction = float(data["ridge"]), data["correction"]
        root = np.sqrt(mass)
        old = hpao_operator_reference(U, coefficients, root, index.shape)
        new = LSMR._linear_operator(U, coefficients, root, index.shape)
        rng = np.random.default_rng(1729)
        x, y = rng.normal(size=index.size), rng.normal(size=I.size)
        np.testing.assert_allclose(new @ x, old @ x, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(
            new.rmatvec(y), old.rmatvec(y), rtol=1e-13, atol=1e-13
        )
        b = (root[:, None] * (I - LSMR._predict(U, index, coefficients))).ravel()
        tol = 1e-10

        def solve(operator):
            return sparse_linalg.lsmr(
                operator,
                b,
                damp=math.sqrt(ridge),
                atol=tol,
                btol=tol,
                maxiter=max(50, 5 * index.size),
            )

        old_result, new_result = solve(old), solve(new)
        normal = (
            new.rmatvec(b) - new.rmatvec(new @ new_result[0]) - ridge * new_result[0]
        )
        ratio = float(np.linalg.norm(normal) / (ridge * np.linalg.norm(new_result[0])))
        transformed, scale = hpao_scaled_ridge(new, U, coefficients, mass, ridge)
        augmented_rhs = np.concatenate((b, np.zeros(index.size)))

        def solve_scaled():
            return sparse_linalg.lsmr(
                transformed,
                augmented_rhs,
                atol=tol,
                btol=tol,
                maxiter=max(50, 5 * index.size),
            )

        scaled_result = solve_scaled()
        scaled_correction = scale * scaled_result[0]
        scaled_normal = (
            new.rmatvec(b)
            - new.rmatvec(new @ scaled_correction)
            - ridge * scaled_correction
        )
        scaled_ratio = float(
            np.linalg.norm(scaled_normal) / (ridge * np.linalg.norm(scaled_correction))
        )
        return {
            "forward_max_error": float(np.max(np.abs(new @ x - old @ x))),
            "adjoint_max_error": float(np.max(np.abs(new.rmatvec(y) - old.rmatvec(y)))),
            "adjoint_identity_error": float(abs((new @ x) @ y - x @ new.rmatvec(y))),
            "correction_relative_difference": float(
                np.linalg.norm(new_result[0] - old_result[0])
                / np.linalg.norm(old_result[0])
            ),
            "reference_relative_difference": float(
                np.linalg.norm(new_result[0] - correction) / np.linalg.norm(correction)
            ),
            "normal_residual_ratio": ratio,
            "scaled_normal_residual_ratio": scaled_ratio,
            "scaled_correction_relative_difference": float(
                np.linalg.norm(scaled_correction - old_result[0])
                / np.linalg.norm(old_result[0])
            ),
            "iterations_old": int(old_result[2]),
            "iterations_new": int(new_result[2]),
            "iterations_scaled": int(scaled_result[2]),
            "forward_old_us": 1e6 * _time(lambda: old @ x),
            "forward_new_us": 1e6 * _time(lambda: new @ x),
            "adjoint_old_us": 1e6 * _time(lambda: old.rmatvec(y)),
            "adjoint_new_us": 1e6 * _time(lambda: new.rmatvec(y)),
            "solve_old_us": 1e6 * _time(lambda: solve(old), repeats=50),
            "solve_new_us": 1e6 * _time(lambda: solve(new), repeats=50),
            "solve_scaled_us": 1e6 * _time(solve_scaled, repeats=50),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifold", type=Path, required=True)
    parser.add_argument("--hpao", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"manifold": manifold(args.manifold), "hpao": hpao(args.hpao)}
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
