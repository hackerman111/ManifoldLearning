"""Capture one real manifold and HPAO subproblem and check dense references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ADP.core.manifold.ADP_Manifold import ADP_Manifold
from ADP.solver import LSMR
from benchmarks.fit_bottlenecks import _one


def manifold_reference(data: dict[str, np.ndarray]) -> dict[str, float]:
    U, I = data["U"], data["I"]
    mass, weights = data["mass"], data["weights"]
    source_projectors, slopes = data["source_projectors"], data["slopes"]
    B, ridge = data["B"], float(data["ridge"])
    gamma = mass * weights
    root = np.sqrt(gamma)
    design = (
        root[:, None, None, None] * slopes[:, None, :, None] * U[:, :, None, :]
    ).reshape(I.size, B.size)
    rhs_data = (root[:, None] * I).ravel()
    average = np.einsum(
        "j,jad,jae->de",
        weights / weights.sum(),
        source_projectors,
        source_projectors,
    )
    penalty = ridge * np.kron(np.eye(B.shape[0]), np.eye(B.shape[1]) - average)
    normal = design.T @ design + penalty
    rhs = design.T @ rhs_data
    expected = np.linalg.solve(normal, rhs)
    operator, _, actual_rhs = ADP_Manifold(
        B.shape[0], lambda_manifold=ridge
    )._build_B_system(U, I, mass, weights, source_projectors, slopes)
    action = np.column_stack([operator @ column for column in np.eye(B.size)])
    objective = (
        np.linalg.norm(design @ B.ravel() - rhs_data) ** 2
        + B.ravel() @ penalty @ B.ravel()
    )
    return {
        "normal_max_error": float(np.max(np.abs(action - normal))),
        "rhs_max_error": float(np.max(np.abs(actual_rhs - rhs))),
        "solution_relative_error": float(
            np.linalg.norm(B.ravel() - expected) / np.linalg.norm(expected)
        ),
        "relative_residual": float(
            np.linalg.norm(normal @ B.ravel() - rhs) / np.linalg.norm(rhs)
        ),
        "objective": float(objective),
        "edges": len(weights),
    }


def hpao_reference(data: dict[str, np.ndarray]) -> dict[str, float]:
    U, I, index = data["U"], data["I"], data["index"]
    coefficients, mass = data["coefficients"], data["mass"]
    correction, ridge = data["correction"], float(data["ridge"])
    root = np.sqrt(mass)
    if index.ndim == 1:
        design = (root * coefficients)[:, None, None] * U
        predicted = coefficients[:, None] * (U @ index)
        design = design.reshape(I.size, index.size)
    else:
        design = (
            root[:, None, None, None]
            * coefficients[:, None, :, None]
            * U[:, :, None, :]
        ).reshape(I.size, index.size)
        predicted = (U @ (coefficients @ index)[..., None]).squeeze(-1)
    residual = (root[:, None] * (I - predicted)).ravel()
    augmented = np.vstack((design, np.sqrt(ridge) * np.eye(index.size)))
    expected = np.linalg.lstsq(
        augmented, np.concatenate((residual, np.zeros(index.size))), rcond=None
    )[0]
    operator = LSMR._linear_operator(U, coefficients, root, index.shape)
    action = np.column_stack([operator @ column for column in np.eye(index.size)])
    normal = design.T @ (residual - design @ correction) - ridge * correction
    denominator = (
        ridge * np.linalg.norm(correction)
        if ridge > 0
        else np.linalg.norm(design.T @ residual)
    )
    return {
        "forward_max_error": float(np.max(np.abs(action - design))),
        "solution_relative_error": float(
            np.linalg.norm(correction - expected)
            / max(np.linalg.norm(expected), np.finfo(float).tiny)
        ),
        "normal_residual_ratio": float(np.linalg.norm(normal) / denominator),
        "adjoint_error": float(
            abs(
                (operator @ correction) @ residual
                - correction @ operator.rmatvec(residual)
            )
        ),
        "ridge": ridge,
        "shape": list(U.shape),
    }


def capture(case: str, path: Path) -> dict[str, object]:
    captured: dict[str, np.ndarray] = {}
    if case.startswith("manifold"):
        original_build = ADP_Manifold._build_B_system
        original_solve = ADP_Manifold._solve_B

        def build(self, U, I, mass, weights, source_projectors, slopes):
            if not captured:
                captured.update(
                    U=U.copy(),
                    I=I.copy(),
                    mass=mass.copy(),
                    weights=weights.copy(),
                    source_projectors=source_projectors.copy(),
                    slopes=slopes.copy(),
                    ridge=np.asarray(self.lambda_manifold),
                )
            return original_build(self, U, I, mass, weights, source_projectors, slopes)

        def solve(self, operator, preconditioner, rhs, initial):
            result = original_solve(self, operator, preconditioner, rhs, initial)
            if "B" not in captured:
                captured["B"] = result[0].copy()
            return result

        with (
            patch.object(ADP_Manifold, "_build_B_system", build),
            patch.object(ADP_Manifold, "_solve_B", solve),
        ):
            fit = _one(case, "off")
        reference = manifold_reference(captured)
    else:
        original = LSMR._global_correction

        def correction(I, U, index, coefficients, mass, ridge, tol, maxiter):
            result = original(I, U, index, coefficients, mass, ridge, tol, maxiter)
            if not captured:
                captured.update(
                    I=I.copy(),
                    U=U.copy(),
                    index=index.copy(),
                    coefficients=coefficients.copy(),
                    mass=mass.copy(),
                    ridge=np.asarray(ridge),
                    correction=result[0].copy(),
                )
            return result

        with patch.object(LSMR, "_global_correction", correction):
            fit = _one(case, "off")
        reference = hpao_reference(captured)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **captured)
    return {
        "case": case,
        "input": str(path),
        "fit_seconds": fit["fit_seconds"],
        "fit_error": fit["error"],
        "stop_reason": fit.get("stop_reason"),
        "reference": reference,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("manifold_base", "multi_base"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(capture(args.case, args.output), indent=2))


if __name__ == "__main__":
    main()
