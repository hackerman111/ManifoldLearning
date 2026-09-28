"""Isolated Gaussian-Stein plus quartic surrogate for observable gradients.

This is an explicit structural candidate for the synthetic polynomial link,
not a replacement for the general manifold ADP estimator.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from experiments.manifold_generalization import geometry, make_data


def monomials(dimension: int, degree: int) -> list[tuple[int, ...]]:
    return [
        indices
        for order in range(degree + 1)
        for indices in itertools.combinations_with_replacement(range(dimension), order)
    ]


def design(values: np.ndarray, terms: list[tuple[int, ...]]) -> np.ndarray:
    return np.column_stack(
        [
            np.prod(values[:, indices], axis=1) if indices else np.ones(len(values))
            for indices in terms
        ]
    )


def derivative(
    values: np.ndarray,
    basis: np.ndarray,
    terms: list[tuple[int, ...]],
    coefficients: np.ndarray,
) -> np.ndarray:
    projected = np.zeros_like(values)
    for indices, coefficient in zip(terms, coefficients, strict=True):
        for variable in set(indices):
            remaining = list(indices)
            remaining.remove(variable)
            projected[:, variable] += (
                coefficient
                * indices.count(variable)
                * (np.prod(values[:, remaining], axis=1) if remaining else 1.0)
            )
    return projected @ basis.T


def ridge_fit(A: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    left, singular, right = np.linalg.svd(A, full_matrices=False)
    factors = singular / (singular**2 + penalty)
    return right.T @ (factors * (left.T @ y))


def cv_penalty(A: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, float]]:
    penalties = (1e-6, 1e-2, 1.0, 10.0, 100.0)
    folds = np.random.default_rng(20260928).permutation(len(y)) % 3
    losses = np.zeros(len(penalties))
    for fold in range(3):
        train = folds != fold
        validation = ~train
        left, singular, right = np.linalg.svd(A[train], full_matrices=False)
        transformed = left.T @ y[train]
        for k, penalty in enumerate(penalties):
            coefficients = right.T @ (singular / (singular**2 + penalty) * transformed)
            losses[k] += float(
                np.mean((A[validation] @ coefficients - y[validation]) ** 2)
            )
    best = int(np.argmin(losses))
    return penalties[best], {
        str(penalty): float(loss / 3)
        for penalty, loss in zip(penalties, losses, strict=True)
    }


def gradient_sine(exact: np.ndarray, estimated: np.ndarray) -> dict[str, float]:
    exact_norm = np.linalg.norm(exact, axis=1)
    estimated_norm = np.linalg.norm(estimated, axis=1)
    if np.any(exact_norm <= 1e-12) or np.any(estimated_norm <= 1e-12):
        raise ValueError("gradient direction undefined")
    cosine = np.sum(exact * estimated, axis=1) / (exact_norm * estimated_norm)
    sine = np.sqrt(np.maximum(0.0, 1.0 - np.minimum(1.0, cosine**2)))
    return {"rms": float(np.sqrt(np.mean(sine**2))), "max": float(sine.max())}


def self_check() -> None:
    rng = np.random.default_rng(123)
    values = rng.normal(size=(5, 3))
    basis = np.linalg.qr(rng.normal(size=(5, 3)))[0]
    terms = monomials(3, 4)
    coefficients = rng.normal(size=len(terms))
    analytic = derivative(values, basis, terms, coefficients)
    X = values @ basis.T
    for j in range(X.shape[1]):
        shift = np.zeros_like(X)
        shift[:, j] = 1e-5
        plus = design((X + shift) @ basis, terms) @ coefficients
        minus = design((X - shift) @ basis, terms) @ coefficients
        np.testing.assert_allclose(
            analytic[:, j], (plus - minus) / 2e-5, rtol=1e-8, atol=1e-8
        )


def run_case(seed: int, m: int, curvature: float, noise: float) -> dict[str, Any]:
    X, Y, _, _, _, model_seed = make_data(seed, 600, 8, m, curvature, noise)
    rotation_stream = np.random.SeedSequence(seed).spawn(5)[1]
    rotation, _ = np.linalg.qr(
        np.random.default_rng(rotation_stream).normal(size=(8, 8))
    )
    _, _, exact_gradient = geometry(X, rotation, m, curvature)
    center_seed = np.random.SeedSequence(model_seed).spawn(2)[0]
    indices = np.random.default_rng(center_seed).choice(600, size=40, replace=False)
    started = time.perf_counter()
    result: dict[str, Any] = {
        "seed": seed,
        "m": m,
        "curvature": curvature,
        "noise": noise,
        "error": None,
    }
    try:
        moment = X.T @ (Y[:, None] * X) / len(X) - float(Y.mean()) * np.eye(X.shape[1])
        eigenvalues, eigenvectors = np.linalg.eigh(moment)
        rank = 2 * m
        basis = eigenvectors[:, -rank:]
        result["stein_gap"] = float(eigenvalues[-rank] - eigenvalues[-rank - 1])
        result["stein_min_selected"] = float(eigenvalues[-rank])
        true_basis = rotation[:, :rank]
        overlap = np.linalg.svd(basis.T @ true_basis, compute_uv=False)
        result["global_active_max_sine"] = float(
            np.sqrt(max(0.0, 1.0 - float(overlap[-1] ** 2)))
        )
        terms = monomials(rank, 4)
        values = X @ basis
        matrix = design(values, terms)
        scale = np.maximum(np.sqrt(np.mean(matrix**2, axis=0)), 1e-12)
        scaled = matrix / scale
        penalty, losses = cv_penalty(scaled, Y)
        coefficients = ridge_fit(scaled, Y, penalty) / scale
        estimated = derivative(values[indices], basis, terms, coefficients)
        result["gradient"] = gradient_sine(exact_gradient[indices], estimated)
        result["penalty"] = penalty
        result["cv_mse"] = losses
        result["train_rmse"] = float(np.sqrt(np.mean((matrix @ coefficients - Y) ** 2)))
        result["feature_count"] = len(terms)
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["seconds"] = time.perf_counter() - started
    result["process_peak_rss_mib"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=81000)
    parser.add_argument("--seed-stop", type=int, default=81005)
    args = parser.parse_args()
    if args.seed_stop <= args.seed_start:
        parser.error("seed-stop must exceed seed-start")
    self_check()
    args.out.mkdir(parents=True, exist_ok=False)
    sources = (
        "experiments/manifold_generalization.py",
        "docs/experiments/manifold_curvature_2026-09-28/polynomial_probe.py",
    )
    with threadpool_limits(limits=1):
        manifest = {
            "seeds": [args.seed_start, args.seed_stop],
            "sources": {
                name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                for name in sources
            },
            "threads": threadpool_info(),
            "candidate": (
                "Gaussian second Stein moment top-2m, quartic polynomial "
                "ridge with 3-fold response CV"
            ),
            "model_assumption": (
                "Gaussian X and scalar regression polynomial degree <=4 "
                "in a rank-2m global active subspace"
            ),
            "metric": "all 40 center gradient directions; no truth in fitting or CV",
            "rss": "cumulative process ru_maxrss MiB",
        }
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        with (args.out / "runs.jsonl").open("w") as output:
            for seed in range(args.seed_start, args.seed_stop):
                for m in (1, 2, 3):
                    for curvature in (0.35, 0.8):
                        for noise in (0.0, 0.1):
                            row = run_case(seed, m, curvature, noise)
                            output.write(json.dumps(row, allow_nan=False) + "\n")
                            output.flush()
                            print(
                                seed,
                                m,
                                curvature,
                                noise,
                                row["error"],
                                row.get("gradient"),
                                flush=True,
                            )


if __name__ == "__main__":
    main()
