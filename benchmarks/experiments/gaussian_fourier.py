"""Experimental spectral SDR for a known X ~ N(0, I) predictor law.

The population objective equals EDR squared when the Fourier moment matrix
has rank m. Its sample version targets an estimated projector, and supplies
no unconditional finite-sample recovery guarantee. See
benchmarks/results/verification/gaussian-fourier-20261025/DESIGN.md. Production ADP is unchanged.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FourierFit:
    basis: np.ndarray  # (d,m), orthonormal columns in original coordinates
    eigenvalues: np.ndarray  # all sample eigenvalues, descending
    diagnostics: dict


def _real(value, name):
    array = np.asarray(value)
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be real numeric")
    array = np.asarray(array, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _inputs(X, y, frequencies):
    X, y, frequencies = (
        _real(v, name) for v, name in ((X, "X"), (y, "y"), (frequencies, "frequencies"))
    )
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 2:
        raise ValueError("X must have shape (n,d), n>=2, d>=2")
    if y.shape != (len(X),):
        raise ValueError("y must have shape (n,)")
    if (
        frequencies.ndim != 2
        or frequencies.shape[1] != X.shape[1]
        or not len(frequencies)
    ):
        raise ValueError("frequencies must have shape (R,d), R>=1")
    return X, y, frequencies


def _moments(X, y, frequencies, batch_size):
    result = np.empty((X.shape[1], len(frequencies)), dtype=complex)
    for start in range(0, len(frequencies), batch_size):
        block = frequencies[start : start + batch_size]
        weighted_phase = np.exp(1j * (X @ block.T))
        weighted_phase *= y[:, None]
        result[:, start : start + len(block)] = X.T @ weighted_phase / len(
            X
        ) - 1j * block.T * weighted_phase.mean(axis=0)
    if not np.isfinite(result).all():
        raise RuntimeError("nonfinite Fourier moments")
    return result


def moments(X, y, frequencies, *, batch_size=64):
    """Return columns mean[Y (X-i t) exp(i t^T X)]; known standard Gaussian law."""
    X, y, frequencies = _inputs(X, y, frequencies)
    batch_size = _integer(batch_size, "batch_size")
    return _moments(X, y, frequencies, batch_size)


def candidate_matrix(X, y, frequencies, *, batch_size=64):
    """Unbiased distinct-observation estimate, conditional on fixed frequencies.

    An indefinite matrix is permitted: silently clipping its spectrum would
    obscure the distinction between a sample matrix and its population limit.
    """
    X, y, frequencies = _inputs(X, y, frequencies)
    batch_size = _integer(batch_size, "batch_size")
    vectors = _moments(X, y, frequencies, batch_size)
    n, R = len(X), len(frequencies)
    gram = (vectors @ vectors.conj().T).real / R
    squared_y = y**2
    diagonal = X.T @ (squared_y[:, None] * X) / n
    diagonal += squared_y.mean() * (frequencies.T @ frequencies) / R
    matrix = (n * gram - diagonal) / (n - 1)
    matrix = (matrix + matrix.T) / 2
    if not np.isfinite(matrix).all():
        raise RuntimeError("nonfinite Fourier candidate matrix")
    return matrix


def solve(matrix, index_dim):
    """Globally minimize the flattened sample-projector objective by eigensolve."""
    matrix = _real(matrix, "matrix")
    m = _integer(index_dim, "index_dim")
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not 1 <= m < len(matrix)
    ):
        raise ValueError("matrix must be square and 1<=index_dim<d")
    scale = float(np.linalg.norm(matrix, ord=2))
    guard = 64 * np.finfo(float).eps * len(matrix) * max(scale, np.finfo(float).tiny)
    if np.max(np.abs(matrix - matrix.T)) > guard:
        raise ValueError("matrix must be symmetric")
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    values, vectors = values[::-1], vectors[:, ::-1]
    gap = float(values[m - 1] - values[m])
    if values[m - 1] <= guard or gap <= guard:
        raise ValueError("selected subspace is not numerically identified")
    basis = vectors[:, :m].copy()
    # Choose deterministic signs; projector/objective are sign invariant.
    signs = np.sign(basis[np.argmax(np.abs(basis), axis=0), np.arange(m)])
    basis *= signs
    return FourierFit(
        basis,
        values,
        {
            "method": "gaussian_fourier_ustat_spectral",
            "predictor_law": "known N(0,I)",
            "empirical_minimum": 0.0,
            "numerical_eigengap": gap,
            "selected_min_eigenvalue": float(values[m - 1]),
            "negative_eigenvalues": int(np.count_nonzero(values < 0)),
            "orthogonality_error": float(np.linalg.norm(basis.T @ basis - np.eye(m))),
            "sample_recovery_certified": False,
        },
    )


def _basis(value, name):
    value = _real(value, name)
    if value.ndim != 2 or not 1 <= value.shape[1] < value.shape[0]:
        raise ValueError(f"{name} must have shape (d,m), 1<=m<d")
    if not np.allclose(value.T @ value, np.eye(value.shape[1]), atol=1e-10, rtol=1e-10):
        raise ValueError(f"{name} must have orthonormal columns")
    return value


def objective(basis, target_basis):
    """Return sample loss and Grassmann gradient; target_basis is estimated.

    With an exact population target this loss is true EDR squared. A sample
    minimum of zero by itself certifies no recovery of the population target.
    """
    basis, target = _basis(basis, "basis"), _basis(target_basis, "target_basis")
    if basis.shape != target.shape:
        raise ValueError("basis shapes must match")
    m = basis.shape[1]
    projected = target @ (target.T @ basis)
    value = float(np.clip(1 - np.sum(basis * projected) / m, 0, 1))
    gradient = -2 / m * (projected - basis @ (basis.T @ projected))
    return value, gradient


def perturbation_bound(selected_min_eigenvalue, operator_error):
    """Conditional upper bound for EDR and uniform squared-loss error.

    The caller must CERTIFY ||M_hat-M||_op <= operator_error for a rank-m
    population matrix M. This function does not estimate or certify that event.
    """
    if not np.isfinite(selected_min_eigenvalue) or selected_min_eigenvalue <= 0:
        raise ValueError("selected eigenvalue must be finite and positive")
    if not np.isfinite(operator_error) or operator_error < 0:
        raise ValueError("operator_error must be finite and nonnegative")
    return float(min(1, operator_error / selected_min_eigenvalue))


def fit(
    X, y, index_dim, *, n_frequencies=256, frequency_scale=2.0, seed=0, batch_size=64
):
    """Estimate a subspace assuming known X ~ N(0,I); never accepts a true basis.

    frequency_scale is the RMS frequency norm. Frequencies are independent
    of the data; a fixed zero frequency retains the average-gradient moment.
    Full fit time includes validation, frequency generation and matrix assembly.
    """
    started = time.perf_counter()
    X = _real(X, "X")
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 2:
        raise ValueError("X must have shape (n,d), n>=2, d>=2")
    m = _integer(index_dim, "index_dim")
    R = _integer(n_frequencies, "n_frequencies")
    if not m < X.shape[1]:
        raise ValueError("require index_dim<d")
    if 2 * R - 1 < m:
        raise ValueError(
            "insufficient population rank: require 2*n_frequencies-1>=index_dim"
        )
    if not np.isfinite(frequency_scale) or frequency_scale <= 0:
        raise ValueError("frequency_scale must be finite and positive")
    frequencies = np.random.default_rng(seed).normal(size=(R, X.shape[1]))
    frequencies *= frequency_scale / np.sqrt(X.shape[1])
    frequencies[0] = 0
    matrix = candidate_matrix(X, y, frequencies, batch_size=batch_size)
    result = solve(matrix, m)
    result.diagnostics.update(
        n_frequencies=R,
        frequency_scale=float(frequency_scale),
        seed=int(seed),
        fit_seconds=time.perf_counter() - started,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NPZ containing X=(n,d), y=(n,)")
    parser.add_argument("--index-dim", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-frequencies", type=int, default=256)
    parser.add_argument("--frequency-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        result = fit(
            data["X"],
            data["y"],
            args.index_dim,
            n_frequencies=args.n_frequencies,
            frequency_scale=args.frequency_scale,
            seed=args.seed,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        basis=result.basis,
        eigenvalues=result.eigenvalues,
        diagnostics=json.dumps(result.diagnostics),
    )
    print(json.dumps(result.diagnostics, indent=2))


if __name__ == "__main__":
    main()
