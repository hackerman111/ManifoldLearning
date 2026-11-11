"""Opt-in moment metrics from docs/tex/data/gpt_ans.md; public ADP is unchanged.

ESTIMATOR: GLS, row whitening and effective-size weighting change the loss.
GLS uses a block covariance, ignoring correlations between centers. For
adaptive weights its fixed-design noise theorem does not apply directly.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ADP.core.ADP_Statistic import ADP_Statistics


def moment_covariances(x, blocks, directions):
    """Return D_j=H_j H_j.T, with normalized, centered noise operators H_j."""
    covariances = np.empty((*directions.shape[:2], directions.shape[1]))
    expected = 0
    for start, weights in blocks:
        if start != expected:
            raise ValueError("weight blocks must be contiguous")
        for offset, w in enumerate(weights):
            j = start + offset
            support = w > 0
            a = w[support] / w.sum()
            dx = x[support] - a @ x[support]
            projected = dx @ directions[j].T
            projected -= a @ projected
            h = a[:, None] * projected
            covariances[j] = h.T @ h
        expected += len(weights)
    if expected != len(directions):
        raise ValueError("weight blocks do not cover directions")
    return covariances


def reweight_statistics(x, blocks, directions, stats: ADP_Statistics, kind):
    """Whiten I/U while retaining original mass for structural adaptation.

    GLS divides the whitened moments by sqrt(mass), exactly canceling the
    solver's outer mass. Row whitening keeps that mass. Near-singular
    eigenvalues are floored at 1e-8 times the largest eigenvalue; such centers
    implement a regularized metric, rather than the exact inverse theorem.
    """
    info = {
        "metric": kind,
        "clipped_eigenvalues": 0,
        "regularized_centers": 0,
        "zero_information_centers": 0,
    }
    if kind == "base":
        return stats, info
    if kind == "neff":
        scale = np.sqrt(stats.n_eff / stats.mass)
        return replace(
            stats, I=stats.I * scale[:, None], U=stats.U * scale[:, None, None]
        ), info
    if kind == "gls":
        covariances = moment_covariances(x, blocks, directions)
    elif kind == "row":
        covariances = stats.U @ stats.U.swapaxes(1, 2)
    else:
        raise ValueError("unknown moment metric")
    values, vectors = np.linalg.eigh(covariances)
    if np.any(values[:, -1] < 0) or not np.isfinite(values).all():
        raise ValueError("moment metric has no positive finite scale")
    zero = values[:, -1] == 0
    if np.any(stats.U[zero] != 0) or np.any(stats.I[zero] != 0):
        raise ValueError("zero covariance is inconsistent with nonzero moments")
    info["zero_information_centers"] = int(zero.sum())
    threshold = 1e-8 * np.where(zero, 1.0, values[:, -1])[:, None]
    clipped = (values < threshold) & ~zero[:, None]
    info["clipped_eigenvalues"] = int(clipped.sum())
    info["regularized_centers"] = int(clipped.any(axis=1).sum())
    inverse_root = 1 / np.sqrt(np.maximum(values, threshold))
    inverse_root[zero] = 0
    whitener = (vectors * inverse_root[:, None, :]) @ vectors.swapaxes(1, 2)
    if kind == "gls":
        whitener /= np.sqrt(stats.mass)[:, None, None]
    return replace(
        stats,
        I=(whitener @ stats.I[..., None]).squeeze(-1),
        U=whitener @ stats.U,
    ), info


def projector_certificate(estimated, t, m):
    """Return the leading subspace and the exact-arithmetic bound (25).

    t must be an established upper bound on ||Vhat-V||_2. This routine does
    not establish that assumption or certify floating-point roundoff.
    """
    if not np.isfinite(t) or t < 0 or not 1 <= m <= min(estimated.shape):
        raise ValueError("invalid certificate inputs")
    left, singular, _ = np.linalg.svd(estimated, full_matrices=False)
    basis = left[:, :m]
    matrix = estimated @ estimated.T
    rayleigh = basis.T @ matrix @ basis
    q = float(np.linalg.eigvalsh(rayleigh)[0])
    if q <= np.finfo(float).eps * max(1.0, singular[0] ** 2):
        raise ValueError("preliminary gradients do not identify the subspace")
    residual = float(np.linalg.norm(matrix @ basis - basis @ rayleigh))
    delta = float(2 * singular[0] * t + t * t)
    raw_bound = (delta + residual / np.sqrt(m)) / q
    return basis, {
        "q": q,
        "residual_fro": residual,
        "t": float(t),
        "raw_edr_bound": float(raw_bound),
        "edr_bound": float(min(1.0, raw_bound)),
        "nonvacuous_step_threshold": bool(raw_bound < 0.5),
    }
