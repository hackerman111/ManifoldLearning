"""General QR local profiles; experiment-only replacement for grassman._profile."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ADP.solver import grassman as base

ORIGINAL_PROFILE = base._profile
COUNTS = {"calls": 0, "qr_centers": 0, "svd_centers": 0, "delegated_centers": 0}


def reset_counts():
    for key in COUNTS:
        COUNTS[key] = 0


def triangular_solve(R: np.ndarray, rhs: np.ndarray, *, transpose=False) -> np.ndarray:
    """Batched triangular substitution, rhs=(J,m,k); never forms R.T@R."""
    result = np.empty_like(rhs)
    m = R.shape[1]
    for i in range(m) if transpose else range(m - 1, -1, -1):
        if transpose:
            correction = np.einsum("ji,jik->jk", R[:, :i, i], result[:, :i])
        else:
            correction = np.einsum("ji,jik->jk", R[:, i, i + 1 :], result[:, i + 1 :])
        result[:, i] = (rhs[:, i] - correction) / R[:, i, i, None]
    return result


@dataclass(slots=True)
class QRProfile(base._Profile):
    qr_mask: np.ndarray | None = None

    def normal_solve(self, rhs: np.ndarray) -> np.ndarray:
        assert self.qr_mask is not None and self.triangular is not None
        if not self.smooth:
            return np.zeros_like(rhs)
        result = np.empty_like(rhs)
        mask = self.qr_mask
        R = self.triangular[mask]
        result[mask] = triangular_solve(
            R, triangular_solve(R, rhs[mask], transpose=True)
        )
        if np.any(~mask):
            assert self.right is not None and self.inverse_squared is not None
            right = self.right[~mask]
            result[~mask] = right.swapaxes(1, 2) @ (
                self.inverse_squared[~mask, :, None] * (right @ rhs[~mask])
            )
        return result


def profile(
    M: np.ndarray, I: np.ndarray, mass: np.ndarray, tau: float
) -> base._Profile:
    """Same profiled LS/ridge target and unsafe SVD semantics, per center."""
    COUNTS["calls"] += 1
    J, p, m = M.shape
    if m <= 2 or p < m:
        COUNTS["delegated_centers"] += J
        return ORIGINAL_PROFILE(M, I, mass, tau)
    if tau:
        design = np.concatenate(
            (M, np.broadcast_to(np.sqrt(tau) * np.eye(m), (J, m, m))), axis=1
        )
        target = np.concatenate((I, np.zeros((J, m))), axis=1)
    else:
        design, target = M, I
    Q, R = np.linalg.qr(design, mode="reduced")
    # Conservative full-rank guard matches the existing m2 conditioning domain.
    singular = np.linalg.svd(R, compute_uv=False)
    absolute_guard = (
        64 * np.finfo(float).eps * np.maximum(1.0, np.linalg.norm(design, axis=(1, 2)))
    )
    safe = singular[:, -1] > np.maximum(
        absolute_guard, np.finfo(float).eps ** 0.25 * singular[:, 0]
    )
    COUNTS["qr_centers"] += int(np.count_nonzero(safe))
    COUNTS["svd_centers"] += int(np.count_nonzero(~safe))
    coefficients = np.empty((J, m))
    coordinates = Q[safe].swapaxes(1, 2) @ target[safe, :, None]
    coefficients[safe] = triangular_solve(R[safe], coordinates).squeeze(-1)
    ranks = np.full(J, m)
    right = inverse_squared = None
    smooth = True
    if np.any(~safe):
        reference = base._profile_svd(M[~safe], I[~safe], mass[~safe], tau)
        coefficients[~safe] = reference.coefficients
        ranks[~safe] = reference.ranks
        smooth = reference.smooth
        right = np.zeros((J, m, m))
        inverse_squared = np.zeros((J, m))
        right[~safe] = reference.right
        inverse_squared[~safe] = reference.inverse_squared
    residual = I - (M @ coefficients[..., None]).squeeze(-1)
    value = float(
        np.sum(
            mass * (np.sum(residual**2, axis=1) + tau * np.sum(coefficients**2, axis=1))
        )
    )
    if not np.isfinite(value) or not np.all(np.isfinite(coefficients)):
        raise RuntimeError("non-finite QR local profile")
    return QRProfile(
        M=M,
        coefficients=coefficients,
        residual=residual,
        right=right,
        inverse_squared=inverse_squared,
        ranks=ranks,
        smooth=smooth,
        value=value,
        triangular=R,
        qr_mask=safe,
    )
