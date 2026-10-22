"""NUMERICAL per-center QR/SVD; APPROXIMATE scalar10/smooth5 solver."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from ADP.solver import grassman as baseline
from ADP.solver.LSMR import _normalize_index, _validate_inputs

from .candidate import AngleProfile


def profile(
    M: np.ndarray, I: np.ndarray, mass: np.ndarray, tau: float
) -> baseline._Profile:
    """Preserve global GN guards; localize nonsmooth refits and SVD factors."""
    J, p, m = M.shape
    if m > 2 or p < m:
        return baseline._profile(M, I, mass, tau)
    design = M
    if tau:
        design = np.concatenate(
            (M, np.broadcast_to(np.sqrt(tau) * np.eye(m), (J, m, m))), axis=1
        )
    norms = np.linalg.norm(design, axis=1)
    a = norms[:, 0]
    guard = 64 * np.finfo(float).eps * np.maximum(1.0, np.linalg.norm(norms, axis=1))
    safe = (a > guard) & np.all(np.isfinite(norms), axis=1)
    q0 = np.divide(
        design[:, :, 0],
        a[:, None],
        out=np.zeros_like(design[:, :, 0]),
        where=safe[:, None],
    )
    b = np.zeros(J)
    c = np.ones(J)
    q1 = np.zeros_like(q0)
    if m == 2:
        b = np.einsum("jp,jp->j", q0, design[:, :, 1])
        q1 = design[:, :, 1] - q0 * b[:, None]
        correction = np.einsum("jp,jp->j", q0, q1)
        q1 -= q0 * correction[:, None]
        b += correction
        c = np.linalg.norm(q1, axis=1)
        largest = (np.hypot(a + c, b) + np.hypot(a - c, b)) * 0.5
        smallest = a * np.divide(c, largest, out=np.zeros_like(c), where=largest > 0)
        safe &= smallest > np.maximum(guard, np.finfo(float).eps ** 0.25 * largest)
        q1 = np.divide(q1, c[:, None], out=np.zeros_like(q1), where=safe[:, None])
    if not np.any(safe):
        return baseline._profile_svd(M, I, mass, tau)
    all_safe = bool(np.all(safe))
    rows = slice(None) if all_safe else safe
    safe_count = J if all_safe else int(np.count_nonzero(safe))
    unsafe = ~safe
    difficult = None
    if np.any(unsafe):
        difficult = baseline._profile_svd(M[unsafe], I[unsafe], mass[unsafe], tau)
        if difficult.smooth:
            return baseline._profile(M, I, mass, tau)
    target = (
        np.concatenate((I[rows], np.zeros((safe_count, m))), axis=1) if tau else I[rows]
    )
    coordinates0 = np.einsum("jp,jp->j", q0[rows], target)
    coefficients_safe = np.empty((safe_count, m))
    if m == 1:
        coefficients_safe[:, 0] = coordinates0 / a[rows]
        R = a[rows, None, None]
    else:
        coordinates1 = np.einsum("jp,jp->j", q1[rows], target)
        coefficients_safe[:, 1] = coordinates1 / c[rows]
        coefficients_safe[:, 0] = (
            coordinates0 - b[rows] * coefficients_safe[:, 1]
        ) / a[rows]
        R = np.zeros((len(coefficients_safe), m, m))
        R[:, 0, 0], R[:, 0, 1], R[:, 1, 1] = a[rows], b[rows], c[rows]
    residual_safe = I[rows] - (M[rows] @ coefficients_safe[..., None]).squeeze(-1)
    if all_safe:
        coefficients, residual, ranks = coefficients_safe, residual_safe, np.full(J, m)
    else:
        assert difficult is not None
        coefficients, residual, ranks = (
            np.empty((J, m)),
            np.empty_like(I),
            np.empty(J, dtype=int),
        )
        coefficients[safe], residual[safe], ranks[safe] = (
            coefficients_safe,
            residual_safe,
            m,
        )
        coefficients[unsafe], residual[unsafe], ranks[unsafe] = (
            difficult.coefficients,
            difficult.residual,
            difficult.ranks,
        )
    value = float(
        np.sum(
            mass * (np.sum(residual**2, axis=1) + tau * np.sum(coefficients**2, axis=1))
        )
    )
    if not np.isfinite(value) or not np.all(np.isfinite(coefficients)):
        raise RuntimeError("non-finite local profile")
    if all_safe:
        return baseline._Profile(
            M, coefficients, residual, None, None, ranks, True, value, R
        )
    return baseline._Profile(
        M,
        coefficients,
        residual,
        np.broadcast_to(np.eye(m), (J, m, m)),
        np.zeros((J, m)),
        ranks,
        False,
        value,
    )


_spec = importlib.util.spec_from_file_location(
    "ADP.solver._selective_core", Path(baseline.__file__)
)
assert _spec is not None and _spec.loader is not None
_core = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _core
_spec.loader.exec_module(_core)
vars(_core).update(_profile=profile, _AngleProfile=AngleProfile)


def solve(index_init, U, I, *, max_steps=10, **kwargs):
    # Cap initial smooth GN to the original budget, not the rank-one budget.
    # Validation remains authoritative in the original solver.
    ridge = kwargs.get("local_ridge", 0.0)
    if (
        hasattr(U, "__cuda_array_interface__")
        or isinstance(max_steps, bool)
        or not isinstance(max_steps, (int, np.integer))
        or max_steps <= 0
        or not np.isfinite(ridge)
        or ridge < 0
    ):
        return _core.solve(index_init, U, I, max_steps=max_steps, **kwargs)
    B, checked_U, checked_I, checked_mass = _validate_inputs(
        index_init, U, I, kwargs.get("mass")
    )
    if B.ndim == 2:
        initial = profile(
            baseline._project(checked_U, _normalize_index(B).T),
            checked_I,
            checked_mass,
            ridge,
        )
        if initial.smooth:
            max_steps = min(max_steps, 5)
    return _core.solve(index_init, U, I, max_steps=max_steps, **kwargs)
