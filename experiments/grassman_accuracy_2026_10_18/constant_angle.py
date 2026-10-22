"""EXACT/NUMERICAL cache for structurally constant rank-one scalar curves."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from ADP.solver import grassman as baseline

from .candidate import AngleProfile as GuardedAngleProfile


class AngleProfile:
    """Retain constant losses; refit all accepted full designs as before."""

    def __init__(self, M, w, a, I, mass, tau):
        self.M, self.w, self.a = M, w, a
        self.u = M @ a
        rows = np.any(M != 0, axis=2) | (w != 0)
        nonzero_rows = np.count_nonzero(rows, axis=1)
        constant = nonzero_rows == 0
        if not tau and M.shape[2] > 1:
            complement = np.linalg.svd(a[None, :], full_matrices=True)[2][1:].T
            constant_norm2 = np.sum((M @ complement) ** 2, axis=(1, 2))
            full_bound2 = (
                constant_norm2
                + (np.linalg.norm(self.u, axis=1) + np.linalg.norm(w, axis=1)) ** 2
            )
            constant |= (nonzero_rows == 1) & (
                constant_norm2 > np.sqrt(np.finfo(float).eps) * full_bound2
            )
        loss = np.sum(np.where(rows, 0.0, I) ** 2, axis=1)
        self.constant_value = float(mass[constant] @ loss[constant])
        self.constant_centers = int(np.count_nonzero(constant))
        active = ~constant
        self.curve = (
            GuardedAngleProfile(M[active], w[active], a, I[active], mass[active], tau)
            if np.any(active)
            else None
        )
        self.evaluations = 0

    def design(self, t):
        return (
            self.M
            + (self.u * (np.cos(t) - 1) + self.w * np.sin(t))[:, :, None] * self.a
        )

    def value(self, t):
        self.evaluations += 1
        return self.constant_value + (
            self.curve.value(t) if self.curve is not None else 0.0
        )


_spec = importlib.util.spec_from_file_location(
    "ADP.solver._constant_angle_core", Path(baseline.__file__)
)
assert _spec is not None and _spec.loader is not None
_core = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _core
_spec.loader.exec_module(_core)
vars(_core).update(_AngleProfile=AngleProfile)
solve = _core.solve
