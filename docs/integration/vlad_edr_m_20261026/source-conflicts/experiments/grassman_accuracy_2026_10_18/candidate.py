"""Изолированный prototype: локальный SVD fallback в скалярном профиле.

Ядро загружается в отдельный namespace; production module не меняется.
Число шагов задаётся явно benchmark-ом. Вывод и границы: MATH.md.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ADP.core.solver_common import HPAOResult
from ADP.solver import grassman as baseline

_spec = importlib.util.spec_from_file_location(
    "ADP.solver._accuracy_core", Path(baseline.__file__)
)
assert _spec is not None and _spec.loader is not None
_core = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _core
_spec.loader.exec_module(_core)


class AngleProfile:
    """Локальный Schur-профиль: unsafe центры refit-ятся независимо от safe."""

    def __init__(
        self,
        M: np.ndarray,
        w: np.ndarray,
        a: np.ndarray,
        I: np.ndarray,
        mass: np.ndarray,
        tau: float,
    ) -> None:
        self.M, self.w, self.a, self.I, self.mass, self.tau = M, w, a, I, mass, tau
        J, p, m = M.shape
        complement = np.linalg.svd(a[None, :], full_matrices=True)[2][1:].T
        C = M @ complement
        self.u = (M @ a).reshape(J, p)
        self.raw_norms = np.stack(
            (np.linalg.norm(self.u, axis=1), np.linalg.norm(w, axis=1)), axis=-1
        )
        data = np.stack((I, self.u, w), axis=-1)
        self.safe = np.ones(J, dtype=bool)
        self.constant_smallest2 = np.full(J, np.inf)
        if m > 1:
            if tau:
                C = np.concatenate(
                    (
                        C,
                        np.broadcast_to(
                            np.sqrt(tau) * np.eye(m - 1), (J, m - 1, m - 1)
                        ),
                    ),
                    axis=1,
                )
                data = np.concatenate((data, np.zeros((J, m - 1, 3))), axis=1)
            left, singular, _ = np.linalg.svd(C, full_matrices=False)
            self.constant_smallest2 = singular[:, -1] ** 2
            keep = singular > np.finfo(float).eps * max(p, m) * singular[:, :1]
            if not tau:
                self.safe = (np.count_nonzero(keep, axis=1) == m - 1) & (
                    singular[:, -1] > np.sqrt(np.finfo(float).eps) * singular[:, 0]
                )
            coordinates = left.swapaxes(1, 2) @ data
            data = data - left @ (coordinates * keep[:, :, None])
        self.constant_norm2 = np.sum(C**2, axis=(1, 2))
        gram = data.swapaxes(1, 2) @ data
        self.k, self.b, self.c = gram[:, 0, 0], gram[:, 0, 1], gram[:, 0, 2]
        self.a2, self.d, self.e = gram[:, 1, 1], gram[:, 1, 2], gram[:, 2, 2]
        self.guard = 64 * np.finfo(float).eps * (self.a2 + self.e + tau)
        self.evaluations = self.refit_centers = 0

    def design(self, t: float) -> np.ndarray:
        delta = self.u * (np.cos(t) - 1) + self.w * np.sin(t)
        return self.M + delta[:, :, None] * self.a

    def value(self, t: float) -> float:
        self.evaluations += 1
        ct, st = np.cos(t), np.sin(t)
        denominator = self.a2 * ct**2 + 2 * self.d * ct * st + self.e * st**2 + self.tau
        safe = self.safe & (denominator > self.guard)
        full_norm_bound2 = (
            self.constant_norm2
            + (abs(ct) * self.raw_norms[:, 0] + abs(st) * self.raw_norms[:, 1]) ** 2
            + self.tau
        )
        # Schur cutoff должен учитывать масштаб всей augmented design.
        cutoff = np.sqrt(np.finfo(float).eps) * full_norm_bound2
        safe &= (denominator > cutoff) & (self.constant_smallest2 > cutoff)
        numerator = self.b[safe] * ct + self.c[safe] * st
        value = float(
            self.mass[safe] @ (self.k[safe] - numerator**2 / denominator[safe])
        )
        unsafe = ~safe
        if np.any(unsafe):
            self.refit_centers += int(np.count_nonzero(unsafe))
            delta = self.u[unsafe] * (ct - 1) + self.w[unsafe] * st
            design = self.M[unsafe] + delta[:, :, None] * self.a
            value += baseline._profile(
                design, self.I[unsafe], self.mass[unsafe], self.tau
            ).value
        return value


vars(_core)["_AngleProfile"] = AngleProfile


def solve(
    index_init: np.ndarray, U: np.ndarray, I: np.ndarray, **kwargs: Any
) -> HPAOResult:
    """Исторический scalar prototype сохраняет настройки исходного solve."""
    defaults = {
        "max_steps": 5,
        "relative_gn": False,
        "quadratic_trials": 0,
        "constant_centers": False,
        "smooth_max_steps": 5,
    }
    return _core.solve(index_init, U, I, **(defaults | kwargs))
