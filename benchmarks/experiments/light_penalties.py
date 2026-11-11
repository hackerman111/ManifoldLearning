"""Opt-in cheap penalties using the unchanged projected-moment solver.

ESTIMATOR: coefficient ridge and a fixed initial-projector prior.
Virtual centers encode the projector penalty exactly when local_ridge=0.
"""

from __future__ import annotations

import time

import numpy as np

from ADP.core.ADP_Statistic import ADP_Statistics
from ADP.core.solver_common import HPAOResult
from ADP.solver import grassman

ARMS = ("base", "pad", "ridge0.001", "ridge0.01", "proj0.01", "proj0.1")


def prepare(stats: ADP_Statistics, index, anchor, arm):
    """Return solver arrays, frozen ridge and scale diagnostics; no mutation."""
    if arm not in ARMS:
        raise ValueError(f"unknown penalty arm: {arm}")
    index, anchor = np.asarray(index), np.asarray(anchor)
    m, d = index.shape
    if anchor.shape != (d, m) or not np.allclose(
        anchor.T @ anchor, np.eye(m), atol=1e-10, rtol=1e-10
    ):
        raise ValueError("anchor must have orthonormal columns with shape (d,m)")
    energy = float(np.sum(stats.mass[:, None] * stats.I**2))
    tau, coefficient, extra = 0.0, 0.0, 0
    info = {
        "arm": arm,
        "target_energy": energy,
        "virtual_centers": 0,
        "tolerance_multiplier": 1.0,
    }
    if arm.startswith("ridge"):
        scale = float(np.median(np.sum((stats.U @ index.T) ** 2, axis=(1, 2))) / m)
        tau = float(arm.removeprefix("ridge")) * scale
        info["ridge_scale"] = scale
    if arm.startswith("proj"):
        beta = float(arm.removeprefix("proj"))
        coefficient = beta * energy / m
        extra = m if coefficient > 0 else 0
        info.update(
            virtual_centers=extra,
            projector_coefficient=coefficient,
            tolerance_multiplier=max(1.0, energy)
            / max(1.0, energy + extra * coefficient),
        )
    info["local_ridge"] = tau
    if arm not in ("pad", "proj0.01", "proj0.1"):
        return stats.U, stats.I, stats.mass, tau, info

    j, p, _ = stats.U.shape
    rows = max(p, d)
    u = np.zeros((j + extra, rows, d))
    target = np.zeros((j + extra, rows))
    u[:j, :p] = stats.U
    target[:j, :p] = stats.I
    mass = np.r_[stats.mass, np.full(extra, coefficient)]
    if extra:
        u[j:, :d] = np.eye(d)
        target[j:, :d] = anchor.T
    info["padded_rows"] = rows - p
    return u, target, mass, tau, info


def solve(index, stats, anchor, arm, *, max_steps=20, tol=1e-6):
    """Fit the penalized profile and return only real-center coefficients."""
    started = time.perf_counter()
    u, target, mass, tau, info = prepare(stats, index, anchor, arm)
    preparation = time.perf_counter() - started
    started = time.perf_counter()
    result = grassman.solve(
        index,
        u,
        target,
        mass=mass,
        local_ridge=tau,
        max_steps=max_steps,
        tol=tol * info["tolerance_multiplier"],
        lambda_prox=0.05,
    )
    diagnostics = dict(result.diagnostics)
    diagnostics["penalty"] = dict(
        info,
        prepare_seconds=preparation,
        solve_seconds=time.perf_counter() - started,
    )
    return HPAOResult(
        result.index,
        result.coefficients[: len(stats.mass)].copy(),
        diagnostics,
    )
