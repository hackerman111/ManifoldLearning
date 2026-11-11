"""Opt-in moment integrations, preserving the live ADP engine/defaults.

The initializer override is process-local and unsuitable for concurrent threads.
Campaign fits run in isolated processes. No truth is passed to fitting code.
"""

from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import numpy as np

from ADP import ADP_Config
from ADP.core.solver_common import HPAOResult
from ADP.engine.common import index_fit
from ADP.engine.common.logger import IndexProfiler
from ADP.engine.common.statistic import calculate_statistics
from ADP.solver.grassman import _profile, _project, solve as grassman
from experiments import local_polynomial as poly
from experiments.weak_moments import fit as moment_fit

ARMS = ("adp", "residual_start", "budget", "proposal", "wls1", "wls2")


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


class ClockProfiler(IndexProfiler):
    """Stage wall time without tracemalloc's per-allocation timing overhead."""

    def __init__(self):
        self.seconds = {}

    @contextmanager
    def stage(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = (
                self.seconds.get(name, 0.0) + time.perf_counter() - start
            )


def frozen_profile(index, u, target, mass):
    """Exactly the current tau=0 inner objective, with its numerical rank rule."""
    return _profile(_project(u, index.T), target, mass, 0.0)


def propose(index, u, target, mass, quadratic):
    """Try ten rank-one rotations using a residual-moment direction.

    Only the fixed U,I,mass profile determines acceptance. Moment eigenvalues
    never enter the objective. Refit local slopes at every trial.
    """
    index, quadratic = np.asarray(index), np.asarray(quadratic)
    m, d = index.shape
    if quadratic.shape != (d, d) or not np.isfinite(quadratic).all():
        raise ValueError("quadratic moment must be a finite d by d matrix")
    if not np.allclose(index @ index.T, np.eye(m), atol=1e-10):
        raise ValueError("index rows must be orthonormal")
    state = frozen_profile(index, u, target, mass)
    before = state.value
    q = (quadratic + quadratic.T) / 2
    q -= index.T @ (index @ q)
    q -= (q @ index.T) @ index
    values, vectors = np.linalg.eigh((q + q.T) / 2)
    k = int(np.argmax(np.abs(values)))
    diag: dict[str, Any] = {
        "before": before,
        "after": before,
        "accepted": False,
        "trials": 0,
        "eigenvalue": float(values[k]),
        "row": None,
        "angle": None,
    }
    guard = 64 * np.finfo(float).eps * max(1.0, np.linalg.norm(quadratic, 2))
    if abs(values[k]) <= guard:
        return index.copy(), diag
    v = vectors[:, k] - index.T @ (index @ vectors[:, k])
    norm = np.linalg.norm(v)
    if norm <= 64 * np.finfo(float).eps:
        return index.copy(), diag
    v /= norm
    if v[np.argmax(np.abs(v))] < 0:
        v = -v
    best = index.copy()
    margin = 64 * np.finfo(float).eps * max(1.0, before)
    for row in range(m):
        for angle in (-np.pi / 8, np.pi / 8, -np.pi / 4, np.pi / 4, np.pi / 2):
            trial = index.copy()
            trial[row] = np.cos(angle) * index[row] + np.sin(angle) * v
            value = frozen_profile(trial, u, target, mass).value
            diag["trials"] += 1
            if value < diag["after"] - margin:
                best = trial
                diag.update(after=value, accepted=True, row=row, angle=float(angle))
    return best, diag


def fit(x, y, seed, arm, *, config_overrides=None):
    if arm not in ARMS:
        raise ValueError(f"unknown integration arm {arm}")
    cache, stages = {}, []
    moment = (
        moment_fit(x, y)
        if arm in ("residual_start", "proposal", "wls1", "wls2")
        else None
    )
    residual_start = arm in ("residual_start", "wls1", "wls2")
    original_init = index_fit._initial_index

    def initialize(*args, **kwargs):
        if residual_start:
            assert moment is not None
            return moment["basis"].T.copy(), np.empty(0)
        return original_init(*args, **kwargs)

    def statistics(X, Y, weights, directions, batch_size=32, *, normalized=False):
        blocks = list(weights)
        cache["weights"] = digest(*(w for _, w in blocks))
        cache["directions"] = digest(directions)
        if arm in ("wls1", "wls2"):
            cache["local"] = poly.neighborhoods(X, Y, blocks)
        return calculate_statistics(
            X, Y, iter(blocks), directions, batch_size=batch_size, normalized=normalized
        )

    def solve(index, stats):
        proposal = None
        before = frozen_profile(index, stats.U, stats.I, stats.mass).value
        if arm in ("wls1", "wls2"):
            result = poly.solve(
                index, cache["local"], degree=int(arm[-1]), ridge=1e-4, max_steps=40
            )
        else:
            result = grassman(index, stats.U, stats.I, mass=stats.mass)
            if arm in ("budget", "proposal"):
                candidate = result.index
                if arm == "proposal":
                    assert moment is not None
                    candidate, proposal = propose(
                        candidate, stats.U, stats.I, stats.mass, moment["quadratic"]
                    )
                candidate_state = frozen_profile(
                    candidate, stats.U, stats.I, stats.mass
                )
                second = grassman(candidate, stats.U, stats.I, mass=stats.mass)
                evaluated = frozen_profile(second.index, stats.U, stats.I, stats.mass)
                first_diag = result.diagnostics
                if evaluated.value > candidate_state.value:
                    second = HPAOResult(
                        candidate,
                        candidate_state.coefficients,
                        {
                            "method": "two_call_guarded_return",
                            "rejected_second_call": second.diagnostics,
                            "converged": False,
                            "stop_reason": "second_call_increase_rejected",
                            "profile_objective": candidate_state.value,
                            "loss": candidate_state.value / 2,
                            "profile_history": (candidate_state.value,),
                        },
                    )
                second.diagnostics["first_call"] = first_diag
                result = second
        after = frozen_profile(result.index, stats.U, stats.I, stats.mass).value
        stages.append(
            {
                "weights": cache["weights"],
                "directions": cache["directions"],
                "solver": result.diagnostics,
                "proposal": proposal,
                "original_before": before,
                "original_after": after,
            }
        )
        return result

    options: dict[str, Any] = {
        "seed": seed,
        "N_lin": 50,
        "N_J": 100,
        "N_loc": 20,
        "N_phi": 12,
        "outer_steps": 8,
        "select_step": "best",
    }
    options.update(config_overrides or {})
    config = ADP_Config(**options)
    if (
        config.gpu
        or config.estimator != "new"
        or config.training_set != "all"
        or config.experimental_centers is not None
    ):
        raise ValueError("adapter requires CPU/new/all without experimental centers")
    profiler = ClockProfiler()
    # Private hook is restored even when fit_index raises. No production edit.
    with patch.object(index_fit, "_initial_index", initialize):
        result = index_fit.fit_index(
            x,
            y,
            config=config,
            mode="multi",
            index_dim=2,
            solve=solve,
            statistics_function=statistics,
            profiler=profiler,
        )
    return {
        "basis": result.index.T,
        "initial_basis": result.initial_index.T,
        "diagnostics": {
            "stop_reason": result.metadata["stop_reason"],
            "selected_iteration": result.metadata["selected_iteration"],
            "initial_hash": digest(result.initial_index),
            "stages": stages,
            "stage_seconds": profiler.seconds,
            "converged": all(s["solver"].get("converged", False) for s in stages),
        },
    }
