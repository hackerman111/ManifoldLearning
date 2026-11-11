"""Один ADP fit; окружение потоков задаёт родитель до импорта NumPy."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from ADP.core.ADP_Config import ADP_Config
from ADP.core.ADP_Solver import ADP_solver
from ADP.core.multi.ADP_multi_index import ADP_multi_index
from ADP.solver.grassman import solve as solve_grassman
from ADP.solver.LSMR import solve as solve_lsmr


def solver_for(method: str):
    if method == "ADP-Grassmann-accurate":
        from experiments.grassman_accuracy_2026_10_18.candidate import solve

        return solve
    if method == "ADP-Grassmann-constant":
        from experiments.grassman_accuracy_2026_10_18.constant_centers import solve

        return solve
    if method == "ADP-Grassmann-armijo":
        from experiments.grassman_accuracy_2026_10_18.backtracking import solve

        return solve
    if method == "ADP-Grassmann-quadratic":
        from experiments.grassman_accuracy_2026_10_18.quadratic import solve

        return solve
    if method == "ADP-Grassmann-intrinsic":
        from experiments.grassman_accuracy_2026_10_18.intrinsic import solve

        return solve
    if method == "ADP-Grassmann-warmup":
        from experiments.grassman_accuracy_2026_10_18.warmup import solve

        return solve
    if method == "ADP-Grassmann-conjugate":
        from experiments.grassman_accuracy_2026_10_18.conjugate_guard import solve

        return solve
    if method in {
        "ADP-Grassmann-relative",
        "ADP-Grassmann-combined10",
        "ADP-Grassmann-combined",
    }:
        return solve_grassman
    return {"ADP": solve_lsmr, "ADP-Grassmann": solve_grassman}.get(method)


def run(job: dict) -> None:
    x = np.loadtxt(job["x"], delimiter=",", ndmin=2)
    y = np.loadtxt(job["y"], delimiter=",", ndmin=1)
    method = job.get("method", "ADP")
    solver_method = solver_for(method)
    if solver_method is None:
        raise ValueError(f"unknown ADP solver variant: {method}")
    model = ADP_multi_index(
        job["m"],
        config=ADP_Config(seed=job["seed"], **job["config"]),
        solver=ADP_solver(solver_method, **job["solver"]),
    )
    started = time.perf_counter()
    model.fit(x, y)
    elapsed = time.perf_counter() - started
    np.savetxt(job["basis"], model.basis_, delimiter=",", fmt="%.17g")
    diagnostics = model.solver_diagnostics_
    meta = {
        "solver_variant": method,
        "fit_seconds": elapsed,
        "iterations": len(model.trace_),
        "converged": None,
        "all_inner_certified": bool(diagnostics)
        and all(d.get("converged", False) for d in diagnostics),
        "n_inner_certified": sum(d.get("converged") is True for d in diagnostics),
        "n_inner_total": len(diagnostics),
        "stop_reason": model.result_.stop_reason,
        "inner_diagnostics": diagnostics,
        "effective_parameters": {
            k: v
            for k, v in model.effective_parameters_.items()
            if k not in {"trace", "beta_init", "beta_final"}
            and isinstance(v, (str, int, float, bool, type(None)))
        },
    }
    Path(job["meta"]).write_text(json.dumps(meta, default=str, allow_nan=False))


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text()))
