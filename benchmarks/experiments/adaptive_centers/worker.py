"""One isolated full-fit observation for the adaptive-center experiment."""

from __future__ import annotations

import argparse
import dataclasses
import json
import resource
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_info

from .common import (
    ARMS,
    CASES,
    FIT_SETTINGS,
    arm_config,
    array_hash,
    environment,
    json_value,
)
from .data import generate


def _compact_trace(trace: list[dict]) -> list[dict]:
    keep = {
        "iteration",
        "k",
        "h",
        "factor",
        "err",
        "alpha",
        "rho",
        "mean_mass",
        "eigenvalues",
        "stop_reason",
        "solver",
        "converged",
        "iterations",
        "certificate",
        "selected_centers",
        "active_centers",
        "coverage",
        "covered_mass",
        "fallback_count",
        "selection_time_seconds",
        "statistics_time_seconds",
        "experimental_centers",
    }
    return [{key: value for key, value in row.items() if key in keep} for row in trace]


def _certificate_summary(diagnostics: list[Any]) -> dict[str, Any]:
    rows = [dict(row) if isinstance(row, dict) else vars(row) for row in diagnostics]
    converged = [bool(row.get("converged", False)) for row in rows]
    return {
        "count": len(rows),
        "converged_count": sum(converged),
        "all_converged": bool(rows) and all(converged),
        "stop_reasons": [row.get("stop_reason") for row in rows],
        "lsmr_iterations_total": [row.get("lsmr_iterations_total") for row in rows],
        "lsmr_solves_total": [row.get("lsmr_solves_total") for row in rows],
        "accepted_steps": [row.get("accepted_steps") for row in rows],
        "certificate_failures": [row.get("certificate_failures") for row in rows],
    }


def run(case_name: str, seed: int, arm_name: str) -> dict[str, Any]:
    from ADP import ADP_Config, ADP_multi_index

    case = next(item for item in CASES if item.name == case_name)
    arm = next(item for item in ARMS if item.name == arm_name)
    pools = threadpool_info()
    unconstrained = [pool for pool in pools if pool.get("num_threads") != 1]
    if unconstrained:
        raise RuntimeError(f"thread limit verification failed: {unconstrained}")

    problem = generate(case, seed)
    experiment_config = arm_config(arm, case)
    config = ADP_Config(
        seed=seed,
        N_J=int(FIT_SETTINGS["N_J"]),
        N_phi=int(FIT_SETTINGS["N_phi"]),
        N_loc=int(FIT_SETTINGS["N_loc"]),
        N_lin=FIT_SETTINGS["N_lin_factor"] * case.d,
        a=float(FIT_SETTINGS["a"]),
        h_min=float(FIT_SETTINGS["h_min"]),
        index_init=str(FIT_SETTINGS["index_init"]),
        estimator=str(FIT_SETTINGS["estimator"]),
        select_step=str(FIT_SETTINGS["select_step"]),
        experimental_centers=experiment_config,
    )
    started = time.perf_counter()
    model = ADP_multi_index(case.index_dim, config).fit(problem.X, problem.Y)
    fit_wall = time.perf_counter() - started

    estimated = np.asarray(model.basis_, dtype=float)
    singular = np.linalg.svd(problem.truth.T @ estimated, compute_uv=False)
    sine_squared = np.maximum(0.0, 1.0 - np.minimum(1.0, singular) ** 2)
    edr_error = float(np.sqrt(np.mean(sine_squared)))
    orthogonality = float(
        np.linalg.norm(estimated.T @ estimated - np.eye(case.index_dim))
    )
    initial = np.asarray(model.result_.beta_init, dtype=float)
    trace = model.trace_
    profile = getattr(model, "profile_", {})
    effective = getattr(model, "effective_parameters_", {})
    solver_diagnostics = getattr(model, "solver_diagnostics_", [])
    last_center_diagnostic = trace[-1].get("experimental_centers", {}) if trace else {}
    selected_pool_indices = last_center_diagnostic.get("selected_pool_indices")
    if selected_pool_indices is None:
        selected_pool_indices = tuple(range(case.n))
    selected_pool_indices = [int(index) for index in selected_pool_indices]
    coefficients = np.asarray(model.coefficients_, dtype=float)
    coefficient_mapping = {
        "semantics": (
            "coefficients_[r] corresponds to selected_pool_indices[r]; with N_J=n, "
            "candidate pool row equals original X row"
        ),
        "selected_pool_indices": selected_pool_indices,
        "coefficient_shape": list(coefficients.shape),
        "coefficient_hash": array_hash(coefficients),
        "row_count_matches_selection": len(selected_pool_indices)
        == (coefficients.shape[0] if coefficients.ndim else 0),
        "values": coefficients,
    }
    selected_metadata = {
        key: value
        for key, value in effective.items()
        if any(
            token in key.lower()
            for token in ("center", "coverage", "selection", "fallback", "mass")
        )
    }
    row = {
        "status": "ok",
        "case": case.name,
        "seed": seed,
        "arm": arm.name,
        "shape": {"n": case.n, "d": case.d, "m": case.index_dim},
        "input_hashes": {
            "X": array_hash(problem.X),
            "Y": array_hash(problem.Y),
            "truth": array_hash(problem.truth),
            "initial_basis": array_hash(initial),
            "final_basis": array_hash(estimated),
        },
        "seed_streams": problem.seed_streams,
        "effective_config": dataclasses.asdict(config),
        "experimental_config": None
        if experiment_config is None
        else dataclasses.asdict(experiment_config),
        "edr_error": edr_error,
        "orthogonality_error": orthogonality,
        "fit_wall_sec": fit_wall,
        "profile": profile,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "outer_stop_reason": model.result_.stop_reason,
        "outer_iterations": len(trace),
        "inner_certificates": _certificate_summary(solver_diagnostics),
        "coefficient_mapping": coefficient_mapping,
        "center_diagnostics": selected_metadata,
        "trace": _compact_trace(trace),
        "environment": environment(),
    }
    return json_value(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        row = run(args.case, args.seed, args.arm)
    except BaseException as exc:
        row = {
            "status": "error",
            "case": args.case,
            "seed": args.seed,
            "arm": args.arm,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=12),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_value(row), allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
