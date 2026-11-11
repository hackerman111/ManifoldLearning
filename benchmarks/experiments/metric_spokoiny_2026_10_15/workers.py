"""Isolated workers for the frozen hard-Spokoiny metric study."""

from __future__ import annotations

import argparse
import json
import resource
import time
from functools import partial
from pathlib import Path
from typing import Any

from .diagnostics import (
    array_hash,
    config_dict,
    environment,
    json_value,
    metric_patch,
    projector_metrics,
    restore_metric,
    sha256_bytes,
    solver_diagnostics,
)

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE_PATHS = (
    "benchmarks/grassman_benchmark.py",
    "benchmarks/experiments/data.py",
    "benchmarks/experiments/models.py",
    "benchmarks/experiments/runner.py",
    "ADP/core/ADP_Config.py",
    "ADP/core/multi/ADP_multi_index.py",
    "ADP/engine/common/index_fit.py",
    "ADP/engine/common/statistic.py",
    "ADP/engine/common/weights.py",
    "ADP/solver/grassman.py",
    "ADP/solver/grassman_research.py",
    "ADP/solver/_grassman_schur.py",
    "ADP/solver/LSMR.py",
    "benchmarks/experiments/metric_spokoiny_2026_10_15/PROTOCOL.md",
    "benchmarks/experiments/metric_spokoiny_2026_10_15/run.py",
    "benchmarks/experiments/metric_spokoiny_2026_10_15/workers.py",
    "benchmarks/experiments/metric_spokoiny_2026_10_15/analysis.py",
    "benchmarks/experiments/metric_spokoiny_2026_10_15/diagnostics.py",
)
ARMS = ("core", "metric2", "metric1", "isotropic2", "isotropic2_matched")
FULLFIT_ARMS = ("core", "metric2", "metric1")


def _solver(arm: str, budget: int):
    from ADP import ADP_solver
    from ADP.solver.grassman import solve as solve_core
    from ADP.solver.grassman_research import solve as solve_research

    if arm == "core":
        method = partial(
            solve_core,
            max_steps=budget,
            tol=1e-6,
            energy_tol=0.05,
            local_ridge=0.0,
            max_angle=0.5,
            workspace_bytes=16 * 1024**2,
        )
    else:
        method = partial(
            solve_research,
            max_steps=budget,
            tol=1e-6,
            rank=1 if arm == "metric1" else 2,
            local_ridge=0.0,
            max_angle=0.5,
            method="metric",
            metric_floor=1.0 if arm == "isotropic2" else 1e-6,
            workspace_bytes=16 * 1024**2,
        )
    return ADP_solver(method)


def _diag_row(diagnostics: dict[str, Any]) -> dict[str, Any]:
    return solver_diagnostics(diagnostics)


def _fullfit(
    seed: int, arm: str, budget: int, capture: bool, output: Path
) -> dict[str, Any]:
    import numpy as np

    from ADP import ADP_multi_index
    from benchmarks.grassman_benchmark import PARAMS, _make_problem

    X, y, truth, config = _make_problem(seed)
    if config.lambda_penalty != 0.05:
        raise RuntimeError("frozen protocol expects lambda_prox=lambda_penalty=0.05")
    original_metric = metric_patch(arm)
    solver = _solver(arm, budget)
    call_times: list[float] = []
    stats_refs: list[tuple[Any, Any, Any, Any]] = []
    captures: dict[int, dict[str, Any]] = {}
    last_capture: dict[str, Any] | None = None
    original_method = solver.method

    def measured(index: Any, U: Any, I: Any, **kwargs: Any):
        nonlocal last_capture
        if not stats_refs:
            stats_refs.append((U, I, kwargs.get("mass"), index))
        if capture:
            item = {
                "U": U,
                "I": I,
                "mass": kwargs["mass"],
                "index_init": index,
                "call_index": len(call_times),
            }
            last_capture = item
            if item["call_index"] in (0, 80, 160):
                captures[item["call_index"]] = {
                    **{
                        key: np.array(value, copy=True)
                        for key, value in item.items()
                        if key != "call_index"
                    },
                    "call_index": item["call_index"],
                }
        started = time.perf_counter()
        result = original_method(index, U, I, **kwargs)
        call_times.append(time.perf_counter() - started)
        return result

    solver.method = measured
    model = ADP_multi_index(truth.shape[1], config, solver)
    started = time.perf_counter()
    model.fit(X, y)
    fit_time = time.perf_counter() - started
    restore_metric(original_metric)

    trace = model.trace_
    solver_rows = [_diag_row(row.get("solver", {})) for row in trace]
    first_stats = stats_refs[0] if stats_refs else None
    hashes = {
        "X": array_hash(X),
        "y": array_hash(y),
        "truth_basis": array_hash(truth),
    }
    if first_stats is not None:
        U, I, mass, index = first_stats
        hashes.update(
            first_index=array_hash(index),
            first_U=array_hash(U),
            first_I=array_hash(I),
            first_mass=array_hash(mass),
        )
    metrics = projector_metrics(model.basis_, truth)
    row: dict[str, Any] = {
        "mode": "capture" if capture else "fullfit",
        "seed": seed,
        "arm": arm,
        "budget": budget,
        "status": "ok",
        "shape": {key: PARAMS[key] for key in ("n", "d", "m", "N_J", "N_phi", "N_lin")},
        "dtype": str(X.dtype),
        "config": config_dict(config),
        "data_hashes": hashes,
        "fit_time_sec": fit_time,
        "sum_solver_time_sec": float(sum(call_times)),
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "outer_iterations": model.effective_parameters_.get("outer_iterations"),
        "selected_iteration": model.effective_parameters_.get("selected_iteration"),
        "fit_stop_reason": model.result_.stop_reason,
        "model_profile": json_value(model.profile_),
        "final_solver": solver_rows[-1] if solver_rows else {},
        "solver_summary": {
            "calls": len(solver_rows),
            "converged_calls": sum(bool(item.get("converged")) for item in solver_rows),
            "stop_reasons": {
                reason: sum(item.get("stop_reason") == reason for item in solver_rows)
                for reason in sorted(
                    {str(item.get("stop_reason")) for item in solver_rows}
                )
            },
            "accepted_steps": sum(
                int(item.get("accepted_steps", 0)) for item in solver_rows
            ),
            "rank_guard_fallbacks": sum(
                int(item.get("rank_guard_fallbacks", 0)) for item in solver_rows
            ),
            "local_rank_loss": sum(
                int(item.get("local_rank_loss", 0)) for item in solver_rows
            ),
        },
        **metrics,
        "environment": environment(ROOT, HERE / "PROTOCOL.md", SOURCE_PATHS),
    }
    if capture:
        if 160 not in captures and last_capture is not None:
            call_index = last_capture["call_index"]
            captures[call_index] = {
                **{
                    key: np.array(value, copy=True)
                    for key, value in last_capture.items()
                    if key != "call_index"
                },
                "call_index": call_index,
            }
        row["capture_curves"] = _capture_curves(captures, trace, truth)
        for call_index, arrays in captures.items():
            path = output / "captures" / f"{arm}_seed{seed}_call{call_index}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(path)
            np.savez_compressed(
                path,
                **{
                    **{k: v for k, v in arrays.items() if k != "call_index"},
                    "truth_basis": np.asarray(truth),
                },
            )
        row["capture_files"] = [
            str(output / "captures" / f"{arm}_seed{seed}_call{index}.npz")
            for index in sorted(captures)
        ]
    return row


def _capture_curves(
    captures: dict[int, dict[str, Any]], trace: list[dict[str, Any]], truth: Any
) -> list[dict[str, Any]]:
    import numpy as np

    from ADP.solver import grassman

    rows = []
    for call_index, fitrow in enumerate(trace):
        basis = np.asarray(fitrow.get("basis"))
        solver = fitrow.get("solver", {})
        if basis.ndim != 2:
            continue
        rows.append(
            {
                "call_index": call_index,
                "h": fitrow.get("h"),
                "rho_or_alpha": fitrow.get("factor"),
                "outer_tensor_eigenvalues": list(fitrow.get("eigenvalues", ())),
                "solver_live_gradient_certificate": solver.get("riemannian_gradient"),
                "stationarity_applicable": solver.get("stationarity_applicable"),
                "local_rank_loss_count": solver.get("local_rank_loss"),
                "stop_reason": solver.get("stop_reason"),
                **projector_metrics(basis, truth),
            }
        )
    for call_index, item in sorted(captures.items()):
        U, I, mass, basis = item["U"], item["I"], item["mass"], item["index_init"].T
        state = grassman._profile(grassman._project(U, basis), I, mass, 0.0)
        gradient = grassman._gradient(basis, U, state, mass)
        scale = max(1.0, float(np.sum(mass[:, None] * I**2)))
        fitrow = trace[call_index] if call_index < len(trace) else {}
        input_row = {
            "capture_kind": "independent_snapshot_input",
            "call_index": call_index,
            "h": fitrow.get("h"),
            "rho_or_alpha": fitrow.get("factor"),
            "profile_objective_at_input": state.value,
            "live_gradient_certificate": float(np.linalg.norm(gradient) / scale),
            "certificate_scale": scale,
            "smooth": state.smooth,
            "local_rank_min": int(np.min(state.ranks)),
            "local_rank_median": float(np.median(state.ranks)),
            "local_rank_loss_count": int(
                np.count_nonzero(state.ranks < basis.shape[1])
            ),
            "R_eigenvalues": np.linalg.eigvalsh(
                state.coefficients.T @ (mass[:, None] * state.coefficients)
            ).tolist(),
        }
        input_row["record_key"] = f"snapshot_{call_index}"
        rows.append(input_row)
    return rows


def _frozen(seed: int, arm: str, budget: int, snapshot: Path) -> dict[str, Any]:
    import numpy as np

    from ADP.solver import grassman
    from ADP.solver.LSMR import _normalize_index

    arrays = np.load(snapshot, allow_pickle=False)
    B, U, I, mass = (arrays[name] for name in ("index_init", "U", "I", "mass"))
    truth = arrays["truth_basis"]
    original_metric = metric_patch(arm)
    solver = _solver(arm, budget)
    kwargs: dict[str, Any] = {"mass": mass, "lambda_prox": 0.05}
    before_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    started = time.perf_counter()
    result = solver.method(B, U, I, **kwargs)
    elapsed = time.perf_counter() - started
    restore_metric(original_metric)
    basis = _normalize_index(result.index).T
    state = grassman._profile(grassman._project(U, basis), I, mass, 0.0)
    gradient = grassman._gradient(basis, U, state, mass)
    scale = max(1.0, float(np.sum(mass[:, None] * I**2)))
    initial = _normalize_index(B).T
    initial_state = grassman._profile(grassman._project(U, initial), I, mass, 0.0)
    diag = _diag_row(result.diagnostics)
    return {
        "mode": "frozen",
        "seed": seed,
        "arm": arm,
        "budget": budget,
        "status": "ok",
        "snapshot": str(snapshot.relative_to(ROOT)),
        "snapshot_sha256": sha256_bytes(snapshot.read_bytes()),
        "input_hashes": {
            name: array_hash(value)
            for name, value in (("B", B), ("U", U), ("I", I), ("mass", mass))
        },
        "shape": {
            "J": int(U.shape[0]),
            "p": int(U.shape[1]),
            "d": int(U.shape[2]),
            "m": int(B.shape[0]),
        },
        "time_sec": elapsed,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "additional_peak_rss_kib": max(
            0, int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) - before_rss
        ),
        "objective_initial": initial_state.value,
        "objective_final": state.value,
        "objective_ratio": state.value / max(initial_state.value, np.finfo(float).tiny),
        "live_gradient_raw": float(np.linalg.norm(gradient)),
        "certificate_scale": scale,
        "live_gradient_certificate": float(np.linalg.norm(gradient) / scale),
        "smooth": state.smooth,
        "local_rank_min": int(np.min(state.ranks)),
        "local_rank_loss_count": int(np.count_nonzero(state.ranks < basis.shape[1])),
        "R_eigenvalues": np.linalg.eigvalsh(
            state.coefficients.T @ (mass[:, None] * state.coefficients)
        ).tolist(),
        **projector_metrics(basis, truth),
        "diagnostics": diag,
        "environment": environment(ROOT, HERE / "PROTOCOL.md", SOURCE_PATHS),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("fullfit", "capture", "frozen"), required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "frozen":
        if args.snapshot is None:
            parser.error("--snapshot is required for frozen mode")
        row = _frozen(args.seed, args.arm, args.budget, args.snapshot)
    else:
        row = _fullfit(
            args.seed, args.arm, args.budget, args.mode == "capture", args.output
        )
    print(json.dumps(json_value(row), allow_nan=False))


if __name__ == "__main__":
    main()
