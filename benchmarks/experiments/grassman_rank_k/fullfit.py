"""Isolated full-fit and first-statistic capture workers."""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections import Counter
from contextlib import suppress
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    OUT,
    SETTINGS,
    array_hash,
    environment,
    json_value,
    make_problem,
    projector_metrics,
)


class _Captured(Exception):
    pass


def _method(arm: str, budget: int, m: int):
    from ADP.solver.grassman import block_rank_k, solve as core
    from ADP.solver.grassman_exp import solve as experimental

    common = {**SETTINGS, "max_steps": budget}
    if arm == "core":
        return partial(
            core, **{k: v for k, v in common.items() if k != "block_max_steps"}
        )
    if arm == "rank_one":
        return partial(
            experimental,
            method="rank_one",
            **{k: v for k, v in common.items() if k != "block_max_steps"},
        )
    if arm == "spectral":
        return partial(
            experimental,
            method="spectral",
            **{k: v for k, v in common.items() if k != "block_max_steps"},
        )
    if arm == "flat":
        return partial(
            experimental,
            method="flat_gn",
            **{k: v for k, v in common.items() if k != "block_max_steps"},
        )
    options = {
        "block1": {"rank_k": 1},
        "block_full": {"rank_k": m, "rank_strategy": "energy"},
        "block_energy": {"rank_k": None, "rank_strategy": "energy"},
        "block_cost": {"rank_k": None, "rank_strategy": "cost"},
        "block_spectral": {
            "rank_k": None,
            "rank_strategy": "energy",
            "block_variant": "spectral",
        },
    }
    if arm not in options:
        raise ValueError(f"unknown arm: {arm}")
    return partial(block_rank_k, **common, **options[arm])


def _trace_summary(trace: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [entry.get("solver", {}) for entry in trace]
    calls = [call for call in calls if isinstance(call, dict)]

    def total(name: str) -> int:
        return sum(int(call.get(name, 0) or 0) for call in calls)

    gradients = [
        float(c["riemannian_gradient"])
        for c in calls
        if c.get("riemannian_gradient") is not None
        and np.isfinite(c["riemannian_gradient"])
    ]
    orthogonality = [
        float(c[name])
        for c in calls
        for name in ("orthogonality_error", "orthogonality")
        if c.get(name) is not None and np.isfinite(c[name])
    ]
    histories = [
        np.asarray(c["profile_history"], dtype=float)
        for c in calls
        if c.get("profile_history") is not None
    ]
    positive_diffs = [
        float(np.max(np.diff(history))) for history in histories if history.size > 1
    ]
    ranks = Counter(int(q) for c in calls for q in c.get("step_ranks", ()))
    return {
        "step_rank_histogram": dict(sorted(ranks.items())),
        "cost_rank_caps": [c["cost_rank_cap"] for c in calls if "cost_rank_cap" in c],
        "iterations_by_call": [int(c.get("iterations", 0) or 0) for c in calls],
        "accepted_steps": total("accepted_steps"),
        "profile_evaluations": total("profile_evaluations") + total("evaluations"),
        "conceptual_total_evaluations_plus_initial_final": total("profile_evaluations")
        + total("evaluations")
        + 2 * len(calls),
        "gn_solves": total("gn_solves"),
        "fallbacks": {
            name: total(name)
            for name in (
                "rank_guard_fallbacks",
                "workspace_fallbacks",
                "flat_fallbacks",
            )
        },
        "cost_probe_evaluations": total("cost_probe_evaluations"),
        "cost_probe_seconds": float(
            sum(float(c.get("cost_probe_seconds", 0.0) or 0.0) for c in calls)
        ),
        "riemannian_gradient_quantiles": np.quantile(
            gradients, [0, 0.5, 0.9, 1]
        ).tolist()
        if gradients
        else None,
        "max_local_orthogonality_error": max(orthogonality, default=None),
        "max_profile_history_positive_difference": max(positive_diffs, default=None),
        "stop_reasons": [c.get("stop_reason") for c in calls],
        "all_inner_converged": bool(calls)
        and all(bool(c.get("converged", False)) for c in calls),
        "converged_calls": sum(bool(c.get("converged", False)) for c in calls),
    }


def run(scenario: str, seed: int, arm: str, *, capture: bool = False) -> dict[str, Any]:
    from ADP import ADP_multi_index, ADP_solver

    X, y, truth, config = make_problem(scenario, seed)
    method = _method(arm, int(SETTINGS["max_steps"]), truth.shape[1])
    solver = ADP_solver(method)
    hashes: dict[str, str | None] = {
        "X": array_hash(X),
        "y": array_hash(y),
        "truth": array_hash(truth),
    }
    solver_wall = 0.0
    hash_overhead = 0.0
    first_statistics_hashes: dict[str, str | None] | None = None
    original = solver.method

    def timed_method(index_init, U, I, *, mass=None, **kwargs):
        nonlocal solver_wall, hash_overhead, first_statistics_hashes
        if first_statistics_hashes is None:
            hashing_started = time.perf_counter()
            first_statistics_hashes = {
                "index_init": array_hash(np.asarray(index_init)),
                "U": array_hash(np.asarray(U)),
                "I": array_hash(np.asarray(I)),
                "mass": None if mass is None else array_hash(np.asarray(mass)),
            }
            hash_overhead += time.perf_counter() - hashing_started
        started = time.perf_counter()
        result = original(index_init, U, I, mass=mass, **kwargs)
        solver_wall += time.perf_counter() - started
        return result

    solver.method = timed_method
    capture_path = OUT / "captures" / f"{scenario}_seed{seed}.npz"
    if capture:
        original = solver.method

        def capture_method(index_init, U, I, *, mass=None, **kwargs):
            if capture_path.exists():
                raise FileExistsError(capture_path)
            payload = {
                "index_init": np.asarray(index_init),
                "U": np.asarray(U),
                "I": np.asarray(I),
                "truth": truth,
            }
            if mass is not None:
                payload["mass"] = np.asarray(mass)
            capture_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(capture_path, **payload)
            raise _Captured

        solver.method = capture_method
        with suppress(_Captured):
            ADP_multi_index(truth.shape[1], config, solver).fit(X, y)
        with np.load(capture_path, allow_pickle=False) as snap:
            hashes.update(
                {key: array_hash(snap[key]) for key in ("index_init", "U", "I")}
            )
            hashes["mass"] = array_hash(snap["mass"]) if "mass" in snap.files else None
        return {
            "scenario": scenario,
            "seed": seed,
            "status": "captured",
            "input_hashes": hashes,
            "effective_config": asdict(config),
            "snapshot": str(capture_path),
        }

    started = time.perf_counter()
    model = ADP_multi_index(truth.shape[1], config, solver).fit(X, y)
    wall = time.perf_counter() - started
    index = np.asarray(model.basis_).T
    projector_error, orthogonality = projector_metrics(index, truth)
    trace = model.trace_
    finite = bool(np.isfinite(index).all() and np.isfinite(wall))
    completed = bool(model.result_ is not None and index.shape == truth.T.shape)
    summary = _trace_summary(trace)
    return {
        "scenario": scenario,
        "seed": seed,
        "arm": arm,
        "status": "ok" if finite and completed else "invalid",
        "finite": finite,
        "completed": completed,
        "converged": summary["all_inner_converged"],
        "shape": {"n": int(X.shape[0]), "d": int(X.shape[1]), "m": int(truth.shape[1])},
        "input_hashes": hashes,
        "first_statistics_hashes": first_statistics_hashes,
        "effective_config": asdict(config),
        "wall_sec": max(0.0, wall - hash_overhead),
        "hash_overhead_sec": hash_overhead,
        "solver_wall_sec": solver_wall,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "projector_error": projector_error,
        "orthogonality_error": orthogonality,
        "outer_stop_reason": getattr(model.result_, "stop_reason", None),
        "outer_iterations": len(trace),
        "solver_summary": summary,
        "environment": environment(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--arm", default="core")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        row = run(args.scenario, args.seed, args.arm, capture=args.capture)
    except BaseException as exc:
        row = {
            "scenario": args.scenario,
            "seed": args.seed,
            "arm": args.arm,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    args.output.write_text(
        json.dumps(json_value(row), allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
