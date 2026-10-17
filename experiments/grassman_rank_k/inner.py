"""Frozen planted-operator workloads and isolated inner solver worker."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    SETTINGS,
    environment,
    json_value,
    projector_metrics,
)


def make_frozen(m: int, d: int, seed: int, *, J: int = 64) -> dict[str, np.ndarray]:
    p = max(12, 2 * m)
    rng = np.random.default_rng(np.random.SeedSequence([seed, m, d, 816]))
    truth = np.linalg.qr(rng.normal(size=(d, m)))[0]
    init = np.linalg.qr(truth + 0.35 * rng.normal(size=(d, m)))[0]
    base_J = 256
    U_all = (
        rng.normal(size=(base_J, p, d))
        + 0.5 * rng.normal(size=(base_J, p, m)) @ truth.T
    )
    coefficient = rng.normal(size=(base_J, m))
    I_all = np.einsum("jpm,jm->jp", U_all @ truth, coefficient)
    I_all += 0.12 * rng.normal(size=(base_J, p))
    mass_all = np.exp(rng.uniform(-1.0, 1.0, size=base_J))
    U, I, mass = U_all[:J], I_all[:J], mass_all[:J]
    return {"index_init": init.T, "U": U, "I": I, "mass": mass, "truth": truth}


def _solver(
    arm: str,
    *,
    budget: int,
    tol: float,
    rank_k: int | None = None,
    variant: str = "optimized",
):
    from ADP.solver.grassman import block_rank_k, solve as core
    from ADP.solver.grassman_exp import solve as exp

    common = {
        "max_steps": budget,
        "tol": tol,
        "lambda_prox": SETTINGS["lambda_prox"],
        "local_ridge": SETTINGS["local_ridge"],
        "max_angle": SETTINGS["max_angle"],
        "workspace_bytes": SETTINGS["workspace_bytes"],
    }
    if arm == "core":
        return core, common
    if arm in {"rank_one", "spectral", "flat"}:
        return exp, {
            **common,
            "method": {
                "rank_one": "rank_one",
                "spectral": "spectral",
                "flat": "flat_gn",
            }[arm],
        }
    if arm in {"block1", "block_full"} and rank_k is None:
        raise ValueError(f"{arm} requires an explicit fixed rank")
    if arm not in {
        "block_fixed",
        "block1",
        "block_full",
        "block_energy",
        "block_cost",
        "block_spectral",
    }:
        raise ValueError(f"unknown arm: {arm}")
    strategy = "cost" if arm == "block_cost" else "energy"
    if arm == "block1":
        rank_k = 1
    return block_rank_k, {
        **common,
        "rank_k": rank_k,
        "rank_strategy": strategy,
        "block_variant": variant,
        "block_max_steps": SETTINGS["block_max_steps"],
    }


def solve_snapshot(
    snapshot: dict[str, np.ndarray],
    arm: str,
    *,
    budget: int,
    tol: float = 1e-6,
    rank_k: int | None = None,
    variant: str = "optimized",
) -> dict[str, Any]:
    from ADP.solver.LSMR import _local_refit, _loss

    method, kwargs = _solver(
        arm, budget=budget, tol=tol, rank_k=rank_k, variant=variant
    )
    args = dict(kwargs)
    args["mass"] = snapshot["mass"]
    from ADP.solver import grassman as core_module, grassman_exp as exp_module

    originals = {
        "core_gradient": core_module._gradient,
        "exp_gradient": exp_module._gradient,
        "core_profile": core_module._profile,
        "exp_profile": exp_module._profile,
    }
    trace: list[dict[str, Any]] = []
    profile_calls = 0
    profile_seconds = 0.0
    gradient_seconds = 0.0
    trace_started = time.perf_counter()

    def tracked_profile(original):
        def call(*call_args, **call_kwargs):
            nonlocal profile_calls, profile_seconds
            started_at = time.perf_counter()
            result_value = original(*call_args, **call_kwargs)
            profile_seconds += time.perf_counter() - started_at
            profile_calls += 1
            return result_value

        return call

    def tracked_gradient(original):
        def call(Y, U, state, mass, *call_args, **call_kwargs):
            nonlocal gradient_seconds
            started_at = time.perf_counter()
            gradient = original(Y, U, state, mass, *call_args, **call_kwargs)
            gradient_seconds += time.perf_counter() - started_at
            error, _ = projector_metrics(Y.T, snapshot["truth"])
            trace.append(
                {
                    "profile_objective": float(state.value),
                    "full_gradient_norm": float(np.linalg.norm(gradient)),
                    "projector_error": error,
                    "profile_calls": profile_calls,
                    "profile_seconds": profile_seconds,
                    "gradient_seconds": gradient_seconds,
                    "elapsed_sec": time.perf_counter() - trace_started,
                }
            )
            return gradient

        return call

    core_module._profile = tracked_profile(originals["core_profile"])
    exp_module._profile = tracked_profile(originals["exp_profile"])
    core_module._gradient = tracked_gradient(originals["core_gradient"])
    exp_module._gradient = tracked_gradient(originals["exp_gradient"])
    started = time.perf_counter()
    try:
        result = method(snapshot["index_init"], snapshot["U"], snapshot["I"], **args)
    finally:
        elapsed = time.perf_counter() - started
        core_module._gradient = originals["core_gradient"]
        exp_module._gradient = originals["exp_gradient"]
        core_module._profile = originals["core_profile"]
        exp_module._profile = originals["exp_profile"]
    index = np.asarray(result.index)
    coefficients, _ = _local_refit(snapshot["I"], snapshot["U"], index)
    objective = 2.0 * float(
        _loss(snapshot["I"], snapshot["U"], index, coefficients, snapshot["mass"])
    )
    error, orthogonality = projector_metrics(index, snapshot["truth"])
    diagnostics = dict(result.diagnostics)
    finite = bool(np.isfinite(index).all() and np.isfinite(objective))
    return {
        "status": "ok" if finite else "invalid",
        "finite": finite,
        "completed": bool(index.shape == snapshot["index_init"].shape),
        "converged": bool(diagnostics.get("converged", False)),
        "arm": arm,
        "rank_k": rank_k,
        "variant": variant,
        "wall_sec": elapsed,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "objective": objective,
        "projector_error": error,
        "orthogonality_error": orthogonality,
        "diagnostics": diagnostics,
        "trace": trace,
        "instrumented_wall_sec": elapsed,
        "instrumented_profile_calls": profile_calls,
        "instrumented_profile_seconds": profile_seconds,
        "instrumented_gradient_seconds": gradient_seconds,
        "timing_note": (
            "wall includes scalar trace instrumentation; kernel totals recorded"
        ),
        "environment": environment(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--budget", type=int, default=1)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--rank-k", type=int)
    parser.add_argument(
        "--variant", choices=("optimized", "spectral"), default="optimized"
    )
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        with np.load(args.snapshot, allow_pickle=False) as data:
            snapshot = {key: data[key] for key in data.files}
        if args.reference:
            from .reference import reference

            row = reference(snapshot, max_steps=args.budget, tol=args.tol)
        else:
            row = solve_snapshot(
                snapshot,
                args.arm,
                budget=args.budget,
                tol=args.tol,
                rank_k=args.rank_k,
                variant=args.variant,
            )
    except BaseException as exc:
        row = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    row.update(
        {
            "snapshot": str(args.snapshot),
            "arm": args.arm,
            "budget": args.budget,
            "rank_k": args.rank_k,
            "variant": args.variant,
        }
    )
    args.output.write_text(
        json.dumps(json_value(row), allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
