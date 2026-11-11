"""Inner-solver and fixed-rank aggregation."""

from __future__ import annotations

import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ._analysis_io import _number
from .common import OUT


def _diag(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("diagnostics", {})
    return value if isinstance(value, dict) else {}


def _difference(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None


@lru_cache(maxsize=128)
def _trace_scale_snapshot(snapshot: str) -> float | None:
    path = Path(snapshot)
    if not path.is_absolute():
        path = OUT / path
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            I, mass = data["I"], data["mass"]
            return max(1.0, float(np.sum(mass[:, None] * I**2)))
    except (OSError, KeyError, ValueError):
        return None


def _trace_scale(row: dict[str, Any]) -> float | None:
    snapshot = row.get("snapshot")
    return _trace_scale_snapshot(snapshot) if isinstance(snapshot, str) else None


def inner_tables(
    rows: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    summary, sweeps, frozen_regressions, traces, references, equal_budgets = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    by_task: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    refs: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for row in rows:
        m, d, seed, J = (row.get(k) for k in ("m", "d", "seed", "J"))
        if not (isinstance(m, int) and isinstance(d, int) and isinstance(seed, int)):
            continue
        J = J if isinstance(J, int) else 64
        key = (int(m), int(d), J, int(seed))
        arm = str(row.get("arm", "unknown"))
        if arm == "reference":
            refs[key] = row
        else:
            by_task[key].append(row)
        diag = _diag(row)
        trace = row.get("trace", [])
        trace = trace if isinstance(trace, list) else []
        profile_objective = _number(diag.get("profile_objective"))
        direct_objective = _number(row.get("objective"))
        objective = (
            profile_objective if profile_objective is not None else direct_objective
        )
        scale = _trace_scale(row)
        tol = _number(row.get("tol")) or 1e-7
        time_to_tol = None
        for point in trace:
            if not isinstance(point, dict):
                continue
            norm, elapsed = (
                _number(point.get("full_gradient_norm")),
                _number(point.get("elapsed_sec")),
            )
            if (
                norm is not None
                and elapsed is not None
                and scale is not None
                and norm <= tol * scale
            ):
                time_to_tol = elapsed
                break
        for step, point in enumerate(trace):
            if not isinstance(point, dict):
                continue
            norm = _number(point.get("full_gradient_norm"))
            traces.append(
                {
                    "m": m,
                    "d": d,
                    "J": J,
                    "seed": seed,
                    "arm": arm,
                    "rank_k": row.get("rank_k"),
                    "variant": row.get("variant", "optimized"),
                    "budget": row.get("budget"),
                    "step": step,
                    "elapsed_sec": point.get("elapsed_sec"),
                    "profile_objective": point.get("profile_objective"),
                    "full_gradient_norm": norm,
                    "normalized_gradient": norm / scale
                    if norm is not None and scale
                    else None,
                    "profile_calls_local_refit": point.get("profile_calls"),
                    "objective_evaluations_all_profile_trials": _number(
                        point.get("objective_evaluations")
                    ),
                    "profile_seconds": point.get("profile_seconds"),
                    "gradient_seconds": point.get("gradient_seconds"),
                    "time_to_fixed_tol_sec": time_to_tol,
                }
            )
        if arm == "reference":
            starts = row.get("starts", [])
            starts = starts if isinstance(starts, list) else []
            start_f = [
                _number(s.get("objective")) for s in starts if isinstance(s, dict)
            ]
            references.append(
                {
                    "m": m,
                    "d": d,
                    "J": J,
                    "seed": seed,
                    "status": row.get("status", row.get("worker_status")),
                    "reference_objective": direct_objective,
                    "best_start": row.get("best_start"),
                    "start_objective_min": min(
                        (x for x in start_f if x is not None), default=None
                    ),
                    "start_objective_max": max(
                        (x for x in start_f if x is not None), default=None
                    ),
                    "normalized_gradient": row.get("normalized_gradient"),
                    "converged": row.get("converged"),
                    "stop_reason": row.get("stop_reason"),
                    "iterations": row.get("iterations"),
                    "profile_evaluations": row.get("profile_evaluations"),
                    "wall_sec": row.get("wall_sec"),
                    "projector_error": row.get("projector_error"),
                }
            )
        objective_count = _number(row.get("objective_evaluation_count"))
        diagnostic_evaluations = _number(diag.get("profile_evaluations"))
        summary.append(
            {
                "m": m,
                "d": d,
                "J": J,
                "seed": seed,
                "arm": arm,
                "rank_k": row.get("rank_k"),
                "variant": row.get("variant", "optimized"),
                "budget": row.get("budget"),
                "status": row.get("status", row.get("worker_status")),
                "completed": row.get("completed"),
                "objective": objective,
                "profile_objective": profile_objective,
                "direct_objective": direct_objective,
                "objective_crosscheck_abs": abs(profile_objective - direct_objective)
                if profile_objective is not None and direct_objective is not None
                else None,
                "projector_error": _number(row.get("projector_error")),
                "converged": row.get("converged", diag.get("converged")),
                "stationarity_applicable": diag.get("stationarity_applicable"),
                "gradient_norm": _number(diag.get("gradient_norm")),
                "orthogonality_error": _number(
                    row.get("orthogonality_error", diag.get("orthogonality_error"))
                ),
                "wall_sec": _number(row.get("wall_sec")),
                "peak_rss_kib": _number(row.get("peak_rss_kib")),
                "trace_points": len(trace),
                "profile_calls_local_refit": _number(
                    row.get("instrumented_profile_calls")
                ),
                "objective_evaluation_count_all_profile_trials": objective_count,
                "objective_evaluation_count_expected_from_diagnostics": (
                    diagnostic_evaluations + 2
                    if diagnostic_evaluations is not None
                    else None
                ),
                "objective_evaluation_count_crosscheck": (
                    objective_count == diagnostic_evaluations + 2
                    if objective_count is not None
                    and diagnostic_evaluations is not None
                    else None
                ),
                "local_refit_trial_calls_estimate": max(
                    0.0, (_number(row.get("instrumented_profile_calls")) or 0.0) - 2.0
                ),
                "solver_objective_evaluations": _number(
                    diag.get("profile_evaluations")
                ),
                "solver_conceptual_evaluations_plus_initial_final": _number(
                    diag.get("conceptual_total_evaluations_plus_initial_final")
                ),
                "schur_or_nonrefit_evaluations_estimate": max(
                    0.0,
                    (_number(diag.get("profile_evaluations")) or 0.0)
                    - max(
                        0.0,
                        (_number(row.get("instrumented_profile_calls")) or 0.0) - 2.0,
                    ),
                ),
                "iterations": _number(diag.get("iterations")),
                "time_to_fixed_tol_sec": time_to_tol,
                "fixed_tol": tol,
                "certificate_at_trace_tol": time_to_tol is not None,
            }
        )
    for key, items in by_task.items():
        reference = refs.get(key, {})
        ref_f = _number(reference.get("objective"))
        for row in items:
            if row.get("arm") != "block_fixed" or row.get("budget") != 1:
                continue
            rank = row.get("rank_k")
            if not isinstance(rank, int):
                continue
            rank_one = next(
                (
                    r
                    for r in items
                    if r.get("arm") == "rank_one" and r.get("budget") == rank
                ),
                None,
            )
            trace = row.get("trace", [])
            trace = trace if isinstance(trace, list) else []
            initial = (
                _number(trace[0].get("profile_objective"))
                if trace and isinstance(trace[0], dict)
                else None
            )
            final = _number(_diag(row).get("profile_objective"))
            if final is None:
                final = _number(row.get("objective"))
            gap = None if initial is None or ref_f is None else initial - ref_f
            threshold = 1e-12 * max(1.0, abs(initial or 0.0), abs(ref_f or 0.0))
            valid = (
                gap is not None
                and gap > threshold
                and ref_f is not None
                and final is not None
                and final >= ref_f - threshold
            )
            rho = (
                (final - ref_f) / gap
                if valid and final is not None and ref_f is not None and gap is not None
                else None
            )
            seq_final = (
                _number(_diag(rank_one).get("profile_objective")) if rank_one else None
            )
            if seq_final is None and rank_one:
                seq_final = _number(rank_one.get("objective"))
            seq_valid = (
                gap is not None
                and gap > threshold
                and ref_f is not None
                and seq_final is not None
                and seq_final >= ref_f - threshold
            )
            seq_rho = (
                (seq_final - ref_f) / gap
                if seq_valid
                and seq_final is not None
                and ref_f is not None
                and gap is not None
                else None
            )
            r = min(key[0], key[1] - key[0])
            sweeps.append(
                {
                    "m": key[0],
                    "d": key[1],
                    "J": key[2],
                    "seed": key[3],
                    "variant": row.get("variant", "optimized"),
                    "k": rank,
                    "r": r,
                    "x_k_over_r": rank / r,
                    "initial_objective": initial,
                    "reference_objective": ref_f,
                    "block_objective": final,
                    "ratio": rho,
                    "block_wall_sec": _number(row.get("wall_sec")),
                    "block_projector_error": _number(row.get("projector_error")),
                    "minus_log_ratio_times_r_over_k": -math.log(rho) * r / rank
                    if rho is not None and rho > 0
                    else None,
                    "sequential_ratio": seq_rho,
                    "sequential_minus_log_ratio_times_r_over_k": -math.log(seq_rho)
                    * r
                    / rank
                    if seq_rho is not None and seq_rho > 0
                    else None,
                    "reference_gap_valid": valid,
                    "invalid_gap_reason": (
                        None
                        if valid
                        else "no reference in cost-only grid"
                        if ref_f is None
                        else "invalid reference gap or arm objective below reference"
                    ),
                    "reference_converged": reference.get("converged"),
                    "reference_normalized_gradient": reference.get(
                        "normalized_gradient"
                    ),
                    "sequential_reference_gap_valid": seq_valid,
                    "sequential_rank_one_budget_k_objective": seq_final,
                    "sequential_rank_one_wall_sec": _number(rank_one.get("wall_sec"))
                    if rank_one
                    else None,
                    "sequential_rank_one_projector_error": _number(
                        rank_one.get("projector_error")
                    )
                    if rank_one
                    else None,
                    "block_minus_sequential_objective": final - seq_final
                    if final is not None and seq_final is not None
                    else None,
                }
            )
        reference_error = _number(reference.get("projector_error"))
        one_steps = [r for r in items if r.get("budget") == 1]

        def label(r: dict[str, Any]) -> str:
            return f"{r.get('arm')}:{r.get('variant', 'optimized')}:k{r.get('rank_k')}"

        for i, ra in enumerate(one_steps):
            fa = _number(_diag(ra).get("profile_objective")) if ra else None
            ea = _number(ra.get("projector_error")) if ra else None
            if fa is None or ea is None:
                continue
            for rb in one_steps[i + 1 :]:
                if label(ra) == label(rb):
                    continue
                fb, eb = (
                    (
                        _number(_diag(rb).get("profile_objective")),
                        _number(rb.get("projector_error")),
                    )
                    if rb
                    else (None, None)
                )
                if (
                    fb is not None
                    and eb is not None
                    and ((fa < fb and ea > eb) or (fb < fa and eb > ea))
                ):
                    frozen_regressions.append(
                        {
                            "m": key[0],
                            "d": key[1],
                            "J": key[2],
                            "seed": key[3],
                            "arm_a": label(ra),
                            "F_a": fa,
                            "EDR_a": ea,
                            "arm_b": label(rb),
                            "F_b": fb,
                            "EDR_b": eb,
                            "reference_EDR": reference_error,
                        }
                    )
    # Compare prefixes at common wall, local refit, all-trial, and iteration budgets.
    long_groups: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        if row.get("budget") == 100 and row.get("arm") != "reference":
            m, d, seed, J = (row.get(k) for k in ("m", "d", "seed", "J"))
            if isinstance(m, int) and isinstance(d, int) and isinstance(seed, int):
                long_groups[
                    (int(m), int(d), J if isinstance(J, int) else 64, int(seed))
                ].append(row)
    for key, items in long_groups.items():
        core = next((r for r in items if r.get("arm") == "core"), None)
        if core is None:
            continue
        for candidate in items:
            if candidate is core:
                continue

            def indexed_trace(row: dict[str, Any]) -> list[Any]:
                source = row.get("trace", [])
                if not isinstance(source, list):
                    return []
                iterations = _number(_diag(row).get("iterations"))
                limit = (
                    int(iterations)
                    if iterations is not None
                    else max(0, len(source) - 1)
                )
                return [
                    {**point, "accepted_iteration": min(ix, limit)}
                    if isinstance(point, dict)
                    else point
                    for ix, point in enumerate(source)
                ]

            ct, bt = indexed_trace(candidate), indexed_trace(core)
            if not isinstance(ct, list) or not isinstance(bt, list) or not ct or not bt:
                continue

            def val(trace: list[Any], field: str, default: float = 0.0) -> float:
                values = [_number(p.get(field)) for p in trace if isinstance(p, dict)]
                return next((v for v in reversed(values) if v is not None), default)

            caps = {
                "wall": min(val(ct, "elapsed_sec"), val(bt, "elapsed_sec")),
                "local_refit_call": min(
                    val(ct, "profile_calls"), val(bt, "profile_calls")
                ),
                "iteration": min(
                    val(ct, "accepted_iteration"), val(bt, "accepted_iteration")
                ),
            }
            objective_counts = [
                next(
                    (
                        value
                        for point in reversed(trace)
                        if isinstance(point, dict)
                        and (value := _number(point.get("objective_evaluations")))
                        is not None
                    ),
                    None,
                )
                for trace in (ct, bt)
            ]
            if objective_counts[0] is not None and objective_counts[1] is not None:
                caps["objective_evaluation"] = min(
                    objective_counts[0], objective_counts[1]
                )
            for kind, cap in caps.items():

                def at(
                    trace: list[Any], kind: str = kind, cap: float = cap
                ) -> dict[str, Any] | None:
                    eligible = []
                    for point in trace:
                        if not isinstance(point, dict):
                            continue
                        measure = (
                            _number(point.get("accepted_iteration"))
                            if kind == "iteration"
                            else _number(
                                point.get(
                                    {
                                        "wall": "elapsed_sec",
                                        "local_refit_call": "profile_calls",
                                        "objective_evaluation": "objective_evaluations",
                                    }[kind]
                                )
                            )
                        )
                        if measure is not None and measure <= cap:
                            eligible.append(point)
                    return eligible[-1] if eligible else None

                cp, bp = at(ct), at(bt)
                cf, bf = (
                    _number(cp.get("profile_objective")) if cp else None,
                    _number(bp.get("profile_objective")) if bp else None,
                )
                equal_budgets.append(
                    {
                        "m": key[0],
                        "d": key[1],
                        "J": key[2],
                        "seed": key[3],
                        "arm": candidate.get("arm"),
                        "variant": candidate.get("variant"),
                        "budget_kind": kind,
                        "common_budget": cap,
                        "candidate_accepted_iteration": _number(
                            cp.get("accepted_iteration")
                        )
                        if cp
                        else None,
                        "core_accepted_iteration": _number(bp.get("accepted_iteration"))
                        if bp
                        else None,
                        "candidate_objective": cf,
                        "core_objective": bf,
                        "candidate_minus_core": cf - bf
                        if cf is not None and bf is not None
                        else None,
                        "candidate_projector_error": _number(cp.get("projector_error"))
                        if cp
                        else None,
                        "core_projector_error": _number(bp.get("projector_error"))
                        if bp
                        else None,
                        "candidate_minus_core_edr": _difference(
                            _number(cp.get("projector_error")) if cp else None,
                            _number(bp.get("projector_error")) if bp else None,
                        ),
                        "candidate_full_gradient_norm": _number(
                            cp.get("full_gradient_norm")
                        )
                        if cp
                        else None,
                        "core_full_gradient_norm": _number(bp.get("full_gradient_norm"))
                        if bp
                        else None,
                        "candidate_elapsed_sec": _number(cp.get("elapsed_sec"))
                        if cp
                        else None,
                        "core_elapsed_sec": _number(bp.get("elapsed_sec"))
                        if bp
                        else None,
                        "candidate_local_refit_calls": _number(cp.get("profile_calls"))
                        if cp
                        else None,
                        "core_local_refit_calls": _number(bp.get("profile_calls"))
                        if bp
                        else None,
                        "candidate_objective_evaluations": _number(
                            cp.get("objective_evaluations")
                        )
                        if cp
                        else None,
                        "core_objective_evaluations": _number(
                            bp.get("objective_evaluations")
                        )
                        if bp
                        else None,
                    }
                )
    return summary, sweeps, frozen_regressions, traces, references, equal_budgets
