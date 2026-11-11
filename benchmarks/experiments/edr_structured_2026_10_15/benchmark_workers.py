"""Synthetic-data and per-arm benchmark workers."""

from __future__ import annotations

import hashlib
import resource
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def array_hash(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(value)
    return digest(str(array.dtype).encode() + repr(array.shape).encode() + array.tobytes())


def _problem(case: str, seed: int, *, tau: float = 0.0):
    """Deterministic finite sketch with a planted row-space basis."""
    import numpy as np

    rng = np.random.default_rng(seed)
    if case in ("near_truth", "weak_signal"):
        J, p, d, m = 80, 8, 12, 2
        truth, _ = np.linalg.qr(rng.standard_normal((d, m)), mode="reduced")
        U = rng.standard_normal((J, p, d))
        coeff = rng.standard_normal((J, m))
        if case == "weak_signal":
            coeff[:, 1] *= 0.03
        I = np.einsum("jpd,dm,jm->jp", U, truth, coeff, optimize=True)
        noise = rng.standard_normal((d, m))
        horizontal = noise - truth @ (truth.T @ noise)
        left, _, right = np.linalg.svd(truth + 0.05 * horizontal, full_matrices=False)
        init = left @ right
        return init.T.copy(), U, I, np.full(J, 1 / J), truth.T.copy(), tau
    J, p, d, m = 100, 8, 40, 3
    q, _ = np.linalg.qr(rng.standard_normal((d, m)), mode="reduced")
    if case == "isotropic":
        covariance = np.eye(d)
    elif case == "weak_index":
        eigenvalues = np.ones(d)
        eigenvalues[:m] = 1.15
        covariance = np.diag(eigenvalues)
    elif case == "correlated":
        rotation, _ = np.linalg.qr(rng.standard_normal((d, d)))
        covariance = (rotation * np.geomspace(0.2, 5.0, d)) @ rotation.T
    else:
        raise ValueError(f"unknown frozen case: {case}")
    root = np.linalg.cholesky(covariance)
    U = (rng.standard_normal((J, p, d)) @ root).astype(np.float64)
    coeff = rng.standard_normal((J, m))
    clean = np.einsum("jpd,md,jm->jp", U, q.T, coeff, optimize=True)
    noise_scale = {"isotropic": 0.35, "weak_index": 0.18, "correlated": 0.22}[case]
    I = clean + noise_scale * rng.standard_normal((J, p))
    I = np.ascontiguousarray(I, dtype=np.float64)
    U = np.ascontiguousarray(U, dtype=np.float64)
    mass = np.full(J, 1.0 / J, dtype=np.float64)
    # Deterministic orthonormal initialization, independent of the planted q.
    init, _ = np.linalg.qr(rng.standard_normal((d, m)), mode="reduced")
    return init.T.copy(), U, I, mass, q.T.copy(), tau


def _solver(variant: str, max_steps: int = 30) -> tuple[Callable[..., Any], dict[str, Any]]:
    settings: dict[str, Any] = {
        "lambda_prox": 0.05,
        "max_steps": max_steps,
        "tol": 1e-6,
        "local_ridge": 0.0,
        "max_angle": 0.5,
        "workspace_bytes": 16 * 1024 * 1024,
    }
    if variant == "core_gn":
        from ADP.solver.grassman import solve

        return solve, settings
    from ADP.solver.grassman_research import solve

    if variant == "schur_gn_preconditioned":
        settings.update(method="schur_gn", precondition=True)
    elif variant == "schur_newton_preconditioned":
        settings.update(method="schur_newton", precondition=True)
    elif variant == "schur_newton_unpreconditioned":
        settings.update(method="schur_newton", precondition=False)
    elif variant == "schur_newton_recycle2":
        settings.update(method="schur_newton", precondition=True, recycle_steps=2)
    elif variant == "metric":
        settings.update(method="metric", precondition=True)
    else:
        raise ValueError(f"unknown variant: {variant}")
    return solve, settings


def _frozen_worker(
    case: str,
    seed: int,
    variant: str,
    tau: float,
    snapshot: Path | None,
    max_steps: int,
) -> dict[str, Any]:
    import numpy as np

    from ADP.solver import grassman

    frozen_hash = None
    if snapshot is None:
        init, U, I, mass, truth, _ = _problem(case, seed, tau=tau)
    else:
        with np.load(snapshot, allow_pickle=False) as arrays:
            init = arrays["index_init"]
            U, I = arrays["U"], arrays["I"]
            mass = arrays["mass"] if "mass" in arrays.files else np.ones(len(I))
            truth = arrays["truth_basis"]
        frozen_hash = digest(snapshot.read_bytes())
    solve, settings = _solver(variant, max_steps)
    settings["local_ridge"] = tau
    rss_before = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    started = time.perf_counter()
    result = solve(init, U, I, mass=mass, **settings)
    elapsed = time.perf_counter() - started
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    basis = np.asarray(result.index, dtype=np.float64)
    Y = grassman._normalize_index(basis).T
    state = grassman._profile(grassman._project(U, Y), I, mass, tau)
    gradient = grassman._gradient(Y, U, state, mass)
    objective = float(state.value)
    live_gradient = float(np.linalg.norm(gradient, ord="fro"))
    truth_rows = truth if truth.shape[0] == basis.shape[0] else truth.T
    projector_distance = float(np.linalg.norm(basis.T @ basis - truth_rows.T @ truth_rows, ord="fro") / np.sqrt(2 * basis.shape[0]))
    diagnostics = getattr(result, "diagnostics", {})
    if hasattr(diagnostics, "__dict__"):
        diagnostics = vars(diagnostics)
    return {
        "stage": "frozen",
        "case": case,
        "seed": seed,
        "variant": variant,
        "status": "ok",
        "tau": tau,
        "shape": {
            "J": int(U.shape[0]),
            "p": int(U.shape[1]),
            "d": int(U.shape[2]),
            "m": int(basis.shape[0]),
        },
        "dtype": str(U.dtype),
        "input_hashes": {
            "init": array_hash(init),
            "U": array_hash(U),
            "I": array_hash(I),
            "mass": array_hash(mass),
            "truth": array_hash(truth),
        },
        "frozen_snapshot": str(snapshot) if snapshot is not None else None,
        "frozen_snapshot_sha256": frozen_hash,
        "solver_settings": settings,
        "wall_sec": elapsed,
        "peak_rss_platform_units": peak_rss,
        "additional_peak_rss_platform_units": max(0, peak_rss - rss_before),
        "objective": objective,
        "live_gradient_fro": live_gradient,
        "live_gradient_scaled_legacy": live_gradient / max(1.0, 2.0 * abs(objective)),
        "certificate_scale": max(1.0, float(np.sum(mass[:, None] * I**2))),
        "live_gradient_certificate": live_gradient / max(1.0, float(np.sum(mass[:, None] * I**2))),
        "projector_error": projector_distance,
        "truth_projector": (truth_rows.T @ truth_rows).tolist() if case == "weak_signal" else None,
        "iterations": diagnostics.get("iterations") if isinstance(diagnostics, dict) else None,
        "accepted_steps": diagnostics.get("accepted_steps") if isinstance(diagnostics, dict) else None,
        "stop_reason": diagnostics.get("stop_reason") if isinstance(diagnostics, dict) else None,
        "converged": diagnostics.get("converged") if isinstance(diagnostics, dict) else None,
        "operator_pass_counts": diagnostics.get("operator_pass_counts") if isinstance(diagnostics, dict) else None,
        "diagnostics": diagnostics,
    }


def _cx_worker(seed: int, support: int, solver_arm: str | None = None) -> dict[str, Any]:
    """Measure exact CSR C@X construction, storage and paired actions."""
    import numpy as np

    from ADP.engine.common.statistic import calculate_statistics
    from ADP.solver.statistic_operator import DenseOperator, factorized_statistics

    n, d, J, p, repeats = 400, 100, 100, 8, 3
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = rng.standard_normal(n)
    direction_frame, _ = np.linalg.qr(rng.standard_normal((d, p)), mode="reduced")
    directions = np.broadcast_to(direction_frame.T, (J, p, d)).copy()
    weights = np.zeros((J, n), dtype=np.float64)
    for j in range(J):
        start = int(rng.integers(0, n - support + 1))
        weights[j, start : start + support] = rng.uniform(0.1, 1.0, support)
    rss_before = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    started = time.perf_counter()
    operator, I, mass = factorized_statistics(X, y, weights, directions, batch_size=32, normalized=False)
    builder_sec = time.perf_counter() - started
    started = time.perf_counter()
    dense_stats = calculate_statistics(X, y, weights, directions, batch_size=32, normalized=False)
    dense_builder_sec = time.perf_counter() - started
    U = dense_stats.U
    np.testing.assert_allclose(I, dense_stats.I, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(mass, dense_stats.mass, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(operator.C @ operator.X).reshape(J, p, d), U, rtol=2e-11, atol=2e-11)
    dense = DenseOperator(U)
    rng_action = np.random.default_rng(seed + 9127)
    V = rng_action.standard_normal((d, 3))
    P = rng_action.standard_normal((J, p, 3))
    # Warm both paths before the alternating paired measurements.
    np.testing.assert_allclose(operator.project(V), dense.project(V), rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(operator.adjoint(P), dense.adjoint(P), rtol=2e-12, atol=2e-12)
    project_times: list[float] = []
    dense_project_times: list[float] = []
    adjoint_times: list[float] = []
    dense_adjoint_times: list[float] = []
    for repeat in range(repeats):
        actions = [
            ("project", operator.project, V, project_times),
            ("dense_project", dense.project, V, dense_project_times),
            ("adjoint", operator.adjoint, P, adjoint_times),
            ("dense_adjoint", dense.adjoint, P, dense_adjoint_times),
        ]
        if repeat % 2:
            actions.reverse()
        for _, fn, value, times in actions:
            started = time.perf_counter()
            fn(value)
            times.append(time.perf_counter() - started)
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    row = {
        "stage": "cx",
        "seed": seed,
        "support": support,
        "status": "ok",
        "shape": {"n": n, "d": d, "J": J, "p": p},
        "dtype": str(X.dtype),
        "input_hashes": {
            "X": array_hash(X),
            "y": array_hash(y),
            "directions": array_hash(directions),
            "weights": array_hash(weights),
        },
        "builder_sec": builder_sec,
        "dense_builder_sec": dense_builder_sec,
        "builder_moments_max_abs_error": float(np.max(np.abs(np.asarray(operator.C @ operator.X).reshape(J, p, d) - U))),
        "operator_project_sec": project_times,
        "dense_project_sec": dense_project_times,
        "operator_adjoint_sec": adjoint_times,
        "dense_adjoint_sec": dense_adjoint_times,
        "project_max_abs_error": float(np.max(np.abs(operator.project(V) - dense.project(V)))),
        "adjoint_max_abs_error": float(np.max(np.abs(operator.adjoint(P) - dense.adjoint(P)))),
        "mass_hash": array_hash(mass),
        "I_hash": array_hash(I),
        "csr_data_bytes": int(operator.C.data.nbytes),
        "csr_indices_bytes": int(operator.C.indices.nbytes),
        "csr_indptr_bytes": int(operator.C.indptr.nbytes),
        "X_bytes": int(operator.X.nbytes),
        "factorized_bytes": int(operator.nbytes),
        "dense_U_bytes": int(U.nbytes),
        "factorized_over_dense_storage": float(operator.nbytes / U.nbytes),
        "peak_rss_platform_units": peak_rss,
        "additional_peak_rss_platform_units": max(0, peak_rss - rss_before),
    }
    if solver_arm is not None:
        from ADP.solver import grassman

        variant = "cx_dense" if solver_arm == "cx_dense" else "cx_factorized"
        solve, settings = _solver("schur_gn_preconditioned")
        init, _ = np.linalg.qr(np.random.default_rng(seed + 81).standard_normal((d, 3)), mode="reduced")
        solve_input = dense if solver_arm == "cx_dense" else operator
        started = time.perf_counter()
        result = solve(init.T, solve_input, I, mass=mass, **settings)
        solver_sec = time.perf_counter() - started
        basis = np.asarray(result.index)
        Y = grassman._normalize_index(basis).T
        state = grassman._profile(grassman._project(U, Y), I, mass, 0.0)
        grad = grassman._gradient(Y, U, state, mass)
        diagnostics = result.diagnostics
        row.update(
            solver_arm=variant,
            solver_sec=solver_sec,
            solver_peak_rss_platform_units=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            objective=float(state.value),
            live_gradient_fro=float(np.linalg.norm(grad)),
            converged=bool(diagnostics.get("converged")),
            solver_stop_reason=diagnostics.get("stop_reason"),
            operator_projects=diagnostics.get("operator_projects"),
            operator_adjoints=diagnostics.get("operator_adjoints"),
            solution_hash=array_hash(basis),
            solver_settings=settings,
        )
    return row


def _fullfit_worker(seed: int, variant: str, snapshot: Path | None) -> dict[str, Any]:
    """Small actual-pipeline fit; captures the first real statistics call."""
    import numpy as np

    from ADP import ADP_Config, ADP_multi_index, ADP_solver
    from experiments.data import _generate_data, _make_seed_bundle
    from experiments.models import ExperimentPoint
    from experiments.runner import _effective_config

    point = ExperimentPoint(
        d=12,
        n_over_d=12.5,
        sigma_x=1.0,
        sigma_eps=0.1,
        link="spokoini_m2",
        x_distribution="beta1_tau",
        noise_distribution="gaussian",
        mode="multi",
        index_dim=2,
        n_samples=150,
        tau=1.0,
        N_loc=6,
        N_lin=20,
        N_J=80,
        N_phi=6,
        a=1.010050167084168,
        index_init="local",
        select_step="last",
        solver_max_steps=3,
        outer_steps=3,
    )
    selector = "mi-spokoini-m2-dhigh"
    seed_bundle = _make_seed_bundle(selector, point, seed)
    generated = _generate_data(selector, point, seed_bundle, seed)
    config = _effective_config(ADP_Config(), point, seed_bundle.init)
    solve, settings = _solver(variant)
    captured: dict[str, Any] = {}

    def method(
        index_init: np.ndarray,
        U: np.ndarray,
        I: np.ndarray,
        *,
        mass: np.ndarray | None = None,
        **kwargs: Any,
    ):
        if not captured:
            captured.update(
                index_init=np.array(index_init, copy=True),
                U=np.array(U, copy=True),
                I=np.array(I, copy=True),
                mass=None if mass is None else np.array(mass, copy=True),
            )
        call_settings = dict(kwargs)
        call_settings.update(settings)
        return solve(index_init, U, I, mass=mass, **call_settings)

    solver = ADP_solver(method)
    started = time.perf_counter()
    model = ADP_multi_index(2, config, solver).fit(generated.X, generated.Y)
    fit_sec = time.perf_counter() - started
    if snapshot is not None and captured:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        np.savez(snapshot, **captured, truth_basis=np.asarray(generated.beta))
    basis = np.asarray(model.basis_)
    truth = np.asarray(generated.beta)
    truth_projector = truth.T @ truth if truth.shape[0] == 2 else truth @ truth.T
    projector = basis.T @ basis if basis.shape[0] == 2 else basis @ basis.T
    distance = float(np.linalg.norm(projector - truth_projector, ord="fro") / np.sqrt(4.0))
    trace = [entry.get("solver", {}) for entry in model.trace_]
    return {
        "stage": "fullfit",
        "seed": seed,
        "variant": variant,
        "status": "ok",
        "shape": {"n": 150, "d": 12, "m": 2, "J": 80, "p": 6, "outer_steps": 3},
        "fit_wall_sec": fit_sec,
        "peak_rss_platform_units": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "input_hashes": {
            "X": array_hash(generated.X),
            "y": array_hash(generated.Y),
            "truth": array_hash(truth),
        },
        "first_statistics_hashes": {key: array_hash(value) for key, value in captured.items() if value is not None},
        "snapshot": str(snapshot) if snapshot is not None else None,
        "projector_error": distance,
        "outer_iterations": model.effective_parameters_.get("outer_iterations"),
        "selected_iteration": model.effective_parameters_.get("selected_iteration"),
        "fit_stop_reason": model.result_.stop_reason,
        "solver_calls": len(trace),
        "last_solver_diagnostics": trace[-1] if trace else {},
    }
