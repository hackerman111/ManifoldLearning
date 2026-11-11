"""Isolated variants of an immutable solver; no process-global monkeypatches."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from dataclasses import replace
from functools import lru_cache, partial
from pathlib import Path
from typing import cast

import numpy as np
from scipy.sparse.linalg import lsmr

from ADP.solver.grassman_exp import ProfileJacobian

from . import lowrank


@lru_cache
def load_reference(name="baseline"):
    path = Path(__file__).with_name("reference.py")
    module_name = f"ADP.solver._accuracy22_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    source = path.read_text()
    if name.startswith("lowrank_"):
        _, mode, rank_text = name.split("_")
        rank_k = None if rank_text == "k" else int(rank_text)
        old = "        V, singular, At = xp.linalg.svd(G, full_matrices=False)"
        assert source.count(old) == 1
        source = source.replace(
            old,
            "        proposal, info, extra = lowrank_step(\n"
            "            Y, U, I, state, mass, G, local_ridge, max_angle,\n"
            "            workspace_bytes)\n"
            "        lowrank_statistics.append(info)\n"
            "        evaluations += extra\n"
            "        if proposal is not None:\n"
            "            Y, state = proposal\n"
            "            accepted += 1\n"
            "            history.append(state.value)\n"
            "            q_history.append(info.get('rank', 0))\n"
            "            continue\n" + old,
        )
        old = '    stop = "max_steps"'
        assert source.count(old) == 1
        source = source.replace(old, old + "\n    lowrank_statistics = []")
        old = '        "gn_solves": gn_solves,'
        assert source.count(old) == 1
        source = source.replace(
            old, old + '\n        "lowrank_statistics": tuple(lowrank_statistics),'
        )
        module.__dict__["lowrank_step"] = partial(
            lowrank.step, mode=mode, rank_k=rank_k
        )
    if name == "smooth_relative":
        old = "if relative_gn and not state.smooth:"
        assert source.count(old) == 1
        source = source.replace(old, "if relative_gn:")
    if name == "stable_jacobian":
        old = "4 * rows * q * m"
        assert source.count(old) == 1
        source = source.replace(old, "16 * rows * q * m")
    if name == "krylov":
        old = "V, singular, At = xp.linalg.svd(G, full_matrices=False)"
        assert source.count(old) == 1
        source = source.replace(
            old,
            "direction = krylov_direction(Y, U, I, state, mass, G, "
            "local_ridge, cached_ranks, lambda_prox, krylov_stops)\n"
            "        V, singular, At = xp.linalg.svd(direction, full_matrices=False)",
        )
        assert source.count('    stop = "max_steps"') == 1
        source = source.replace(
            '    stop = "max_steps"',
            '    stop = "max_steps"\n    krylov_stops = []',
        )
        assert source.count('        "gn_solves": gn_solves,') == 1
        source = source.replace(
            '        "gn_solves": gn_solves,',
            '        "gn_solves": gn_solves,\n'
            '        "krylov_lsmr": tuple(krylov_stops),',
        )
    if name == "gn_quadratic":
        old = (
            "                    if candidate is not None:\n"
            "                        break\n"
            "                    damping *= 10"
        )
        assert source.count(old) == 1
        source = source.replace(
            old,
            "                    if candidate is not None:\n"
            "                        candidate, extra = polish_core(\n"
            "                            Y, state, V, W, K, I, mass, local_ridge,\n"
            "                            candidate, backtrack, slope, tangent_limit)\n"
            "                        evaluations += extra\n"
            "                        break\n"
            "                    damping *= 10",
        )
    exec(compile(source, str(path), "exec"), module.__dict__)
    if name == "stable_jacobian":
        module.__dict__["_core_jacobian"] = orthogonal_jacobian
    if name == "projected_jacobian":
        module.__dict__["_core_jacobian"] = projected_jacobian
    if name == "gn_quadratic":
        module.__dict__["polish_core"] = polish_core
    if name == "krylov":
        module.__dict__["krylov_direction"] = krylov_direction
    if name == "diagonal":
        old = "V, singular, At = xp.linalg.svd(G, full_matrices=False)"
        assert source.count(old) == 1
        source = source.replace(
            old,
            "V, singular, At = xp.linalg.svd("
            "diagonal_direction(Y, U, state, mass, G), full_matrices=False)",
        )
        module.__dict__["diagonal_direction"] = diagonal_direction
        exec(compile(source, str(path), "exec"), module.__dict__)
    return module


reference = load_reference()


def projected_jacobian(state, W, mass, tau, *, root_mass=None):
    """Approximate dr using only its part orthogonal to the local design.

    It retains J.T r exactly at a least-squares profile, but omits a PSD
    contribution to J.T J. This changes the proposal, never the true profile.
    """
    if not state.smooth:
        raise ValueError("projected Jacobian requires safe full local rank")
    J, p, m = state.M.shape
    q = W.shape[2]
    rhs = -(state.M.swapaxes(1, 2) @ W)[..., None]
    rhs = (rhs * state.coefficients[:, None, None, :]).reshape(J, m, q * m)
    dg = state.normal_solve(rhs)
    direct = (-W[..., None] * state.coefficients[:, None, None, :]).reshape(J, p, q * m)
    jac = direct - state.M @ dg
    residual = state.residual
    if tau:
        jac = np.concatenate((jac, -np.sqrt(tau) * dg), axis=1)
        residual = np.concatenate(
            (residual, -np.sqrt(tau) * state.coefficients), axis=1
        )
    root = np.sqrt(mass)[:, None] if root_mass is None else root_mass
    return (root[:, :, None] * jac).reshape(-1, q * m), (root * residual).ravel()


def orthogonal_jacobian(state, W, mass, tau, *, root_mass=None):
    """dr = -(1-QQ.T) E g - Q R^-T E.T r; no inverse squared."""
    if not state.smooth:
        raise ValueError("full-rank profile derivative requires smooth local systems")
    J, p, m = state.M.shape
    q = W.shape[2]
    rows = p + (m if tau else 0)
    design = (
        np.concatenate(
            (state.M, np.broadcast_to(np.sqrt(tau) * np.eye(m), (J, m, m))), axis=1
        )
        if tau
        else state.M
    )
    Q, R = np.linalg.qr(design, mode="reduced")
    direct = np.zeros((J, rows, q * m))
    direct[:, :p] = (W[:, :, :, None] * state.coefficients[:, None, None, :]).reshape(
        J, p, q * m
    )
    cross = np.zeros((J, m, q, m))
    wr = (state.residual[:, None, :] @ W).squeeze(1)
    for l in range(m):
        cross[:, l, :, l] = wr
    solve = np.linalg.solve(R.swapaxes(1, 2), cross.reshape(J, m, q * m))
    jac = -direct + Q @ ((Q.swapaxes(1, 2) @ direct) - solve)
    residual = (
        np.concatenate((state.residual, -np.sqrt(tau) * state.coefficients), axis=1)
        if tau
        else state.residual
    )
    root = np.sqrt(mass)[:, None] if root_mass is None else root_mass
    jac *= root[:, :, None]
    return jac.reshape(J * rows, q * m), (root * residual).ravel()


def diagonal_direction(Y, U, state, mass, G, *, workspace_bytes=4 * 1024**2):
    J, p, d = U.shape
    chunk = max(1, workspace_bytes // (8 * (p * d + d)))
    diagonal = np.zeros(d)
    for start in range(0, J, chunk):
        stop = min(start + chunk, J)
        weights = mass[start:stop] * np.sum(state.coefficients[start:stop] ** 2, axis=1)
        block = U[start:stop]
        diagonal += weights @ np.einsum("jpd,jpd->jd", block, block)
    floor = 0.05 * float(diagonal.mean())
    if not np.isfinite(floor) or floor <= 0:
        return G
    direction = G / (diagonal[:, None] + floor)
    return direction - Y @ (Y.T @ direction)


def polish_core(Y, state, V, W, K, I, mass, tau, candidate, backtrack, slope, limit):
    t = 0.5**backtrack
    curvature = 2 * (candidate[1].value - state.value - slope * t)
    if not np.isfinite(curvature) or curvature <= 0:
        return candidate, 0
    upper = min(2 * t, limit / np.linalg.norm(K, ord=2))
    proposed = float(np.clip(-slope * t**2 / curvature, t / 8, upper))
    if abs(proposed - t) <= 1e-3 * t:
        return candidate, 0
    Yn, Mn = reference._polar(Y, state.M, V, W, K * proposed)
    trial = reference._profile(Mn, I, mass, tau)
    if (
        trial.value < candidate[1].value
        and trial.value <= state.value + 1e-4 * slope * proposed
    ):
        return (Yn, trial), 1
    return candidate, 1


def krylov_direction(Y, U, I, state, mass, G, tau, cached_ranks, damping, stops):
    model = state
    if not state.smooth:
        subset = (
            reference._intrinsic_model(U, I, mass, state, cached_ranks)
            if cached_ranks is not None and not tau
            else None
        )
        if subset is None:
            return G
        mask, regular = subset
        J, _, m = state.M.shape
        coefficients, residual = (
            np.zeros_like(state.coefficients),
            np.zeros_like(state.residual),
        )
        coefficients[mask], residual[mask] = regular.coefficients, regular.residual
        triangular = right = inverse_squared = None
        if regular.triangular is not None:
            triangular = np.broadcast_to(np.eye(m), (J, m, m)).copy()
            triangular[mask] = regular.triangular
        else:
            right = np.broadcast_to(np.eye(m), (J, m, m)).copy()
            inverse_squared = np.zeros((J, m))
            right[mask], inverse_squared[mask] = regular.right, regular.inverse_squared
        # Zero rows are the regular-center derivative, not a full-profile model.
        model = replace(
            state,
            coefficients=coefficients,
            residual=residual,
            right=right,
            inverse_squared=inverse_squared,
            triangular=triangular,
            smooth=True,
        )
        mass = mass * mask
    operator = ProfileJacobian(Y, U, model, mass, tau)
    residual = (
        model.residual
        if not tau
        else np.concatenate(
            (model.residual, -np.sqrt(tau) * model.coefficients), axis=1
        )
    )
    rhs = (np.sqrt(mass)[:, None] * residual).ravel()
    rayleigh = np.linalg.norm(operator.matvec(G.ravel())) ** 2 / np.sum(G**2)
    mu = damping * rayleigh
    if not np.isfinite(mu) or mu <= 0:
        return G
    krylov_solve = cast(Callable[..., tuple], lsmr)
    result = krylov_solve(
        operator, rhs, damp=np.sqrt(mu), atol=1e-6, btol=1e-6, maxiter=2
    )
    stops.append((int(result[1]), int(result[2]), float(result[3]), float(result[4])))
    Z = result[0].reshape(Y.shape)
    Z -= Y @ (Y.T @ Z)
    return Z if np.all(np.isfinite(Z)) and np.sum(G * Z) > 0 else G


def solver_for(name):
    if name == "budget10":
        return partial(reference.solve, max_steps=10)
    if name in {"projected10", "projected20"}:
        return partial(
            load_reference("projected_jacobian").solve, max_steps=int(name[-2:])
        )
    lowrank_methods = {
        "lora1_10": ("lora", "1", 10),
        "lora_k10": ("lora", "k", 10),
        "lora_k20": ("lora", "k", 20),
        "svd1_10": ("svd", "1", 10),
        "svd_k10": ("svd", "k", 10),
        "svd_k20": ("svd", "k", 20),
        "enriched10": ("enriched", "k", 10),
        "enriched20": ("enriched", "k", 20),
        "svd_greedy10": ("greedy", "k", 10),
    }
    if name in lowrank_methods:
        mode, rank_text, steps = lowrank_methods[name]
        return partial(
            load_reference(f"lowrank_{mode}_{rank_text}").solve, max_steps=steps
        )
    if name in {"krylov10", "krylov20"}:
        return partial(load_reference("krylov").solve, max_steps=int(name[-2:]))
    if name in {"ridge6", "ridge4", "ridge2"}:
        return partial(reference.solve, local_ridge=10.0 ** -int(name[-1]))
    if name not in {
        "baseline",
        "relative10",
        "smooth_relative",
        "stable_jacobian",
        "diagonal",
        "gn_quadratic",
    }:
        raise ValueError(f"unknown accuracy candidate: {name}")
    if name == "relative10":
        return partial(
            reference.solve,
            max_steps=10,
            smooth_max_steps=5,
            quadratic_trials=0,
            constant_centers=False,
        )
    return load_reference(name).solve
