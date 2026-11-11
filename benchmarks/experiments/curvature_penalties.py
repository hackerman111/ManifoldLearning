"""Research-only quadratic profiles with Hessian coupling and literal R_quad."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, cg
from scipy.spatial.distance import cdist

from ADP.core.solver_common import HPAOResult

ARMS = ("wls2", "hgraph", "hcommon", "m1quad", "rquad")


@dataclass
class Group:
    indices: np.ndarray
    delta: np.ndarray
    response: np.ndarray
    weights: np.ndarray
    transform: np.ndarray | None


@dataclass
class Problem:
    groups: list[Group]
    pi: np.ndarray
    laplacian: csr_matrix
    edges: np.ndarray
    edge_weights: np.ndarray
    ridge: float
    coupling: float
    arm: str
    scale: float
    information: dict


@dataclass
class Profile:
    value: float
    gradient: np.ndarray
    theta: np.ndarray
    hessians: np.ndarray
    components: dict
    cg_iterations: int
    normal_residual: float
    data_ranks: np.ndarray | None


def features(projected, weights):
    """Coordinates with ||theta_quadratic||²=||H||F²/2."""
    m = projected.shape[-1]
    a, b = np.triu_indices(m, 1)
    result = np.concatenate(
        (projected, projected**2 / np.sqrt(2), projected[..., a] * projected[..., b]),
        axis=-1,
    )
    return result - weights[:, None, :] @ result


def hessians(theta, m):
    result = np.zeros((len(theta), m, m))
    indices = np.arange(m)
    result[:, indices, indices] = np.sqrt(2) * theta[:, m : 2 * m]
    a, b = np.triu_indices(m, 1)
    result[:, a, b] = result[:, b, a] = theta[:, 2 * m :]
    return result


def build(data, basis, directions, arm, *, ridge=1e-4):
    if arm not in ARMS:
        raise ValueError(f"invalid curvature arm: {arm}")
    j, d, m = len(data.local), basis.shape[0], basis.shape[1]
    if ridge <= 0 or not np.isfinite(ridge):
        raise ValueError("positive finite ridge is required")
    pi = data.mass / data.mass.sum()
    ranks, saturated, second_variances = [], [], []
    q = directions.shape[1]
    if directions.shape != (j, q, d) or not np.isfinite(directions).all():
        raise ValueError("directions must have finite shape (J,q,d)")
    groups, energies = [], []
    # Equal-support buckets avoid padding every window to the largest support.
    buckets = np.ceil(np.log2(data.support)).astype(int)
    for bucket in np.unique(buckets):
        indices = np.flatnonzero(buckets == bucket)
        maximum = int(data.support[indices].max())
        delta = np.zeros((len(indices), maximum, d))
        response, weights = (
            np.zeros((len(indices), maximum)),
            np.zeros((len(indices), maximum)),
        )
        transform = (
            np.zeros((len(indices), d + (q if arm == "rquad" else 0), maximum))
            if arm in ("m1quad", "rquad")
            else None
        )
        for k, node in enumerate(indices):
            local = data.local[node]
            n = len(local.weights)
            delta[k, :n], response[k, :n], weights[k, :n] = (
                local.delta,
                local.response,
                local.weights,
            )
            if transform is not None:
                root = np.sqrt(local.weights)
                left, singular, _ = np.linalg.svd(
                    root[:, None] * local.delta, full_matrices=False
                )
                keep = singular**2 > 1e-12 * max(
                    float(singular[0] ** 2), np.finfo(float).tiny
                )
                rank = int(keep.sum())
                transform[k, :rank, :n] = left[:, keep].T * root
                ranks.append(rank)
                saturated.append(rank == n - 1)
                if arm == "rquad":
                    probes = (local.delta @ directions[node].T) ** 2
                    probes -= local.weights @ probes
                    variance = local.weights @ probes**2
                    denominator = np.maximum(
                        variance, 1e-12 * max(1.0, float(variance.max()))
                    )
                    transform[k, d:, :n] = (
                        probes.T * local.weights / np.sqrt(q * denominator[:, None])
                    )
                    second_variances.extend(variance)
        initial_features = features(delta @ basis, weights)
        energies.extend(
            np.einsum("jn,jnq->j", weights, initial_features[:, :, m:] ** 2)
        )
        groups.append(Group(indices, delta, response, weights, transform))
    distances = cdist(data.mean, data.mean, metric="sqeuclidean")
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1, kind="stable")[:, : min(4, j - 1)]
    edges = np.unique(
        np.sort(
            np.c_[np.repeat(np.arange(j), neighbors.shape[1]), neighbors.ravel()],
            axis=1,
        ),
        axis=0,
    )
    if len(edges):
        distances = distances[edges[:, 0], edges[:, 1]]
        edge_weights = np.exp(-distances / max(float(np.median(distances)), 1e-12))
        edge_weights /= edge_weights.sum()
        adjacency = csr_matrix(
            (np.tile(edge_weights, 2), (edges.T.ravel(), edges[:, ::-1].T.ravel())),
            shape=(j, j),
        )
        laplacian = diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    else:
        edge_weights, laplacian = np.empty(0), csr_matrix((j, j))
    parameter_count = m + m * (m + 1) // 2
    hscale = float(np.median(energies)) / (parameter_count - m)
    coupling = 0.1 * hscale if arm in ("hgraph", "hcommon") else 0.0
    information = {
        "support_quantiles": np.quantile(data.support, [0, 0.1, 0.5, 0.9, 1]),
        "effective_n_quantiles": np.quantile(data.effective_n, [0, 0.1, 0.5, 0.9, 1]),
        "support_below_parameters": int(
            np.count_nonzero(data.support < parameter_count + 1)
        ),
        "parameter_count_with_intercept": parameter_count + 1,
        "covariance_rank_histogram": np.bincount(ranks, minlength=d + 1),
        "saturated_first_moment_windows": int(sum(saturated)),
        "windows": j,
        "second_variance_quantiles": np.quantile(second_variances, [0, 0.5, 1])
        if second_variances
        else [],
        "second_moment_weight": 1.0 if arm == "rquad" else 0.0,
        "hessian_scale": hscale,
    }
    scale = max(
        1.0,
        float(
            sum(
                np.einsum(
                    "j,jn,jn->", pi[group.indices], group.weights, group.response**2
                )
                for group in groups
            )
        ),
    )
    return Problem(
        groups,
        pi,
        laplacian.tocsr(),
        edges,
        edge_weights,
        ridge,
        coupling,
        arm,
        scale,
        information,
    )


def profile(basis, problem, *, diagnostics=False):
    p = problem
    j, m = len(p.pi), basis.shape[1]
    q = m + m * (m + 1) // 2
    blocks, rhs, group_states = np.empty((j, q, q)), np.empty((j, q)), []
    for group in p.groups:
        projected = group.delta @ basis
        phi = features(projected, group.weights)
        if group.transform is None:
            design = np.sqrt(group.weights)[..., None] * phi
            target = np.sqrt(group.weights) * group.response
        else:
            design = group.transform @ phi
            target = (group.transform @ group.response[..., None]).squeeze(-1)
        indices = group.indices
        blocks[indices] = p.pi[indices, None, None] * (
            design.swapaxes(1, 2) @ design + p.ridge * np.eye(q)
        )
        rhs[indices] = p.pi[indices, None] * (
            design.swapaxes(1, 2) @ target[..., None]
        ).squeeze(-1)
        group_states.append((group, projected, design, target))
    iterations = 0

    def multiply(flat):
        theta = flat.reshape(j, q)
        output = (blocks @ theta[..., None]).squeeze(-1)
        if p.arm == "hgraph":
            output[:, m:] += 2 * p.coupling * (p.laplacian @ theta[:, m:])
        elif p.arm == "hcommon":
            output[:, m:] += (
                2 * p.coupling * p.pi[:, None] * (theta[:, m:] - p.pi @ theta[:, m:])
            )
        return output.ravel()

    if not p.coupling:
        theta = np.linalg.solve(blocks, rhs[..., None]).squeeze(-1)
    else:
        diagonal = blocks.copy()
        degree = p.laplacian.diagonal() if p.arm == "hgraph" else p.pi * (1 - p.pi)
        indices = np.arange(m, q)
        diagonal[:, indices, indices] += 2 * p.coupling * degree[:, None]
        inverse = np.linalg.inv(diagonal)
        operator = LinearOperator((j * q, j * q), matvec=multiply, dtype=float)
        preconditioner = LinearOperator(
            operator.shape,
            matvec=lambda v: (inverse @ v.reshape(j, q, 1)).ravel(),
            dtype=float,
        )

        def count(_):
            nonlocal iterations
            iterations += 1

        flat, info = cg(
            operator,
            rhs.ravel(),
            M=preconditioner,
            rtol=1e-10,
            atol=0,
            maxiter=500,
            callback=count,
        )
        if info:
            raise RuntimeError(f"curvature CG failed: info={info}")
        theta = flat.reshape(j, q)
    normal = float(
        np.linalg.norm(multiply(theta.ravel()) - rhs.ravel())
        / max(np.linalg.norm(rhs), np.finfo(float).tiny)
    )
    if normal > 1e-8:
        raise RuntimeError(f"curvature normal residual {normal}")
    h = hessians(theta, m)
    first, second, gradient = 0.0, 0.0, np.zeros_like(basis)
    ranks = np.empty(j, dtype=int) if diagnostics else None
    for group, projected, design, target in group_states:
        indices = group.indices
        residual = target - (design @ theta[indices, :, None]).squeeze(-1)
        if group.transform is None:
            weighted_residual = np.sqrt(group.weights) * residual
            first += float(np.sum(p.pi[indices, None] * residual**2))
        else:
            weighted_residual = (
                group.transform.swapaxes(1, 2) @ residual[..., None]
            ).squeeze(-1)
            d = basis.shape[0]
            first += float(np.sum(p.pi[indices, None] * residual[:, :d] ** 2))
            second += float(np.sum(p.pi[indices, None] * residual[:, d:] ** 2))
        weighted_residual -= group.weights * weighted_residual.sum(axis=1)[:, None]
        local_gradient = theta[indices, None, :m] + projected @ h[indices]
        local_products = group.delta.swapaxes(1, 2) @ (
            weighted_residual[..., None] * local_gradient
        )
        gradient -= 2 * np.einsum("j,jdm->dm", p.pi[indices], local_products)
        if diagnostics:
            ranks[indices] = np.linalg.matrix_rank(design)
    coupling_value = 0.0
    if p.arm == "hgraph":
        difference = h[p.edges[:, 0]] - h[p.edges[:, 1]]
        coupling_value = float(
            p.coupling * np.einsum("e,eab->", p.edge_weights, difference**2)
        )
    elif p.arm == "hcommon":
        difference = h - np.einsum("j,jab->ab", p.pi, h)
        coupling_value = float(p.coupling * np.einsum("j,jab->", p.pi, difference**2))
    components = {
        "first_or_wls": first,
        "second_moment": second,
        "ridge": float(p.ridge * np.sum(p.pi[:, None] * theta**2)),
        "hessian_coupling": coupling_value,
    }
    value = sum(components.values())
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise RuntimeError("nonfinite curvature profile")
    return Profile(value, gradient, theta, h, components, iterations, normal, ranks)


def solve(index, data, directions, arm, *, max_steps=40, tol=1e-6):
    started = time.perf_counter()
    initial = np.asarray(index, dtype=float).T
    d, m = initial.shape
    if not np.allclose(initial.T @ initial, np.eye(m), atol=1e-10):
        raise ValueError("initial index must have orthonormal rows")
    problem = build(data, initial, directions, arm)
    full = np.linalg.qr(initial, mode="complete")[0]
    reference, complement = full[:, :m], full[:, m:]
    shape = (d - m, m)
    preparation_seconds = time.perf_counter() - started
    cache, history, cg_effort, residuals = {}, [], [], []

    def evaluate(flat):
        if "point" not in cache or not np.array_equal(flat, cache["point"]):
            basis, triangular = np.linalg.qr(
                reference + complement @ flat.reshape(shape), mode="reduced"
            )
            state = profile(basis, problem)
            horizontal = state.gradient - basis @ (basis.T @ state.gradient)
            gradient = complement.T @ np.linalg.solve(triangular, horizontal.T).T
            cache.update(point=flat.copy(), basis=basis, state=state, gradient=gradient)
            cg_effort.append(state.cg_iterations)
            residuals.append(state.normal_residual)
        return cache["state"].value / problem.scale, cache[
            "gradient"
        ].ravel() / problem.scale

    initial_point = np.zeros(np.prod(shape))
    history.append(evaluate(initial_point)[0] * problem.scale)

    def accepted(flat):
        value = evaluate(flat)[0] * problem.scale
        if value > history[-1] + 1e-10 * max(1.0, history[-1]):
            raise RuntimeError("curvature chart accepted increasing objective")
        history.append(value)

    result = minimize(
        evaluate,
        initial_point,
        jac=True,
        method="L-BFGS-B",
        callback=accepted,
        options={"maxiter": max_steps, "maxls": 20, "ftol": 1e-12, "gtol": 1e-8},
    )
    evaluate(result.x)
    basis, state = cache["basis"], cache["state"]
    final = profile(basis, problem, diagnostics=True)
    horizontal = final.gradient - basis @ (basis.T @ final.gradient)
    norm = float(np.linalg.norm(horizontal) / problem.scale)
    return HPAOResult(
        basis.T,
        state.theta[:, :m],
        {
            "method": f"curvature_{arm}_chart_lbfgs",
            "arm": arm,
            "profile_objective": final.value,
            "loss": final.value / 2,
            "profile_history": history,
            "iterations": int(result.nit),
            "profile_evaluations": len(cg_effort) + 1,
            "converged": norm <= tol,
            "riemannian_gradient": norm,
            "scipy_success": bool(result.success),
            "stop_reason": "stationary" if norm <= tol else str(result.message),
            "components": final.components,
            "cg_iterations_total": int(sum(cg_effort)),
            "cg_iterations_max": max(cg_effort),
            "normal_residual_max": max(residuals + [final.normal_residual]),
            "data_rank_histogram": np.bincount(
                final.data_ranks, minlength=final.theta.shape[1] + 1
            ),
            "hessian_rms": float(
                np.sqrt(np.einsum("j,jab->", problem.pi, final.hessians**2))
            ),
            "penalty": {
                "local_ridge": problem.ridge,
                "hessian_coupling": problem.coupling,
                "edges": len(problem.edges),
                "prepare_seconds": preparation_seconds,
            },
            "information": problem.information,
        },
    )
