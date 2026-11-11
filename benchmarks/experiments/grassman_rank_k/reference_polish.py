"""Supplementary independent chart L-BFGS reference, tau=0 only.

B=Y0+horizontal_Y0(Z) always has full column rank. Its column space has the
same unregularized profile as QR(B); use the Euclidean envelope derivative
followed by the fixed chart projection. No production numerical kernels.
"""

from __future__ import annotations

import argparse
import json
from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .common import environment, json_value
from .reference import _cg


def chart_profile(z, base, U, I, mass):
    Z = z.reshape(base.shape)
    B = base + Z - base @ (base.T @ Z)
    M = U @ B
    coefficients = np.stack(
        [
            np.linalg.lstsq(local, target, rcond=None)[0]
            for local, target in zip(M, I, strict=True)
        ]
    )
    residual = I - np.einsum("jpm,jm->jp", M, coefficients)
    value = float(np.sum(mass[:, None] * residual**2))
    adjoint = np.einsum("jpd,jp->jd", U, residual, optimize=True)
    gradient = -2 * adjoint.T @ (mass[:, None] * coefficients)
    gradient -= base @ (base.T @ gradient)
    return value, gradient.ravel()


def polish(snapshot, max_steps: int = 5000, tol: float = 1e-9) -> dict[str, Any]:
    U, I, mass = (snapshot[k] for k in ("U", "I", "mass"))
    initial, truth = snapshot["index_init"].T, snapshot["truth"]
    rng = np.random.default_rng(816159)
    random = np.linalg.qr(rng.normal(size=initial.shape))[0]
    scale = max(1.0, float(np.sum(mass[:, None] * I**2)))
    started = perf_counter()
    starts, solutions = [], []
    for name, base in (("initial", initial), ("truth", truth), ("random", random)):
        z0 = np.zeros(base.size)
        initial_f = chart_profile(z0, base, U, I, mass)[0]
        result = minimize(
            chart_profile,
            z0,
            args=(base, U, I, mass),
            jac=True,
            method="L-BFGS-B",
            options={
                "gtol": tol * scale * 0.01,
                "ftol": 1e-15,
                "maxiter": max_steps,
                "maxls": 50,
                "maxcor": 20,
            },
        )
        Z = result.x.reshape(base.shape)
        B = base + Z - base @ (base.T @ Z)
        Q = np.linalg.qr(B, mode="reduced")[0]
        Y, row = _cg(Q, U, I, mass, max_steps=1000, tol=tol)
        row.update(
            {
                "start": name,
                "initial_objective": initial_f,
                "lbfgs_iterations": int(result.nit),
                "lbfgs_evaluations": int(result.nfev),
                "lbfgs_success": bool(result.success),
                "lbfgs_message": str(result.message),
                "chart_basis_condition": float(np.linalg.cond(B)),
            }
        )
        starts.append(row)
        solutions.append(Y)
    best = min(range(len(starts)), key=lambda k: starts[k]["objective"])
    Y = solutions[best]
    return {
        **starts[best],
        "starts": starts,
        "best_start": starts[best]["start"],
        "status": "ok",
        "wall_sec": perf_counter() - started,
        "projector_error": float(
            np.sqrt(max(0.0, 1 - np.linalg.norm(Y.T @ truth) ** 2 / Y.shape[1]))
        ),
        "reference_kind": (
            "independent fixed-chart L-BFGS plus Riemannian CG; "
            "numerical upper reference"
        ),
        "environment": environment(),
    }


def main() -> None:
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.snapshot, allow_pickle=False) as data:
        snapshot = {k: data[k] for k in data.files}
    row = polish(snapshot)
    row["snapshot"] = str(args.snapshot)
    args.output.write_text(json.dumps(json_value(row), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
