"""Small independent references for the isolated SVD candidate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_solver(filename: str):
    path = Path(__file__).parent / filename
    spec = importlib.util.spec_from_file_location(f"ADP.solver.{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    solver = load_solver("candidate_svd.py")
    rng = np.random.default_rng(772)
    J, p, d, m = 9, 4, 12, 3
    U = rng.normal(size=(J, p, d)) * np.geomspace(0.01, 100, d)
    U[..., -1] = 0
    U[..., 1] = U[..., 0] + 1e-7 * U[..., 1]
    P = np.linalg.qr(rng.normal(size=(d, m)))[0].T
    alpha = rng.normal(size=J)
    a = rng.normal(size=m)
    a /= np.linalg.norm(a)
    root_mass = np.sqrt(np.geomspace(1e-3, 1e3, J))
    target = rng.normal(size=J * p)
    rows = np.repeat(root_mass * alpha, p)
    for ridge in (0.0, 1e-4, 0.3, 1e3):
        current_U = U if ridge else rng.normal(size=(J, p, d))
        current_U[..., -1] = 0
        design = rows[:, None] * current_U.reshape(-1, d)
        flat = solver._FlatU(current_U)
        diagonal = flat.normal_diagonal(rows)
        np.testing.assert_allclose(diagonal, np.sum(design**2, axis=0), rtol=1e-13)
        # Force the bounded streamed path and ensure it agrees with the cache.
        limit = solver._DIRECT_WORKSPACE_BYTES
        solver._DIRECT_WORKSPACE_BYTES = 2 * d * U.itemsize
        streamed = solver._FlatU(current_U)
        np.testing.assert_allclose(streamed.normal_diagonal(rows), diagonal, rtol=1e-13)
        assert streamed.column_energy is None
        solver._DIRECT_WORKSPACE_BYTES = limit
        scale = 1 / np.sqrt(diagonal + ridge) if ridge else np.ones(d)
        operator = solver._v_augmented_operator(flat, rows, np.sqrt(ridge), scale)
        augmented = np.vstack((design, np.sqrt(ridge) * np.eye(d)))
        primal = rng.normal(size=d)
        dual = rng.normal(size=J * p + d)
        np.testing.assert_allclose(
            operator @ primal, augmented @ (scale * primal), rtol=1e-12
        )
        np.testing.assert_allclose(
            operator.rmatvec(dual), scale * (augmented.T @ dual), rtol=1e-12
        )
        np.testing.assert_allclose(
            primal @ operator.rmatvec(dual), (operator @ primal) @ dual, rtol=1e-12
        )
        z = P.T @ a if ridge else np.zeros(d)
        expected = np.linalg.lstsq(
            augmented, np.r_[target, np.sqrt(ridge) * z], rcond=None
        )[0]
        for warm in (None, rng.normal(size=d)):
            _, raw, _, _, _, args = solver._v_step(
                flat,
                target,
                P,
                alpha,
                a,
                root_mass,
                ridge,
                1e-8,
                None,
                x0=warm,
                precondition=True,
                krylov_tol=1e-13,
            )
            relative_error = np.linalg.norm(raw - expected) / max(
                1, np.linalg.norm(expected)
            )
            assert relative_error < 1e-7, (ridge, relative_error)
            certificate = solver._v_certificate(flat, raw, *args, ridge=ridge)
            assert np.isfinite(certificate) and certificate < 1e-7
        print(
            f"ridge={ridge:g}: dense, warm start, adjoint, diagonal/cache/stream pass",
            flush=True,
        )


if __name__ == "__main__":
    main()
