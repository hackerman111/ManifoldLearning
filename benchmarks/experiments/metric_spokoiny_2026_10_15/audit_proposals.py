"""Post-timing audit of proposal scales on the nine frozen inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ADP.solver import grassman
from ADP.solver._grassman_schur import metric_direction
from ADP.solver.LSMR import _normalize_index

HERE = Path(__file__).resolve().parent


def audit(output: Path) -> list[dict]:
    """Compare pre-line-search proposals, not accepted trajectory steps."""
    rows = []
    for path in sorted((output / "captures").glob("core_seed*_call*.npz")):
        with np.load(path, allow_pickle=False) as data:
            U, I, mass = (data[key] for key in ("U", "I", "mass"))
            Y = _normalize_index(data["index_init"]).T
        state = grassman._profile(grassman._project(U, Y), I, mass, 0.0)
        G = grassman._gradient(Y, U, state, mass)
        singular = np.linalg.svd(G, compute_uv=False)
        available = np.count_nonzero(
            singular > np.finfo(float).eps * max(G.shape) * singular[0]
        )
        q = min(2, int(available))
        if q == 0:
            raise RuntimeError(f"zero gradient rank at {path}")
        coeff = state.coefficients
        eigenvalues = np.linalg.eigvalsh(coeff.T @ (mass[:, None] * coeff))
        Z = metric_direction(G, coeff, mass, q)
        V = np.linalg.svd(Z, full_matrices=False)[0][:, :q]
        V = np.linalg.qr(V - Y @ (Y.T @ V), mode="reduced")[0]
        reduced = V.T @ G
        K = metric_direction(reduced, coeff, mass, q)
        isotropic = metric_direction(reduced, coeff, mass, q, 1.0)
        # A common mass rescaling preserves the minimizer, but rescales proposals.
        scaled = metric_direction(100 * G, coeff, 100 * mass, q)
        np.testing.assert_allclose(scaled, Z / 10, rtol=2e-12, atol=2e-12)
        assert np.sum(G * Z) < 0
        limit = float(np.tan(0.5))
        cosine = -float(np.sum(G * Z) / (np.linalg.norm(G) * np.linalg.norm(Z)))
        rows.append(
            {
                "snapshot": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "actual_q": q,
                "smooth": bool(state.smooth),
                "local_rank_loss": int(np.count_nonzero(state.ranks < Y.shape[1])),
                "mass_sum": float(mass.sum()),
                "R_eigenvalues": eigenvalues.tolist(),
                "R_condition": float(eigenvalues[-1] / eigenvalues[0]),
                "metric_floor_active": bool(eigenvalues[0] < 1e-6 * eigenvalues[-1]),
                "metric_K_norm2": float(np.linalg.norm(K, ord=2)),
                "isotropic_K_norm2": float(np.linalg.norm(isotropic, ord=2)),
                "spectral_cap": limit,
                "metric_cap_active": bool(np.linalg.norm(K, ord=2) > limit),
                "isotropic_cap_active": bool(np.linalg.norm(isotropic, ord=2) > limit),
                "negative_gradient_direction_cosine": cosine,
            }
        )
    if len(rows) != 9:
        raise RuntimeError(f"expected nine input snapshots, got {len(rows)}")
    target = output / "proposal_audit.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "scope": (
                    "sampled input proposals before line search; "
                    "not per-iteration telemetry"
                ),
                "audit_source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "mass_rescaling_check": "100*mass and 100*G -> Z/10 on all nine inputs",
                "rows": rows,
            },
            stream,
            indent=2,
            allow_nan=False,
        )
    return rows


if __name__ == "__main__":
    result = audit(HERE / "results_repaired")
    print(json.dumps(result, indent=2))
