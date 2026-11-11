"""Масштаб первого GN на сохранённых входах; не бенчмарк времени."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ADP.solver.grassman import _core_jacobian, _gradient, _profile, _project
from ADP.solver.LSMR import _normalize_index
from experiments.grassman_accuracy_2026_10_18.backtracking import _intrinsic_model
from experiments.grassman_accuracy_2026_10_18.pilot_intrinsic import ROOT, digest

OUT = Path(__file__).resolve().parent


def run() -> None:
    output = OUT / "relative_gn_scales.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    records = []
    for path in sorted((OUT / "frozen").glob("*.npz")):
        with np.load(path) as data:
            U, I, mass = data["U"], data["I"], data["mass"]
            Y = _normalize_index(data["index_init"]).T
        state = _profile(_project(U, Y), I, mass, 0.0)
        record = {"input": path.name, "sha256": digest(path), "smooth": state.smooth}
        if state.smooth:
            records.append(record)
            continue
        model = _intrinsic_model(U, I, mass, state, np.full(len(I), -1))
        record["eligible_intrinsic"] = model is not None
        if model is None:
            records.append(record)
            continue
        regular, regular_state = model
        G = _gradient(Y, U, state, mass)
        V, s, _ = np.linalg.svd(G, full_matrices=False)
        available = min(
            Y.shape[1],
            Y.shape[0] - Y.shape[1],
            np.count_nonzero(s > np.finfo(float).eps * max(Y.shape) * s[0]),
        )
        q = min(available, np.searchsorted(np.cumsum(s**2), 0.95 * np.sum(s**2)) + 1)
        V = V[:, :q] - Y @ (Y.T @ V[:, :q])
        V, _ = np.linalg.qr(V, mode="reduced")
        jac, residual = _core_jacobian(
            regular_state, _project(U[regular], V), mass[regular], 0.0
        )
        coordinates = jac.shape[1]
        scale = float(np.einsum("ij,ij->", jac, jac)) / coordinates
        proposals = {}
        for name, damping in (("absolute", 0.05), ("relative", 0.05 * scale)):
            augmented = np.vstack((jac, np.sqrt(damping) * np.eye(coordinates)))
            rhs = np.concatenate((-residual, np.zeros(coordinates)))
            K = np.linalg.lstsq(augmented, rhs, rcond=None)[0].reshape(q, Y.shape[1])
            length = float(np.linalg.norm(K, ord=2))
            K *= min(1.0, np.tan(0.5) / length)
            proposals[name] = {
                "damping": damping,
                "uncapped_step_norm": length,
                "slope": float(np.sum((V.T @ G) * K)),
            }
        record.update(
            jac_shape=list(jac.shape),
            jac_mean_square_column_norm=scale,
            proposals=proposals,
        )
        records.append(record)
    sources = (
        "ADP/solver/grassman.py",
        "benchmarks/experiments/grassman_accuracy_2026_10_18/backtracking.py",
        "benchmarks/experiments/grassman_accuracy_2026_10_18/diagnose_relative_gn.py",
    )
    output.write_text(
        json.dumps(
            {
                "seed": 90800,
                "note": "first proposals only; no full-fit quality or timing claim",
                "source_sha256": {p: digest(ROOT / p) for p in sources},
                "calls": records,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "calls": len(records),
                "eligible": [r for r in records if r.get("eligible_intrinsic")],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
