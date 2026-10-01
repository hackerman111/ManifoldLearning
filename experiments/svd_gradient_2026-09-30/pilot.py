"""Bounded paired fixed-g pilot; run from repo root with PYTHONPATH=."""

import csv
import hashlib
import json
import platform
import subprocess
import time
import tracemalloc
from pathlib import Path

import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

from ADP.solver.SVD import solve_fixed_coefficients

OUT = Path(__file__).parent
SETTINGS = {
    "rank": 2,
    "lambda_penalty": 0.7,
    "rank_tol": 1e-8,
    "inner_tol": 1e-6,
    "inner_maxiter": 20,
}
rows = []
with threadpool_limits(limits=1):
    metadata = {
        "scope": "fixed-g synthetic pilot; no adaptive EDR recovery claim",
        "dimensions": [100, 300],
        "seeds": [11, 23, 37],
        "J": 60,
        "p": 8,
        "m": 4,
        "dtype": "float64",
        "settings": SETTINGS,
        "threads": threadpool_info(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": subprocess.check_output(["git", "status", "--short"], text=True),
        "solver_sha256": hashlib.sha256(
            Path("ADP/solver/SVD.py").read_bytes()
        ).hexdigest(),
        "data": "Gaussian U with geomspace(.2,5,d) column scales; Gaussian g; "
        "positive geomspace(.2,3,J) mass; I=U Btrue.T g + .05 Gaussian noise",
        "memory": "separate deterministic rerun, tracemalloc peak; excludes input "
        "arrays allocated before start and may miss native BLAS allocations",
        "warmup": "one unmeasured call for each configuration",
    }
    for d in metadata["dimensions"]:
        for seed in metadata["seeds"]:
            rng = np.random.default_rng(seed)
            P = np.linalg.qr(rng.normal(size=(d, 4)))[0].T
            truth = np.linalg.qr(rng.normal(size=(d, 4)))[0].T
            U = rng.normal(size=(60, 8, d)) * np.geomspace(0.2, 5, d)
            g, mass = rng.normal(size=(60, 4)), np.geomspace(0.2, 3, 60)
            I = np.array([U[j] @ truth.T @ g[j] for j in range(60)])
            I += 0.05 * rng.normal(size=I.shape)
            for target in ("matrix", "correction"):
                for search in ("alternating", "gradient"):
                    row = {
                        "d": d,
                        "seed": seed,
                        "target": target,
                        "search": search,
                        "seconds": None,
                        "peak_bytes": None,
                        "objective": None,
                        "relative_matrix_error": None,
                        "rank": None,
                        "stop": None,
                        "u_passes": None,
                        "lsmr_iterations": None,
                        "inner_converged": None,
                        "error": "",
                    }
                    kwargs = {
                        **SETTINGS,
                        "low_rank_target": target,
                        "rank_one_search": search,
                    }
                    try:
                        solve_fixed_coefficients(P, U, I, g, mass, **kwargs)
                        start = time.perf_counter()
                        B, diag = solve_fixed_coefficients(P, U, I, g, mass, **kwargs)
                        row["seconds"] = time.perf_counter() - start
                        tracemalloc.start()
                        try:
                            replay, _ = solve_fixed_coefficients(
                                P, U, I, g, mass, **kwargs
                            )
                            row["peak_bytes"] = tracemalloc.get_traced_memory()[1]
                        finally:
                            tracemalloc.stop()
                        np.testing.assert_array_equal(B, replay)
                        assert np.all(np.diff(diag["rank_objective_history"]) <= 1e-8)
                        row.update(
                            objective=diag["rank_objective_history"][-1],
                            relative_matrix_error=float(
                                np.linalg.norm(B - truth) / np.linalg.norm(truth)
                            ),
                            rank=diag["effective_rank"],
                            stop=diag["rank_stop_reason"],
                            u_passes=diag["u_vector_passes"],
                            lsmr_iterations=diag["lsmr_iterations_total"],
                            inner_converged=str(diag["inner_converged"]),
                        )
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    rows.append(row)
    metadata["solver_sha256_after"] = hashlib.sha256(
        Path("ADP/solver/SVD.py").read_bytes()
    ).hexdigest()
    assert metadata["solver_sha256"] == metadata["solver_sha256_after"]
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (OUT / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
print(json.dumps({"rows": len(rows), "failures": sum(bool(r["error"]) for r in rows)}))
