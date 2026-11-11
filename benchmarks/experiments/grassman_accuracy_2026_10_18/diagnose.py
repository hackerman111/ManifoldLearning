"""Inspect actual R-SDR Grassmann inputs and local numerical guards."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ADP.core.ADP_Config import ADP_Config
from ADP.core.ADP_Solver import ADP_solver
from ADP.core.multi.ADP_multi_index import ADP_multi_index
from ADP.solver import grassman
from experiments.r_sdr.design import Case, generate
from experiments.r_sdr.runner import adp_settings

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SEED = 90800
EPS = np.finfo(float).eps


def finite(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None


def rank_stats(s: np.ndarray, scale: int) -> dict:
    ranks = np.count_nonzero(s > EPS * scale * s[:, :1], axis=1)
    cond = np.divide(s[:, 0], s[:, -1], out=np.full(len(s), np.inf), where=s[:, -1] > 0)
    keys, counts = np.unique(ranks, return_counts=True)
    return {
        "ranks": dict(zip(map(str, keys), map(int, counts), strict=True)),
        "condition_median": finite(float(np.median(cond))),
    }


def diagnose(index, U, I, mass, number, result):
    frozen = OUT / "frozen" / f"{number:03d}.npz"
    np.savez_compressed(frozen, index_init=index, U=U, I=I, mass=mass)
    Y = grassman._normalize_index(index).T
    M = grassman._project(U, Y)
    fast = grassman._profile(M, I, mass, 0.0)
    ref = grassman._profile_svd(M, I, mass, 0.0)
    g_fast = grassman._gradient(Y, U, fast, mass)
    g_ref = grassman._gradient(Y, U, ref, mass)
    sm = np.linalg.svd(M, compute_uv=False)
    su = np.linalg.svd(U, compute_uv=False)
    rank = np.count_nonzero(sm > EPS * max(U.shape[1], M.shape[1]) * sm[:, :1], axis=1)
    rank_loss = rank < M.shape[2]
    relative = sm[:, -1] <= np.sqrt(EPS) * sm[:, 0]
    absolute = sm[:, -1] <= 64 * EPS * np.maximum(1.0, np.linalg.norm(M, axis=(1, 2)))
    nonsmooth = rank_loss | relative | absolute
    diag = result.diagnostics
    live = grassman._profile(grassman._project(U, result.index.T), I, mass, 0.0).value
    history = diag.get("profile_history", ())
    cached = float(history[-1]) if history else None
    return {
        "frozen": str(frozen.relative_to(OUT)),
        "zeros": {
            "mass_centers": int(np.count_nonzero(mass == 0)),
            "U_rows": int(np.count_nonzero(np.all(U == 0, axis=2))),
            "I_rows": int(np.count_nonzero(np.all(I == 0, axis=1))),
        },
        "UY": rank_stats(sm, max(U.shape[1], M.shape[1])),
        "Uj": rank_stats(su, max(U.shape[1:])),
        "guard": {
            "nonsmooth": int(nonsmooth.sum()),
            "rank_loss": int(rank_loss.sum()),
            "relative": int(relative.sum()),
            "absolute": int(absolute.sum()),
        },
        "qr_svd": {
            "path": "qr" if fast.triangular is not None else "svd_fallback",
            "coef": finite(float(np.max(np.abs(fast.coefficients - ref.coefficients)))),
            "objective": finite(abs(fast.value - ref.value)),
            "gradient": finite(float(np.max(np.abs(g_fast - g_ref)))),
        },
        "returned": {
            "orth": finite(
                float(
                    np.linalg.norm(result.index @ result.index.T - np.eye(len(index)))
                )
            ),
            "history": finite(cached) if cached is not None else None,
            "live": finite(live),
            "cached_minus_live": finite(cached - live) if cached is not None else None,
            "diagnostics": diag,
        },
    }


def run() -> None:
    (OUT / "frozen").mkdir(parents=True, exist_ok=True)
    seed = int(np.random.SeedSequence([SEED, 1]).generate_state(1)[0]) % (2**31 - 1)
    cases, number = [], 0
    for d in (5, 20, 50):
        case = Case(f"d-{d}", 500, d)
        x, y, _ = generate(case, SEED)
        settings = adp_settings(case, {})
        config = ADP_Config(seed=seed, **settings)
        calls: list[dict] = []

        def capture(index, U, I, _calls=calls, **kw):
            nonlocal number
            number += 1
            mass = np.asarray(kw["mass"], dtype=float)
            start = time.perf_counter()
            out = grassman.solve(
                index,
                U,
                I,
                mass=mass,
                lambda_prox=kw["lambda_prox"],
                max_steps=5,
                tol=1e-6,
                local_ridge=0.0,
            )
            elapsed = time.perf_counter() - start
            item = diagnose(index, U, I, mass, number, out)
            item["solver_seconds_excluding_diagnostics"] = elapsed
            _calls.append(item)
            return out

        model = ADP_multi_index(2, config=config, solver=ADP_solver(capture))
        model.fit(x, y)
        cfg = {
            "adp": settings,
            "seed": seed,
            "estimator": config.estimator,
            "lambda_penalty": config.lambda_penalty,
            "local_ridge": config.local_ridge,
        }
        cases.append(
            {
                "case": case.name,
                "shape": [case.n, d, 2],
                "config": cfg,
                "outer_calls": calls,
                "stop_reason": model.result_.stop_reason,
            }
        )
    sources = (
        "benchmarks/experiments/r_sdr/design.py",
        "benchmarks/experiments/r_sdr/runner.py",
        "ADP/core/ADP_Config.py",
        "ADP/engine/common/index_fit.py",
        "ADP/solver/grassman.py",
        "benchmarks/experiments/grassman_accuracy_2026_10_18/diagnose.py",
    )
    artifact = {
        "seed": SEED,
        "method_seed": seed,
        "dtype": "float64",
        "blas_threads": 1,
        "solver": {"max_steps": 5, "tol": 1e-6, "local_ridge": 0.0},
        "timing_note": "solver timer excludes diagnostics; not benchmark timing",
        "source_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources
        },
        "cases": cases,
    }
    (OUT / "diagnostics.json").write_text(
        json.dumps(artifact, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    run()
