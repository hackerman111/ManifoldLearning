"""Frozen paired study; run with BLAS threads=1 via the documented command."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy

from experiments.weak_moments import fit as moment_fit

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/results/verification/weak-moment-20261030"
CASES = ("quad_strong", "quad_weak", "cubic", "quartic", "uniform", "correlated")
ARMS = ("raw_full", "raw_split", "residual_split", "refined", "adp", "wls1")
SOURCES = (
    "benchmarks/experiments/weak_moments.py",
    "benchmarks/experiments/weak_moment_study.py",
    "benchmarks/experiments/shared_polynomial.py",
    "benchmarks/experiments/local_polynomial.py",
    "ADP/solver/grassman.py",
    "ADP/engine/common/index_fit.py",
    "ADP/engine/common/weights.py",
    "ADP/core/ADP_Config.py",
    "ADP/engine/common/statistic.py",
    "ADP/engine/common/initialize.py",
    "ADP/engine/common/calculus.py",
)


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def fingerprints():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES}


def arrays(case, seed):
    streams = np.random.SeedSequence([seed, 20261030]).spawn(3)
    rng = np.random.default_rng(streams[0])
    x = rng.normal(size=(2000, 20))
    if case == "uniform":
        x = rng.uniform(-np.sqrt(3), np.sqrt(3), size=x.shape)
    if case == "correlated":
        # Known-to-generator covariance; the estimator does not receive it.
        x *= np.geomspace(1, 0.1, 20)
    truth = np.linalg.qr(np.random.default_rng(streams[1]).normal(size=(20, 2)))[0]
    z = x @ truth
    strength = 5 if case in ("quad_strong", "uniform", "correlated") else 1
    delta = 0.12 if case == "quad_weak" else 0.2
    weak = z[:, 1] ** 2 - 1
    if case == "cubic":
        delta, weak = 0.4, (z[:, 1] ** 3 - 3 * z[:, 1]) / np.sqrt(6)
    if case == "quartic":
        delta, weak = 0.4, (z[:, 1] ** 4 - 6 * z[:, 1] ** 2 + 3) / np.sqrt(24)
    noise = np.random.default_rng(streams[2]).normal(size=len(x))
    y = strength * z[:, 0] + delta * weak + 0.3 * noise
    return x, y, truth


def metrics(basis, truth):
    p = basis @ basis.T
    e = float(np.linalg.norm(p - truth @ truth.T))
    return {
        "edr": e,
        "weak_coverage": float(np.linalg.norm(basis.T @ truth[:, 1]) ** 2),
        "max_sine": float(np.linalg.norm(truth - p @ truth, ord=2)),
        "orthogonality": float(np.linalg.norm(basis.T @ basis - np.eye(2))),
    }


def adp_fit(x, y, seed, arm):
    from ADP import ADP_Config
    from ADP.engine.common.index_fit import fit_index
    from ADP.engine.common.statistic import calculate_statistics
    from ADP.solver.grassman import solve as grassman
    from experiments import local_polynomial as poly
    from experiments.two_sided_meanmave import TimeOnlyProfiler

    cache = {}
    stages = []

    def statistics(X, Y, weights, directions, batch_size=32, *, normalized=False):
        blocks = list(weights)
        cache["weight_hash"] = digest(*(w for _, w in blocks))
        if arm == "wls1":
            cache["local"] = poly.neighborhoods(X, Y, blocks)
        return calculate_statistics(
            X, Y, iter(blocks), directions, batch_size=batch_size, normalized=normalized
        )

    def solve(index, stats):
        if arm == "adp":
            result = grassman(index, stats.U, stats.I, mass=stats.mass)
        else:
            result = poly.solve(index, cache["local"], degree=1, ridge=0, max_steps=40)
        stages.append({"weights": cache["weight_hash"], "solver": result.diagnostics})
        return result

    config = ADP_Config(
        seed=seed,
        N_lin=50,
        N_J=100,
        N_loc=20,
        N_phi=12,
        outer_steps=8,
        select_step="best",
    )
    result = fit_index(
        x,
        y,
        config=config,
        mode="multi",
        index_dim=2,
        solve=solve,
        statistics_function=statistics,
        profiler=TimeOnlyProfiler(),
    )
    return result.index.T, {
        "stop_reason": result.metadata["stop_reason"],
        "selected_iteration": result.metadata["selected_iteration"],
        "initial_hash": digest(result.initial_index),
        "stages": stages,
        "converged": all(s["solver"].get("converged", False) for s in stages),
    }


def worker(case, seed, arm):
    x, y, truth = arrays(case, seed)
    row = {"case": case, "seed": seed, "arm": arm, "data_hash": digest(x, y, truth)}
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            if arm in ("adp", "wls1"):
                basis, diag = adp_fit(x, y, seed, arm)
            else:
                start = moment_fit(
                    x, y, arm="residual_split" if arm == "refined" else arm
                )
                basis, diag = start["basis"], start["diagnostics"]
                row["moment_singular_values"] = start["singular_values"].tolist()
                if arm == "refined":
                    from experiments.shared_polynomial import refine

                    result = refine(x, y, basis, degree=2, max_steps=40)
                    basis, diag = result.basis, result.diagnostics
        row.update(status="ok", basis=basis.tolist(), diagnostics=diag)
        row.update(metrics(basis, truth))
    except Exception as error:
        row.update(status="error", error=f"{type(error).__name__}: {error}")
    row["fit_seconds"] = time.perf_counter() - started
    row["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return row


def bootstrap_ratio(candidate, baseline):
    log = np.log(np.maximum(candidate, 1e-15) / np.maximum(baseline, 1e-15))
    rng = np.random.default_rng(3010999)
    draws = rng.choice(log, size=(10000, len(log)), replace=True).mean(axis=1)
    return [
        float(np.exp(log.mean())),
        *np.exp(np.quantile(draws, [0.025, 0.975])).tolist(),
    ]


def summarize(rows):
    groups = {}
    for case in CASES:
        groups[case] = {}
        for arm in ARMS:
            subset = [r for r in rows if r["case"] == case and r["arm"] == arm]
            ok = [r for r in subset if r["status"] == "ok"]
            groups[case][arm] = {
                "count": len(subset),
                "failures": len(subset) - len(ok),
                "mean_edr": float(np.mean([r["edr"] for r in ok])) if ok else None,
                "recovery": sum(r["edr"] <= 0.3 for r in ok),
                "mean_weak_coverage": float(np.mean([r["weak_coverage"] for r in ok]))
                if ok
                else None,
                "median_seconds": float(np.median([r["fit_seconds"] for r in subset])),
                "median_rss_mib": float(np.median([r["peak_rss_mib"] for r in subset])),
                "converged": sum(bool(r["diagnostics"].get("converged")) for r in ok),
            }
    gate_rows = [r for r in rows if r["case"] in ("quad_strong", "quad_weak")]
    pairs = {(r["case"], r["seed"], r["arm"]): r for r in gate_rows}
    keys = sorted({(r["case"], r["seed"]) for r in gate_rows})
    causal = {}
    for arm in ("raw_split", "residual_split", "refined", "adp", "wls1"):
        candidate = [pairs[c, s, arm] for c, s in keys]
        baseline = [pairs[c, s, "raw_full"] for c, s in keys]
        if any(r["status"] != "ok" for r in candidate + baseline):
            causal[arm] = {"pass": False, "reason": "failed fit"}
            continue
        edr = bootstrap_ratio(
            [r["edr"] for r in candidate], [r["edr"] for r in baseline]
        )
        timing = float(
            np.median(
                [
                    a["fit_seconds"] / b["fit_seconds"]
                    for a, b in zip(candidate, baseline, strict=True)
                ]
            )
        )
        rss = float(
            np.median(
                [
                    a["peak_rss_mib"] / b["peak_rss_mib"]
                    for a, b in zip(candidate, baseline, strict=True)
                ]
            )
        )
        causal[arm] = {
            "edr_ratio_ci": edr,
            "time_ratio": timing,
            "rss_ratio": rss,
            "pass": bool(edr[2] < 1 and timing <= 3 and rss <= 1.5),
        }
    return {
        "groups": groups,
        "quadratic_comparisons": causal,
        "gate_pass": causal["residual_split"]["pass"],
        "rows": len(rows),
    }


def run(phase):
    protocol_path = OUT / "PROTOCOL.json"
    if not protocol_path.exists():
        if phase != "selection":
            raise ValueError("selection must be frozen first")
        data_hashes = {
            f"{c}:{s}": digest(*arrays(c, s))
            for c in CASES
            for s in range(3010100, 3010108)
        }
        protocol = {
            "sources": fingerprints(),
            "cases": CASES,
            "arms": ARMS,
            "n": 2000,
            "d": 20,
            "m": 2,
            "gamma": 1,
            "pilot_fraction": 0.5,
            "selection": list(range(3010100, 3010108)),
            "validation": list(range(3011100, 3011108)),
            "selection_hashes": data_hashes,
            "seed_streams": "SeedSequence([seed,20261030]).spawn(3)",
            "dtype": "float64",
            "block_size": 256,
            "threads": 1,
            "environment": {
                "python": sys.version,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "platform": platform.platform(),
            },
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "git_dirty": subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ),
            "gate": (
                "quadratic cases pooled; residual/raw_full upper95CI<1;"
                "time<=3,RSS<=1.5;zero failures"
            ),
            "adp": "N_lin=50,N_J=100,N_loc=20,N_phi=12,outer_steps=8;Grassmann20",
            "wls1": (
                "same config/start/first weights; ridge=0,L-BFGS40;"
                " optimizer also differs"
            ),
            "kernel": (
                "Epanechnikov K(u)=max(1-u**2,0);"
                " u=squared tensor distance; source hashes recorded"
            ),
            "refined": (
                "residual split start; degree2 polynomial;Schur GN40;"
                " no true link supplied"
            ),
            "resources": (
                "sequential separate processes;900 sec/phase;"
                "1024 MiB worker RSS checked after fit"
            ),
        }
        protocol_path.write_text(json.dumps(protocol, indent=2))
    protocol = json.loads(protocol_path.read_text())
    if protocol["sources"] != fingerprints():
        raise ValueError("frozen source fingerprint mismatch")
    if (
        phase == "validation"
        and not json.loads((OUT / "selection_summary.json").read_text())["gate_pass"]
    ):
        raise ValueError("selection failed; validation forbidden")
    destination = OUT / f"{phase}.jsonl"
    if destination.exists():
        raise ValueError(
            "phase already exists; historical results cannot be overwritten"
        )
    jobs = [(c, s, a) for c in CASES for s in protocol[phase] for a in ARMS]
    np.random.default_rng(3010001).shuffle(jobs)
    rows = []
    started = time.perf_counter()
    env = dict(
        os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    with destination.open("x") as output:
        for i, (case, seed, arm) in enumerate(jobs):
            if time.perf_counter() - started > 900:
                raise TimeoutError("phase budget exhausted; partial rows retained")
            command = [
                sys.executable,
                "-m",
                "experiments.weak_moment_study",
                "worker",
                "--case",
                case,
                "--seed",
                str(seed),
                "--arm",
                arm,
            ]
            before = time.perf_counter()
            result = subprocess.run(
                command,
                env=env,
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
            if result.returncode:
                raise RuntimeError(f"worker failed: {result.stderr[-2000:]}")
            row = json.loads(result.stdout)
            row["process_seconds"] = time.perf_counter() - before
            if row["peak_rss_mib"] > 1024:
                row.update(status="error", error="RSS budget exceeded")
            if (
                phase == "selection"
                and row["data_hash"] != protocol["selection_hashes"][f"{case}:{seed}"]
            ):
                raise ValueError("input fingerprint mismatch")
            output.write(json.dumps(row) + "\n")
            output.flush()
            rows.append(row)
            if (i + 1) % 24 == 0:
                print(f"{phase}: {i + 1}/{len(jobs)} rows", flush=True)
    summary = summarize(rows)
    summary["phase_seconds"] = time.perf_counter() - started
    (OUT / f"{phase}_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {"rows": len(rows), "gate": summary["quadratic_comparisons"]}, indent=2
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("worker", "selection", "validation"))
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.mode == "worker":
        print(json.dumps(worker(args.case, args.seed, args.arm)))
    else:
        run(args.mode)


if __name__ == "__main__":
    main()
