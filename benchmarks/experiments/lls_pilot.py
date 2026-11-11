"""Isolated paired pilot of moment and QR local-regression statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ADP import ADP_Config, ADP_solver
from ADP.core.ADP_Statistic import ADP_Statistics
from ADP.engine.common.index_fit import fit_index
from ADP.engine.common.logger import IndexProfiler
from ADP.engine.common.statistic import calculate_statistics
from ADP.solver.grassman import solve
from experiments.r_sdr.design import Case, generate, metrics
from experiments.r_sdr.protocol import ROOT, THREAD_ENV, digest, source_files
from experiments.r_sdr.runner import adp_settings, run_process

SOLVER = {
    "max_steps": 20,
    "tol": 1e-6,
    "rank": None,
    "energy_tol": 0.05,
    "local_ridge": 0.0,
    "max_angle": 0.5,
    "workspace_bytes": 16 * 1024**2,
    "relative_gn": True,
    "quadratic_trials": 3,
    "constant_centers": True,
    "smooth_max_steps": None,
}
BASE = Case("base", 500, 10)
CASES = [
    BASE,
    replace(BASE, name="d-20", d=20),
    replace(BASE, name="d-50", d=50),
    replace(BASE, name="corr-0.9", rho=0.9),
    replace(BASE, name="product-m2", link="product"),
    Case("rank-3", 500, 20, 3),
]
ARMS = ["base", "phid", "identity", "lls"]


def lls_statistics(X, Y, weights, directions, batch_size=32, *, normalized=False):
    """QR compression of centered WLS; mass remains the outer solver weight."""
    J, P, d = directions.shape
    if P != d:
        raise ValueError("LLS requires N_phi=d")
    I, U = np.zeros((J, P)), np.zeros((J, P, d))
    mass, mean = np.empty(J), np.empty((J, d))
    n_eff, S = np.empty(J), np.empty(J)
    blocks = [(0, weights)] if isinstance(weights, np.ndarray) else weights
    expected = 0
    for start, W in blocks:
        W = np.asarray(W)
        if (
            start != expected
            or W.ndim != 2
            or W.shape[1] != len(X)
            or start + len(W) > J
            or not np.isfinite(W).all()
            or np.any(W < 0)
        ):
            raise ValueError("invalid weights or block coverage")
        for b, w_full in enumerate(W):
            j = start + b
            idx = np.flatnonzero(w_full)
            w = w_full[idx]
            mj = w.sum()
            if mj <= 0:
                raise ValueError("local mass must be positive")
            a = w / mj
            mu, yb = a @ X[idx], a @ Y[idx]
            sa = np.sqrt(a)
            q, r = np.linalg.qr(sa[:, None] * (X[idx] - mu), mode="reduced")
            k = r.shape[0]
            U[j, :k], I[j, :k] = r, q.T @ (sa * (Y[idx] - yb))
            mass[j], mean[j], n_eff[j], S[j] = mj, mu, 1 / (a @ a), yb
        expected += len(W)
    if expected != J:
        raise ValueError("incomplete weight blocks")
    if not normalized:
        I *= mass[:, None]
        U *= mass[:, None, None]
    return ADP_Statistics(I, U, mass, mean, n_eff, np.zeros((J, P)), S)


def identity_statistics(X, Y, weights, directions, batch_size=32, *, normalized=False):
    J, P, d = directions.shape
    if P != d:
        raise ValueError("identity moments require N_phi=d")
    return calculate_statistics(
        X,
        Y,
        weights,
        np.broadcast_to(np.eye(d), (J, d, d)),
        batch_size=batch_size,
        normalized=normalized,
    )


def run_fit(x, y, case, seed, arm):
    settings = adp_settings(case, {})
    if arm != "base":
        settings["N_phi"] = case.d
    config = ADP_Config(seed=seed, **settings)
    solver = ADP_solver(solve, **SOLVER)
    statistic = {"lls": lls_statistics, "identity": identity_statistics}.get(arm)
    profiler = IndexProfiler()
    started = time.perf_counter()
    try:
        fitted = fit_index(
            x,
            y,
            config=config,
            mode="multi",
            index_dim=case.m,
            solve=lambda index, stats: solver.fit_current(
                stats, index, normalized=True, lambda_prox=config.lambda_penalty
            ),
            profiler=profiler,
            trace_indices=True,
            statistics_function=statistic,
        )
    finally:
        profiler.finish()
    return fitted, time.perf_counter() - started


def worker(task_path):
    task = json.loads(task_path.read_text())
    data = np.load(task["input"])
    case = Case(**task["case"])
    fitted, elapsed = run_fit(data["X"], data["y"], case, task["seed"], task["arm"])
    trace = fitted.metadata["trace"]
    truth = data["truth"]
    step_errors = [metrics(r["basis"], truth)["projector_distance"] for r in trace]
    diag = [r["solver"] for r in trace]
    row = {
        "edr": metrics(fitted.index.T, truth)["projector_distance"],
        "initial_edr": metrics(fitted.initial_index.T, truth)["projector_distance"],
        "oracle_step_edr": min(step_errors),
        "last_edr": step_errors[-1],
        "steps": len(trace),
        "selected_step": fitted.metadata["selected_iteration"],
        "stop_reason": fitted.metadata["stop_reason"],
        "fit_seconds": elapsed,
        "orthogonality_error": metrics(fitted.index.T, truth)["orthogonality_error"],
        "inner_calls": len(diag),
        "inner_certified": sum(r["converged"] for r in diag),
        "inner_gradient_max": max(r["riemannian_gradient"] for r in diag),
        "inner_rank_loss_max": max(r["local_rank_loss"] for r in diag),
    }
    np.savez_compressed(
        task["basis"],
        basis=fitted.index.T,
        initial=fitted.initial_index.T,
        trace=np.stack([r["basis"] for r in trace]),
    )
    details = [{k: v for k, v in r.items() if k != "basis"} for r in trace]
    Path(task["trace"]).write_text(json.dumps(details, indent=2, allow_nan=False))
    Path(task["result"]).write_text(json.dumps(row, indent=2, allow_nan=False))


def summarize(output):
    protocol = json.loads((output / "manifest.json").read_text())
    rows = [
        json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines()
    ]
    expected = len(protocol["cases"]) * len(protocol["seeds"]) * len(protocol["arms"])
    keys = {(r["case"], r["seed"], r["arm"]) for r in rows}
    if len(keys) != len(rows):
        raise ValueError("duplicate results")
    summaries = []
    lines = [
        "# LLS paired recovery pilot",
        "",
        f"Completed {len(rows)}/{expected} fits.",
        "EDR = RMS principal sine; lower is better. Recovery threshold = 0.1.",
        "Oracle is diagnostic only. All arms use the unchanged adaptive outer loop.",
        "",
        "| Case | Arm | Valid/planned | Median EDR | Recovery | Fit s | Process s | RSS MiB | Inner certificates |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in protocol["cases"]:
        for arm in protocol["arms"]:
            group = [r for r in rows if r["case"] == c["name"] and r["arm"] == arm]
            ok = [r for r in group if r["status"] == "ok"]
            item = {
                "case": c["name"],
                "arm": arm,
                "n_ok": len(ok),
                "n_planned": len(protocol["seeds"]),
                "recovered": sum(r["edr"] <= 0.1 for r in ok),
            }
            for key in [
                "edr",
                "fit_seconds",
                "process_seconds",
                "peak_rss_mib",
                "initial_edr",
                "last_edr",
                "oracle_step_edr",
            ]:
                item[key] = float(np.median([r[key] for r in ok])) if ok else None
            item["inner_certified"] = sum(r["inner_certified"] for r in ok)
            item["inner_calls"] = sum(r["inner_calls"] for r in ok)
            summaries.append(item)
            values = [
                f"{item[k]:.6g}" if item[k] is not None else "NA"
                for k in ["edr", "fit_seconds", "process_seconds", "peak_rss_mib"]
            ]
            lines.append(
                f"| {c['name']} | {arm} | {len(ok)}/{item['n_planned']} | {values[0]} | "
                f"{item['recovered']}/{item['n_planned']} | {' | '.join(values[1:])} | "
                f"{item['inner_certified']}/{item['inner_calls']} |"
            )
    base = {
        (r["case"], r["seed"]): r
        for r in rows
        if r["arm"] == "base" and r["status"] == "ok"
    }
    comparisons = []
    for arm in protocol["arms"]:
        if arm == "base":
            continue
        paired = [
            (r, base[(r["case"], r["seed"])])
            for r in rows
            if r["arm"] == arm
            and r["status"] == "ok"
            and (r["case"], r["seed"]) in base
        ]
        logs = np.array([np.log(r["edr"] / b["edr"]) for r, b in paired])
        by_seed = [
            np.mean(
                [v for v, (r, _) in zip(logs, paired, strict=True) if r["seed"] == s]
            )
            for s in protocol["seeds"]
            if any(r["seed"] == s for r, _ in paired)
        ]
        if by_seed:
            boot = np.random.default_rng(91823).choice(
                by_seed, size=(10000, len(by_seed)), replace=True
            )
            ci = np.exp(np.quantile(np.mean(boot, axis=1), [0.025, 0.975])).tolist()
            ratio = float(np.exp(logs.mean()))
        else:
            ci, ratio = [None, None], None
        comp = {
            "arm": arm,
            "pairs": len(paired),
            "geometric_edr_ratio": ratio,
            "ci95": ci,
            "wins": sum(r["edr"] < b["edr"] for r, b in paired),
            "median_edr_delta": float(
                np.median([r["edr"] - b["edr"] for r, b in paired])
            )
            if paired
            else None,
            "median_fit_ratio": float(
                np.median([r["fit_seconds"] / b["fit_seconds"] for r, b in paired])
            )
            if paired
            else None,
        }
        comparisons.append(comp)
    lls = next(r for r in comparisons if r["arm"] == "lls")
    complete_ok = len(rows) == expected and all(r["status"] == "ok" for r in rows)
    gate = complete_ok and lls["geometric_edr_ratio"] <= 0.95 and lls["ci95"][1] < 1.0
    lines += [
        "",
        "Paired ratios use returned bases, relative to base. CI resamples seed blocks across all six cases.",
        "",
        "```json",
        json.dumps(comparisons, indent=2),
        "```",
        "",
        f"Prespecified LLS quality gate: {'PASS' if gate else 'FAIL'}.",
        "Gate: all fits valid; geometric EDR ratio <= 0.95; upper 95% seed-block bootstrap CI < 1.",
        "Inner certificates are reported separately and do not certify outer convergence.",
        "This experiment changes the estimator and does not isolate optimizer quality.",
    ]
    result = {
        "complete": len(rows) == expected,
        "planned": expected,
        "recorded": len(rows),
        "failed": sum(r["status"] != "ok" for r in rows),
        "quality_gate": bool(gate),
        "summaries": summaries,
        "comparisons": comparisons,
    }
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return result


def run_stage(output, stage):
    arms = ARMS if stage == "selection" else ["base", "lls"]
    seeds = (
        list(range(630000, 630010))
        if stage == "selection"
        else list(range(640000, 640010))
    )
    if stage == "validation":
        selection = json.loads((output / "selection/report.json").read_text())
        if not selection["quality_gate"]:
            raise ValueError(
                "selection quality gate failed; validation stays untouched"
            )
    files = sorted(set(source_files() + [Path(__file__).resolve()]))
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
    config = asdict(ADP_Config())
    config["kernel"] = "ADP.core.ADP_Config.epanechnikov"
    protocol = {
        "stage": stage,
        "cases": [asdict(c) for c in CASES],
        "arms": arms,
        "seeds": seeds,
        "default_config": config,
        "case_settings": {c.name: adp_settings(c, {}) for c in CASES},
        "solver": SOLVER,
        "source_sha256": hashes,
        "python": sys.version,
        "numpy": np.__version__,
        "gate": "all valid; paired geometric EDR ratio <=0.95; upper seed-block CI95 <1",
        "threads": 1,
        "timeout_seconds": 120,
        "memory_mib": 4096,
    }
    output = output / stage
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != protocol:
            raise ValueError("source or protocol drift; choose a new output")
    else:
        manifest.write_text(json.dumps(protocol, indent=2, allow_nan=False))
        with zipfile.ZipFile(
            output / "sources.zip", "x", zipfile.ZIP_DEFLATED
        ) as archive:
            for path in files:
                archive.write(path, path.relative_to(ROOT))
        with zipfile.ZipFile(output / "sources.zip") as archive:
            archive.extractall(output / "source")
    env = os.environ.copy()
    env.update({key: "1" for key in THREAD_ENV})
    env.update(
        OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE", PYTHONPATH=str(output / "source")
    )
    journal = output / "runs.jsonl"
    previous = (
        [json.loads(line) for line in journal.read_text().splitlines()]
        if journal.exists()
        else []
    )
    done = {(r["case"], r["seed"], r["arm"]) for r in previous}
    if len(done) != len(previous):
        raise ValueError("duplicate journal keys")
    count, total = len(done), len(CASES) * len(seeds) * len(arms)
    with journal.open("a") as stream:
        for case in CASES:
            for seed in seeds:
                x, y, truth = generate(case, seed)
                data_hash = hashlib.sha256(
                    x.tobytes() + y.tobytes() + truth.tobytes()
                ).hexdigest()
                input_path = output / "inputs" / f"{case.name}-{seed}.npz"
                input_path.parent.mkdir(exist_ok=True)
                if not input_path.exists():
                    np.savez_compressed(input_path, X=x, y=y, truth=truth)
                order = np.random.default_rng(seed).permutation(arms)
                for arm in order:
                    if (case.name, seed, arm) in done:
                        continue
                    folder = output / "fits" / f"{case.name}-{seed}-{arm}"
                    folder.mkdir(parents=True, exist_ok=True)
                    task = {
                        "input": str(input_path),
                        "case": asdict(case),
                        "seed": seed,
                        "arm": str(arm),
                        "result": str(folder / "result.json"),
                        "basis": str(folder / "basis.npz"),
                        "trace": str(folder / "trace.json"),
                    }
                    (folder / "task.json").write_text(json.dumps(task, indent=2))
                    row = {
                        "case": case.name,
                        "seed": seed,
                        "arm": str(arm),
                        "data_sha256": data_hash,
                        "input_sha256": digest(input_path),
                    }
                    row.update(
                        run_process(
                            [
                                sys.executable,
                                "-m",
                                "experiments.lls_pilot",
                                "--worker",
                                str(folder / "task.json"),
                            ],
                            env,
                            folder / "worker.log",
                            timeout=120,
                            memory_mib=4096,
                            cwd=output / "source",
                        )
                    )
                    if row["status"] == "ok":
                        row.update(json.loads((folder / "result.json").read_text()))
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    count += 1
                    print(
                        f"[{count}/{total}] {stage} {case.name} {seed} {arm}: {row['status']} "
                        f"EDR={row.get('edr')} fit={row.get('fit_seconds')}",
                        flush=True,
                    )
    result = summarize(output)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ["complete", "failed", "quality_gate", "comparisons"]
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/outputs/lls_pilot_20261023")
    )
    parser.add_argument(
        "--stage", choices=["selection", "validation"], default="selection"
    )
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        run_stage(args.output.resolve(), args.stage)


if __name__ == "__main__":
    main()
