"""Paired full-moment/ridge versus MEANMAVE benchmark in one existing grid cell."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import scipy

from ADP import ADP_Config
from ADP.engine.common.index_fit import fit_index
from ADP.engine.common.logger import IndexProfiler
from ADP.engine.common.statistic import calculate_statistics
from experiments import coupled_penalties as coupled
from experiments.light_penalties import solve as baseline
from experiments.mave_grassmann_grid import arrays, case, settings
from experiments.r_sdr.design import metrics
from experiments.r_sdr.protocol import ROOT, digest, environment
from experiments.r_sdr.runner import read_worker_meta, run_process

OUT = ROOT / "benchmarks/results/verification/full-moments-vs-meanmave-20261025/main"
SEEDS = tuple(range(651030, 651040))
M_VALUES = (2, 3, 5)
ARMS = ("Grassmann", "Full-moments-ridge", "MEANMAVE")
BENCHMARK_CASE = case(25, 2, 5000, 0.3, "additive", "large")


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, default=_json_default, allow_nan=False) + "\n")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_paths():
    paths = [
        *sorted((ROOT / "ADP").rglob("*.py")),
        *sorted((ROOT / "benchmarks/experiments/r_sdr").glob("*.py")),
        *sorted((ROOT / "benchmarks/experiments/r_sdr").glob("*.R")),
        ROOT / "benchmarks/experiments/mave_grassmann_grid.py",
        ROOT / "benchmarks/experiments/full_moments_meanmave.py",
        ROOT / "benchmarks/experiments/coupled_penalties.py",
        ROOT / "benchmarks/experiments/light_penalties.py",
        ROOT / "benchmarks/experiments/edr_metric_study.py",
        OUT.parent / "PROTOCOL.md",
    ]
    return sorted(set(paths))


def freeze():
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    (OUT / "inputs").mkdir(parents=True)
    (OUT / "fits").mkdir()
    r_environment, env = environment("Rscript", 1)
    env.update({key: "1" for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")})
    input_hashes = {}
    for seed in SEEDS:
        for m in M_VALUES:
            c = case(25, m, 5000, 0.3, "additive", "large")
            x, y, truth = arrays(c, seed)
            path = OUT / "inputs" / f"m{m}-{seed}.npz"
            np.savez_compressed(path, x=x, y=y, truth=truth)
            input_hashes[str(path.relative_to(ROOT))] = sha(path)
    files = source_paths()
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in files}
    with zipfile.ZipFile(OUT / "sources.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT))
    (OUT / "source").mkdir()
    with zipfile.ZipFile(OUT / "sources.zip") as archive:
        archive.extractall(OUT / "source")
    jobs = [(m, seed, arm) for m in M_VALUES for seed in SEEDS for arm in ARMS]
    np.random.default_rng(20261026).shuffle(jobs)
    cases = {}
    config, solver = settings(BENCHMARK_CASE, "Grassmann")
    for m in M_VALUES:
        c = case(25, m, 5000, 0.3, "additive", "large")
        cases[str(m)] = {"case": c, "config": settings(c, "Grassmann")[0],
                         "solver": settings(c, "Grassmann")[1]}
    protocol = {
        "benchmark": "existing mave_grassmann_grid additive-d25-m{2,3,5}-n5000-s0.3",
        "seeds": SEEDS, "arms": ARMS, "jobs": jobs, "cases": cases,
        "input_hashes": input_hashes, "source_hashes": source_hashes,
        "source_archive_sha256": sha(OUT / "sources.zip"),
        "r_environment": r_environment,
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": sys.platform,
        "max_steps": 20, "tol": 1e-6, "timeout_seconds": 180,
        "memory_limit_mib": 3072, "workers": 1, "blas_threads": 1,
        "meanmave": 'MAVE::mave.compute(method="MEANMAVE",max.dim=m,screen=d)',
        "full_moments": {"relative_eigen_cutoff": 1e-12, "ridge_alpha": 0.01,
                         "solver": "QR-chart L-BFGS, same 20 step budget"},
        "data_seed_scheme": "SeedSequence([seed,d,20261025]).spawn(3), existing benchmark arrays()",
        "edr": "||P_est-P_true||F/sqrt(2m); recovered iff EDR<=0.1",
        "random_job_order_seed": 20261026,
    }
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True, default=_json_default).encode()).hexdigest()
    write_json(OUT / "manifest.json", {"fingerprint": fingerprint, "protocol": protocol})
    return env, fingerprint


def worker(run_dir: Path, arm: str, m: int, seed: int, r_env: dict[str, str]):
    with np.load(run_dir / "inputs" / f"m{m}-{seed}.npz") as saved:
        x, y, truth = saved["x"], saved["y"], saved["truth"]
    method_seed = int(np.random.SeedSequence([seed, 1]).generate_state(1)[0]) % (2**31 - 1)
    c = case(25, m, 5000, 0.3, "additive", "large")
    config_kwargs, solver_kwargs = settings(c, "Grassmann")
    config = ADP_Config(seed=method_seed, **config_kwargs)
    context, stages = {}, []
    profiler = IndexProfiler()
    r_info = {}

    if arm == "MEANMAVE":
        with tempfile.TemporaryDirectory(prefix="full-mave-") as name:
            scratch = Path(name)
            xp, yp = scratch / "x.csv", scratch / "y.csv"
            bp, mp, log = scratch / "basis.csv", scratch / "meta.tsv", scratch / "worker.log"
            np.savetxt(xp, x, delimiter=",", fmt="%.17g")
            np.savetxt(yp, y, delimiter=",", fmt="%.17g")
            command = ["Rscript", "--vanilla", str(run_dir / "source/experiments/r_sdr/worker.R"),
                       "MAVE-package", str(xp), str(yp), str(bp), str(mp), str(method_seed), str(m), "20", "1e-5"]
            r_info = run_process(command, r_env, log, timeout=180, memory_mib=3072,
                                 cwd=run_dir / "source")
            if r_info["status"] != "ok":
                raise RuntimeError(f"MEANMAVE {r_info['status']}: {log.read_text()[-4000:]}")
            basis = np.loadtxt(bp, delimiter=",", ndmin=2)
            meta = read_worker_meta("MAVE-package", mp)
            fit_seconds = meta["fit_seconds"]
            diagnostics = {"r": meta, "r_process_seconds": r_info["process_seconds"],
                           "r_peak_rss_mib": r_info["peak_rss_mib"]}
            initial = None
    else:
        def statistics(X, Y, weights, directions, batch_size=32, *, normalized=False):
            blocks = [(0, weights)] if isinstance(weights, np.ndarray) else list(weights)
            first_hash = hashlib.sha256()
            for _, block in blocks:
                first_hash.update(np.ascontiguousarray(block).tobytes())
            stats = calculate_statistics(X, Y, iter(blocks), directions, batch_size,
                                         normalized=normalized)
            if not stages:
                context.update(first_weights_sha256=first_hash.hexdigest(),
                               first_directions_sha256=hashlib.sha256(np.ascontiguousarray(directions).tobytes()).hexdigest())
            if arm == "Full-moments-ridge":
                covariances, cross = [], []
                for _, block in blocks:
                    for row in block:
                        support = row > 0
                        a = row[support] / row.sum()
                        z = X[support] - a @ X[support]
                        dy = Y[support] - a @ Y[support]
                        covariances.append(z.T @ (a[:, None] * z))
                        cross.append(z.T @ (a * dy))
                context["moments"] = np.array(covariances), np.array(cross)
            return stats

        def solve(index, stats):
            result = (baseline(index, stats, index.T, "base") if arm == "Grassmann" else
                      coupled.solve(index, stats, "full", moments=context["moments"]))
            stages.append(result.diagnostics)
            return result

        started = time.perf_counter()
        try:
            fitted = fit_index(x, y, config=config, mode="multi", index_dim=m,
                               solve=solve, profiler=profiler,
                               statistics_function=statistics, trace_indices=True)
        finally:
            profiler.finish()
        fit_seconds = time.perf_counter() - started
        basis, initial = fitted.index.T, fitted.initial_index.T
        diagnostics = {"stages": stages, "profile": profiler.model_profile(),
                       "selected_iteration": fitted.metadata["selected_iteration"],
                       "stop_reason": fitted.metadata["stop_reason"]}
    if basis.shape != (25, m) or not np.isfinite(basis).all():
        raise ValueError(f"invalid basis shape/values: {basis.shape}")
    return {
        "method": arm, "m": m, "seed": seed, "method_seed": method_seed,
        "case": c["name"], "status": "ok", "basis": basis,
        "initial_basis": initial, "edr": metrics(basis, truth)["projector_distance"],
        "initial_edr": None if initial is None else metrics(initial, truth)["projector_distance"],
        "orthogonality_error": float(np.linalg.norm(basis.T @ basis - np.eye(m))),
        "recovered": bool(metrics(basis, truth)["projector_distance"] <= 0.1),
        "fit_seconds": float(fit_seconds), "diagnostics": diagnostics,
        "first_weights_sha256": context.get("first_weights_sha256"),
        "first_directions_sha256": context.get("first_directions_sha256"),
        "input_sha256": hashlib.sha256(x.tobytes() + y.tobytes() + truth.tobytes()).hexdigest(),
        "r_process": r_info,
        "worker_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


def run():
    env, fingerprint = freeze()
    rows, started = [], time.perf_counter()
    jobs = [(m, seed, arm) for m in M_VALUES for seed in SEEDS for arm in ARMS]
    np.random.default_rng(20261026).shuffle(jobs)
    for i, (m, seed, arm) in enumerate(jobs, 1):
        manifest = json.loads((OUT / "manifest.json").read_text())
        for rel, expected in manifest["protocol"]["source_hashes"].items():
            if sha(ROOT / rel) != expected:
                raise RuntimeError(f"frozen source drift: {rel}")
        before = time.perf_counter()
        log = OUT / "fits" / f"m{m}-{seed}-{arm}.log"
        command = [sys.executable, "-m", "experiments.full_moments_meanmave",
                   "worker", str(OUT), arm, str(m), str(seed)]
        result = run_process(command, env, log, timeout=180, memory_mib=3072,
                             cwd=OUT / "source")
        if result["status"] == "ok":
            try:
                row = json.loads(log.read_text().splitlines()[-1])
                if (row["method"], row["m"], row["seed"]) != (arm, m, seed):
                    raise ValueError("worker result key mismatch")
                row["status"] = "ok"
            except Exception as error:
                row = {"method": arm, "m": m, "seed": seed, "status": "invalid_output", "error": repr(error)}
        else:
            row = {"method": arm, "m": m, "seed": seed, **result,
                   "error": log.read_text()[-4000:]}
        row.update(process_seconds=time.perf_counter() - before,
                   process_peak_rss_mib=result["peak_rss_mib"], fingerprint=fingerprint)
        path = OUT / "fits" / f"m{m}-{seed}-{arm}.json"
        write_json(path, row)
        rows.append(row)
        if i % 6 == 0 or row["status"] != "ok":
            print(f"{i}/90 {arm} m={m} seed={seed}: {row['status']}; elapsed={time.perf_counter()-started:.1f}s", flush=True)
    write_json(OUT / "execution.json", {"planned": 90, "recorded": len(rows),
              "successful": sum(r["status"] == "ok" for r in rows),
              "failed": sum(r["status"] != "ok" for r in rows),
              "wall_seconds": time.perf_counter() - started})
    analyze()


def bootstrap(values, *, rng_seed=20261026, draws=10000):
    values = np.asarray(values, dtype=float)
    sample = np.random.default_rng(rng_seed).integers(0, len(values), (draws, len(values)))
    estimates = values[sample].mean(axis=1)
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def analyze():
    manifest = json.loads((OUT / "manifest.json").read_text())
    rows = [json.loads(path.read_text()) for path in sorted((OUT / "fits").glob("*.json"))]
    planned = {(m, seed, arm) for m in M_VALUES for seed in SEEDS for arm in ARMS}
    keys = [(row["m"], row["seed"], row["method"]) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != planned:
        raise ValueError("fit schedule is incomplete or has duplicate keys")
    if any(row["fingerprint"] != manifest["fingerprint"] for row in rows):
        raise ValueError("fit fingerprint mismatch")
    comparisons, cases = [], []
    lookup = {(row["m"], row["seed"], row["method"]): row for row in rows}
    for m in M_VALUES:
        for arm in ("Full-moments-ridge", "Grassmann"):
            pairs = [(lookup[m, seed, arm], lookup[m, seed, "MEANMAVE"])
                     for seed in SEEDS if lookup[m, seed, arm]["status"] == lookup[m, seed, "MEANMAVE"]["status"] == "ok"]
            differences = [a["edr"] - b["edr"] for a, b in pairs]
            ratios = [max(a["edr"], 1e-12) / max(b["edr"], 1e-12) for a, b in pairs]
            time_ratios = [a["fit_seconds"] / b["fit_seconds"] for a, b in pairs]
            comparisons.append({
                "m": m, "method": arm, "control": "MEANMAVE",
                "planned_pairs": len(SEEDS), "complete_pairs": len(pairs),
                "edr_difference_mean": float(np.mean(differences)) if pairs else None,
                "edr_difference_ci": bootstrap(differences) if pairs else None,
                "edr_ratio_geometric": float(np.exp(np.mean(np.log(ratios)))) if pairs else None,
                "edr_ratio_ci": bootstrap(np.log(ratios)) if pairs else None,
                "full_wins": sum(a["edr"] < b["edr"] for a, b in pairs),
                "recovery": [sum(a["recovered"] for a, _ in pairs), sum(b["recovered"] for _, b in pairs)],
                "fit_time_ratio_geometric": float(np.exp(np.mean(np.log(time_ratios)))) if pairs else None,
                "fit_time_seconds_median": [float(np.median([a["fit_seconds"] for a, _ in pairs])),
                                             float(np.median([b["fit_seconds"] for _, b in pairs]))] if pairs else None,
                "process_seconds_median": [float(np.median([a["process_seconds"] for a, _ in pairs])),
                                            float(np.median([b["process_seconds"] for _, b in pairs]))] if pairs else None,
                "valid": len(pairs) == len(SEEDS),
            })
        for arm in ARMS:
            group = [lookup[m, seed, arm] for seed in SEEDS]
            successful = [row for row in group if row["status"] == "ok"]
            cases.append({"m": m, "method": arm, "planned": len(group),
                          "successful": len(successful),
                          "mean_edr": float(np.mean([r["edr"] for r in successful])) if successful else None,
                          "median_fit_seconds": float(np.median([r["fit_seconds"] for r in successful])) if successful else None,
                          "median_process_seconds": float(np.median([r["process_seconds"] for r in successful])) if successful else None,
                          "recovery": sum(r["recovered"] for r in successful)})
    summary = {"fingerprint": manifest["fingerprint"], "planned": len(planned),
               "recorded": len(rows), "successful": sum(r["status"] == "ok" for r in rows),
               "failures": [r for r in rows if r["status"] != "ok"],
               "comparisons": comparisons, "cases": cases,
               "validation": False, "production_changed": False}
    write_json(OUT / "summary.json", summary)
    lines = ["# Full moments + ridge versus MEANMAVE", "",
             "Existing benchmark cell: additive,d=25,n=5000,noise=.3. Ten new paired seeds per m. EDR below is better.", "",
             "| m | Full method | EDR full−MEANMAVE [95% CI] | Geometric EDR ratio [95% CI] | Wins / 10 | Recovery full/MAVE | Fit seconds median full/MAVE | Time ratio |",
             "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        if row["method"] != "Full-moments-ridge":
            continue
        if not row["complete_pairs"]:
            lines.append(f"| {row['m']} | Full-moments-ridge | — | — | — | — | — | — |")
            continue
        diff_ci = row["edr_difference_ci"]
        ratio_ci = np.exp(row["edr_ratio_ci"])
        t = row["fit_time_seconds_median"]
        lines.append(f"| {row['m']} | Full-moments-ridge | {row['edr_difference_mean']:.4f} [{diff_ci[0]:.4f},{diff_ci[1]:.4f}] | {row['edr_ratio_geometric']:.4f} [{ratio_ci[0]:.4f},{ratio_ci[1]:.4f}] | {row['full_wins']}/{row['complete_pairs']} | {row['recovery'][0]}/{row['complete_pairs']} / {row['recovery'][1]}/{row['complete_pairs']} | {t[0]:.2f}/{t[1]:.2f} | {row['fit_time_ratio_geometric']:.3f} |")
    lines += ["", "The existing Grassmann anchor and each method/cell's raw averages are in summary.json. Pair intervals resample the ten seeds; each m has an exploratory unadjusted 95% interval. MEANMAVE is the installed MAVE package method, not a hand-coded MAVE approximation. Full-moment and Grassmann use ADP's same local initialization and same-sample best-iteration selection; both were timed end-to-end. MEANMAVE package fit time excludes R/package startup; process time includes it.", "", f"Outcomes: {summary['successful']}/{summary['planned']} successful. Failed/timeouts remain recorded and are not replaced. No held-out validation or production change."]
    (OUT.parent / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"planned": summary["planned"], "successful": summary["successful"],
                      "comparisons": [r for r in comparisons if r["method"] == "Full-moments-ridge"]},
                     default=_json_default, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "worker", "analyze"))
    parser.add_argument("args", nargs="*")
    options = parser.parse_args()
    if options.command == "run":
        run()
    elif options.command == "analyze":
        analyze()
    else:
        run_dir, arm, m, seed = options.args
        r_env = dict(os.environ)
        print(json.dumps(worker(Path(run_dir), arm, int(m), int(seed), r_env),
                         default=_json_default, allow_nan=False))
