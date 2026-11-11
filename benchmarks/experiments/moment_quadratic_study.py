"""Frozen paired full-moment and local-quadratic recovery experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy

from ADP import ADP_Config, ADP_solver
from ADP.engine.common.index_fit import fit_index
from ADP.engine.common.logger import IndexProfiler
from ADP.engine.common.statistic import calculate_statistics
from ADP.solver import grassman
from experiments import local_polynomial as poly
from experiments.r_sdr.design import metrics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/results/verification/moment-quadratic-20261023"
FUNCTIONS = ("original", "sine", "quadratic", "product")
DIMS = (2, 3, 5)
ARMS = ("base", "identity", "wls1", "wls2")
SEEDS = {
    "development": tuple(range(438100, 438110)),
    "validation": tuple(range(439100, 439110)),
}


def digest(*arrays):
    return hashlib.sha256(b"".join(a.tobytes() for a in arrays)).hexdigest()


def input_arrays(seed):
    streams = np.random.SeedSequence([seed, 20261023]).spawn(3)
    x = np.random.default_rng(streams[0]).normal(size=(5000, 25))
    basis = np.linalg.qr(np.random.default_rng(streams[1]).normal(size=(25, 5)))[0]
    noise = np.random.default_rng(streams[2]).normal(size=5000)
    return x, basis, noise


def response(x, truth, noise, function):
    z = x @ truth
    if function == "original":
        raw = (
            np.sin(z[:, 0]) + 0.5 * z[:, 1] ** 2 + 0.35 * np.tanh(z[:, 2:]).sum(axis=1)
        )
    elif function == "sine":
        raw = np.sin(z).sum(axis=1)
    elif function == "quadratic":
        raw = (z * z).sum(axis=1)
    elif function == "product":
        raw = np.prod(z, axis=1)
    else:
        raise ValueError(function)
    return (raw - raw.mean()) / raw.std() + 0.3 * noise


def fit(x, y, m, method_seed, arm, truth=None):
    config = ADP_Config(
        seed=method_seed,
        N_J=300,
        N_phi=25 if arm == "identity" else 12,
        N_loc=20,
        N_lin=50,
    )
    cache, stages = {}, []
    profiler = IndexProfiler()
    degree = 2 if arm == "wls2" else 1

    def statistics(X, Y, weights, directions, batch_size=32, *, normalized=False):
        if not normalized:
            raise ValueError("this protocol requires normalized statistics")
        blocks = list(weights)
        cache["weight_hash"] = digest(*(block for _, block in blocks))
        if arm.startswith("wls"):
            cache["data"] = poly.neighborhoods(X, Y, blocks)
        if arm == "identity":
            directions = np.broadcast_to(
                np.eye(X.shape[1]), (len(directions), X.shape[1], X.shape[1])
            )
        stats = calculate_statistics(
            X, Y, iter(blocks), directions, batch_size=batch_size, normalized=True
        )
        cache["effective_n"] = np.quantile(stats.n_eff, [0, 0.1, 0.5, 0.9, 1]).tolist()
        return stats

    def solve(index, stats):
        if arm.startswith("wls"):
            result = poly.solve(
                index, cache["data"], degree=degree, ridge=1e-4, max_steps=40
            )
        else:
            result = ADP_solver(grassman.solve).fit_current(
                stats, index, normalized=True, lambda_prox=config.lambda_penalty
            )
        stage = {
            "weights_sha256": cache["weight_hash"],
            "effective_n_quantiles": cache["effective_n"],
            "solver": result.diagnostics,
        }
        if truth is not None:
            stage.update(
                before_edr=metrics(index.T, truth)["projector_distance"],
                after_edr=metrics(result.index.T, truth)["projector_distance"],
            )
        stages.append(stage)
        return result

    started = time.perf_counter()
    try:
        fitted = fit_index(
            x,
            y,
            config=config,
            mode="multi",
            index_dim=m,
            solve=solve,
            profiler=profiler,
            trace_indices=True,
            statistics_function=statistics,
        )
    finally:
        profiler.finish()
    return fitted, stages, time.perf_counter() - started, profiler.model_profile()


def worker(function, m, seed, arm, phase):
    with np.load(OUT / phase / "inputs" / f"{seed}.npz") as saved:
        x, full_basis, noise = saved["x"], saved["basis"], saved["noise"]
    truth = full_basis[:, :m]
    y = response(x, truth, noise, function)
    method_seed = int(np.random.SeedSequence([seed, 1]).generate_state(1)[0]) % (
        2**31 - 1
    )
    fitted, stages, elapsed, profile = fit(x, y, m, method_seed, arm, truth)
    initial, basis = fitted.initial_index.T, fitted.index.T
    return {
        "function": function,
        "m": m,
        "seed": seed,
        "arm": arm,
        "phase": phase,
        "status": "ok",
        "data_sha256": digest(x, y, truth),
        "initial_sha256": digest(initial),
        "initial_edr": metrics(initial, truth)["projector_distance"],
        "edr": metrics(basis, truth)["projector_distance"],
        "orthogonality_error": float(np.linalg.norm(basis.T @ basis - np.eye(m))),
        "basis": basis.tolist(),
        "initial_basis": initial.tolist(),
        "selected_iteration": fitted.metadata["selected_iteration"],
        "stop_reason": fitted.metadata["stop_reason"],
        "stages": stages,
        "trace": [
            {
                "h": r["h"],
                "alpha": r["factor"],
                "eigenvalues": r["eigenvalues"],
                "edr": metrics(r["basis"], truth)["projector_distance"],
            }
            for r in fitted.metadata["trace"]
        ],
        "fit_seconds": elapsed,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "profile": profile,
    }


def pilot():
    x, basis, noise = input_arrays(90211)
    y = response(x, basis, noise, "sine")
    rows = []
    for arm in ARMS:
        _, stages, seconds, _ = fit(x, y, 5, 90211, arm)
        rows.append(
            {
                "arm": arm,
                "fit_seconds": seconds,
                "inner_calls": len(stages),
                "profile_evaluations": sum(
                    s["solver"]["profile_evaluations"] for s in stages
                ),
            }
        )
        print(rows[-1], flush=True)
    (OUT / "pilot.json").write_text(json.dumps(rows, indent=2))


def jobs(phase, arms):
    result = [
        (f, m, seed, arm)
        for f in FUNCTIONS
        for m in DIMS
        for seed in SEEDS[phase]
        for arm in arms
    ]
    np.random.default_rng(20261023).shuffle(result)
    return result


def sources():
    return [
        *sorted((ROOT / "ADP").rglob("*.py")),
        Path(__file__),
        ROOT / "benchmarks/experiments/local_polynomial.py",
        ROOT / "benchmarks/experiments/r_sdr/design.py",
        ROOT / "tests/test_local_polynomial.py",
        OUT / "DERIVATION.md",
        OUT / "PROTOCOL.md",
        OUT / "AUDIT.md",
    ]


def freeze(phase, arms):
    folder = OUT / phase
    folder.mkdir(exist_ok=True)
    protocol = {
        "phase": phase,
        "tasks": jobs(phase, arms),
        "n": 5000,
        "d": 25,
        "noise_sd": 0.3,
        "config": {
            "N_J": 300,
            "N_phi_base": 12,
            "N_phi_identity": 25,
            "N_loc": 20,
            "N_lin": 50,
        },
        "ridge_wls": 1e-4,
        "max_steps_wls": 40,
        "max_steps_base": 20,
        "threads": 1,
        "workers": 3,
        "timeout": 300,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources()
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()
    protocol["fingerprint"] = fingerprint
    manifest = folder / "manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text())["fingerprint"] == fingerprint, (
            "protocol/source drift"
        )
    else:
        manifest.write_text(json.dumps(protocol, indent=2))
        with zipfile.ZipFile(
            folder / "sources.zip", "w", zipfile.ZIP_DEFLATED
        ) as archive:
            for p in sources():
                archive.write(p, str(p.relative_to(ROOT)))
    (folder / "inputs").mkdir(exist_ok=True)
    for seed in SEEDS[phase]:
        x, basis, noise = input_arrays(seed)
        path = folder / "inputs" / f"{seed}.npz"
        if path.exists():
            with np.load(path) as saved:
                assert all(
                    np.array_equal(saved[k], v)
                    for k, v in zip(
                        ("x", "basis", "noise"), (x, basis, noise), strict=True
                    )
                )
        else:
            np.savez_compressed(path, x=x, basis=basis, noise=noise)
    return protocol


def run(phase="development"):
    run_start = time.perf_counter()
    arms = list(ARMS)
    if phase == "validation":
        gates = analyze("development")["gates"]
        arms = ["base"]
        if gates["identity"]["passed"]:
            arms.append("identity")
        if gates["wls2"]["passed"]:
            arms += ["wls1", "wls2"]
        if arms == ["base"]:
            print("Development gates failed: validation remains untouched", flush=True)
            return
    protocol = freeze(phase, arms)
    folder = OUT / phase
    (folder / "fits").mkdir(exist_ok=True)
    env = os.environ.copy()
    for key in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = "1"

    def launch(task):
        function, m, seed, arm = task
        target = folder / "fits" / f"{function}-m{m}-{seed}-{arm}.json"
        if target.exists():
            old = json.loads(target.read_text())
            assert old["fingerprint"] == protocol["fingerprint"]
            return old["status"]
        start = time.perf_counter()
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.moment_quadratic_study",
                    "worker",
                    function,
                    str(m),
                    str(seed),
                    arm,
                    phase,
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            row = (
                json.loads(result.stdout)
                if result.returncode == 0
                else {
                    "function": function,
                    "m": m,
                    "seed": seed,
                    "arm": arm,
                    "phase": phase,
                    "status": "failed",
                    "error": result.stderr[-6000:],
                }
            )
        except subprocess.TimeoutExpired:
            row = {
                "function": function,
                "m": m,
                "seed": seed,
                "arm": arm,
                "phase": phase,
                "status": "timeout",
            }
        row.update(
            fingerprint=protocol["fingerprint"],
            process_seconds=time.perf_counter() - start,
        )
        target.write_text(json.dumps(row, indent=2, allow_nan=False))
        return row["status"]

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(launch, task) for task in protocol["tasks"]]
        for i, future in enumerate(as_completed(futures), 1):
            status = future.result()
            if i % 20 == 0 or status != "ok":
                print(f"{phase} {i}/{len(futures)}: {status}", flush=True)
    (folder / "execution.json").write_text(
        json.dumps({"wall_seconds": time.perf_counter() - run_start}, indent=2)
    )
    analyze(phase)


def contrast(lookup, seeds, arm, baseline):
    case_logs, deltas, worst_sentinel, worst_main = [], [], 0.0, 0.0
    passed_counts = {arm: 0, baseline: 0}
    for function in FUNCTIONS:
        for m in DIMS:
            a, b = (
                [lookup.get((function, m, seed, method)) for seed in seeds]
                for method in (arm, baseline)
            )
            if not all(r and r["status"] == "ok" for r in a + b):
                return {"passed": False, "reason": "missing/failed pairs"}
            delta = np.array([x["edr"] - y["edr"] for x, y in zip(a, b, strict=True)])
            if m == 2:
                worst_sentinel = max(worst_sentinel, float(delta.mean()))
            else:
                worst_main = max(worst_main, float(delta.mean()))
                logs = np.log(np.maximum([x["edr"] for x in a], 1e-12)) - np.log(
                    np.maximum([x["edr"] for x in b], 1e-12)
                )
                case_logs.append(logs)
                deltas.extend(delta.tolist())
                passed_counts[arm] += sum(x["edr"] <= 0.1 for x in a)
                passed_counts[baseline] += sum(x["edr"] <= 0.1 for x in b)
    block = np.mean(case_logs, axis=0)
    rng = np.random.default_rng(20261023)
    boot = np.exp(block[rng.integers(0, len(block), (10000, len(block)))].mean(axis=1))
    ratio = float(np.exp(block.mean()))
    ci = np.quantile(boot, [0.025, 0.975]).tolist()
    passed = (
        ratio <= 0.9
        and ci[1] < 1
        and worst_sentinel <= 0.02
        and worst_main <= 0.05
        and passed_counts[arm] >= passed_counts[baseline]
    )
    return {
        "passed": bool(passed),
        "geometric_edr_ratio": ratio,
        "ratio_ci": ci,
        "mean_main_delta": float(np.mean(deltas)),
        "worst_m2_mean_delta": worst_sentinel,
        "worst_main_mean_delta": worst_main,
        "quality_pass_main": passed_counts,
    }


def analyze(phase="development"):
    folder = OUT / phase
    manifest = json.loads((folder / "manifest.json").read_text())
    assert all(
        hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha
        for name, sha in manifest["sources"].items()
    )
    rows = [json.loads(p.read_text()) for p in sorted((folder / "fits").glob("*.json"))]
    expected = {tuple(t) for t in manifest["tasks"]}
    keys = [(r["function"], r["m"], r["seed"], r["arm"]) for r in rows]
    assert len(keys) == len(set(keys)) == len(expected) and set(keys) == expected
    assert all(r["fingerprint"] == manifest["fingerprint"] for r in rows)
    lookup = dict(zip(keys, rows, strict=True))
    good = [r for r in rows if r["status"] == "ok"]
    for row in good:
        base = lookup[row["function"], row["m"], row["seed"], "base"]
        if base["status"] == "ok":
            assert row["data_sha256"] == base["data_sha256"]
            assert row["initial_sha256"] == base["initial_sha256"]
            assert (
                row["stages"][0]["weights_sha256"]
                == base["stages"][0]["weights_sha256"]
            )
        assert row["orthogonality_error"] < 1e-8
        for stage in row["stages"]:
            history = np.array(stage["solver"]["profile_history"])
            assert np.all(np.diff(history) <= 1e-9 * max(1, history[0]))
    summary = []
    arms = sorted({r["arm"] for r in rows})
    for function in FUNCTIONS:
        for m in DIMS:
            for arm in arms:
                group = [
                    r
                    for r in good
                    if r["function"] == function and r["m"] == m and r["arm"] == arm
                ]
                summary.append(
                    {
                        "function": function,
                        "m": m,
                        "arm": arm,
                        "ok": len(group),
                        "planned": 10,
                        "mean_edr": float(np.mean([r["edr"] for r in group]))
                        if group
                        else None,
                        "quality_pass_count": sum(r["edr"] <= 0.1 for r in group),
                        "mean_initial_edr": float(
                            np.mean([r["initial_edr"] for r in group])
                        )
                        if group
                        else None,
                        "median_fit_seconds": float(
                            np.median([r["fit_seconds"] for r in group])
                        )
                        if group
                        else None,
                        "median_process_seconds": float(
                            np.median([r["process_seconds"] for r in group])
                        )
                        if group
                        else None,
                        "median_rss_mib": float(
                            np.median([r["peak_rss_mib"] for r in group])
                        )
                        if group
                        else None,
                        "inner_calls": sum(len(r["stages"]) for r in group),
                        "inner_certified": sum(
                            s["solver"]["converged"] for r in group for s in r["stages"]
                        ),
                    }
                )
    gates = {}
    if "identity" in arms:
        gates["identity"] = contrast(lookup, SEEDS[phase], "identity", "base")
    if "wls2" in arms:
        vs_linear = contrast(lookup, SEEDS[phase], "wls2", "wls1")
        vs_base = contrast(lookup, SEEDS[phase], "wls2", "base")
        gates["wls2"] = {
            "passed": vs_linear["passed"] and vs_base["passed"],
            "vs_wls1": vs_linear,
            "vs_base": vs_base,
        }
    result = {
        "phase": phase,
        "completed": len(rows),
        "successful": len(good),
        "planned": len(expected),
        "failures": [r for r in rows if r["status"] != "ok"],
        "summary": summary,
        "gates": gates,
    }
    (folder / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    with (folder / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(
        json.dumps(
            {
                "phase": phase,
                "completed": len(rows),
                "successful": len(good),
                "gates": gates,
            },
            indent=2,
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "command", choices=("pilot", "run", "worker", "analyze", "validate")
    )
    parser.add_argument("args", nargs="*")
    args = parser.parse_args()
    if args.command == "worker":
        function, m, seed, arm, phase = args.args
        print(
            json.dumps(worker(function, int(m), int(seed), arm, phase), allow_nan=False)
        )
    elif args.command == "pilot":
        pilot()
    elif args.command == "validate":
        run("validation")
    elif args.command == "analyze":
        analyze(args.args[0] if args.args else "development")
    else:
        run()
