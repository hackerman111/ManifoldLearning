"""Frozen paired campaign for the three opt-in ADP integrations."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import os
import platform
import resource
import subprocess
import sys
import time
import zipfile
from dataclasses import fields
from pathlib import Path

import numpy as np
import scipy

from ADP import ADP_Config
from experiments.adp_moment_integration import ARMS, digest, fit
from experiments.weak_moment_study import arrays, metrics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/results/verification/adp-moment-integration-20261030"
CASES = ("quad_strong", "quad_weak", "cubic", "quartic", "correlated")
CONTRASTS = {
    "start": ("residual_start", "adp"),
    "proposal": ("proposal", "budget"),
    "quadratic": ("wls2", "wls1"),
}
SEEDS = {
    "selection": list(range(3020100, 3020108)),
    "validation": list(range(3021100, 3021108)),
}


def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, default=encode, indent=2, allow_nan=False) + "\n")


def fingerprints():
    paths = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "ADP").rglob("*.py"))
    paths += [
        "benchmarks/experiments/adp_moment_integration.py",
        "benchmarks/experiments/integration_study.py",
        "benchmarks/experiments/local_polynomial.py",
        "benchmarks/experiments/weak_moments.py",
        "benchmarks/experiments/weak_moment_study.py",
        "tests/test_adp_moment_integration.py",
    ]
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def worker(case, seed, arm):
    x, y, truth = arrays(case, seed)
    row = {"case": case, "seed": seed, "arm": arm, "data_hash": digest(x, y, truth)}
    start = time.perf_counter()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = fit(x, y, seed, arm)
        row.update(status="ok", **result, **metrics(result["basis"], truth))
        if not np.isfinite(row["edr"]) or row["orthogonality"] > 1e-8:
            raise RuntimeError("invalid returned projector")
    except Exception as error:
        row.update(status="error", error=f"{type(error).__name__}: {error}")
    row["fit_seconds"] = time.perf_counter() - start
    row["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if row["peak_rss_mib"] > 1024:
        row.update(status="error", error="RSS exceeded 1GiB")
    return row


def summarize(rows, contrasts=CONTRASTS):
    groups = {}
    for case in CASES:
        groups[case] = {}
        for arm in sorted({r["arm"] for r in rows}):
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
                "median_process_seconds": float(
                    np.median([r["process_seconds"] for r in subset])
                ),
                "median_rss_mib": float(np.median([r["peak_rss_mib"] for r in subset])),
                "all_inner_converged": sum(r["diagnostics"]["converged"] for r in ok),
            }
    pairs = {(r["case"], r["seed"], r["arm"]): r for r in rows}
    gates = {}
    seeds = sorted({r["seed"] for r in rows})
    for name, (a, b) in contrasts.items():
        candidate = [pairs[c, s, a] for s in seeds for c in CASES[:2]]
        baseline = [pairs[c, s, b] for s in seeds for c in CASES[:2]]
        all_relevant = [r for r in rows if r["arm"] in (a, b)]
        if any(r["status"] != "ok" for r in all_relevant):
            gates[name] = {"pass": False, "reason": "failed fit"}
            continue
        logs = (
            np.array(
                [
                    np.log(max(r["edr"], 1e-15) / max(t["edr"], 1e-15))
                    for r, t in zip(candidate, baseline, strict=True)
                ]
            )
            .reshape(len(seeds), 2)
            .mean(axis=1)
        )
        rng = np.random.default_rng(3020999)
        samples = np.exp(rng.choice(logs, (10000, len(seeds))).mean(axis=1))
        interval = np.quantile(samples, [0.025, 0.975])
        timing = float(
            np.median(
                [
                    r["fit_seconds"] / t["fit_seconds"]
                    for r, t in zip(candidate, baseline, strict=True)
                ]
            )
        )
        rss = float(
            np.median(
                [
                    r["peak_rss_mib"] / t["peak_rss_mib"]
                    for r, t in zip(candidate, baseline, strict=True)
                ]
            )
        )
        worst = max(
            r["edr"] - t["edr"] for r, t in zip(candidate, baseline, strict=True)
        )
        gates[name] = {
            "edr_ratio": float(np.exp(logs.mean())),
            "seed_cluster_percentile95": interval.tolist(),
            "time_ratio": timing,
            "rss_ratio": rss,
            "worst_pair_increase": worst,
            "pass": bool(
                interval[1] < 1 and timing <= 3 and rss <= 1.5 and worst <= 0.25
            ),
        }
    return {"rows": len(rows), "groups": groups, "gates": gates}


def prepare():
    path = OUT / "PROTOCOL.json"
    if path.exists():
        raise RuntimeError("protocol already frozen")
    config = ADP_Config(
        seed=0, N_lin=50, N_J=100, N_loc=20, N_phi=12, outer_steps=8, select_step="best"
    )
    configuration = {}
    for field in fields(config):
        value = getattr(config, field.name)
        configuration[field.name] = (
            inspect.getsource(value) if callable(value) else value
        )
    sources = fingerprints()
    with zipfile.ZipFile(OUT / "SOURCE_SNAPSHOT.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sources:
            z.write(ROOT / p, p)
    protocol = {
        "seeds": SEEDS,
        "cases": CASES,
        "arms": ARMS,
        "contrasts": CONTRASTS,
        "configuration": configuration,
        "sources": sources,
        "data_hashes": {
            phase: {f"{c}/{s}": digest(*arrays(c, s)) for s in seeds for c in CASES}
            for phase, seeds in SEEDS.items()
        },
        "gate": (
            "all fits valid; quadratic-case seed-cluster ratio upper95<1, "
            "time<=3,RSS<=1.5,worst increase<=.25"
        ),
        "bootstrap": "10000 seed clusters RNG3020999; percentile diagnostic",
        "kernel": (
            "max(1-u^2,0), u squared tensor distance; "
            "original bandwidth/anisotropy/centers"
        ),
        "dtype": "float64",
        "block_size": 256,
        "threads": {
            k: os.getenv(k)
            for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ),
        "notes": (
            "fit time includes moments, excludes data generation/import/startup; "
            "process time includes them; RSS full process; "
            "fresh sequential subprocesses; no truth in fit"
        ),
    }
    write_json(path, protocol)
    write_json(OUT / "NUMPY_CONFIG.json", np.show_config(mode="dicts"))


def phase_run(phase):
    protocol = json.loads((OUT / "PROTOCOL.json").read_text())
    if protocol["sources"] != fingerprints():
        raise RuntimeError("source fingerprint changed")
    arms, contrasts = list(ARMS), CONTRASTS
    if phase == "validation":
        selected = json.loads((OUT / "selection_summary.json").read_text())["gates"]
        contrasts = {k: v for k, v in CONTRASTS.items() if selected[k]["pass"]}
        arms = sorted({a for pair in contrasts.values() for a in pair})
        if not arms:
            raise RuntimeError("no passing selection contrast; validation forbidden")
    destination = OUT / f"{phase}.jsonl"
    if destination.exists():
        raise RuntimeError("refuse to overwrite historical results")
    rows = []
    start = time.perf_counter()
    with destination.open("x") as output:
        for seed in SEEDS[phase]:
            for case in CASES:
                # Deterministic rotation avoids always giving one arm a warm host.
                rotation = (seed + CASES.index(case)) % len(arms)
                for arm in arms[rotation:] + arms[:rotation]:
                    if time.perf_counter() - start > 900:
                        raise RuntimeError(
                            "phase exceeded 15min; incomplete results retained"
                        )
                    tick = time.perf_counter()
                    command = [
                        sys.executable,
                        "-m",
                        "experiments.integration_study",
                        "worker",
                        "--case",
                        case,
                        "--seed",
                        str(seed),
                        "--arm",
                        arm,
                    ]
                    env = {
                        **os.environ,
                        "OPENBLAS_NUM_THREADS": "1",
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                    }
                    try:
                        result = subprocess.run(
                            command,
                            cwd=ROOT,
                            env=env,
                            text=True,
                            capture_output=True,
                            timeout=120,
                            check=True,
                        )
                        row = json.loads(result.stdout)
                    except (subprocess.SubprocessError, ValueError) as error:
                        row = {
                            "case": case,
                            "seed": seed,
                            "arm": arm,
                            "status": "error",
                            "error": str(error),
                            "fit_seconds": time.perf_counter() - tick,
                            "peak_rss_mib": 0,
                        }
                    row["process_seconds"] = time.perf_counter() - tick
                    if (
                        row["status"] == "ok"
                        and row["data_hash"]
                        != protocol["data_hashes"][phase][f"{case}/{seed}"]
                    ):
                        raise RuntimeError("data fingerprint mismatch")
                    output.write(
                        json.dumps(row, default=encode, allow_nan=False) + "\n"
                    )
                    output.flush()
                    rows.append(row)
            print(f"{phase} seed {seed} complete: {len(rows)} fits", flush=True)
    summary = summarize(rows, contrasts)
    write_json(OUT / f"{phase}_summary.json", summary)
    print(json.dumps(summary["gates"], indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "command", choices=("prepare", "selection", "validation", "worker")
    )
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.command == "worker":
        print(
            json.dumps(
                worker(args.case, args.seed, args.arm), default=encode, allow_nan=False
            )
        )
    elif args.command == "prepare":
        prepare()
    else:
        phase_run(args.command)


if __name__ == "__main__":
    main()
