"""Frozen paired complete fits of whitened versus direct full-moment ridge."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import resource
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import scipy

from ADP import ADP_Config
from ADP.engine.common.index_fit import fit_index
from ADP.engine.common.logger import IndexProfiler
from ADP.engine.common.statistic import calculate_statistics
from experiments import coupled_penalties as reference
from experiments import full_ridge_fast as fast
from experiments.mave_grassmann_grid import arrays, case, settings
from experiments.r_sdr.design import metrics
from experiments.r_sdr.runner import run_process

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/results/verification/full-ridge-fast-20261025"
CASES = tuple(itertools.product(("additive", "quadratic", "sine"), (2, 3, 5)))
SEEDS = {
    "selection": tuple(range(973000, 973004)),
    "validation": tuple(range(973100, 973104)),
}
ARMS = ("reference", "direct")


def default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write(path, value):
    path.write_text(
        json.dumps(value, indent=2, default=default, allow_nan=False) + "\n"
    )


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(*values):
    return hashlib.sha256(
        b"".join(np.ascontiguousarray(v).tobytes() for v in values)
    ).hexdigest()


def fit(x, y, truth, seed, arm):
    method_seed = int(np.random.SeedSequence([seed, 1]).generate_state(1)[0]) % (
        2**31 - 1
    )
    config = ADP_Config(
        seed=method_seed,
        **settings(case(x.shape[1], truth.shape[1], len(x), 0.3), "Grassmann")[0],
    )
    context, stages = {}, []
    profiler = IndexProfiler()

    def statistics(X, Y, weights, directions, batch_size=32, *, normalized=False):
        first = not stages
        hasher = hashlib.sha256()

        def observed():
            for start, block in weights:
                if first:
                    hasher.update(np.ascontiguousarray(block).tobytes())
                yield start, block

        if arm == "direct":
            stats = fast.calculate_statistics(
                X, Y, observed(), directions, batch_size, normalized=normalized
            )
        else:
            blocks = list(observed())
            stats = calculate_statistics(
                X, Y, iter(blocks), directions, batch_size, normalized=normalized
            )
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
        if first:
            context.update(
                first_weights_sha256=hasher.hexdigest(),
                first_directions_sha256=digest(directions),
            )
        return stats

    def solve(index, stats):
        result = (
            fast.solve(index, stats)
            if arm == "direct"
            else reference.solve(index, stats, "full", moments=context["moments"])
        )
        stages.append(result.diagnostics)
        return result

    started = time.perf_counter()
    try:
        fitted = fit_index(
            x,
            y,
            config=config,
            mode="multi",
            index_dim=truth.shape[1],
            solve=solve,
            statistics_function=statistics,
            profiler=profiler,
        )
    finally:
        profiler.finish()
    return {
        "basis": fitted.index.T,
        "initial_basis": fitted.initial_index.T,
        "edr": metrics(fitted.index.T, truth)["projector_distance"],
        "fit_seconds": time.perf_counter() - started,
        "worker_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "selected_iteration": fitted.metadata["selected_iteration"],
        "stop_reason": fitted.metadata["stop_reason"],
        "stages": stages,
        "profile": profiler.model_profile(),
        "method_seed": method_seed,
        "initial_sha256": digest(fitted.initial_index),
        "first_weights_sha256": context["first_weights_sha256"],
        "first_directions_sha256": context["first_directions_sha256"],
        "data_sha256": digest(x, y, truth),
        "orthogonality_error": float(
            np.linalg.norm(fitted.index @ fitted.index.T - np.eye(truth.shape[1]))
        ),
    }


def freeze(phase):
    directory = OUT / phase
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite {directory}")
    (directory / "inputs").mkdir(parents=True)
    (directory / "fits").mkdir()
    sources = sorted(
        {
            *ROOT.joinpath("ADP").rglob("*.py"),
            *ROOT.joinpath("experiments").glob("*.py"),
            *ROOT.joinpath("benchmarks/experiments/r_sdr").glob("*.py"),
            ROOT / "tests/test_full_ridge_fast.py",
            OUT / "PROTOCOL.md",
            OUT / "R1_AUDIT.md",
        }
    )
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    with zipfile.ZipFile(
        directory / "sources.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for p in sources:
            archive.write(p, p.relative_to(ROOT))
    inputs = {}
    for link, m in CASES:
        for seed in SEEDS[phase]:
            path = directory / "inputs" / f"{link}-m{m}-{seed}.npz"
            x, y, truth = arrays(case(25, m, 5000, 0.3, link), seed)
            np.savez_compressed(path, x=x, y=y, truth=truth)
            inputs[str(path.relative_to(ROOT))] = sha(path)
    jobs = [
        (link, m, seed, arm)
        for link, m in CASES
        for seed in SEEDS[phase]
        for arm in ARMS
    ]
    np.random.default_rng(973).shuffle(jobs)
    manifest = {
        "phase": phase,
        "jobs": jobs,
        "sources": source_hashes,
        "inputs": inputs,
        "archive_sha256": sha(directory / "sources.zip"),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": sys.version,
        "timeout_seconds": 60,
        "memory_mib": 2048,
        "blas_threads": 1,
        "workers": 1,
        "load_average": os.getloadavg(),
    }
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    write(directory / "manifest.json", manifest)
    return directory, manifest


def analyze(phase):
    directory = OUT / phase
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = [
        json.loads(p.read_text()) for p in sorted((directory / "fits").glob("*.json"))
    ]
    keys = [(r["link"], r["m"], r["seed"], r["arm"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != {
        tuple(job) for job in manifest["jobs"]
    }:
        raise ValueError("incomplete or duplicate fit schedule")
    if any(r["fingerprint"] != manifest["fingerprint"] for r in rows):
        raise ValueError("row fingerprint mismatch")
    lookup = {k: r for k, r in zip(keys, rows, strict=True)}
    pairs = []
    for link, m in CASES:
        for seed in SEEDS[phase]:
            a, b = (lookup[link, m, seed, arm] for arm in ARMS)
            if a["status"] != "ok" or b["status"] != "ok":
                continue
            pairs.append(
                {
                    "link": link,
                    "m": m,
                    "seed": seed,
                    "edr_reference": a["edr"],
                    "edr_direct": b["edr"],
                    "edr_difference": b["edr"] - a["edr"],
                    "fit_ratio": b["fit_seconds"] / a["fit_seconds"],
                    "process_ratio": b["process_seconds"] / a["process_seconds"],
                    "rss_ratio": b["process_peak_rss_mib"] / a["process_peak_rss_mib"],
                    "same_pairing": all(
                        a[k] == b[k]
                        for k in (
                            "data_sha256",
                            "initial_sha256",
                            "first_weights_sha256",
                            "first_directions_sha256",
                        )
                    ),
                    "same_recovery": (a["edr"] <= 0.1) == (b["edr"] <= 0.1),
                    "same_selected_iteration": a["selected_iteration"]
                    == b["selected_iteration"],
                    "same_stage_count": len(a["stages"]) == len(b["stages"]),
                    "no_convergence_loss": len(a["stages"]) == len(b["stages"])
                    and all(
                        old["converged"] == new["converged"]
                        for old, new in zip(a["stages"], b["stages"], strict=True)
                    ),
                    "projector_difference": float(
                        np.linalg.norm(
                            np.array(a["basis"]) @ np.array(a["basis"]).T
                            - np.array(b["basis"]) @ np.array(b["basis"]).T
                        )
                    ),
                    "reference_fit_seconds": a["fit_seconds"],
                    "direct_fit_seconds": b["fit_seconds"],
                    "reference_rss_mib": a["worker_peak_rss_mib"],
                    "direct_rss_mib": b["worker_peak_rss_mib"],
                }
            )
    gates = {
        "complete": len(pairs) == 36 and all(r["status"] == "ok" for r in rows),
        "paired": all(p["same_pairing"] for p in pairs),
        "edr_parity": max(
            (abs(p["edr_difference"]) for p in pairs), default=float("inf")
        )
        <= 1e-4,
        "recovery_parity": all(p["same_recovery"] for p in pairs),
        "convergence": all(p["no_convergence_loss"] for p in pairs),
        "orthogonality": all(r.get("orthogonality_error", 1) <= 1e-10 for r in rows),
        "normal_residual": all(
            s["normal_residual_max"] <= 1e-8
            for r in rows
            if r["status"] == "ok"
            for s in r["stages"]
        ),
        "fit_time": bool(pairs) and np.median([p["fit_ratio"] for p in pairs]) <= 0.90,
        "rss": bool(pairs) and np.median([p["rss_ratio"] for p in pairs]) <= 1.02,
    }
    summary = {
        "phase": phase,
        "planned": 72,
        "recorded": len(rows),
        "pairs": pairs,
        "gates": gates,
        "passed": all(gates.values()),
        "median_fit_ratio": float(np.median([p["fit_ratio"] for p in pairs]))
        if pairs
        else None,
        "median_process_ratio": float(np.median([p["process_ratio"] for p in pairs]))
        if pairs
        else None,
        "median_rss_ratio": float(np.median([p["rss_ratio"] for p in pairs]))
        if pairs
        else None,
    }
    write(directory / "summary.json", summary)
    print(
        json.dumps({k: v for k, v in summary.items() if k != "pairs"}, default=default),
        flush=True,
    )
    return summary


def run(phase):
    if (
        phase == "validation"
        and not json.loads((OUT / "selection/summary.json").read_text())["passed"]
    ):
        raise RuntimeError("selection failed; validation remains untouched")
    directory, manifest = freeze(phase)
    if phase == "validation":
        selected = json.loads((OUT / "selection/manifest.json").read_text())
        if manifest["sources"] != selected["sources"]:
            raise RuntimeError("source changed after selection")
    env = dict(
        os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    started = time.perf_counter()
    for i, (link, m, seed, arm) in enumerate(manifest["jobs"], 1):
        for relative, expected in manifest["sources"].items():
            if sha(ROOT / relative) != expected:
                raise RuntimeError(f"frozen source drift: {relative}")
        input_path = directory / "inputs" / f"{link}-m{m}-{seed}.npz"
        if sha(input_path) != manifest["inputs"][str(input_path.relative_to(ROOT))]:
            raise RuntimeError("frozen input drift")
        name = f"{link}-m{m}-{seed}-{arm}"
        log = directory / "fits" / f"{name}.log"
        result = run_process(
            [
                sys.executable,
                "-m",
                "experiments.full_ridge_fast_study",
                "worker",
                phase,
                link,
                str(m),
                str(seed),
                arm,
            ],
            env,
            log,
            timeout=60,
            memory_mib=2048,
            cwd=ROOT,
        )
        if result["status"] == "ok":
            try:
                row = json.loads(log.read_text().splitlines()[-1])
            except (ValueError, IndexError) as error:
                row = {"status": "invalid_output", "error": repr(error)}
        else:
            row = {"status": result["status"], "error": log.read_text()[-4000:]}
        row.update(
            link=link,
            m=m,
            seed=seed,
            arm=arm,
            fingerprint=manifest["fingerprint"],
            process_seconds=result["process_seconds"],
            process_peak_rss_mib=result["peak_rss_mib"],
        )
        write(directory / "fits" / f"{name}.json", row)
        if i % 12 == 0 or row["status"] != "ok":
            print(
                f"{phase}: {i}/72, {time.perf_counter() - started:.1f}s, {row['status']}",
                flush=True,
            )
    write(
        directory / "execution.json",
        {"wall_seconds": time.perf_counter() - started, "recorded": i},
    )
    return analyze(phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pilot", "run", "worker", "analyze"))
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.action == "worker":
        phase, link, m, seed, arm = args.arguments
        with np.load(OUT / phase / "inputs" / f"{link}-m{m}-{seed}.npz") as saved:
            row = fit(saved["x"], saved["y"], saved["truth"], int(seed), arm)
        print(json.dumps({**row, "status": "ok"}, default=default, allow_nan=False))
    elif args.action == "pilot":
        x, y, truth = arrays(case(25, 5, 5000, 0.3, "sine"), 973900)
        for arm in ARMS:
            row = fit(x, y, truth, 973900, arm)
            # Numerical/timing feasibility only: pilot EDR is not a selection result.
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "fit_seconds": row["fit_seconds"],
                        "stages": len(row["stages"]),
                        "statistics_seconds": row["profile"],
                        "fallbacks": [
                            s["penalty"]["full_moments"].get("spectral_fallbacks")
                            for s in row["stages"]
                        ],
                    },
                    default=default,
                )
            )
    elif args.action == "run":
        run(args.arguments[0])
    else:
        analyze(args.arguments[0])


if __name__ == "__main__":
    main()
