"""Frozen paired study of shared cubic regression; never tunes on selection."""

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

from experiments import full_ridge_batched, shared_polynomial
from experiments.full_ridge_fast_study import digest, sha, write
from experiments.mave_grassmann_grid import arrays, case, settings
from experiments.r_sdr.runner import run_process
from experiments.two_sided_meanmave import TimeOnlyProfiler, method_seed

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/results/verification/shared-polynomial-20261028"
CASES = tuple(itertools.product(("additive", "quadratic", "sine"), (3, 5)))
ARMS = ("projected", "full", "two_sided", "cubic_euclidean", "cubic")
SEEDS = {
    "selection": tuple(range(2811000, 2811010)),
    "validation": tuple(range(2812000, 2812010)),
}


def fit(x, y, m, seed, arm):
    """No truth/link argument is accepted; includes all initialization work."""
    from ADP import ADP_Config
    from ADP.engine.common.index_fit import fit_index
    from ADP.solver.grassman import solve as projected_solve

    started = time.perf_counter()
    method = method_seed(seed)
    if arm in ("two_sided", "cubic", "cubic_euclidean"):
        initial = shared_polynomial.initialize(x, y, m, seed=method)
        if arm == "two_sided":
            basis = initial.index.T
            diagnostics = initial.metadata["diagnostics"]
        else:
            result = shared_polynomial.refine(
                x,
                y,
                initial.index.T,
                geometry="schur" if arm == "cubic" else "euclidean",
            )
            basis, diagnostics = result.basis, result.diagnostics
        refinement_start = initial.index.T
    else:
        config = ADP_Config(
            seed=method, **settings(case(x.shape[1], m, len(x), 0.3), "Grassmann")[0]
        )
        if arm == "full":
            solver = full_ridge_batched.solve
            statistics = full_ridge_batched.calculate_statistics
        elif arm == "projected":

            def solver(index, stats):
                return projected_solve(index, stats.U, stats.I, mass=stats.mass)

            statistics = None
        else:
            raise ValueError(arm)
        initial = fit_index(
            x,
            y,
            config=config,
            mode="multi",
            index_dim=m,
            solve=solver,
            statistics_function=statistics,
            profiler=TimeOnlyProfiler(),
        )
        basis, diagnostics = initial.index.T, initial.metadata["diagnostics"]
        refinement_start = basis
    elapsed = time.perf_counter() - started
    return {
        "basis": basis,
        "initial_basis": initial.initial_index.T,
        "refinement_start": refinement_start,
        "initial_sha256": digest(initial.initial_index.T),
        "refinement_start_sha256": digest(refinement_start),
        "basis_sha256": digest(basis),
        "diagnostics": diagnostics,
        "method_seed": method,
        "fit_seconds": elapsed,
        "selected_adp_iteration": initial.metadata["selected_iteration"],
        "outer_stop_reason": initial.metadata["stop_reason"],
        "adp_trace": initial.metadata["trace"],
        "worker_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
    }


def summarize(rows, jobs):
    expected = {tuple(job) for job in jobs}
    keys = [(r["link"], r["m"], r["seed"], r["arm"]) for r in rows]
    complete = (
        len(keys) == len(expected)
        and len(set(keys)) == len(keys)
        and set(keys) == expected
    )
    integrity = complete and all(
        r["status"] == "ok" and r.get("checked", False) for r in rows
    )
    lookup = dict(zip(keys, rows))
    comparisons = {}
    planned = sum(job[-1] == "cubic" for job in jobs)
    numerical = []
    candidates = [r for r in rows if r["arm"] == "cubic" and r["status"] == "ok"]
    for row in candidates:
        diag = row["diagnostics"]
        numerical.append(
            diag["normal_residual_max"] <= 1e-10
            and diag["gradient_rank_ok"]
            and diag["tangent_rank_ok"]
            and all(
                b <= a + 1e-12 * max(1, abs(a))
                for a, b in itertools.pairwise(diag["history"])
            )
        )
    convergence = sum(r["diagnostics"]["converged"] for r in candidates) / max(
        planned, 1
    )
    global_gates = {
        "complete": complete,
        "integrity": integrity,
        "numerical": len(numerical) == planned and all(numerical),
        "stationarity": convergence >= 0.9,
    }
    for control in ("two_sided", "full", "projected", "cubic_euclidean"):
        pairs = []
        for link, m, seed, arm in jobs:
            if arm != "cubic":
                continue
            new = lookup.get((link, m, seed, "cubic"))
            old = lookup.get((link, m, seed, control))
            two = lookup.get((link, m, seed, "two_sided"))
            if (
                not new
                or not old
                or not two
                or any(r["status"] != "ok" for r in (new, old, two))
            ):
                continue
            paired = (
                new["data_sha256"] == old["data_sha256"] == two["data_sha256"]
                and new["initial_sha256"]
                == old["initial_sha256"]
                == two["initial_sha256"]
                and new["method_seed"] == old["method_seed"] == two["method_seed"]
                and new["refinement_start_sha256"] == two["basis_sha256"]
                and (
                    control != "cubic_euclidean"
                    or old["refinement_start_sha256"] == two["basis_sha256"]
                )
            )
            pairs.append(
                {
                    "link": link,
                    "m": m,
                    "seed": seed,
                    "paired": paired,
                    "control_edr": old["edr"],
                    "candidate_edr": new["edr"],
                    "delta_edr": new["edr"] - old["edr"],
                    **{
                        key: new[field] / old[field]
                        for key, field in (
                            ("fit_ratio", "fit_seconds"),
                            ("process_ratio", "process_seconds"),
                            ("rss_ratio", "peak_rss_mib"),
                        )
                    },
                }
            )
        full = integrity and len(pairs) == planned
        deltas = np.array([p["delta_edr"] for p in pairs])
        clusters = [
            np.mean([p["delta_edr"] for p in pairs if p["seed"] == seed])
            for seed in sorted({p["seed"] for p in pairs})
        ]
        interval, upper = None, None
        if full:
            bootstrap = (
                np.random.default_rng(281028)
                .choice(clusters, (50000, len(clusters)))
                .mean(axis=1)
            )
            interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
            upper = float(np.quantile(bootstrap, 0.975))
        cells = {
            f"{link}-m{m}": float(
                np.mean(
                    [p["delta_edr"] for p in pairs if p["link"] == link and p["m"] == m]
                )
            )
            for link, m in CASES
            if any(p["link"] == link and p["m"] == m for p in pairs)
        }
        recovery = [
            sum(p[key] <= 0.1 for p in pairs)
            for key in ("control_edr", "candidate_edr")
        ]
        mean_delta = float(deltas.mean()) if len(pairs) else None
        gates = {
            "complete": full,
            "pairing": full and all(p["paired"] for p in pairs),
            "mean_edr": full and mean_delta <= -0.03,
            "cluster_upper": full and upper < 0,
            "worst_pair": full and float(deltas.max()) <= 0.05,
            "cell_means": full
            and len(cells) == len(CASES)
            and max(cells.values()) <= 0.01,
            "recovery": full and recovery[1] >= recovery[0] + 10,
        }
        if control == "two_sided":
            gates.update(
                {
                    "fit_median": full
                    and np.median([p["fit_ratio"] for p in pairs]) <= 4,
                    "fit_max": full and max(p["fit_ratio"] for p in pairs) <= 12,
                    "process_median": full
                    and np.median([p["process_ratio"] for p in pairs]) <= 4,
                    "rss_median": full
                    and np.median([p["rss_ratio"] for p in pairs]) <= 1.5,
                }
            )
        comparisons[control] = {
            "pairs_count": len(pairs),
            "mean_delta_edr": mean_delta,
            "cluster_ci95": interval,
            "familywise_upper": upper,
            "cell_mean_deltas": cells,
            "recovery": recovery,
            "median_fit_ratio": float(np.median([p["fit_ratio"] for p in pairs]))
            if pairs
            else None,
            "max_fit_ratio": max((p["fit_ratio"] for p in pairs), default=None),
            "median_process_ratio": float(
                np.median([p["process_ratio"] for p in pairs])
            )
            if pairs
            else None,
            "median_rss_ratio": float(np.median([p["rss_ratio"] for p in pairs]))
            if pairs
            else None,
            "worst_pair": max(pairs, key=lambda p: p["delta_edr"]) if pairs else None,
            "gates": gates,
            "pairs": pairs,
        }
    global_gates["all_arms_paired"] = all(
        comparisons[arm]["gates"]["pairing"] for arm in comparisons
    )
    passed = all(global_gates.values()) and all(
        all(comparisons[arm]["gates"].values()) for arm in ("two_sided", "full")
    )
    cells = {}
    for link, m in CASES:
        cells[f"{link}-m{m}"] = {}
        for arm in ARMS:
            subset = [
                r
                for r in rows
                if (r["link"], r["m"], r["arm"]) == (link, m, arm)
                and r["status"] == "ok"
            ]
            if subset:
                cells[f"{link}-m{m}"][arm] = {
                    "count": len(subset),
                    "mean_edr": float(np.mean([r["edr"] for r in subset])),
                    "recovery": sum(r["edr"] <= 0.1 for r in subset),
                    "median_fit_seconds": float(
                        np.median([r["fit_seconds"] for r in subset])
                    ),
                }
    return {
        "passed": bool(passed),
        "outcomes": len(rows),
        "planned": len(jobs),
        "global_gates": global_gates,
        "candidate_stationarity": convergence,
        "comparisons": comparisons,
        "cells": cells,
    }


def verify(manifest, *, inputs=False):
    copy = {key: value for key, value in manifest.items() if key != "fingerprint"}
    if (
        hashlib.sha256(json.dumps(copy, sort_keys=True).encode()).hexdigest()
        != manifest["fingerprint"]
    ):
        raise RuntimeError("manifest fingerprint mismatch")
    for name, expected in manifest["sources"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f"frozen source drift: {name}")
    directory = OUT / manifest["phase"]
    if sha(directory / "sources.zip") != manifest["archive_sha256"]:
        raise RuntimeError("archive drift")
    if inputs:
        for name, expected in manifest["inputs"].items():
            if sha(directory / name) != expected:
                raise RuntimeError(f"input drift: {name}")


def freeze(phase):
    directory = OUT / phase
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite {directory}")
    if phase == "validation":
        selected = json.loads((OUT / "selection/summary.json").read_text())
        audited = json.loads((OUT / "selection/independent_audit.json").read_text())
        if (
            not selected["passed"]
            or not audited["integrity_ok"]
            or not audited["gate_agreement"]
        ):
            raise RuntimeError("selection and independent audit must pass")
    sources = sorted(
        {
            *ROOT.joinpath("ADP").rglob("*.py"),
            *ROOT.joinpath("experiments").glob("*.py"),
            *ROOT.joinpath("benchmarks/experiments/r_sdr").glob("*.py"),
            *ROOT.joinpath("tests").glob("test_shared_polynomial*.py"),
            *(OUT / name for name in ("DESIGN.md", "PROTOCOL.md", "R1_AUDIT.md")),
        }
    )
    source_map = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    if phase == "validation":
        previous = json.loads((OUT / "selection/manifest.json").read_text())
        verify(previous, inputs=True)
        if source_map != previous["sources"]:
            raise RuntimeError("selection/validation source maps differ")
    (directory / "inputs").mkdir(parents=True)
    for name in ("fits", "raw", "logs"):
        (directory / name).mkdir()
    with zipfile.ZipFile(
        directory / "sources.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sources:
            archive.write(path, path.relative_to(ROOT))
    inputs = {}
    for link, m in CASES:
        for seed in SEEDS[phase]:
            x, y, truth = arrays(case(25, m, 5000, 0.3, link), seed)
            path = directory / "inputs" / f"{link}-m{m}-{seed}.npz"
            np.savez_compressed(path, x=x, y=y, truth=truth)
            inputs[str(path.relative_to(directory))] = sha(path)
    jobs = [
        [link, m, seed, arm]
        for link, m in CASES
        for seed in SEEDS[phase]
        for arm in ARMS
    ]
    np.random.default_rng(281028).shuffle(jobs)
    manifest = {
        "phase": phase,
        "sources": source_map,
        "inputs": inputs,
        "jobs": jobs,
        "archive_sha256": sha(directory / "sources.zip"),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "timeout_seconds": 60,
        "memory_mib": 2048,
        "workers": 1,
        "blas_threads": 1,
        "settings": {
            str(m): settings(case(25, m, 5000, 0.3), "Grassmann")[0] for m in (3, 5)
        },
        "candidate": {
            "degree": 3,
            "max_steps": 60,
            "tol": 1e-7,
            "damping_relative": 1e-4,
        },
    }
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    write(directory / "manifest.json", manifest)
    return manifest


def worker(phase, link, m, seed, arm):
    directory = OUT / phase
    manifest = json.loads((directory / "manifest.json").read_text())
    verify(manifest)
    name = f"{link}-m{m}-{seed}"
    path = directory / "inputs" / f"{name}.npz"
    if sha(path) != manifest["inputs"][f"inputs/{name}.npz"]:
        raise RuntimeError("worker input hash mismatch")
    with np.load(path) as saved:
        x, y = saved["x"], saved["y"]
        row = fit(x, y, m, seed, arm)
        truth = saved["truth"]
    basis = row["basis"]
    row.update(
        {
            "link": link,
            "m": m,
            "seed": seed,
            "arm": arm,
            "status": "ok",
            "data_sha256": digest(x, y),
            "fingerprint": manifest["fingerprint"],
            "input_sha256": sha(path),
            "edr": float(
                np.linalg.norm(basis @ basis.T - truth @ truth.T) / np.sqrt(2 * m)
            ),
            "max_sine": float(
                np.linalg.norm((np.eye(len(basis)) - basis @ basis.T) @ truth, 2)
            ),
        }
    )
    write(directory / "raw" / f"{name}-{arm}.json", row)


def analyze(phase):
    directory = OUT / phase
    manifest = json.loads((directory / "manifest.json").read_text())
    verify(manifest, inputs=True)
    rows = [
        json.loads(path.read_text())
        for path in sorted((directory / "fits").glob("*.json"))
    ]
    for row in rows:
        row["checked"] = False
        if row["status"] != "ok":
            continue
        with np.load(
            directory / "inputs" / f"{row['link']}-m{row['m']}-{row['seed']}.npz"
        ) as saved:
            x, y, truth = saved["x"], saved["y"], saved["truth"]
        b, initial, start = (
            np.asarray(row[key])
            for key in ("basis", "initial_basis", "refinement_start")
        )
        finite = all(
            np.isfinite(v).all()
            and v.shape == truth.shape
            and np.linalg.norm(v.T @ v - np.eye(row["m"])) <= 1e-9
            for v in (b, initial, start)
        )
        actual = np.linalg.norm(b @ b.T - truth @ truth.T) / np.sqrt(2 * row["m"])
        max_sine = np.linalg.norm((np.eye(len(b)) - b @ b.T) @ truth, 2)
        row["checked"] = bool(
            finite
            and abs(actual - row["edr"]) <= 1e-10
            and abs(max_sine - row["max_sine"]) <= 1e-10
            and row["data_sha256"] == digest(x, y)
            and row["basis_sha256"] == digest(b)
            and row["initial_sha256"] == digest(initial)
            and row["refinement_start_sha256"] == digest(start)
            and row["fingerprint"] == manifest["fingerprint"]
            and all(
                np.isfinite(row[key]) and row[key] > 0
                for key in ("fit_seconds", "process_seconds", "peak_rss_mib")
            )
        )
    summary = summarize(rows, manifest["jobs"])
    write(directory / "summary.json", summary)
    return summary


def run(phase):
    manifest = freeze(phase)
    directory = OUT / phase
    environment = os.environ.copy()
    environment.update(
        {
            key: "1"
            for key in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        }
    )
    environment["PYTHONPATH"] = str(ROOT)
    environment.pop("PYTHONTRACEMALLOC", None)
    started = time.perf_counter()
    for index, (link, m, seed, arm) in enumerate(manifest["jobs"], 1):
        name = f"{link}-m{m}-{seed}-{arm}"
        outcome = run_process(
            [
                sys.executable,
                "-m",
                "experiments.shared_polynomial_study",
                "worker",
                phase,
                "--link",
                link,
                "--m",
                str(m),
                "--seed",
                str(seed),
                "--arm",
                arm,
            ],
            environment,
            directory / "logs" / f"{name}.log",
            timeout=60,
            memory_mib=2048,
            cwd=ROOT,
        )
        path = directory / "raw" / f"{name}.json"
        row = (
            json.loads(path.read_text())
            if outcome["status"] == "ok" and path.exists()
            else {
                "link": link,
                "m": m,
                "seed": seed,
                "arm": arm,
                "fingerprint": manifest["fingerprint"],
            }
        )
        row.update(outcome)
        if outcome["status"] == "ok" and not path.exists():
            row["status"] = "missing_worker_output"
        if row["status"] == "ok":
            row["peak_rss_mib"] = max(
                row["peak_rss_mib"] or 0, row["worker_peak_rss_mib"]
            )
        write(directory / "fits" / f"{name}.json", row)
        if index % 10 == 0 or row["status"] != "ok":
            print(
                phase,
                index,
                "/",
                len(manifest["jobs"]),
                "last",
                name,
                row["status"],
                flush=True,
            )
    write(
        directory / "execution.json",
        {"process_seconds": time.perf_counter() - started, "attempts": index},
    )
    summary = analyze(phase)
    print(
        json.dumps(
            {
                "phase": phase,
                "passed": summary["passed"],
                "outcomes": summary["outcomes"],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "worker", "analyze"))
    parser.add_argument("phase", choices=tuple(SEEDS))
    parser.add_argument("--link", choices=("additive", "quadratic", "sine"))
    parser.add_argument("--m", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.command == "worker":
        worker(args.phase, args.link, args.m, args.seed, args.arm)
    elif args.command == "run":
        run(args.phase)
    else:
        summary = analyze(args.phase)
        print(
            json.dumps(
                {"passed": summary["passed"], "global_gates": summary["global_gates"]}
            )
        )


if __name__ == "__main__":
    main()
