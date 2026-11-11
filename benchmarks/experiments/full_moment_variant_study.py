"""Four frozen paired experiments; run each estimator gate independently."""

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
from experiments import full_moment_variants as candidate
from experiments import full_ridge_batched as ridge
from experiments.full_ridge_fast_study import default, digest, sha, write
from experiments.mave_grassmann_grid import arrays, case, settings
from experiments.r_sdr.runner import run_process

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/results/verification/full-moment-variants-20261026"
COMPARISONS = {
    "unregularized": ("ridge", "full0"),
    "correlated": ("full0", "metric"),
    "smoothness": ("metric", "coherence"),
    "optimizer": ("full0", "tangent"),
}
CASES = tuple(itertools.product(("additive", "quadratic", "sine"), (3, 5)))
BASE_SEEDS = {name: 986000 + 1000 * i for i, name in enumerate(COMPARISONS)}
PHASE_OFFSET = {"selection": 0, "validation": 100}


def edr(basis, truth):
    return float(
        np.linalg.norm(basis @ basis.T - truth @ truth.T) / np.sqrt(2 * truth.shape[1])
    )


def fit(x, y, truth, seed, arm, kind="fit"):
    method_seed = int(np.random.SeedSequence([seed, 1]).generate_state(1)[0]) % (
        2**31 - 1
    )
    config = ADP_Config(
        seed=method_seed,
        **settings(case(x.shape[1], truth.shape[1], len(x), 0.3), "Grassmann")[0],
    )
    context, stages = {}, []
    profiler = IndexProfiler()

    class OracleDone(Exception):
        pass

    def statistics(X, Y, weights, directions, batch_size=32, *, normalized=False):
        hasher = hashlib.sha256()

        def observed():
            for start, block in weights:
                if not stages:
                    hasher.update(np.ascontiguousarray(block).tobytes())
                yield start, block

        stats = candidate.calculate_statistics(
            X, Y, observed(), directions, batch_size, normalized=normalized
        )
        if not stages:
            context.update(
                first_weights_sha256=hasher.hexdigest(),
                first_directions_sha256=digest(directions),
            )
        return stats

    def solve(index, stats):
        if not stages:
            context["initial_basis"] = index.T.copy()
        incoming = truth.T if kind == "oracle" else index
        result = (
            ridge.solve(incoming, stats)
            if arm == "ridge"
            else candidate.solve(incoming, stats, arm)
        )
        stages.append(result.diagnostics)
        if kind == "oracle":
            context["result"] = result
            raise OracleDone
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
        basis = fitted.index.T
        selected_iteration, stop_reason = (
            fitted.metadata["selected_iteration"],
            fitted.metadata["stop_reason"],
        )
    except OracleDone:
        basis = context["result"].index.T
        selected_iteration, stop_reason = None, "single_solve_from_truth"
    finally:
        profiler.finish()
    return {
        "basis": basis,
        "initial_basis": context["initial_basis"],
        "edr": edr(basis, truth),
        "fit_seconds": time.perf_counter() - started,
        "worker_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "selected_iteration": selected_iteration,
        "stop_reason": stop_reason,
        "stages": stages,
        "profile": profiler.model_profile(),
        "method_seed": method_seed,
        "initial_sha256": digest(context["initial_basis"]),
        **{k: v for k, v in context.items() if k.endswith("sha256")},
        "data_sha256": digest(x, y, truth),
        "orthogonality_error": float(
            np.linalg.norm(basis.T @ basis - np.eye(truth.shape[1]))
        ),
    }


def sources():
    return sorted(
        {
            *ROOT.joinpath("ADP").rglob("*.py"),
            *ROOT.joinpath("experiments").glob("*.py"),
            *ROOT.joinpath("benchmarks/experiments/r_sdr").glob("*.py"),
            ROOT / "tests/test_full_moment_variants.py",
            OUT / "PROTOCOL.md",
            OUT / "R1_AUDIT.md",
        }
    )


def freeze(comparison, phase):
    directory = OUT / comparison / phase
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite {directory}")
    if phase == "validation":
        selection = directory.parent / "selection"
        if not json.loads((selection / "summary.json").read_text())["passed"]:
            raise RuntimeError("selection gate failed; validation remains untouched")
        if not json.loads((selection / "audit.json").read_text())["integrity_ok"]:
            raise RuntimeError("selection integrity audit must pass first")
    (directory / "inputs").mkdir(parents=True)
    (directory / "fits").mkdir()
    paths = sources()
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    if phase == "validation":
        selected = json.loads(
            (directory.parent / "selection/manifest.json").read_text()
        )
        if selected["sources"] != hashes:
            raise RuntimeError("source changed after selection")
    with zipfile.ZipFile(
        directory / "sources.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for p in paths:
            archive.write(p, p.relative_to(ROOT))
    seeds = tuple(
        range(
            BASE_SEEDS[comparison] + PHASE_OFFSET[phase],
            BASE_SEEDS[comparison] + PHASE_OFFSET[phase] + 3,
        )
    )
    inputs = {}
    for link, m in CASES:
        for seed in seeds:
            path = directory / "inputs" / f"{link}-m{m}-{seed}.npz"
            x, y, truth = arrays(case(25, m, 5000, 0.3, link), seed)
            np.savez_compressed(path, x=x, y=y, truth=truth)
            inputs[str(path.relative_to(ROOT))] = sha(path)
    # Full fits and oracle-start diagnostics use separate cold worker processes.
    jobs = [
        (kind, link, m, seed, arm)
        for kind in ("fit", "oracle")
        for link, m in CASES
        for seed in seeds
        for arm in COMPARISONS[comparison]
    ]
    np.random.default_rng(BASE_SEEDS[comparison]).shuffle(jobs)
    manifest = {
        "comparison": comparison,
        "phase": phase,
        "jobs": jobs,
        "seeds": seeds,
        "sources": hashes,
        "inputs": inputs,
        "archive_sha256": sha(directory / "sources.zip"),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": sys.version,
        "timeout_seconds": 90,
        "memory_mib": 2048,
        "blas_threads": 1,
        "workers": 1,
        "load_average": os.getloadavg(),
        "quality_gate": {
            "mean_edr_delta_max": 0.005 if comparison == "optimizer" else -0.005,
            "cell_mean_edr_delta_max": 0.02,
            "pair_edr_delta_max": 0.03 if comparison == "optimizer" else 0.08,
            "recovery_threshold": 0.1,
            "convergence_rate_loss_max": 0.05,
        },
        "median_fit_ratio_max": {
            "unregularized": 3.0,
            "correlated": 2.0,
            "smoothness": 3.0,
            "optimizer": 0.9,
        }[comparison],
        "median_rss_ratio_max": 1.5,
    }
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    write(directory / "manifest.json", manifest)
    return directory, manifest


def verify(directory, manifest):
    for relative, expected in manifest["sources"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f"frozen source drift: {relative}")
    if sha(directory / "sources.zip") != manifest["archive_sha256"]:
        raise RuntimeError("frozen archive drift")


def analyze(comparison, phase):
    directory = OUT / comparison / phase
    manifest = json.loads((directory / "manifest.json").read_text())
    verify(directory, manifest)
    rows = [
        json.loads(p.read_text()) for p in sorted((directory / "fits").glob("*.json"))
    ]
    keys = [(r["kind"], r["link"], r["m"], r["seed"], r["arm"]) for r in rows]
    expected = {tuple(job) for job in manifest["jobs"]}
    if len(keys) != len(set(keys)) or not set(keys) <= expected:
        raise RuntimeError("duplicate or foreign rows")
    if any(r["fingerprint"] != manifest["fingerprint"] for r in rows):
        raise RuntimeError("row fingerprint mismatch")
    lookup = dict(zip(keys, rows, strict=True))
    pairs, oracle = [], []
    for kind in ("fit", "oracle"):
        for link, m in CASES:
            for seed in manifest["seeds"]:
                a, b = (
                    lookup.get((kind, link, m, seed, arm), {})
                    for arm in COMPARISONS[comparison]
                )
                if a.get("status") != "ok" or b.get("status") != "ok":
                    continue
                pair = {
                    "link": link,
                    "m": m,
                    "seed": seed,
                    "reference_edr": a["edr"],
                    "candidate_edr": b["edr"],
                    "edr_delta": b["edr"] - a["edr"],
                    "fit_ratio": b["fit_seconds"] / a["fit_seconds"],
                    "process_ratio": b["process_seconds"] / a["process_seconds"],
                    "rss_ratio": b["worker_peak_rss_mib"] / a["worker_peak_rss_mib"],
                    "paired": all(
                        a[k] == b[k]
                        for k in (
                            "data_sha256",
                            "initial_sha256",
                            "first_weights_sha256",
                            "first_directions_sha256",
                        )
                    ),
                    "recovery_gain": int(b["edr"] <= 0.1) - int(a["edr"] <= 0.1),
                    "reference_converged_rate": float(
                        np.mean([s["converged"] for s in a["stages"]])
                    ),
                    "candidate_converged_rate": float(
                        np.mean([s["converged"] for s in b["stages"]])
                    ),
                }
                (pairs if kind == "fit" else oracle).append(pair)
    q = manifest["quality_gate"]
    gates = {
        "complete": set(keys) == expected
        and len(pairs) == len(oracle) == 18
        and all(r["status"] == "ok" for r in rows),
        "paired": len(pairs) == 18 and all(p["paired"] for p in pairs + oracle),
        "mean_edr": bool(pairs)
        and float(np.mean([p["edr_delta"] for p in pairs])) <= q["mean_edr_delta_max"],
        "cell_mean_edr": bool(pairs)
        and all(
            float(np.mean([p["edr_delta"] for p in pairs if (p["link"], p["m"]) == c]))
            <= q["cell_mean_edr_delta_max"]
            for c in CASES
        ),
        "worst_pair_edr": bool(pairs)
        and max(p["edr_delta"] for p in pairs) <= q["pair_edr_delta_max"],
        "recovery": bool(pairs) and sum(p["recovery_gain"] for p in pairs) >= 0,
        "convergence": bool(pairs)
        and float(
            np.mean(
                [
                    p["candidate_converged_rate"] - p["reference_converged_rate"]
                    for p in pairs
                ]
            )
        )
        >= -q["convergence_rate_loss_max"],
        "orthogonality": all(r.get("orthogonality_error", 1) <= 1e-10 for r in rows),
        "coefficient_certificate": all(
            s["normal_residual_max"] <= 2e-7
            for r in rows
            if r["status"] == "ok"
            for s in r["stages"]
        ),
        "fit_time": bool(pairs)
        and float(np.median([p["fit_ratio"] for p in pairs]))
        <= manifest["median_fit_ratio_max"],
        "rss": bool(pairs)
        and float(np.median([p["rss_ratio"] for p in pairs]))
        <= manifest["median_rss_ratio_max"],
    }
    cells = []
    for link, m in CASES:
        cell = [p for p in pairs if (p["link"], p["m"]) == (link, m)]
        cells.append(
            {
                "link": link,
                "m": m,
                "complete_pairs": len(cell),
                **(
                    {
                        "mean_reference_edr": float(
                            np.mean([p["reference_edr"] for p in cell])
                        ),
                        "mean_candidate_edr": float(
                            np.mean([p["candidate_edr"] for p in cell])
                        ),
                        "mean_edr_delta": float(
                            np.mean([p["edr_delta"] for p in cell])
                        ),
                        "median_fit_ratio": float(
                            np.median([p["fit_ratio"] for p in cell])
                        ),
                    }
                    if cell
                    else {}
                ),
            }
        )
    summary = {
        "comparison": comparison,
        "phase": phase,
        "planned": len(expected),
        "recorded": len(rows),
        "failed_rows": [
            {
                k: r.get(k)
                for k in ("kind", "link", "m", "seed", "arm", "status", "error")
            }
            for r in rows
            if r["status"] != "ok"
        ],
        "gates": gates,
        "passed": all(gates.values()),
        "cells": cells,
        "pairs": pairs,
        "oracle_pairs": oracle,
        "mean_edr_delta": float(np.mean([p["edr_delta"] for p in pairs]))
        if pairs
        else None,
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
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ("pairs", "oracle_pairs", "cells", "failed_rows")
            }
        ),
        flush=True,
    )
    return summary


def audit(comparison, phase):
    """Recompute geometry, identities and hashes from raw worker files."""
    directory = OUT / comparison / phase
    manifest = json.loads((directory / "manifest.json").read_text())
    verify(directory, manifest)
    checked, max_error = 0, 0.0
    for relative, expected in manifest["inputs"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError("input hash mismatch")
    for job in manifest["jobs"]:
        kind, link, m, seed, arm = job
        path = directory / "fits" / f"{kind}-{link}-m{m}-{seed}-{arm}.json"
        if not path.exists():
            raise RuntimeError("missing worker row")
        row = json.loads(path.read_text())
        if (
            row["fingerprint"] != manifest["fingerprint"]
            or [row[k] for k in ("kind", "link", "m", "seed", "arm")] != job
        ):
            raise RuntimeError("worker identity mismatch")
        if row["status"] != "ok":
            continue
        with np.load(directory / "inputs" / f"{link}-m{m}-{seed}.npz") as saved:
            if row["data_sha256"] != digest(saved["x"], saved["y"], saved["truth"]):
                raise RuntimeError("data pairing mismatch")
            b = np.array(row["basis"])
            truth = saved["truth"]
            # Independent equivalent principal-angle form, not runner's projector distance.
            independently = np.sqrt(max(0, 1 - np.linalg.norm(b.T @ truth) ** 2 / m))
            error = abs(independently - row["edr"])
            max_error = max(max_error, error)
            if error > 1e-9 or np.linalg.norm(b.T @ b - np.eye(m)) > 1e-10:
                raise RuntimeError("EDR or orthogonality audit mismatch")
        if digest(np.array(row["initial_basis"])) != row["initial_sha256"]:
            raise RuntimeError("initializer hash mismatch")
        checked += 1
    result = {
        "integrity_ok": True,
        "successful_rows_recomputed": checked,
        "max_edr_difference": max_error,
        "scope": "separate deterministic recomputation, not an independent researcher review",
    }
    write(directory / "audit.json", result)
    print(json.dumps(result), flush=True)
    return result


def run(comparison, phase):
    directory, manifest = freeze(comparison, phase)
    env = dict(
        os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    started = time.perf_counter()
    for i, job in enumerate(manifest["jobs"], 1):
        verify(directory, manifest)
        kind, link, m, seed, arm = job
        input_path = directory / "inputs" / f"{link}-m{m}-{seed}.npz"
        if sha(input_path) != manifest["inputs"][str(input_path.relative_to(ROOT))]:
            raise RuntimeError("input drift")
        name = f"{kind}-{link}-m{m}-{seed}-{arm}"
        log = directory / "fits" / f"{name}.log"
        result = run_process(
            [
                sys.executable,
                "-m",
                "experiments.full_moment_variant_study",
                "worker",
                comparison,
                phase,
                kind,
                link,
                str(m),
                str(seed),
                arm,
            ],
            env,
            log,
            timeout=manifest["timeout_seconds"],
            memory_mib=manifest["memory_mib"],
            cwd=ROOT,
        )
        verify(directory, manifest)
        if result["status"] == "ok":
            try:
                row = json.loads(log.read_text().splitlines()[-1])
            except (ValueError, IndexError) as error:
                row = {"status": "invalid_output", "error": repr(error)}
        else:
            row = {"status": result["status"], "error": log.read_text()[-4000:]}
        row.update(
            comparison=comparison,
            kind=kind,
            link=link,
            m=m,
            seed=seed,
            arm=arm,
            fingerprint=manifest["fingerprint"],
            process_seconds=result["process_seconds"],
            process_peak_rss_mib=result["peak_rss_mib"],
        )
        write(directory / "fits" / f"{name}.json", row)
        print(
            f"{comparison}/{phase} {i}/{len(manifest['jobs'])} {name}: {row['status']}, {time.perf_counter() - started:.1f}s",
            flush=True,
        )
    write(
        directory / "execution.json",
        {"wall_seconds": time.perf_counter() - started, "recorded": i},
    )
    result = analyze(comparison, phase)
    audit(comparison, phase)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("pilot", "worker", "run", "analyze", "audit")
    )
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.action == "worker":
        comparison, phase, kind, link, m, seed, arm = args.arguments
        with np.load(
            OUT / comparison / phase / "inputs" / f"{link}-m{m}-{seed}.npz"
        ) as saved:
            row = fit(saved["x"], saved["y"], saved["truth"], int(seed), arm, kind)
        print(json.dumps({**row, "status": "ok"}, default=default, allow_nan=False))
    elif args.action == "pilot":
        x, y, truth = arrays(case(25, 5, 5000, 0.3, "sine"), 985900)
        for arm in args.arguments or (
            "ridge",
            "full0",
            "metric",
            "coherence",
            "tangent",
        ):
            row = fit(x, y, truth, 985900, arm)
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "fit_seconds": row["fit_seconds"],
                        "stages": len(row["stages"]),
                        "inner_iterations": sum(
                            s.get("inner_iterations", 0) for s in row["stages"]
                        ),
                    }
                ),
                flush=True,
            )
    else:
        {"run": run, "analyze": analyze, "audit": audit}[args.action](*args.arguments)


if __name__ == "__main__":
    main()
