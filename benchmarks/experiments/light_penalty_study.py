"""Frozen, sequential full-fit EDR/time probe for lightweight penalties.

Run once: python -m experiments.light_penalty_study run
Read saved results: python -m experiments.light_penalty_study analyze
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import resource
import subprocess
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
from experiments.edr_metric_study import arrays, digest, response, write_json
from experiments.light_penalties import ARMS, solve
from experiments.r_sdr.design import metrics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/results/verification/light-penalties-20261025"
RUN = OUT / "run"
SEEDS = tuple(range(910600, 910606))
CASES = tuple(itertools.product(("additive", "quadratic"), (2, 3, 5)))


def sha(path):
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(link, m, seed, arm):
    with np.load(RUN / "inputs" / f"{seed}.npz") as data:
        x, full_basis, noise = data["x"], data["basis"], data["noise"]
    truth = full_basis[:, :m]
    y, _ = response(x, truth, noise, link)
    method_seed = int(np.random.SeedSequence([seed, 1]).generate_state(1)[0]) % (
        2**31 - 1
    )
    config = ADP_Config(seed=method_seed, N_J=300, N_phi=12, N_loc=20, N_lin=50)
    context, stages = {}, []
    profiler = IndexProfiler()

    def statistics(X, Y, weights, directions, batch_size=32, *, normalized=False):
        import hashlib

        first = not stages
        hasher = hashlib.sha256()

        def blocks():
            for start, block in weights:
                if first:
                    hasher.update(np.ascontiguousarray(block).tobytes())
                yield start, block

        stats = calculate_statistics(
            X, Y, blocks(), directions, batch_size, normalized=normalized
        )
        if first:
            context.update(
                first_weights_sha256=hasher.hexdigest(),
                first_directions_sha256=digest(directions),
            )
        return stats

    def solver(index, stats):
        if "anchor" not in context:
            context["anchor"] = index.T.copy()
        result = solve(index, stats, context["anchor"], arm)
        stages.append(result.diagnostics)
        return result

    start = time.perf_counter()
    try:
        fitted = fit_index(
            x,
            y,
            config=config,
            mode="multi",
            index_dim=m,
            solve=solver,
            profiler=profiler,
            statistics_function=statistics,
            trace_indices=True,
        )
    finally:
        profiler.finish()
    seconds = time.perf_counter() - start
    estimate, initial = fitted.index.T, fitted.initial_index.T
    return {
        "link": link,
        "m": m,
        "seed": seed,
        "arm": arm,
        "status": "ok",
        "method_seed": method_seed,
        "data_sha256": digest(x, y, truth),
        "initial_sha256": digest(initial),
        "first_weights_sha256": context["first_weights_sha256"],
        "first_directions_sha256": context["first_directions_sha256"],
        "basis": estimate,
        "initial_basis": initial,
        "edr": metrics(estimate, truth)["projector_distance"],
        "initial_edr": metrics(initial, truth)["projector_distance"],
        "orthogonality_error": float(np.linalg.norm(estimate.T @ estimate - np.eye(m))),
        "selected_iteration": fitted.metadata["selected_iteration"],
        "stop_reason": fitted.metadata["stop_reason"],
        "fit_seconds": seconds,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "stages": stages,
        "profile": profiler.model_profile(),
        "load_average": os.getloadavg(),
    }


def freeze():
    if RUN.exists():
        raise FileExistsError(f"refusing to overwrite {RUN}")
    (RUN / "inputs").mkdir(parents=True)
    (RUN / "fits").mkdir()
    inputs = {}
    for seed in SEEDS:
        x, basis, noise = arrays(seed)
        path = RUN / "inputs" / f"{seed}.npz"
        np.savez_compressed(path, x=x, basis=basis, noise=noise)
        inputs[str(path.relative_to(ROOT))] = sha(path)
    paths = [
        *sorted((ROOT / "ADP").rglob("*.py")),
        Path(__file__),
        ROOT / "benchmarks/experiments/light_penalties.py",
        ROOT / "benchmarks/experiments/edr_metric_study.py",
        ROOT / "benchmarks/experiments/edr_metric_corrections.py",
        ROOT / "benchmarks/experiments/r_sdr/design.py",
        ROOT / "tests/test_light_penalties.py",
        OUT / "PROTOCOL.md",
        OUT / "R1_AUDIT.md",
    ]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    with zipfile.ZipFile(RUN / "sources.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT))
    jobs = [(link, m, seed, arm) for link, m in CASES for seed in SEEDS for arm in ARMS]
    np.random.default_rng(20261025).shuffle(jobs)
    manifest = {
        "scope": "exploratory only; no held-out validation or default change",
        "jobs": jobs,
        "sources": hashes,
        "inputs": inputs,
        "seeds": SEEDS,
        "arms": ARMS,
        "n": 5000,
        "d": 25,
        "config": {"N_J": 300, "N_loc": 20, "N_phi": 12, "N_lin": 50},
        "workers": 1,
        "blas_threads": 1,
        "timeout_seconds": 180,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "source_archive_sha256": sha(RUN / "sources.zip"),
    }
    import hashlib

    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    write_json(RUN / "manifest.json", manifest)
    return manifest


def run():
    manifest = freeze()
    env = dict(
        os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    started = time.perf_counter()
    for i, job in enumerate(manifest["jobs"], 1):
        drift = [p for p, h in manifest["sources"].items() if sha(ROOT / p) != h]
        if drift:
            raise RuntimeError(f"source drift: {drift}")
        link, m, seed, arm = job
        before = time.perf_counter()
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.light_penalty_study",
                    "worker",
                    link,
                    str(m),
                    str(seed),
                    arm,
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            if completed.returncode:
                raise RuntimeError(completed.stderr[-8000:])
            row = json.loads(completed.stdout)
        except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError) as error:
            row = dict(zip(("link", "m", "seed", "arm"), job, strict=True))
            row.update(status="failed", error=str(error))
        row.update(
            process_seconds=time.perf_counter() - before,
            fingerprint=manifest["fingerprint"],
        )
        write_json(RUN / "fits" / f"{link}-m{m}-{seed}-{arm}.json", row)
        if i % 6 == 0 or row["status"] != "ok":
            print(
                f"{i}/{len(manifest['jobs'])}: {row['status']}, elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    write_json(
        RUN / "execution.json",
        {
            "completed": len(manifest["jobs"]),
            "wall_seconds": time.perf_counter() - started,
        },
    )
    analyze()


def interval(values, *, geometric=False):
    values = np.array(values)
    if geometric:
        values = np.log(values)
    rng = np.random.default_rng(20261025)
    means = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    bounds = np.quantile(means, [0.025, 0.975])
    return (np.exp(bounds) if geometric else bounds).tolist()


def compare(rows, arm, control="base"):
    lookup = {(r["link"], r["m"], r["seed"], r["arm"]): r for r in rows}
    pairs = []
    for link, m in CASES:
        for seed in SEEDS:
            a, b = (
                lookup.get((link, m, seed, arm)),
                lookup.get((link, m, seed, control)),
            )
            if a and b and a["status"] == b["status"] == "ok":
                pairs.append((a, b))
    primary = [(a, b) for a, b in pairs if a["m"] in (3, 5)]
    if not primary:
        return {
            "arm": arm,
            "control": control,
            "complete_pairs": len(pairs),
            "promising": False,
        }
    seed_ratios, seed_deltas = [], []
    for seed in SEEDS:
        block = [(a, b) for a, b in primary if a["seed"] == seed]
        if block:
            seed_ratios.append(
                float(
                    np.exp(
                        np.mean(
                            [
                                np.log(max(a["edr"], 1e-12) / max(b["edr"], 1e-12))
                                for a, b in block
                            ]
                        )
                    )
                )
            )
            seed_deltas.append(float(np.mean([a["edr"] - b["edr"] for a, b in block])))
    ratio = float(np.exp(np.mean(np.log(seed_ratios))))
    ci = interval(seed_ratios, geometric=True)
    fit_ratio = float(
        np.exp(np.mean([np.log(a["fit_seconds"] / b["fit_seconds"]) for a, b in pairs]))
    )
    process_ratio = float(
        np.exp(
            np.mean(
                [np.log(a["process_seconds"] / b["process_seconds"]) for a, b in pairs]
            )
        )
    )
    case_deltas = [
        float(
            np.mean(
                [a["edr"] - b["edr"] for a, b in pairs if (a["link"], a["m"]) == case]
            )
        )
        for case in CASES
        if any((a["link"], a["m"]) == case for a, b in pairs)
    ]
    recovery = [
        sum(a["edr"] <= 0.1 for a, _ in primary),
        sum(b["edr"] <= 0.1 for _, b in primary),
    ]
    checks = {
        "all_36_pairs": len(pairs) == 36,
        "primary_edr_ratio_le_0.95": ratio <= 0.95,
        "primary_ratio_ci_below_1": ci[1] < 1,
        "no_case_mean_regression_above_0.02": max(case_deltas) <= 0.02,
        "primary_recovery_not_lower": recovery[0] >= recovery[1],
        "fit_time_ratio_le_1.25": fit_ratio <= 1.25,
    }
    return {
        "arm": arm,
        "control": control,
        "complete_pairs": len(pairs),
        "primary_pairs": len(primary),
        "primary_edr_ratio": ratio,
        "ratio_ci": ci,
        "primary_mean_delta": float(np.mean(seed_deltas)),
        "delta_ci": interval(seed_deltas),
        "primary_wins": sum(a["edr"] < b["edr"] for a, b in primary),
        "primary_recovery_candidate_control": recovery,
        "fit_ratio": fit_ratio,
        "process_ratio": process_ratio,
        "checks": checks,
        "promising": all(checks.values()),
    }


def analyze():
    manifest = json.loads((RUN / "manifest.json").read_text())
    rows = [json.loads(p.read_text()) for p in sorted((RUN / "fits").glob("*.json"))]
    summary = []
    for arm in ARMS:
        for link, m in CASES:
            group = [
                r
                for r in rows
                if (r["arm"], r["link"], r["m"]) == (arm, link, m)
                and r["status"] == "ok"
            ]
            if not group:
                continue
            summary.append(
                {
                    "arm": arm,
                    "link": link,
                    "m": m,
                    "successful": len(group),
                    "planned": 6,
                    "mean_edr": float(np.mean([r["edr"] for r in group])),
                    "recovery": sum(r["edr"] <= 0.1 for r in group),
                    "median_fit_seconds": float(
                        np.median([r["fit_seconds"] for r in group])
                    ),
                    "median_process_seconds": float(
                        np.median([r["process_seconds"] for r in group])
                    ),
                    "median_rss_mib": float(
                        np.median([r["peak_rss_mib"] for r in group])
                    ),
                    "inner_converged_calls": sum(
                        bool(s["converged"]) for r in group for s in r["stages"]
                    ),
                    "inner_calls": sum(len(r["stages"]) for r in group),
                }
            )
    comparisons = [compare(rows, arm) for arm in ARMS if arm != "base"]
    pad_comparisons = [
        compare(rows, arm, "pad") for arm in ARMS if arm.startswith("proj")
    ]
    result = {
        "fingerprint": manifest["fingerprint"],
        "planned": len(manifest["jobs"]),
        "recorded": len(rows),
        "successful": sum(r["status"] == "ok" for r in rows),
        "failures": [r for r in rows if r["status"] != "ok"],
        "comparisons": comparisons,
        "projector_vs_pad": pad_comparisons,
        "cases": summary,
        "validation_run": False,
        "production_changed": False,
    }
    write_json(OUT / "summary.json", result)
    lines = [
        "# Lightweight EDR penalties: exploratory results",
        "",
        f"Recorded {result['recorded']}/{result['planned']}, successful {result['successful']}. No held-out validation or production promotion.",
        "",
        "Primary EDR: additive/quadratic, m=3,5, six paired seed blocks; smaller is better. Fit/process ratios cover all six cases including m=2.",
        "",
        "| Arm | EDR ratio vs base [95% CI] | Full-fit time ratio | Process ratio | Promising |",
        "|---|---:|---:|---:|---|",
    ]
    for item in comparisons:
        if "primary_edr_ratio" in item:
            lines.append(
                f"| {item['arm']} | {item['primary_edr_ratio']:.4f} [{item['ratio_ci'][0]:.4f}, {item['ratio_ci'][1]:.4f}] | {item['fit_ratio']:.3f} | {item['process_ratio']:.3f} | {item['promising']} |"
            )
    lines += [
        "",
        "| Arm | Link | m | Mean EDR | Recovery /6 | Median fit seconds |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['arm']} | {item['link']} | {item['m']} | {item['mean_edr']:.5f} | {item['recovery']} | {item['median_fit_seconds']:.3f} |"
        )
    lines += [
        "",
        "Padding is a representation control, not a statistical penalty. Projector-vs-pad comparisons are saved in summary.json. The baseline is the actual current public solver with effective local_ridge=0. All timings include penalty preparation. Missing/failed tasks remain in planned denominators.",
        "",
        "Confidence intervals bootstrap six whole seed blocks and are exploratory, without multiplicity correction. Inner convergence is reported separately in summary.json. A passing probe would require fresh independent confirmation; no universal EDR improvement is implied.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "recorded": len(rows),
                "successful": result["successful"],
                "comparisons": comparisons,
            },
            indent=2,
        )
    )


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
        link, m, seed, arm = options.args
        # Serialize NumPy through the existing helper without worker-side files.
        from experiments.edr_metric_study import json_default

        print(
            json.dumps(
                worker(link, int(m), int(seed), arm),
                default=json_default,
                allow_nan=False,
            )
        )
