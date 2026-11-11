"""Frozen paired full-ADP evaluation for anisotropic geometry metrics.

This module reuses the full-ridge statistics/profile and the ADP fit harness.
It does not implement a statistical estimator or manifold optimizer.
"""

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
from experiments import full_ridge_batched as ridge
from experiments import full_ridge_fast_study as harness
from experiments.mave_grassmann_grid import arrays, case, settings
from experiments.r_sdr.design import metrics

ROOT = harness.ROOT
OUT = ROOT / "benchmarks/results/verification/anisotropic-metrics-20261026"
CASES = tuple(itertools.product(("additive", "quadratic", "sine"), (3, 5)))
SEEDS = {
    "selection": tuple(range(1286000, 1286010)),
    "validation": tuple(range(1286100, 1286110)),
}
BASELINE = "euclidean"
CANDIDATES = ("left", "two_sided", "schur", "varpro", "kernel")
PRACTICAL = "lbfgs"
ARMS = (BASELINE, *CANDIDATES, PRACTICAL)
BOOTSTRAP_REPLICATES = 50_000
BOOTSTRAP_SEED = 1_286_026
FAMILYWISE_ALPHA = 0.05


def default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write(path: Path, value):
    path.write_text(
        json.dumps(value, indent=2, default=default, allow_nan=False) + "\n"
    )


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(*values):
    return hashlib.sha256(
        b"".join(np.ascontiguousarray(value).tobytes() for value in values)
    ).hexdigest()


def source_paths():
    paths = set(ROOT.joinpath("ADP").rglob("*.py"))
    paths.update(
        ROOT / relative
        for relative in (
            "benchmarks/experiments/anisotropic_metrics.py",
            "benchmarks/experiments/anisotropic_geometry_study.py",
            "benchmarks/experiments/anisotropic_fixed_windows.py",
            "benchmarks/experiments/anisotropic_geometry_check.py",
            "benchmarks/experiments/coupled_penalties.py",
            "benchmarks/experiments/full_ridge_fast.py",
            "benchmarks/experiments/full_ridge_batched.py",
            "benchmarks/experiments/full_ridge_fast_study.py",
            "benchmarks/experiments/mave_grassmann_grid.py",
            "benchmarks/experiments/r_sdr/design.py",
            "benchmarks/experiments/r_sdr/runner.py",
            "tests/test_anisotropic_geometry_study.py",
            "tests/test_anisotropic_metrics.py",
            "benchmarks/results/verification/anisotropic-metrics-20261026/PROOF.md",
            "benchmarks/results/verification/anisotropic-metrics-20261026/PLAN.md",
            "benchmarks/results/verification/anisotropic-metrics-20261026/PROTOCOL.md",
            "benchmarks/results/verification/anisotropic-metrics-20261026/R1_AUDIT.md",
            "benchmarks/results/verification/anisotropic-metrics-20261026/formula_checks.json",
            "benchmarks/results/verification/anisotropic-metrics-20261026/audit_results.py",
        )
    )
    return sorted(paths)


def cluster_bootstrap_upper(
    deltas, *, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED
):
    """Bonferroni one-sided upper bound resampling whole seed clusters."""
    by_seed = {}
    for row in deltas:
        by_seed.setdefault(int(row["seed"]), []).append(float(row["delta_edr"]))
    seeds = sorted(by_seed)
    if not seeds or any(len(by_seed[seed]) != len(CASES) for seed in seeds):
        return {"upper": None, "interval95": None, "cluster_means": {}}
    cluster_means = np.array([np.mean(by_seed[seed]) for seed in seeds], dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(cluster_means, size=(replicates, len(seeds)), replace=True)
    bootstrap_means = sampled.mean(axis=1)
    quantile = 1 - FAMILYWISE_ALPHA / len(CANDIDATES)
    return {
        "upper": float(np.quantile(bootstrap_means, quantile)),
        "interval95": np.quantile(bootstrap_means, [0.025, 0.975]).tolist(),
        "quantile": quantile,
        "cluster_means": {
            str(seed): float(value)
            for seed, value in zip(seeds, cluster_means, strict=True)
        },
    }


def summarize_candidate(rows, candidate, seed_values):
    lookup = {(row["link"], row["m"], row["seed"], row["arm"]): row for row in rows}
    pairs = []
    candidate_rows, baseline_rows = [], []
    expected = list(
        itertools.product(("additive", "quadratic", "sine"), (3, 5), seed_values)
    )
    for link, m, seed in expected:
        old = lookup.get((link, m, seed, BASELINE))
        new = lookup.get((link, m, seed, candidate))
        if old is not None:
            baseline_rows.append(old)
        if new is not None:
            candidate_rows.append(new)
        if (
            old is None
            or new is None
            or old.get("status") != "ok"
            or new.get("status") != "ok"
        ):
            continue
        pairs.append(
            {
                "link": link,
                "m": m,
                "seed": seed,
                "delta_edr": new["edr"] - old["edr"],
                "control_edr": old["edr"],
                "candidate_edr": new["edr"],
                "fit_ratio": new["fit_seconds"] / old["fit_seconds"],
                "process_ratio": new["process_seconds"] / old["process_seconds"],
                "rss_ratio": new["process_peak_rss_mib"] / old["process_peak_rss_mib"],
                "paired": all(
                    old[key] == new[key]
                    for key in (
                        "data_sha256",
                        "method_seed",
                        "initial_sha256",
                        "first_weights_sha256",
                        "first_directions_sha256",
                        "first_moments_sha256",
                        "first_tau",
                    )
                ),
                "same_selected_iteration": old["selected_iteration"]
                == new["selected_iteration"],
                "same_stage_count": len(old["stages"]) == len(new["stages"]),
                "same_stage_iterations": [s["iterations"] for s in old["stages"]]
                == [s["iterations"] for s in new["stages"]],
                "same_stage_convergence": [s["converged"] for s in old["stages"]]
                == [s["converged"] for s in new["stages"]],
            }
        )

    complete = (
        len(pairs) == len(expected)
        and len(baseline_rows) == len(expected)
        and len(candidate_rows) == len(expected)
    )
    deltas = [pair["delta_edr"] for pair in pairs]
    bootstrap = (
        cluster_bootstrap_upper(pairs)
        if complete
        else {"upper": None, "interval95": None, "cluster_means": {}}
    )
    cell_means = {
        f"{link}-m{m}": float(
            np.mean(
                [p["delta_edr"] for p in pairs if p["link"] == link and p["m"] == m]
            )
        )
        for link, m in CASES
        if any(p["link"] == link and p["m"] == m for p in pairs)
    }

    def convergence_fraction(fits):
        stages = [
            stage
            for fit in fits
            if fit.get("status") == "ok"
            for stage in fit.get("stages", [])
        ]
        return (
            sum(bool(stage["converged"]) for stage in stages) / len(stages)
            if stages
            else 0.0
        )

    recovery_base = sum(
        row.get("status") == "ok" and row["edr"] <= 0.1 for row in baseline_rows
    )
    recovery_candidate = sum(
        row.get("status") == "ok" and row["edr"] <= 0.1 for row in candidate_rows
    )
    convergence_base, convergence_candidate = (
        convergence_fraction(baseline_rows),
        convergence_fraction(candidate_rows),
    )
    fit_ratio = (
        float(np.median([pair["fit_ratio"] for pair in pairs])) if pairs else None
    )
    process_ratio = (
        float(np.median([pair["process_ratio"] for pair in pairs])) if pairs else None
    )
    rss_ratio = (
        float(np.median([pair["rss_ratio"] for pair in pairs])) if pairs else None
    )
    gates = {
        "complete": complete,
        "pairing": complete and all(pair["paired"] for pair in pairs),
        "mean_edr": complete and float(np.mean(deltas)) <= -0.005,
        "familywise_cluster_upper": complete
        and bootstrap["upper"] is not None
        and bootstrap["upper"] < 0,
        "worst_pair_edr": complete and max(deltas) <= 0.08,
        "cell_mean_edr": complete
        and len(cell_means) == len(CASES)
        and max(cell_means.values()) <= 0.02,
        "recovery_nondecrease": complete and recovery_candidate >= recovery_base,
        "convergence_loss": complete
        and convergence_candidate >= convergence_base - 0.05,
        "fit_time": complete and fit_ratio <= 1.5,
        "rss": complete and rss_ratio <= 1.1,
    }
    return {
        "candidate": candidate,
        "planned_pairs": len(expected),
        "complete_pairs": len(pairs),
        "mean_delta_edr": float(np.mean(deltas)) if deltas else None,
        "familywise_upper": bootstrap["upper"],
        "cluster_interval95": bootstrap["interval95"],
        "cluster_means": bootstrap["cluster_means"],
        "cell_mean_deltas": cell_means,
        "worst_pair": max(pairs, key=lambda pair: pair["delta_edr"]) if pairs else None,
        "recovery_baseline": recovery_base,
        "recovery_candidate": recovery_candidate,
        "convergence_baseline": convergence_base,
        "convergence_candidate": convergence_candidate,
        "median_fit_ratio": fit_ratio,
        "median_process_ratio": process_ratio,
        "median_rss_ratio": rss_ratio,
        "gates": gates,
        "passed": all(gates.values()),
        "pairs": pairs,
    }


def fit(x, y, truth, seed, arm):
    """Shared full-ADP harness with explicit solver metric context."""
    from experiments import anisotropic_metrics as geometry

    method_seed = int(np.random.SeedSequence([seed, 1]).generate_state(1)[0]) % (
        2**31 - 1
    )
    config = ADP_Config(
        seed=method_seed,
        **settings(case(x.shape[1], truth.shape[1], len(x), 0.3), "Grassmann")[0],
    )
    stages, context = [], {}
    profiler = IndexProfiler()

    def statistics(X, Y, weights, directions, batch_size=32, *, normalized=False):
        first = not stages
        weight_hash = hashlib.sha256()

        def observed():
            for start, block in weights:
                if first:
                    weight_hash.update(np.ascontiguousarray(block).tobytes())
                yield start, block

        stats = ridge.calculate_statistics(
            X, Y, observed(), directions, batch_size, normalized=normalized
        )
        if first:
            context.update(
                first_weights_sha256=weight_hash.hexdigest(),
                first_directions_sha256=digest(directions),
                first_moments_sha256=digest(stats.U, stats.I, stats.mass),
            )
        return stats

    def solve(index, stats, **metric_context):
        if arm == PRACTICAL:
            result = ridge.solve(index, stats)
        else:
            metric_tensor = None
            if arm == "kernel":
                metric_tensor = geometry.localization_tensor(
                    index.T,
                    metric_context["metric_alpha"],
                    metric_context["metric_eigenvalues"],
                    metric_context["metric_tensor"],
                )
            result = geometry.solve(
                index,
                stats,
                arm,
                metric_tensor=metric_tensor,
            )
        if not stages:
            context["first_tau"] = result.diagnostics["penalty"]["local_ridge"]
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
            solver_metric_context=True,
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
        "first_moments_sha256": context["first_moments_sha256"],
        "first_tau": context["first_tau"],
        "data_sha256": digest(x, y, truth),
        "orthogonality_error": float(
            np.linalg.norm(fitted.index @ fitted.index.T - np.eye(truth.shape[1]))
        ),
    }


def freeze(phase, arms):
    directory = OUT / phase
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite {directory}")
    (directory / "inputs").mkdir(parents=True)
    (directory / "fits").mkdir()
    sources = source_paths()
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in sources}
    with zipfile.ZipFile(
        directory / "sources.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sources:
            archive.write(path, path.relative_to(ROOT))
    inputs = {}
    for link, m in CASES:
        for seed in SEEDS[phase]:
            path = directory / "inputs" / f"{link}-m{m}-{seed}.npz"
            x, y, truth = arrays(case(25, m, 5000, 0.3, link), seed)
            np.savez_compressed(path, x=x, y=y, truth=truth)
            inputs[str(path.relative_to(ROOT))] = sha(path)
    jobs = []
    for seed_index, seed in enumerate(SEEDS[phase]):
        for cell_index, (link, m) in enumerate(CASES):
            order = (
                arms if (seed_index + cell_index) % 2 == 0 else tuple(reversed(arms))
            )
            jobs.extend((link, m, seed, arm) for arm in order)
    manifest = {
        "phase": phase,
        "purpose": "frozen paired full-ADP geometry comparison",
        "arms": list(arms),
        "baseline": BASELINE,
        "candidates": [arm for arm in CANDIDATES if arm in arms],
        "practical_control": PRACTICAL,
        "jobs": jobs,
        "sources": source_hashes,
        "inputs": inputs,
        "archive_sha256": sha(directory / "sources.zip"),
        "fixed_prerequisite_hashes": {
            str(path.relative_to(ROOT)): sha(path)
            for path in (
                OUT / "R1_AUDIT.md",
                OUT / "fixed/manifest.json",
                OUT / "fixed/results.json",
                OUT / "fixed/REPORT.md",
                OUT / "fixed/sources.zip",
            )
        },
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": sys.version,
        "timeout_seconds": 60,
        "memory_mib": 2048,
        "blas_threads": 1,
        "workers": 1,
        "settings": {
            str(m): settings(case(25, m, 5000, 0.3), "Grassmann")[0] for m in (3, 5)
        },
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "familywise_alpha": FAMILYWISE_ALPHA,
            "candidates": len(CANDIDATES),
            "unit": "seed cluster aggregated across six cells",
        },
        "load_average": os.getloadavg(),
    }
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    write(directory / "manifest.json", manifest)
    return directory, manifest


def verify(directory, manifest):
    for relative, expected in manifest["sources"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f"source drift: {relative}")
    if sha(directory / "sources.zip") != manifest["archive_sha256"]:
        raise RuntimeError("source archive drift")
    for relative, expected in manifest.get("fixed_prerequisite_hashes", {}).items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f"fixed prerequisite drift: {relative}")
    for relative, expected in manifest["inputs"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f"input drift: {relative}")


def analyze(phase):
    directory = OUT / phase
    manifest = json.loads((directory / "manifest.json").read_text())
    verify(directory, manifest)
    rows = [
        json.loads(path.read_text())
        for path in sorted((directory / "fits").glob("*.json"))
    ]
    keys = [(row["link"], row["m"], row["seed"], row["arm"]) for row in rows]
    expected = {tuple(job) for job in manifest["jobs"]}
    schedule_ok = len(keys) == len(set(keys)) and set(keys) == expected
    for row in rows:
        if row["fingerprint"] != manifest["fingerprint"]:
            raise RuntimeError("row fingerprint mismatch")
    row_checks = {}
    for row in rows:
        key = (row["link"], row["m"], row["seed"], row["arm"])
        if row.get("status") != "ok":
            row_checks[key] = {"numerical": False, "geometry": False, "budget": False}
            continue
        basis = np.asarray(row.get("basis"), dtype=float)
        geometry_ok = bool(
            basis.ndim == 2
            and basis.shape == (25, row["m"])
            and np.isfinite(basis).all()
            and np.linalg.norm(basis.T @ basis - np.eye(row["m"])) <= 1e-10
        )
        with np.load(
            directory / "inputs" / f"{row['link']}-m{row['m']}-{row['seed']}.npz"
        ) as saved:
            actual_edr = metrics(basis, saved["truth"])["projector_distance"]
        geometry_ok &= bool(
            np.isfinite(actual_edr) and abs(actual_edr - row["edr"]) <= 1e-10
        )
        numerical_ok = bool(row.get("stages"))
        budget_ok = bool(row.get("stages"))
        for stage in row.get("stages", []):
            history = np.asarray(stage.get("profile_history", []), dtype=float)
            metric_residual = stage.get(
                "metric_residual_max", 0.0 if row["arm"] == PRACTICAL else np.inf
            )
            numerical_ok &= bool(
                np.isfinite(history).all()
                and np.isfinite(stage.get("profile_objective", np.nan))
                and np.isfinite(stage.get("riemannian_gradient", np.nan))
                and stage.get("normal_residual_max", np.inf) <= 1e-8
                and metric_residual <= 1e-8
                and np.all(
                    np.diff(history) <= 1e-10 * np.maximum(1.0, np.abs(history[:-1]))
                )
            )
            if row["arm"] == PRACTICAL:
                budget_ok &= bool(stage.get("iterations", 21) <= 20)
            else:
                budget_ok &= bool(
                    stage.get("iterations", 21) <= 20
                    and stage.get("profile_evaluations", 402) <= 401
                )
        row_checks[key] = {
            "numerical": bool(numerical_ok),
            "geometry": bool(geometry_ok),
            "budget": bool(budget_ok),
        }
    candidate_summaries = {
        candidate: summarize_candidate(rows, candidate, SEEDS[phase])
        for candidate in manifest["candidates"]
    }
    practical_full = summarize_candidate(rows, PRACTICAL, SEEDS[phase])
    practical_summary = {
        key: practical_full[key]
        for key in (
            "complete_pairs",
            "mean_delta_edr",
            "recovery_baseline",
            "recovery_candidate",
            "convergence_baseline",
            "convergence_candidate",
            "median_fit_ratio",
            "median_process_ratio",
            "median_rss_ratio",
        )
    }
    for candidate, result in candidate_summaries.items():
        relevant = [
            row_checks[(row["link"], row["m"], row["seed"], row["arm"])]
            for row in rows
            if row["arm"] in (BASELINE, candidate)
        ]
        result["gates"].update(
            geometry=bool(relevant) and all(check["geometry"] for check in relevant),
            numerical=bool(relevant) and all(check["numerical"] for check in relevant),
            budget=bool(relevant) and all(check["budget"] for check in relevant),
        )
        result["passed"] = all(result["gates"].values())
    practical_checks = [
        row_checks[(row["link"], row["m"], row["seed"], row["arm"])]
        for row in rows
        if row["arm"] == PRACTICAL
    ]
    practical_diagnostics = {
        "rows": len(practical_checks),
        "geometry": bool(practical_checks)
        and all(check["geometry"] for check in practical_checks),
        "numerical": bool(practical_checks)
        and all(check["numerical"] for check in practical_checks),
        "budget": bool(practical_checks)
        and all(check["budget"] for check in practical_checks),
    }
    summary = {
        "phase": phase,
        "planned_fits": len(expected),
        "recorded_fits": len(rows),
        "schedule_ok": schedule_ok,
        "status_counts": {
            status: sum(row.get("status") == status for row in rows)
            for status in sorted({row.get("status") for row in rows})
        },
        "practical_control_diagnostics": practical_diagnostics,
        "practical_control_vs_shared_solver_descriptive": practical_summary,
        "candidates": candidate_summaries,
        "passed_candidates": [
            arm
            for arm, result in candidate_summaries.items()
            if schedule_ok and result["passed"]
        ],
    }
    write(directory / "summary.json", summary)
    write_report()
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "candidates"}
            | {
                "gates": {
                    arm: value["gates"] for arm, value in candidate_summaries.items()
                }
            },
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_report():
    lines = [
        "# Anisotropic geometry full-ADP benchmark",
        "",
        "The geometry arms preserve the full-moment ridge profile and share one polar Armijo solver. A (`euclidean`) is the causal baseline. The existing batched L-BFGS arm is a separate practical control because it changes the optimizer.",
        "",
        "Fixed-window diagnostics",
        "",
        "See [fixed-window report](fixed/REPORT.md). Those SPD/mixed-rank microtimings are descriptive and are not ADP recovery evidence or a speed gate.",
    ]
    for phase in ("selection", "validation"):
        path = OUT / phase / "summary.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        lines.extend(
            [
                "",
                f"## {phase.title()}",
                "",
                f"Recorded {summary['recorded_fits']}/{summary['planned_fits']} scheduled fits; schedule integrity: {summary['schedule_ok']}; status counts: `{summary['status_counts']}`.",
                "",
                "| Candidate vs shared-solver A | Complete pairs | Mean ΔEDR | Familywise upper | Worst ΔEDR | Recovery A/candidate | Convergence A/candidate | Fit ratio | Process ratio | RSS ratio | Passed |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for arm, result in summary["candidates"].items():
            worst = result["worst_pair"]["delta_edr"] if result["worst_pair"] else None
            lines.append(
                f"| {arm} | {result['complete_pairs']}/{result['planned_pairs']} | {result['mean_delta_edr']} | {result['familywise_upper']} | {worst} | {result['recovery_baseline']}/{result['recovery_candidate']} | {result['convergence_baseline']:.4f}/{result['convergence_candidate']:.4f} | {result['median_fit_ratio']} | {result['median_process_ratio']} | {result['median_rss_ratio']} | {result['passed']} |"
            )
        practical = summary["practical_control_vs_shared_solver_descriptive"]
        lines.extend(
            [
                "",
                f"Practical batched L-BFGS control vs shared-solver A (descriptive, not causal): {practical['complete_pairs']} complete pairs; mean ΔEDR {practical['mean_delta_edr']}; recovery {practical['recovery_baseline']}/{practical['recovery_candidate']}; fit/process/RSS median ratios {practical['median_fit_ratio']}/{practical['median_process_ratio']}/{practical['median_rss_ratio']}.",
                f"Candidates passing frozen gates: `{summary['passed_candidates']}`.",
            ]
        )
    lines.extend(
        [
            "",
            "All paired full-fit artifacts, source/input hashes, and stage diagnostics are retained under `selection/` and, only when authorized by an independently audited selection, `validation/`. Failed or missing rows remain in completeness denominators. This evaluation does not change production defaults.",
            "",
        ]
    )
    (OUT / "REPORT.md").write_text("\n".join(lines))


def run(phase):
    arms = list(ARMS)
    if phase == "selection":
        r1_path = OUT / "R1_AUDIT.md"
        if not r1_path.exists() or "Status: PASS" not in r1_path.read_text():
            raise RuntimeError("independent R1 review must pass before selection")
        fixed_path = OUT / "fixed/results.json"
        if not fixed_path.exists():
            raise RuntimeError("fixed-window diagnostics must pass before selection")
        from experiments import anisotropic_fixed_windows as fixed

        fixed_manifest = json.loads((OUT / "fixed/manifest.json").read_text())
        fixed.verify_sources(fixed_manifest)
        fixed_results = json.loads(fixed_path.read_text())["results"]
        if not fixed_results or not all(
            case["passed"] for case in fixed_results.values()
        ):
            raise RuntimeError(
                "fixed-window numerical checks failed; selection not run"
            )
    if phase == "validation":
        selected = json.loads((OUT / "selection/summary.json").read_text())
        audit_path = OUT / "selection/independent_audit.json"
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        selected_candidates = selected.get("passed_candidates", [])
        if (
            not selected.get("schedule_ok")
            or not selected_candidates
            or not audit.get("integrity_ok")
            or audit.get("agreed_passed_candidates") != selected_candidates
        ):
            raise RuntimeError(
                "independent selection audit not passed; validation remains untouched"
            )
        selection_manifest = json.loads((OUT / "selection/manifest.json").read_text())
        verify(OUT / "selection", selection_manifest)
        if audit.get("manifest_sha256") != sha(
            OUT / "selection/manifest.json"
        ) or audit.get("summary_sha256") != sha(OUT / "selection/summary.json"):
            raise RuntimeError("stale selection audit; validation remains untouched")
        arms = [BASELINE, *selected["passed_candidates"], PRACTICAL]
    directory, manifest = freeze(phase, arms)
    if phase == "validation":
        selected = json.loads((OUT / "selection/manifest.json").read_text())
        if (
            manifest["sources"] != selected["sources"]
            or manifest["fixed_prerequisite_hashes"]
            != selected["fixed_prerequisite_hashes"]
        ):
            raise RuntimeError("source/prerequisite map changed since selection")
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONPATH=str(ROOT),
    )
    started = time.perf_counter()
    for index, (link, m, seed, arm) in enumerate(manifest["jobs"], 1):
        verify(directory, manifest)
        name = f"{link}-m{m}-{seed}-{arm}"
        log = directory / "fits" / f"{name}.log"
        process = harness.run_process(
            [
                sys.executable,
                "-m",
                "experiments.anisotropic_geometry_study",
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
        verify(directory, manifest)
        if process["status"] == "ok":
            try:
                row = json.loads(log.read_text().splitlines()[-1])
            except (ValueError, IndexError) as error:
                row = {"status": "invalid_output", "error": repr(error)}
        else:
            row = {"status": process["status"], "error": log.read_text()[-4000:]}
        row.update(
            link=link,
            m=m,
            seed=seed,
            arm=arm,
            fingerprint=manifest["fingerprint"],
            process_seconds=process["process_seconds"],
            process_peak_rss_mib=process["peak_rss_mib"],
        )
        write(directory / "fits" / f"{name}.json", row)
        print(
            f"{phase}: {index}/{len(manifest['jobs'])} {link} m={m} {arm}: {row['status']}",
            flush=True,
        )
    write(
        directory / "execution.json",
        {"wall_seconds": time.perf_counter() - started, "recorded": index},
    )
    return analyze(phase)


def worker(phase, link, m, seed, arm):
    with np.load(OUT / phase / "inputs" / f"{link}-m{m}-{seed}.npz") as saved:
        row = fit(saved["x"], saved["y"], saved["truth"], int(seed), arm)
    return {**row, "status": "ok"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "worker", "analyze"))
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.action == "run":
        run(args.arguments[0])
    elif args.action == "worker":
        result = worker(
            args.arguments[0],
            args.arguments[1],
            int(args.arguments[2]),
            args.arguments[3],
            args.arguments[4],
        )
        print(json.dumps(result, default=default, allow_nan=False), flush=True)
    else:
        analyze(args.arguments[0])


if __name__ == "__main__":
    main()
