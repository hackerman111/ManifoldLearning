"""Frozen paired full fits for coupled Hessians and exact second-moment penalties."""

from __future__ import annotations

import argparse
import hashlib
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
from experiments import coupled_penalties as coupled
from experiments import curvature_penalties as curvature
from experiments import local_polynomial as poly
from experiments.light_penalties import solve as baseline
from experiments.mave_grassmann_grid import arrays, case, settings
from experiments.r_sdr.design import metrics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/results/verification/curvature-penalties-20261025"
RUN = OUT / "run"
SEEDS = tuple(range(912000, 912006))
CASES = tuple(itertools.product(("additive", "quadratic", "sine"), (2, 3, 5)))
ARMS = ("base", "full", *curvature.ARMS)


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


def benchmark(link, m):
    return case(25, m, 5000, 0.3, link, "curvature")


def fit(x, y, truth, method_seed, arm):
    m = truth.shape[1]
    config_kwargs, _ = settings(benchmark("additive", m), "Grassmann")
    config = ADP_Config(seed=method_seed, **config_kwargs)
    context, stages = {}, []
    profiler = IndexProfiler()

    def statistics(X, Y, weights, directions, batch_size=32, *, normalized=False):
        blocks = [(0, weights)] if isinstance(weights, np.ndarray) else list(weights)
        if not stages:
            hasher = hashlib.sha256()
            for _, block in blocks:
                hasher.update(np.ascontiguousarray(block).tobytes())
            context.update(
                first_weights_sha256=hasher.hexdigest(),
                first_directions_sha256=digest(directions),
            )
        stats = calculate_statistics(
            X, Y, iter(blocks), directions, batch_size, normalized=normalized
        )
        if arm in curvature.ARMS or arm == "full":
            data = poly.neighborhoods(X, Y, blocks)
            context.update(data=data, directions=directions.copy())
            if arm == "full":
                covariances, cross = [], []
                for local in data.local:
                    covariances.append(
                        local.delta.T @ (local.weights[:, None] * local.delta)
                    )
                    cross.append(local.delta.T @ (local.weights * local.response))
                context["moments"] = np.array(covariances), np.array(cross)
        return stats

    def solver(index, stats):
        if arm == "base":
            result = baseline(index, stats, index.T, "base")
        elif arm == "full":
            result = coupled.solve(index, stats, "full", moments=context["moments"])
        else:
            result = curvature.solve(index, context["data"], context["directions"], arm)
        stages.append(
            {
                **result.diagnostics,
                "before_edr": metrics(index.T, truth)["projector_distance"],
                "after_edr": metrics(result.index.T, truth)["projector_distance"],
            }
        )
        return result

    started = time.perf_counter()
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
    elapsed = time.perf_counter() - started
    basis, initial = fitted.index.T, fitted.initial_index.T
    return {
        "basis": basis,
        "initial_basis": initial,
        "edr": metrics(basis, truth)["projector_distance"],
        "initial_edr": metrics(initial, truth)["projector_distance"],
        "orthogonality_error": float(np.linalg.norm(basis.T @ basis - np.eye(m))),
        "initial_sha256": digest(initial),
        "first_weights_sha256": context["first_weights_sha256"],
        "first_directions_sha256": context["first_directions_sha256"],
        "fit_seconds": elapsed,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "selected_iteration": fitted.metadata["selected_iteration"],
        "stop_reason": fitted.metadata["stop_reason"],
        "stages": stages,
        "profile": profiler.model_profile(),
    }


def worker(link, m, seed, arm):
    with np.load(RUN / "inputs" / f"{link}-m{m}-{seed}.npz") as saved:
        x, y, truth = saved["x"], saved["y"], saved["truth"]
    method_seed = int(np.random.SeedSequence([seed, 1]).generate_state(1)[0]) % (
        2**31 - 1
    )
    row = fit(x, y, truth, method_seed, arm)
    row.update(
        link=link,
        m=m,
        seed=seed,
        arm=arm,
        status="ok",
        method_seed=method_seed,
        data_sha256=digest(x, y, truth),
        load_average=os.getloadavg(),
    )
    return row


def pilot():
    # Timing/numerical feasibility only; no pilot EDR is printed or selected on.
    x, y, truth = arrays(benchmark("sine", 5), 912900)
    rows = []
    for arm in ARMS:
        result = fit(x, y, truth, 912900, arm)
        row = {
            "arm": arm,
            "fit_seconds": result["fit_seconds"],
            "calls": len(result["stages"]),
            "max_normal_residual": max(
                s.get("normal_residual_max", 0) for s in result["stages"]
            ),
            "max_cg_iterations": max(
                s.get("cg_iterations_max", 0) for s in result["stages"]
            ),
        }
        rows.append(row)
        print(json.dumps(row, default=default), flush=True)
    write(OUT / "PILOT.json", rows)


def source_paths():
    return sorted(
        {
            *ROOT.joinpath("ADP").rglob("*.py"),
            *ROOT.joinpath("benchmarks/experiments/r_sdr").glob("*.py"),
            ROOT / "benchmarks/experiments/__init__.py",
            Path(__file__),
            ROOT / "benchmarks/experiments/curvature_penalties.py",
            ROOT / "benchmarks/experiments/local_polynomial.py",
            ROOT / "benchmarks/experiments/coupled_penalties.py",
            ROOT / "benchmarks/experiments/light_penalties.py",
            ROOT / "benchmarks/experiments/mave_grassmann_grid.py",
            ROOT / "tests/test_curvature_penalties.py",
            OUT / "PROTOCOL.md",
            OUT / "R1_AUDIT.md",
            OUT / "audit_results.py",
        }
    )


def freeze():
    if RUN.exists():
        raise FileExistsError(f"refusing to overwrite {RUN}")
    (RUN / "inputs").mkdir(parents=True)
    (RUN / "fits").mkdir()
    inputs = {}
    for link, m in CASES:
        for seed in SEEDS:
            x, y, truth = arrays(benchmark(link, m), seed)
            path = RUN / "inputs" / f"{link}-m{m}-{seed}.npz"
            np.savez_compressed(path, x=x, y=y, truth=truth)
            inputs[str(path.relative_to(ROOT))] = sha(path)
    paths = source_paths()
    sources = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    with zipfile.ZipFile(RUN / "sources.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT))
    jobs = [(link, m, seed, arm) for link, m in CASES for seed in SEEDS for arm in ARMS]
    np.random.default_rng(20261027).shuffle(jobs)
    manifest = {
        "scope": "exploratory ESTIMATOR; no independent validation or production promotion",
        "jobs": jobs,
        "sources": sources,
        "inputs": inputs,
        "seeds": SEEDS,
        "arms": ARMS,
        "cases": CASES,
        "n": 5000,
        "d": 25,
        "noise": 0.3,
        "config": {
            "N_J": 300,
            "N_loc": 20,
            "N_phi": 12,
            "N_lin": 50,
            "select_step": "best",
            "index_init": "local",
        },
        "workers": 1,
        "blas_threads": 1,
        "timeout_seconds": 180,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "archive_sha256": sha(RUN / "sources.zip"),
    }
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    write(RUN / "manifest.json", manifest)
    return manifest


def run():
    manifest = freeze()
    env = dict(os.environ)
    env.update(
        {
            key: "1"
            for key in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "BLIS_NUM_THREADS",
            )
        }
    )
    started = time.perf_counter()
    for i, (link, m, seed, arm) in enumerate(manifest["jobs"], 1):
        drift = [p for p, h in manifest["sources"].items() if sha(ROOT / p) != h]
        if drift:
            raise RuntimeError(f"source drift: {drift}")
        before = time.perf_counter()
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.curvature_penalty_study",
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
            if result.returncode:
                raise RuntimeError(result.stderr[-8000:])
            row = json.loads(result.stdout)
        except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError) as error:
            row = {
                "link": link,
                "m": m,
                "seed": seed,
                "arm": arm,
                "status": "failed",
                "error": str(error),
            }
        row.update(
            process_seconds=time.perf_counter() - before,
            fingerprint=manifest["fingerprint"],
        )
        write(RUN / "fits" / f"{link}-m{m}-{seed}-{arm}.json", row)
        if i % 7 == 0 or row["status"] != "ok":
            print(
                f"{i}/{len(manifest['jobs'])}: {row['status']}, elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    write(
        RUN / "execution.json",
        {
            "completed": len(manifest["jobs"]),
            "wall_seconds": time.perf_counter() - started,
        },
    )
    analyze()


def interval(seed_values):
    values = np.asarray(seed_values)
    indices = np.random.default_rng(20261027).integers(
        0, len(values), (10000, len(values))
    )
    means = values[indices].mean(axis=1)
    return float(np.exp(values.mean())), np.exp(
        np.quantile(means, [0.025, 0.975])
    ).tolist()


def comparison(rows, candidate, control):
    by_key = {
        (r["link"], r["m"], r["seed"], r["arm"]): r for r in rows if r["status"] == "ok"
    }
    pairs = [
        (by_key[(link, m, seed, candidate)], by_key[(link, m, seed, control)])
        for link, m in CASES
        for seed in SEEDS
        if (link, m, seed, candidate) in by_key and (link, m, seed, control) in by_key
    ]
    complete = len(pairs) == len(CASES) * len(SEEDS)
    if not complete:
        return {
            "candidate": candidate,
            "control": control,
            "complete": False,
            "paired": len(pairs),
            "promising": False,
        }
    primary = [(a, b) for a, b in pairs if a["m"] in (3, 5)]
    seed_logs = [
        np.mean(
            [
                np.log(max(a["edr"], 1e-12) / max(b["edr"], 1e-12))
                for a, b in primary
                if a["seed"] == seed
            ]
        )
        for seed in SEEDS
    ]
    ratio, ci = interval(seed_logs)
    time_logs = [
        np.mean(
            [
                np.log(a["fit_seconds"] / b["fit_seconds"])
                for a, b in pairs
                if a["seed"] == seed
            ]
        )
        for seed in SEEDS
    ]
    time_ratio, time_ci = interval(time_logs)
    deltas = {
        f"{link}-m{m}": float(
            np.mean(
                [
                    a["edr"] - b["edr"]
                    for a, b in pairs
                    if a["link"] == link and a["m"] == m
                ]
            )
        )
        for link, m in CASES
    }
    recovery = [
        sum(r["edr"] <= 0.1 for r, _ in primary),
        sum(r["edr"] <= 0.1 for _, r in primary),
    ]
    orthogonality = max(
        max(a["orthogonality_error"], b["orthogonality_error"]) for a, b in pairs
    )
    gates = {
        "coverage_pairing": complete,
        "orthogonality": orthogonality <= 1e-8,
        "edr_improvement": ratio <= 0.95 and ci[1] < 1,
        "case_regression": max(deltas.values()) <= 0.02,
        "recovery": recovery[0] >= recovery[1],
        "time": time_ratio <= 2,
    }
    return {
        "candidate": candidate,
        "control": control,
        "complete": complete,
        "paired": len(pairs),
        "primary_edr_ratio": ratio,
        "ratio_ci": ci,
        "fit_ratio": time_ratio,
        "fit_ratio_ci": time_ci,
        "primary_mean_delta": float(np.mean([a["edr"] - b["edr"] for a, b in primary])),
        "case_deltas": deltas,
        "primary_recovery": recovery,
        "max_orthogonality": orthogonality,
        "gates": gates,
        "promising": all(gates.values()),
    }


def analyze():
    rows = [
        json.loads(path.read_text()) for path in sorted((RUN / "fits").glob("*.json"))
    ]
    contrasts = [(arm, "base") for arm in ARMS if arm != "base"]
    contrasts += [(arm, "full") for arm in curvature.ARMS]
    contrasts += [
        ("hgraph", "wls2"),
        ("hcommon", "wls2"),
        ("rquad", "m1quad"),
        ("m1quad", "wls2"),
    ]
    comparisons = [comparison(rows, a, b) for a, b in contrasts]
    cases, costs = [], []
    for arm in ARMS:
        successful = [r for r in rows if r["arm"] == arm and r["status"] == "ok"]
        stages = [stage for row in successful for stage in row["stages"]]
        costs.append(
            {
                "arm": arm,
                "successful": len(successful),
                "planned": len(SEEDS) * len(CASES),
                "median_fit_seconds": float(
                    np.median([r["fit_seconds"] for r in successful])
                ),
                "median_process_seconds": float(
                    np.median([r["process_seconds"] for r in successful])
                ),
                "median_rss_mib": float(
                    np.median([r["peak_rss_mib"] for r in successful])
                ),
                "converged": sum(s.get("converged", False) for s in stages),
                "calls": len(stages),
                "max_normal_residual": max(
                    (s.get("normal_residual_max", 0) for s in stages), default=0
                ),
                "max_cg_iterations": max(
                    (s.get("cg_iterations_max", 0) for s in stages), default=0
                ),
                "moment_saturated_windows": sum(
                    s.get("information", {}).get("saturated_first_moment_windows", 0)
                    for s in stages
                ),
                "moment_windows": sum(
                    s.get("information", {}).get("windows", 0) for s in stages
                ),
            }
        )
        for link, m in CASES:
            selected = [r for r in successful if r["link"] == link and r["m"] == m]
            cases.append(
                {
                    "arm": arm,
                    "link": link,
                    "m": m,
                    "successful": len(selected),
                    "planned": len(SEEDS),
                    "mean_edr": float(np.mean([r["edr"] for r in selected]))
                    if selected
                    else None,
                    "recovery": sum(r["edr"] <= 0.1 for r in selected),
                    "median_fit_seconds": float(
                        np.median([r["fit_seconds"] for r in selected])
                    )
                    if selected
                    else None,
                }
            )
    result = {
        "planned": len(SEEDS) * len(CASES) * len(ARMS),
        "recorded": len(rows),
        "successful": sum(r["status"] == "ok" for r in rows),
        "failed": [
            {k: r[k] for k in ("link", "m", "seed", "arm", "error")}
            for r in rows
            if r["status"] != "ok"
        ],
        "comparisons": comparisons,
        "cases": cases,
        "costs": costs,
        "validation_run": False,
        "production_changed": False,
    }
    write(OUT / "summary.json", result)
    lines = [
        "# Связанные гессианы и R_quad: результаты разведки",
        "",
        f"Успешно {result['successful']}/{result['planned']} полных fits; seeds 912000–912005.",
        "",
        "Основные сравнения изолируют новые штрафы. EDR — меньше лучше; CI — bootstrap целых seed-блоков.",
        "",
        "| Кандидат / контроль | EDR ratio (m=3,5), 95% CI | Fit time ratio | Гейт |",
        "|---|---:|---:|---|",
    ]
    for c in comparisons:
        if c["complete"]:
            lines.append(
                f"| {c['candidate']} / {c['control']} | {c['primary_edr_ratio']:.4f} [{c['ratio_ci'][0]:.4f}, {c['ratio_ci'][1]:.4f}] | {c['fit_ratio']:.3f} | {'PASS' if c['promising'] else 'FAIL'} |"
            )
        else:
            lines.append(
                f"| {c['candidate']} / {c['control']} | incomplete {c['paired']} pairs | — | FAIL |"
            )
    lines += [
        "",
        "| Функция | m | " + " | ".join(ARMS) + " |",
        "|---|---:|" + "---:|" * len(ARMS),
    ]
    for link, m in CASES:
        values = [
            next(c for c in cases if (c["link"], c["m"], c["arm"]) == (link, m, arm))
            for arm in ARMS
        ]
        lines.append(
            f"| {link} | {m} | "
            + " | ".join(
                f"{v['mean_edr']:.5f}" if v["mean_edr"] is not None else "failed"
                for v in values
            )
            + " |"
        )
    lines += [
        "",
        "| Arm | Медиана fit, s | Медиана process, s | RSS, MiB | Внутренняя стационарность |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in costs:
        lines.append(
            f"| {c['arm']} | {c['median_fit_seconds']:.3f} | {c['median_process_seconds']:.3f} | {c['median_rss_mib']:.1f} | {c['converged']}/{c['calls']} |"
        )
    lines += [
        "",
        "Сравнения с base/full включают различия оптимизатора, ridge и бюджета. Только hgraph/wls2, hcommon/wls2 и rquad/m1quad изолируют добавленный штраф. Окна совпадают на первом шаге, далее адаптируются. Низкий objective и внутренний normal residual не являются сертификатом EDR. Небольшое число seed-блоков; интервалы без поправки на множественные сравнения. Validation и production promotion не проводились. Подробнее — PROTOCOL.md, summary.json и AUDIT.md.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "successful": result["successful"],
                "planned": result["planned"],
                "causal_comparisons": comparisons[-4:],
            },
            indent=2,
            default=default,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pilot", "run", "worker", "analyze"))
    parser.add_argument("args", nargs="*")
    options = parser.parse_args()
    if options.command == "worker":
        link, m, seed, arm = options.args
        print(
            json.dumps(
                worker(link, int(m), int(seed), arm), default=default, allow_nan=False
            )
        )
    else:
        {"pilot": pilot, "run": run, "analyze": analyze}[options.command]()
