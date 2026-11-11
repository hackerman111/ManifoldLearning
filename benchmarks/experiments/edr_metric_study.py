"""Reproducible paired ablations for the ideas in docs/tex/data/gpt_ans.md.

Run: python -m experiments.edr_metric_study run
Audit saved bases: python -m experiments.edr_metric_study audit
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
from ADP.core.experimental_centers import AdaptiveCentersConfig
from ADP.engine.common.calculus import pairwise_distance2, search_bandwidth
from ADP.engine.common.index_fit import fit_index
from ADP.engine.common.logger import IndexProfiler
from ADP.engine.common.statistic import calculate_statistics
from ADP.solver import grassman
from experiments.edr_metric_corrections import (
    projector_certificate,
    reweight_statistics,
)
from experiments.r_sdr.design import metrics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/results/verification/edr-metric-20261025"
SEEDS = {
    "pilot": range(765000, 765003),
    "selection": range(766000, 766010),
    "validation": range(767000, 767020),
}
ARMS = ["base", "gls", "row", "neff"] + [
    f"{prefix}floor{eta}" for prefix in ("", "gls-") for eta in (0.01, 0.05, 0.1, 0.2)
]
CASES = [(link, m) for link in ("additive", "quadratic") for m in (2, 3, 5)]


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, default=json_default, allow_nan=False) + "\n"
    )


def digest(*arrays):
    return hashlib.sha256(
        b"".join(np.ascontiguousarray(a).tobytes() for a in arrays)
    ).hexdigest()


def arrays(seed):
    streams = np.random.SeedSequence([seed, 20261025, 815]).spawn(3)
    x = np.random.default_rng(streams[0]).normal(size=(5000, 25))
    basis = np.linalg.qr(np.random.default_rng(streams[1]).normal(size=(25, 5)))[0]
    noise = np.random.default_rng(streams[2]).normal(size=5000)
    return x, basis, noise


def response(x, basis, noise, link):
    z = x @ basis
    raw = (
        np.sin(z[:, 0]) + 0.5 * z[:, 1] ** 2 + 0.35 * np.tanh(z[:, 2:]).sum(axis=1)
        if link == "additive"
        else np.square(z).sum(axis=1)
    )
    return (raw - raw.mean()) / raw.std() + 0.3 * noise, float(raw.std())


def certificate_diagnostic(x, y, truth, seed, link, signal_scale):
    """Post-fit diagnostic with known simulation constants; never used by fit."""
    config = ADP_Config(seed=seed, N_J=300, N_lin=50)
    center_seed = np.random.SeedSequence(seed).spawn(3)[0]
    centers = x[np.random.default_rng(center_seed).choice(len(x), 300, replace=False)]
    distance2 = pairwise_distance2(x, centers)
    h = search_bandwidth(distance2, 50, config.kernel, lower=np.finfo(float).eps)
    gradients, masses, bias_terms, variance_terms, exact = [], [], [], [], []
    curvature = (1.0 if link == "additive" else 2.0) / signal_scale
    for j, center in enumerate(centers):
        w = config.kernel(distance2[j] / h**2)
        active = w > 0
        a = w[active] / w.sum()
        dx = x[active] - a @ x[active]
        covariance = dx.T @ (a[:, None] * dx)
        least = np.linalg.eigvalsh(covariance)[0]
        if least <= 1e-12:
            return {"status": "singular_C", "min_eigenvalue": float(least)}
        c = dx.T @ (a * (y[active] - a @ y[active]))
        gradients.append(np.linalg.solve(covariance, c))
        masses.append(w.sum())
        radius = np.linalg.norm(x[active] - center, axis=1).max()
        bias_terms.append(curvature**2 * radius**6 / least**2)
        inverse_dx = np.linalg.solve(covariance, dx.T).T
        variance_terms.append(0.3**2 * np.sum(a**2 * np.sum(inverse_dx**2, axis=1)))
        z = center @ truth
        if link == "additive":
            derivative = np.concatenate(
                ([np.cos(z[0]), z[1]], 0.35 / np.cosh(z[2:]) ** 2)
            )
        else:
            derivative = 2 * z
        exact.append(truth @ derivative / signal_scale)
    pi = np.asarray(masses) / np.sum(masses)
    estimated = np.asarray(gradients).T * np.sqrt(pi)
    exact = np.asarray(exact).T * np.sqrt(pi)
    t = float(np.sqrt(pi @ bias_terms) + np.sqrt(pi @ variance_terms / 0.05))
    basis, info = projector_certificate(estimated, t, truth.shape[1])
    info.update(
        status="ok",
        rho=0.05,
        known_curvature_bound=curvature,
        actual_gradient_error=float(np.linalg.norm(estimated - exact, 2)),
        preliminary_edr=metrics(basis, truth)["projector_distance"],
    )
    return info


def fit(x, y, m, seed, arm):
    eta = float(arm.split("floor")[-1]) if "floor" in arm else 0.0
    kind = (
        "gls" if arm.startswith("gls") else (arm if arm in ("row", "neff") else "base")
    )
    config = ADP_Config(
        seed=seed,
        N_J=300,
        N_phi=12,
        N_loc=20,
        N_lin=50,
        experimental_centers=AdaptiveCentersConfig(lambda_floor=eta) if eta else None,
    )
    stages = []
    cache = {}
    profiler = IndexProfiler()

    def statistics(X, Y, weights, directions, batch_size=32, *, normalized=False):
        blocks = list(weights)
        original = calculate_statistics(
            X, Y, iter(blocks), directions, batch_size, normalized=normalized
        )
        transformed, info = reweight_statistics(X, blocks, directions, original, kind)
        cache.update(
            info=info,
            weights_sha256=digest(*(w for _, w in blocks)),
            directions_sha256=digest(directions),
        )
        return transformed

    def solve(index, stats):
        result = ADP_solver(grassman.solve).fit_current(
            stats, index, normalized=True, lambda_prox=config.lambda_penalty
        )
        stages.append(dict(cache, solver=result.diagnostics))
        return result

    start = time.perf_counter()
    try:
        result = fit_index(
            x,
            y,
            config=config,
            mode="multi",
            index_dim=m,
            solve=solve,
            profiler=profiler,
            statistics_function=statistics,
            trace_indices=True,
        )
    finally:
        profiler.finish()
    return result, stages, time.perf_counter() - start


def worker(phase, link, m, seed, arm):
    with np.load(OUT / phase / "inputs" / f"{seed}.npz") as data:
        x, full_basis, noise = data["x"], data["basis"], data["noise"]
    truth = full_basis[:, :m]
    y, signal_scale = response(x, truth, noise, link)
    method_seed = int(np.random.SeedSequence([seed, 1]).generate_state(1)[0]) % (
        2**31 - 1
    )
    fitted, stages, seconds = fit(x, y, m, method_seed, arm)
    basis, initial = fitted.index.T, fitted.initial_index.T
    row = {
        "phase": phase,
        "link": link,
        "m": m,
        "seed": seed,
        "arm": arm,
        "status": "ok",
        "method_seed": method_seed,
        "data_sha256": digest(x, y, truth),
        "initial_sha256": digest(initial),
        "basis": basis,
        "initial_basis": initial,
        "edr": metrics(basis, truth)["projector_distance"],
        "initial_edr": metrics(initial, truth)["projector_distance"],
        "orthogonality_error": float(np.linalg.norm(basis.T @ basis - np.eye(m))),
        "fit_seconds": seconds,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "selected_iteration": fitted.metadata["selected_iteration"],
        "stop_reason": fitted.metadata["stop_reason"],
        "stages": stages,
        "trace": fitted.metadata["trace"],
    }
    diagnostic_start = time.perf_counter()
    if arm == "base":
        row["certificate"] = certificate_diagnostic(
            x, y, truth, method_seed, link, signal_scale
        )
    row["diagnostic_seconds"] = time.perf_counter() - diagnostic_start
    return row


def freeze(phase, arms):
    folder = OUT / phase
    if folder.exists():
        raise FileExistsError(f"refusing to overwrite frozen phase {folder}")
    (folder / "inputs").mkdir(parents=True)
    (folder / "fits").mkdir()
    inputs = {}
    for seed in SEEDS[phase]:
        x, basis, noise = arrays(seed)
        path = folder / "inputs" / f"{seed}.npz"
        np.savez_compressed(path, x=x, basis=basis, noise=noise)
        inputs[str(seed)] = hashlib.sha256(path.read_bytes()).hexdigest()
    sources = [
        *sorted((ROOT / "ADP").rglob("*.py")),
        Path(__file__),
        ROOT / "benchmarks/experiments/edr_metric_corrections.py",
        ROOT / "benchmarks/experiments/r_sdr/design.py",
        ROOT / "tests/test_edr_metric_corrections.py",
        OUT / "PROTOCOL.md",
        ROOT / "docs/tex/data/gpt_ans.md",
    ]
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sources
    }
    with zipfile.ZipFile(folder / "sources.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, path.relative_to(ROOT))
    jobs = [
        (link, m, seed, arm)
        for link, m in CASES
        for seed in SEEDS[phase]
        for arm in arms
    ]
    np.random.default_rng(815).shuffle(jobs)
    manifest = {
        "phase": phase,
        "arms": arms,
        "seeds": list(SEEDS[phase]),
        "jobs": jobs,
        "sources": hashes,
        "inputs": inputs,
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "workers": 3,
        "blas_threads": 1,
        "timeout_seconds": 180,
    }
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    write_json(folder / "manifest.json", manifest)
    return jobs


def run_job(phase, job):
    link, m, seed, arm = job
    start = time.perf_counter()
    env = dict(
        os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.edr_metric_study",
                "worker",
                phase,
                link,
                str(m),
                str(seed),
                arm,
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr[-6000:])
        row = json.loads(completed.stdout)
    except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError) as error:
        row = {
            "phase": phase,
            "link": link,
            "m": m,
            "seed": seed,
            "arm": arm,
            "status": "failed",
            "error": str(error),
        }
    row["process_seconds"] = time.perf_counter() - start
    row["fingerprint"] = json.loads((OUT / phase / "manifest.json").read_text())[
        "fingerprint"
    ]
    write_json(OUT / phase / "fits" / f"{link}-m{m}-{seed}-{arm}.json", row)
    return row


def run_phase(phase, arms):
    jobs = freeze(phase, arms)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_job, phase, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            if i % 12 == 0 or row["status"] != "ok":
                print(f"{phase} {i}/{len(jobs)} {row['status']}", flush=True)
    return read_rows(phase)


def read_rows(phase):
    return [
        json.loads(p.read_text()) for p in sorted((OUT / phase / "fits").glob("*.json"))
    ]


def bootstrap(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(815)
    means = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def comparison(rows, arm):
    indexed = {(r["link"], r["m"], r["seed"], r["arm"]): r for r in rows}
    summary = []
    for link, m in CASES:
        seeds = sorted({r["seed"] for r in rows})
        pairs = [
            (indexed.get((link, m, s, "base")), indexed.get((link, m, s, arm)))
            for s in seeds
        ]
        valid = [
            (a, b) for a, b in pairs if a and b and a["status"] == b["status"] == "ok"
        ]
        result = {
            "link": link,
            "m": m,
            "arm": arm,
            "expected_pairs": len(seeds),
            "complete_pairs": len(valid),
        }
        if valid:
            base = np.array([a["edr"] for a, _ in valid])
            candidate = np.array([b["edr"] for _, b in valid])
            result.update(
                base_edr=float(base.mean()),
                candidate_edr=float(candidate.mean()),
                ratio_of_means=float(candidate.mean() / base.mean()),
                geometric_ratio=float(np.exp(np.log(candidate / base).mean())),
                delta_ci=bootstrap(candidate - base),
                wins=int((candidate < base).sum()),
                base_initial_edr=float(np.mean([a["initial_edr"] for a, _ in valid])),
                base_fit_seconds=float(np.median([a["fit_seconds"] for a, _ in valid])),
                candidate_fit_seconds=float(
                    np.median([b["fit_seconds"] for _, b in valid])
                ),
                base_process_seconds=float(
                    np.median([a["process_seconds"] for a, _ in valid])
                ),
                candidate_process_seconds=float(
                    np.median([b["process_seconds"] for _, b in valid])
                ),
                base_rss_mib=float(np.median([a["peak_rss_mib"] for a, _ in valid])),
                candidate_rss_mib=float(
                    np.median([b["peak_rss_mib"] for _, b in valid])
                ),
            )
        summary.append(result)
    return summary


def gate(rows, arm):
    summaries = comparison(rows, arm)
    complete = all(r["complete_pairs"] == r["expected_pairs"] for r in summaries)
    primary = [r for r in summaries if r["link"] == "additive" and r["m"] in (3, 5)]
    primary_rows = [r for r in rows if r["link"] == "additive" and r["m"] in (3, 5)]
    seed_deltas = []
    for seed in sorted({r["seed"] for r in rows}):
        values = [r for r in primary_rows if r["seed"] == seed and r["status"] == "ok"]
        if len(values) == 4:
            seed_deltas.append(
                np.mean([r["edr"] for r in values if r["arm"] == arm])
                - np.mean([r["edr"] for r in values if r["arm"] == "base"])
            )
    interval = bootstrap(seed_deltas) if seed_deltas else None
    checks = {
        "complete_pairs": complete,
        "orthogonality": all(r.get("orthogonality_error", 1) < 1e-8 for r in rows),
        "primary_reduction": all(r.get("ratio_of_means", 2) <= 0.95 for r in primary),
        "primary_ci": bool(interval and interval[1] < 0),
        "other_cases": all(
            r.get("ratio_of_means", 2) <= 1.05 for r in summaries if r not in primary
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "primary_delta_ci": interval,
        "cases": summaries,
    }


def run():
    pilot = run_phase("pilot", ARMS)
    scores = {}
    for arm in ARMS[1:]:
        cases = comparison(pilot, arm)
        primary = [r for r in cases if r["link"] == "additive" and r["m"] in (3, 5)]
        if all(r["complete_pairs"] == r["expected_pairs"] for r in cases):
            scores[arm] = float(
                np.mean([np.log(r["geometric_ratio"]) for r in primary])
            )
    if not scores:
        write_json(
            OUT / "decision.json", {"status": "pilot_failed", "validation_run": False}
        )
        return
    candidate = min(scores, key=scores.get)
    decision = {
        "candidate": candidate,
        "pilot_log_ratio_scores": scores,
        "validation_run": False,
    }
    write_json(OUT / "decision.json", decision)
    print(f"Frozen candidate: {candidate}", flush=True)
    selected = run_phase("selection", ["base", candidate])
    decision["selection"] = gate(selected, candidate)
    write_json(OUT / "decision.json", decision)
    if decision["selection"]["passed"]:
        validated = run_phase("validation", ["base", candidate])
        decision["validation_run"] = True
        decision["validation"] = gate(validated, candidate)
        write_json(OUT / "decision.json", decision)
    audit()


def audit():
    """Recompute EDR from saved bases; verify pairing and frozen file hashes."""
    decision = json.loads((OUT / "decision.json").read_text())
    audit_rows, summaries = [], []
    for phase in SEEDS:
        if not (OUT / phase / "manifest.json").exists():
            continue
        manifest = json.loads((OUT / phase / "manifest.json").read_text())
        fingerprint = manifest.pop("fingerprint")
        assert (
            hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
            == fingerprint
        )
        with zipfile.ZipFile(OUT / phase / "sources.zip") as archive:
            for name, expected in manifest["sources"].items():
                assert hashlib.sha256(archive.read(name)).hexdigest() == expected
        for seed, expected in manifest["inputs"].items():
            assert (
                hashlib.sha256(
                    (OUT / phase / "inputs" / f"{seed}.npz").read_bytes()
                ).hexdigest()
                == expected
            )
        rows = read_rows(phase)
        assert len(rows) == len(manifest["jobs"])
        assert len({(r["link"], r["m"], r["seed"], r["arm"]) for r in rows}) == len(
            rows
        )
        for r in rows:
            assert r["fingerprint"] == fingerprint
            if r["status"] != "ok":
                continue
            with np.load(OUT / phase / "inputs" / f"{r['seed']}.npz") as data:
                truth = data["basis"][:, : r["m"]]
                y, _ = response(data["x"], truth, data["noise"], r["link"])
                assert digest(data["x"], y, truth) == r["data_sha256"]
            basis = np.asarray(r["basis"])
            # Direct projector formula, independent of the angles routine used by worker.
            edr = np.linalg.norm(basis - truth @ (truth.T @ basis)) / np.sqrt(r["m"])
            assert abs(edr - r["edr"]) < 2e-12
            base = next(
                b
                for b in rows
                if b["link"] == r["link"]
                and b["m"] == r["m"]
                and b["seed"] == r["seed"]
                and b["arm"] == "base"
            )
            if base["status"] == "ok":
                assert r["initial_sha256"] == base["initial_sha256"]
                assert (
                    r["stages"][0]["weights_sha256"]
                    == base["stages"][0]["weights_sha256"]
                )
                assert (
                    r["stages"][0]["directions_sha256"]
                    == base["stages"][0]["directions_sha256"]
                )
        for arm in manifest["arms"]:
            if arm != "base":
                summaries.extend(dict(phase=phase, **s) for s in comparison(rows, arm))
        audit_rows.append(
            {
                "phase": phase,
                "expected": len(rows),
                "successful": sum(r["status"] == "ok" for r in rows),
                "failed": sum(r["status"] != "ok" for r in rows),
                "hashes_and_saved_edr": "verified",
            }
        )
        if phase != "pilot":
            assert gate(rows, decision["candidate"]) == decision[phase]
    write_json(OUT / "audit.json", audit_rows)
    write_json(OUT / "summary.json", summaries)
    keys = sorted({key for row in summaries for key in row})
    with (OUT / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)
    print(
        json.dumps(
            {
                "audit": audit_rows,
                "candidate": decision.get("candidate"),
                "selection_passed": decision.get("selection", {}).get("passed"),
                "validation_run": decision["validation_run"],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "audit", "worker"))
    parser.add_argument("args", nargs="*")
    args = parser.parse_args()
    if args.command == "worker":
        phase, link, m, seed, arm = args.args
        print(
            json.dumps(
                worker(phase, link, int(m), int(seed), arm),
                default=json_default,
                allow_nan=False,
            )
        )
    elif args.command == "run":
        run()
    else:
        audit()


if __name__ == "__main__":
    main()
