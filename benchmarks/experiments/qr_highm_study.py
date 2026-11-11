"""Paired plain full fits and frozen checks for general QR local profiles."""

from __future__ import annotations

import argparse
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
from unittest.mock import patch

import numpy as np

from ADP.core.ADP_Config import ADP_Config
from ADP.core.ADP_Solver import ADP_solver
from ADP.core.multi.ADP_multi_index import ADP_multi_index
from ADP.solver import grassman
from experiments import qr_profile as qr
from experiments.init_rank_diagnosis import geometry, input_data

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/results/verification/qr-highm-20261023"
PREVIOUS = ROOT / "benchmarks/results/verification/init-rank-20261023"


def jobs():
    result = []
    for m in (2, 3, 4, 5, 8):
        for offset in range(100 if m in (3, 5) else 20):
            seed = (412000 if m in (2, 3, 5) else 413000) + offset
            result.extend((m, seed, arm) for arm in ("current", "qr"))
    np.random.default_rng(20261024).shuffle(result)
    return result


def worker(m, seed, arm):
    x, y, truth = input_data(m, seed)
    prior = json.loads((PREVIOUS / "fits" / f"m{m}-{seed}-current.json").read_text())
    digest = hashlib.sha256(x.tobytes() + y.tobytes() + truth.tobytes()).hexdigest()
    assert digest == prior["data_sha256"]
    method_seed = int(np.random.SeedSequence([seed, 1]).generate_state(1)[0]) % (
        2**31 - 1
    )
    config = ADP_Config(seed=method_seed, N_lin=50, N_J=300, N_phi=12, N_loc=20)
    model = ADP_multi_index(
        m,
        config=config,
        solver=ADP_solver(
            grassman.solve,
            max_steps=20,
            tol=1e-6,
            local_ridge=0.0,
            relative_gn=True,
            quadratic_trials=3,
            constant_centers=True,
            smooth_max_steps=None,
        ),
    )
    qr.reset_counts()

    def current(M, I, mass, tau):
        qr.COUNTS["calls"] += 1
        return qr.ORIGINAL_PROFILE(M, I, mass, tau)

    started = time.perf_counter()
    with patch.object(grassman, "_profile", qr.profile if arm == "qr" else current):
        model.fit(x, y)
    elapsed = time.perf_counter() - started
    metric = geometry(model.basis_, truth)
    result = {
        "m": m,
        "seed": seed,
        "arm": arm,
        "status": "ok",
        "data_sha256": digest,
        "initial_sha256": hashlib.sha256(model.result_.beta_init.tobytes()).hexdigest(),
        "initial": geometry(model.result_.beta_init, truth),
        **metric,
        "basis": model.basis_.tolist(),
        "fit_seconds": elapsed,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "profile": model.profile_,
        "counts": dict(qr.COUNTS),
        "stop_reason": model.result_.stop_reason,
        "selected_iteration": model.effective_parameters_["selected_iteration"],
        "trace": [
            {
                "edr": geometry(t["basis"], truth)["edr"],
                "h": t["h"],
                "selection_error": t["err"],
                "solver": t["solver"],
            }
            for t in model.trace_
        ],
    }
    if arm == "current":
        result["replay_edr_difference"] = abs(metric["edr"] - prior["edr"])
        assert result["replay_edr_difference"] < 1e-10
    return result


def freeze_sources():
    sources = sorted((ROOT / "ADP").rglob("*.py")) + [
        Path(__file__),
        ROOT / "benchmarks/experiments/qr_profile.py",
        ROOT / "benchmarks/experiments/init_rank_diagnosis.py",
        ROOT / "benchmarks/experiments/r_sdr/design.py",
        OUT / "PROTOCOL.md",
    ]
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sources
    }
    old = json.loads((PREVIOUS / "manifest.json").read_text())["sources"]
    assert all(hashes[k] == v for k, v in old.items() if k.startswith("ADP/"))
    content = {
        "sources": hashes,
        "jobs": jobs(),
        "workers": 3,
        "threads": 1,
        "timeout": 300,
        "numpy": np.__version__,
    }
    fingerprint = hashlib.sha256(
        json.dumps(content, sort_keys=True).encode()
    ).hexdigest()
    content["fingerprint"] = fingerprint
    manifest = OUT / "manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text())["fingerprint"] == fingerprint, (
            "manifest changed"
        )
    else:
        manifest.write_text(json.dumps(content, indent=2))
        with zipfile.ZipFile(OUT / "sources.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sources:
                archive.write(path, str(path.relative_to(ROOT)))
    return fingerprint


def run():
    (OUT / "fits").mkdir(exist_ok=True)
    fingerprint = freeze_sources()
    environment = os.environ.copy()
    for key in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[key] = "1"

    def launch(job):
        m, seed, arm = job
        path = OUT / "fits" / f"m{m}-{seed}-{arm}.json"
        if path.exists():
            existing = json.loads(path.read_text())
            assert existing["fingerprint"] == fingerprint
            if existing["status"] == "ok":
                return "cached"
        start = time.perf_counter()
        try:
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.qr_highm_study",
                    "worker",
                    str(m),
                    str(seed),
                    arm,
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            row = (
                json.loads(process.stdout)
                if process.returncode == 0
                else {
                    "m": m,
                    "seed": seed,
                    "arm": arm,
                    "status": "failed",
                    "error": process.stderr[-4000:],
                }
            )
        except subprocess.TimeoutExpired:
            row = {"m": m, "seed": seed, "arm": arm, "status": "timeout"}
        row.update(fingerprint=fingerprint, process_seconds=time.perf_counter() - start)
        path.write_text(json.dumps(row, indent=2, allow_nan=False))
        return row["status"]

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(launch, job) for job in jobs()]
        for i, future in enumerate(as_completed(futures), 1):
            status = future.result()
            if i % 20 == 0 or status not in ("ok", "cached"):
                print(f"{i}/{len(jobs())}: {status}", flush=True)
    analyze()


def frozen():
    """Alternating warm timings and numerical checks; run before full study."""
    rows = []
    for path in sorted((PREVIOUS / "frozen").glob("*.npz")):
        with np.load(path) as data:
            U, I, mass, index, truth = (
                data[k] for k in ("U", "I", "mass", "index", "truth")
            )
        M = grassman._project(U, index.T)
        reference = qr.ORIGINAL_PROFILE(M, I, mass, 0.0)
        candidate = qr.profile(M, I, mass, 0.0)
        np.testing.assert_allclose(
            candidate.coefficients, reference.coefficients, rtol=1e-10, atol=1e-10
        )
        np.testing.assert_array_equal(candidate.ranks, reference.ranks)
        assert candidate.smooth == reference.smooth
        assert abs(candidate.value - reference.value) < 1e-10 * max(
            1.0, reference.value
        )
        profile_times = {"current": [], "qr": []}
        solve_times = {"current": [], "qr": []}
        solutions = {}
        for repetition in range(10):
            order = ("current", "qr") if repetition % 2 == 0 else ("qr", "current")
            for arm in order:
                function = qr.ORIGINAL_PROFILE if arm == "current" else qr.profile
                start = time.perf_counter()
                for _ in range(10):
                    function(M, I, mass, 0.0)
                profile_times[arm].append((time.perf_counter() - start) / 10)
                if repetition < 4:
                    with patch.object(grassman, "_profile", function):
                        start = time.perf_counter()
                        result = grassman.solve(index, U, I, mass=mass)
                        solve_times[arm].append(time.perf_counter() - start)
                    solutions[arm] = {
                        "edr": geometry(result.index.T, truth)["edr"],
                        "solver": result.diagnostics,
                    }
        rows.append(
            {
                "name": path.stem,
                "m": len(index),
                "coefficient_max_difference": float(
                    np.max(np.abs(candidate.coefficients - reference.coefficients))
                ),
                "gradient_max_difference": float(
                    np.max(
                        np.abs(
                            grassman._gradient(index.T, U, candidate, mass)
                            - grassman._gradient(index.T, U, reference, mass)
                        )
                    )
                ),
                "objective_difference": candidate.value - reference.value,
                "profile_times": profile_times,
                "solve_times": solve_times,
                "solutions": solutions,
            }
        )
    assert len(rows) == 15
    (OUT / "frozen.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
    print("15 frozen numerical checks passed", flush=True)


def analyze():
    manifest = json.loads((OUT / "manifest.json").read_text())
    rows = [json.loads(p.read_text()) for p in sorted((OUT / "fits").glob("*.json"))]
    expected = {tuple(job) for job in manifest["jobs"]}
    keys = [(r["m"], r["seed"], r["arm"]) for r in rows]
    assert len(keys) == len(set(keys)) and set(keys) <= expected
    assert all(r["fingerprint"] == manifest["fingerprint"] for r in rows)
    assert all(
        hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value
        for name, value in manifest["sources"].items()
    )
    by_key = dict(zip(keys, rows, strict=True))
    summary = []
    for m in (2, 3, 4, 5, 8):
        planned = [t[1] for t in manifest["jobs"] if t[0] == m and t[2] == "current"]
        pairs = []
        for seed in planned:
            current, candidate = (
                by_key.get((m, seed, "current")),
                by_key.get((m, seed, "qr")),
            )
            if (
                current
                and candidate
                and current["status"] == candidate["status"] == "ok"
            ):
                assert current["data_sha256"] == candidate["data_sha256"]
                assert current["initial_sha256"] == candidate["initial_sha256"]
                pairs.append((current, candidate))
        if not pairs:
            continue
        delta = np.array([b["edr"] - a["edr"] for a, b in pairs])
        rng = np.random.default_rng(20261024)
        blocks = rng.integers(0, len(pairs), size=(10000, len(pairs)))
        time_ratio = np.array([b["fit_seconds"] / a["fit_seconds"] for a, b in pairs])
        process_ratio = np.array(
            [b["process_seconds"] / a["process_seconds"] for a, b in pairs]
        )
        candidates = [b for _, b in pairs]
        qr_count = sum(b["counts"]["qr_centers"] for b in candidates)
        svd_count = sum(b["counts"]["svd_centers"] for b in candidates)
        summary.append(
            {
                "m": m,
                "planned_pairs": len(planned),
                "pairs": len(pairs),
                "current_mean_edr": float(np.mean([a["edr"] for a, _ in pairs])),
                "qr_mean_edr": float(np.mean([b["edr"] for _, b in pairs])),
                "mean_delta": float(delta.mean()),
                "delta_ci": np.quantile(
                    delta[blocks].mean(axis=1), [0.025, 0.975]
                ).tolist(),
                "max_abs_delta": float(np.max(np.abs(delta))),
                "wins": int(np.count_nonzero(delta < 0)),
                "geomean_fit_ratio": float(np.exp(np.log(time_ratio).mean())),
                "fit_ratio_ci": np.exp(
                    np.quantile(np.log(time_ratio)[blocks].mean(axis=1), [0.025, 0.975])
                ).tolist(),
                "median_fit_ratio": float(np.median(time_ratio)),
                "median_process_ratio": float(np.median(process_ratio)),
                "current_median_fit_seconds": float(
                    np.median([a["fit_seconds"] for a, _ in pairs])
                ),
                "qr_median_fit_seconds": float(
                    np.median([b["fit_seconds"] for _, b in pairs])
                ),
                "current_median_process_seconds": float(
                    np.median([a["process_seconds"] for a, _ in pairs])
                ),
                "qr_median_process_seconds": float(
                    np.median([b["process_seconds"] for _, b in pairs])
                ),
                "current_median_rss_mib": float(
                    np.median([a["peak_rss_mib"] for a, _ in pairs])
                ),
                "qr_median_rss_mib": float(
                    np.median([b["peak_rss_mib"] for _, b in pairs])
                ),
                "general_qr_center_fraction": qr_count / (qr_count + svd_count)
                if qr_count + svd_count
                else None,
                "current_recovery": sum(a["edr"] <= 0.1 for a, _ in pairs),
                "qr_recovery": sum(b["edr"] <= 0.1 for _, b in pairs),
                "current_certified_calls": sum(
                    t["solver"]["converged"] for a, _ in pairs for t in a["trace"]
                ),
                "qr_certified_calls": sum(
                    t["solver"]["converged"] for _, b in pairs for t in b["trace"]
                ),
            }
        )
    result = {
        "completed": len(rows),
        "planned": len(expected),
        "summary": summary,
        "failures": [r for r in rows if r["status"] != "ok"],
        "max_baseline_difference": max(
            (r.get("replay_edr_difference", 0.0) for r in rows), default=0.0
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=("run", "worker", "frozen", "analyze"))
    parser.add_argument("args", nargs="*")
    args = parser.parse_args()
    if args.command == "worker":
        print(
            json.dumps(
                worker(int(args.args[0]), int(args.args[1]), args.args[2]),
                allow_nan=False,
            )
        )
    else:
        {"run": run, "frozen": frozen, "analyze": analyze}[args.command]()
