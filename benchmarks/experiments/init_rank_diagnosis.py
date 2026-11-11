"""Reproducible initialization/step-rank diagnosis; never changes ADP defaults."""

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
from scipy.integrate import quad

from ADP.core.ADP_Config import ADP_Config
from ADP.core.ADP_Solver import ADP_solver
from ADP.core.multi.ADP_multi_index import ADP_multi_index
from ADP.engine.common import index_fit, initialize
from ADP.solver import grassman
from experiments.r_sdr.design import Case, generate, metrics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/results/verification/init-rank-20261023"
SAVED = ROOT / "benchmarks/results/outputs/r_sdr/d25-n5000-20261023"
ARMS = ("current", "rank1", "oracle", "budget100", "init500")


def population_opg(m: int) -> np.ndarray:
    """Unscaled E[grad f grad f.T] in true Gaussian coordinates."""

    def expect(f):
        return quad(
            lambda z: f(z) * np.exp(-z * z / 2) / np.sqrt(2 * np.pi),
            -12,
            12,
            epsabs=1e-12,
            epsrel=1e-12,
        )[0]

    a = expect(lambda z: 1 - np.tanh(z) ** 2)
    b = expect(lambda z: (1 - np.tanh(z) ** 2) ** 2)
    mean = np.array([np.exp(-0.5), 0.0] + [0.35 * a] * (m - 2))
    diagonal = np.array([(1 + np.exp(-2)) / 2, 1.0] + [0.35**2 * b] * (m - 2))
    return np.outer(mean, mean) + np.diag(diagonal - mean**2)


def geometry(basis: np.ndarray, truth: np.ndarray) -> dict:
    basis = np.linalg.qr(basis)[0]
    missing = truth - basis @ (basis.T @ truth)
    result = {
        "edr": metrics(basis, truth)["projector_distance"],
        "coordinate_leakage": np.sum(missing**2, axis=0).tolist(),
    }
    k = truth.shape[1] - 2
    if k:
        common = truth[:, 2:].sum(axis=1) / np.sqrt(k)
        result["common_leakage"] = float(
            np.linalg.norm(common - basis @ (basis.T @ common)) ** 2
        )
    if k > 1:
        # Orthonormal basis for sum-zero contrasts among the tanh coordinates.
        contrast = np.linalg.svd(np.ones((1, k)), full_matrices=True)[2][1:].T
        residual = missing[:, 2:] @ contrast
        result["contrast_leakage"] = float(np.sum(residual**2) / (k - 1))
    return result


def paired(rows: list[dict], m: int, arm: str) -> dict:
    selected = [r for r in rows if r["m"] == m and r["arm"] in ("current", arm)]
    keys = [(r["seed"], r["arm"]) for r in selected]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate paired rows")
    by_key = dict(zip(keys, selected, strict=True))
    seeds = sorted({r["seed"] for r in selected if r["arm"] == arm})
    differences = []
    ratios = []
    for seed in seeds:
        a, b = by_key.get((seed, arm), {}), by_key.get((seed, "current"), {})
        if a.get("status") == b.get("status") == "ok":
            differences.append(a["edr"] - b["edr"])
            if "fit_seconds" in a and "fit_seconds" in b:
                ratios.append(a["fit_seconds"] / b["fit_seconds"])
    if not differences:
        return {"n_attempts": len(seeds), "n_pairs": 0}
    delta = np.array(differences)
    rng = np.random.default_rng(20261023)
    bootstrap = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(axis=1)
    return {
        "n_attempts": len(seeds),
        "n_pairs": len(delta),
        "mean_delta": float(delta.mean()),
        "ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "wins": int(np.count_nonzero(delta < 0)),
        "median_time_ratio": float(np.median(ratios)) if ratios else None,
    }


def input_data(m: int, seed: int):
    if m in (2, 3, 5):
        with np.load(SAVED / f"m{m}/inputs/d25-n5000-m{m}-{seed}.npz") as data:
            return data["x"], data["y"], data["truth"]
    return generate(Case(name=f"d25-n5000-m{m}", n=5000, d=25, m=m), seed)


def worker(m: int, seed: int, arm: str) -> dict:
    x, y, truth = input_data(m, seed)
    method_seed = int(np.random.SeedSequence([seed, 1]).generate_state(1)[0]) % (
        2**31 - 1
    )
    config = ADP_Config(
        seed=method_seed,
        N_lin=500 if arm == "init500" else 50,
        N_J=300,
        N_phi=12,
        N_loc=20,
        index_init="local",
        select_step="best",
    )
    initial = {}
    outer = []
    frozen = []
    original_init = index_fit._initial_index
    original_pca = initialize._principal_gradient_basis_with_spectrum

    def capture_pca(gradients, index_dim, *, mass=None):
        basis, spectrum = original_pca(gradients, index_dim, mass=mass)
        initial["spectrum"] = (spectrum / spectrum[0]).tolist()
        w = mass if mass is not None else np.ones(len(gradients))
        opg = gradients.T @ (w[:, None] * gradients) / np.sum(w)
        initial["opg_true_coordinates"] = (truth.T @ opg @ truth).tolist()
        initial["off_subspace_energy_fraction"] = float(
            max(0.0, np.trace(opg) - np.trace(truth.T @ opg @ truth)) / np.trace(opg)
        )
        return basis, spectrum

    def capture_init(*args, **kwargs):
        if arm == "oracle":
            return truth.T.copy(), np.empty(0)
        index, spectrum = original_init(*args, **kwargs)
        initial.update(geometry(index.T, truth))
        return index, spectrum

    def solve(index, U, I, **kwargs):
        mass = kwargs.get("mass")
        if mass is None:
            mass = np.ones(len(I))
        start = grassman._profile(grassman._project(U, index.T), I, mass, 0.0)
        entry = {
            "before": geometry(index.T, truth),
            "before_objective": start.value,
            "zero_centers": int(np.count_nonzero(~np.any(U != 0, axis=(1, 2)))),
            "mass_quantiles": np.quantile(mass, [0, 0.1, 0.5, 0.9, 1]).tolist(),
            "before_local_rank_histogram": np.bincount(
                start.ranks, minlength=m + 1
            ).tolist(),
            "statistics_sha256": hashlib.sha256(
                U.tobytes() + I.tobytes() + mass.tobytes()
            ).hexdigest(),
        }
        if m == 2 and arm == "current":
            svd = grassman._profile_svd(start.M, I, mass, 0.0)
            entry["qr_svd"] = {
                "objective_abs": abs(start.value - svd.value),
                "coefficients_max_abs": float(
                    np.max(np.abs(start.coefficients - svd.coefficients))
                ),
                "gradient_max_abs": float(
                    np.max(
                        np.abs(
                            grassman._gradient(index.T, U, start, mass)
                            - grassman._gradient(index.T, U, svd, mass)
                        )
                    )
                ),
            }
        result = grassman.solve(index, U, I, **kwargs)
        entry.update(after=geometry(result.index.T, truth), solver=result.diagnostics)
        if not outer and arm == "current" and seed % 1000 < 3:
            frozen.append(
                (index.copy(), U.copy(), I.copy(), mass.copy(), kwargs.copy())
            )
        outer.append(entry)
        return result

    model = ADP_multi_index(
        m,
        config=config,
        solver=ADP_solver(
            solve,
            max_steps=100 if arm == "budget100" else 20,
            tol=1e-6,
            rank=1 if arm == "rank1" else None,
            relative_gn=True,
            quadratic_trials=3,
            constant_centers=True,
            smooth_max_steps=None,
            local_ridge=0.0,
        ),
    )
    started = time.perf_counter()
    with (
        patch.object(index_fit, "_initial_index", capture_init),
        patch.object(
            initialize, "_principal_gradient_basis_with_spectrum", capture_pca
        ),
    ):
        model.fit(x, y)
    elapsed = time.perf_counter() - started
    if frozen:
        frozen_index, U, I, mass, settings = frozen[0]
        (OUT / "frozen").mkdir(exist_ok=True)
        np.savez_compressed(
            OUT / "frozen" / f"m{m}-{seed}.npz",
            U=U,
            I=I,
            mass=mass,
            index=frozen_index,
            truth=truth,
        )
        alternatives = {}
        for name, extra in (("rank1", {"rank": 1}), ("budget100", {"max_steps": 100})):
            alt = grassman.solve(frozen_index, U, I, **(settings | extra))
            alternatives[name] = {
                "geometry": geometry(alt.index.T, truth),
                "solver": alt.diagnostics,
            }
        outer[0]["frozen_alternatives"] = alternatives
        outer[0]["truth_frozen_objective"] = grassman._profile(
            grassman._project(U, truth), I, mass, 0.0
        ).value
    initial.update(geometry(model.result_.beta_init, truth))
    trace = [
        {
            "iteration": t["iteration"],
            "h": t["h"],
            "alpha": t["factor"],
            "selection_error": t["err"],
            "eigenvalues": t["eigenvalues"],
            **geometry(t["basis"], truth),
        }
        for t in model.trace_
    ]
    selected = model.effective_parameters_["selected_iteration"]
    assert selected == int(np.argmin([t["selection_error"] for t in trace]))
    metric = geometry(model.basis_, truth)
    row = {
        "m": m,
        "seed": seed,
        "method_seed": method_seed,
        "arm": arm,
        "status": "ok",
        "data_sha256": hashlib.sha256(
            x.tobytes() + y.tobytes() + truth.tobytes()
        ).hexdigest(),
        **metric,
        "initial": initial,
        "trace": trace,
        "outer": outer,
        "last_edr": trace[-1]["edr"],
        "oracle_min_trace_edr": min(t["edr"] for t in trace),
        "selected_iteration": selected,
        "fit_seconds": elapsed,
        "profile": model.profile_,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "stop_reason": model.result_.stop_reason,
    }
    if m in (2, 3, 5):
        saved = [
            json.loads(line)
            for line in (SAVED / f"m{m}/runs.jsonl").read_text().splitlines()
        ]
        old = next(
            r
            for r in saved
            if r["seed"] == seed and r["method"] == "ADP-Grassmann-combined"
        )
        assert row["data_sha256"] == old["data_sha256"]
        if arm == "current":
            row["baseline_abs_difference"] = abs(row["edr"] - old["projector_distance"])
            assert row["baseline_abs_difference"] < 1e-10
    return row


def tasks():
    jobs = []
    for m in (2, 3, 5, 4, 8):
        for offset in range(100 if m in (2, 3, 5) else 20):
            seed = (412000 if m in (2, 3, 5) else 413000) + offset
            for arm in ARMS if offset < 20 else ARMS[:2]:
                jobs.append((m, seed, arm))
    np.random.default_rng(20261023).shuffle(jobs)
    return jobs


def run():
    OUT.mkdir(exist_ok=True)
    for name in ("fits", "frozen"):
        (OUT / name).mkdir(exist_ok=True)
    paths = sorted((ROOT / "ADP").rglob("*.py")) + [
        Path(__file__),
        ROOT / "benchmarks/experiments/r_sdr/design.py",
        OUT / "PROTOCOL.md",
    ]
    sources = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }
    old_manifest = json.loads((SAVED / "m3/manifest.json").read_text())["protocol"][
        "sources"
    ]
    assert all(
        sources[k] == v
        for k, v in old_manifest.items()
        if k.startswith("ADP/") and k in sources
    )
    manifest = {
        "sources": sources,
        "tasks": tasks(),
        "threads": 1,
        "workers": 3,
        "timeout": 300,
        "numpy": np.__version__,
    }
    fingerprint = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    manifest["fingerprint"] = fingerprint
    target = OUT / "manifest.json"
    if target.exists():
        assert json.loads(target.read_text())["fingerprint"] == fingerprint, (
            "manifest changed"
        )
    else:
        target.write_text(json.dumps(manifest, indent=2))
        with zipfile.ZipFile(OUT / "sources.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                archive.write(path, str(path.relative_to(ROOT)))
    env = os.environ.copy()
    for key in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        env[key] = "1"

    def launch(job):
        m, seed, arm = job
        path = OUT / "fits" / f"m{m}-{seed}-{arm}.json"
        if path.exists() and json.loads(path.read_text()).get("status") == "ok":
            return "cached"
        start = time.perf_counter()
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.init_rank_diagnosis",
                    "worker",
                    str(m),
                    str(seed),
                    arm,
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
                    "m": m,
                    "seed": seed,
                    "arm": arm,
                    "status": "failed",
                    "error": result.stderr[-4000:],
                }
            )
        except subprocess.TimeoutExpired:
            row = {"m": m, "seed": seed, "arm": arm, "status": "timeout"}
        row["process_seconds"] = time.perf_counter() - start
        row["fingerprint"] = fingerprint
        path.write_text(json.dumps(row, indent=2, allow_nan=False))
        return row["status"]

    jobs = tasks()
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(launch, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            status = future.result()
            if i % 20 == 0 or status not in ("ok", "cached"):
                print(f"{i}/{len(jobs)}: {status}", flush=True)
    analyze()


def analyze():
    rows = [json.loads(p.read_text()) for p in sorted((OUT / "fits").glob("*.json"))]
    summary = []
    comparisons = []
    for m in (2, 3, 4, 5, 8):
        for arm in ARMS:
            all_rows = [r for r in rows if r["m"] == m and r["arm"] == arm]
            good = [r for r in all_rows if r["status"] == "ok"]
            if not good:
                continue
            summary.append(
                {
                    "m": m,
                    "arm": arm,
                    "attempts": len(all_rows),
                    "ok": len(good),
                    "mean_edr": float(np.mean([r["edr"] for r in good])),
                    "median_edr": float(np.median([r["edr"] for r in good])),
                    "mean_init_edr": float(
                        np.mean([r["initial"]["edr"] for r in good])
                    ),
                    "mean_last_edr": float(np.mean([r["last_edr"] for r in good])),
                    "mean_oracle_min_trace_edr": float(
                        np.mean([r["oracle_min_trace_edr"] for r in good])
                    ),
                    "recovery_count": sum(r["edr"] <= 0.1 for r in good),
                    "median_fit_seconds": float(
                        np.median([r["fit_seconds"] for r in good])
                    ),
                    "median_process_seconds": float(
                        np.median([r["process_seconds"] for r in good])
                    ),
                    "median_rss_mib": float(
                        np.median([r["peak_rss_mib"] for r in good])
                    ),
                    "all_inner_certified": sum(
                        all(t["solver"]["converged"] for t in r["outer"]) for r in good
                    ),
                    "max_orthogonality_error": max(
                        t["solver"]["orthogonality_error"]
                        for r in good
                        for t in r["outer"]
                    ),
                }
            )
            if arm != "current":
                comparisons.append({"m": m, "arm": arm, **paired(rows, m, arm)})
    output = {
        "completed": len(rows),
        "planned": len(tasks()),
        "summary": summary,
        "paired": comparisons,
        "failures": [
            {k: r.get(k) for k in ("m", "seed", "arm", "status", "error")}
            for r in rows
            if r["status"] != "ok"
        ],
        "population_opg": {
            str(m): {
                "matrix": population_opg(m).tolist(),
                "eigenvalues": np.linalg.eigvalsh(population_opg(m))[::-1].tolist(),
            }
            for m in (2, 3, 4, 5, 8)
        },
    }
    (OUT / "summary.json").write_text(json.dumps(output, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                "completed": len(rows),
                "failures": len(output["failures"]),
                "summary": summary,
                "paired": comparisons,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=("run", "worker", "analyze"))
    parser.add_argument("args", nargs="*")
    args = parser.parse_args()
    if args.command == "worker":
        print(
            json.dumps(
                worker(int(args.args[0]), int(args.args[1]), args.args[2]),
                default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x),
                allow_nan=False,
            )
        )
    elif args.command == "run":
        run()
    else:
        analyze()
