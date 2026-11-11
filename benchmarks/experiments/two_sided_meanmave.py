"""Fixed paired comparison of saved two-sided ADP and package MEANMAVE."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import tempfile
import time
import tracemalloc
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from experiments import anisotropic_geometry_study as geometry_study
from experiments.r_sdr.protocol import ROOT, THREAD_ENV, environment
from experiments.r_sdr.runner import read_worker_meta, run_process

OUT = ROOT / "benchmarks/results/verification/two-sided-meanmave-20261026"
BASE_OUT = ROOT / "benchmarks/results/verification/anisotropic-metrics-20261026"
SEED_VALUES = tuple(range(1286100, 1286110))
CELLS = tuple(itertools.product(("additive", "quadratic", "sine"), (3, 5)))
ARMS = ("two_sided", "MEANMAVE")
BASELINE_ARM = "two_sided"
R_METHOD = "MAVE-package"
R_SCRIPT = "Rscript"
BOOTSTRAP_DRAWS = 50_000
BOOTSTRAP_SEED = 1_286_026


class TimeOnlyProfiler:
    """Match stage timing while leaving tracemalloc disabled."""

    def __init__(self):
        self.times = {}

    @contextmanager
    def stage(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.times[name] = self.times.get(name, 0.0) + time.perf_counter() - started

    def finish(self):
        pass

    def model_profile(self):
        return {"allocation_tracing": False, "stage_seconds": self.times}


def default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path: Path, value):
    path.write_text(
        json.dumps(value, indent=2, default=default, allow_nan=False) + "\n"
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def array_sha(*arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def method_seed(seed: int) -> int:
    return int(np.random.SeedSequence([seed, 1]).generate_state(1)[0]) % (2**31 - 1)


def input_path(link: str, m: int, seed: int) -> Path:
    return BASE_OUT / "validation/inputs" / f"{link}-m{m}-{seed}.npz"


def baseline_path(link: str, m: int, seed: int) -> Path:
    return BASE_OUT / "validation/fits" / f"{link}-m{m}-{seed}-two_sided.json"


def source_map():
    validation_manifest = json.loads(
        (BASE_OUT / "validation/manifest.json").read_text()
    )
    paths = {ROOT / rel for rel in validation_manifest["sources"]}
    paths.update(
        {
            ROOT / "benchmarks/experiments/two_sided_meanmave.py",
            ROOT / "benchmarks/experiments/r_sdr/protocol.py",
            ROOT / "benchmarks/experiments/r_sdr/runner.py",
            ROOT / "benchmarks/experiments/r_sdr/worker.R",
            ROOT / "benchmarks/experiments/r_sdr/methods.R",
            ROOT / "tests/test_two_sided_meanmave.py",
            OUT / "PROTOCOL.md",
        }
    )
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(paths)}


def verify_frozen(manifest):
    validation_manifest = json.loads(
        (BASE_OUT / "validation/manifest.json").read_text()
    )
    geometry_study.verify(BASE_OUT / "validation", validation_manifest)
    if (
        sha(BASE_OUT / "validation/manifest.json")
        != manifest["validation_manifest_sha256"]
    ):
        raise RuntimeError("validation manifest changed")
    if (
        sha(BASE_OUT / "validation/independent_audit.json")
        != manifest["validation_audit_sha256"]
    ):
        raise RuntimeError("validation audit changed")
    for relative, expected in manifest["sources"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f"frozen source drift: {relative}")
    for row in manifest["baseline_rows"]:
        path = ROOT / row["path"]
        if sha(path) != row["sha256"]:
            raise RuntimeError(f"saved baseline row drift: {path}")
    for relative, expected in manifest["inputs"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f"saved input drift: {relative}")


def freeze():
    if OUT.exists():
        present = sorted(path.name for path in OUT.iterdir())
        if present != ["PROTOCOL.md"]:
            raise FileExistsError(f"refusing to overwrite non-protocol output {OUT}")
    else:
        OUT.mkdir(parents=True)
    validation_dir = BASE_OUT / "validation"
    validation_manifest = json.loads((validation_dir / "manifest.json").read_text())
    geometry_study.verify(validation_dir, validation_manifest)
    validation_audit_path = validation_dir / "independent_audit.json"
    validation_audit = json.loads(validation_audit_path.read_text())
    selected = json.loads((BASE_OUT / "selection/summary.json").read_text())
    if (
        not validation_audit.get("integrity_ok")
        or validation_audit.get("agreed_passed_candidates")
        != selected.get("passed_candidates")
        or not validation_audit.get("agreed_passed_candidates")
    ):
        raise RuntimeError("independent validation audit is not passed")

    (OUT / "fits").mkdir(parents=True, exist_ok=False)
    (OUT / "raw").mkdir()
    expected_inputs, baseline_rows, jobs = {}, [], []
    for seed in SEED_VALUES:
        for link, m in CELLS:
            ipath, bpath = input_path(link, m, seed), baseline_path(link, m, seed)
            key = f"{link}-m{m}-{seed}"
            baseline = json.loads(bpath.read_text())
            if (
                baseline.get("status") != "ok"
                or baseline.get("arm") != "two_sided"
                or baseline.get("fingerprint") != validation_manifest["fingerprint"]
            ):
                raise RuntimeError(f"invalid saved two-sided baseline: {bpath}")
            if (
                baseline.get("seed") != seed
                or baseline.get("m") != m
                or baseline.get("link") != link
            ):
                raise RuntimeError(f"baseline key mismatch: {bpath}")
            with np.load(ipath) as saved:
                if array_sha(saved["x"], saved["y"], saved["truth"]) != baseline.get(
                    "data_sha256"
                ):
                    raise RuntimeError(f"baseline/input content mismatch: {bpath}")
            expected_inputs[str(ipath.relative_to(ROOT))] = sha(ipath)
            baseline_rows.append(
                {
                    "key": key,
                    "path": str(bpath.relative_to(ROOT)),
                    "sha256": sha(bpath),
                    "input_sha256": sha(ipath),
                    "edr": float(baseline["edr"]),
                    "selected_iteration": int(baseline["selected_iteration"]),
                    "stage_iterations": [
                        int(stage["iterations"]) for stage in baseline["stages"]
                    ],
                    "stage_convergence": [
                        bool(stage["converged"]) for stage in baseline["stages"]
                    ],
                    "method_seed": int(baseline["method_seed"]),
                    "initial_sha256": baseline["initial_sha256"],
                    "first_weights_sha256": baseline["first_weights_sha256"],
                    "first_directions_sha256": baseline["first_directions_sha256"],
                    "first_moments_sha256": baseline["first_moments_sha256"],
                    "first_tau": float(baseline["first_tau"]),
                }
            )
            arms = list(ARMS)
            jobs.append((link, m, seed, arms))

    rng = np.random.default_rng(1_286_026)
    for job in jobs:
        rng.shuffle(job[3])

    r_environment, r_env = environment(R_SCRIPT, 1)
    sources = source_map()
    for relative in sources:
        if sha(ROOT / relative) != sources[relative]:
            raise RuntimeError(f"source changed while freezing: {relative}")
    with zipfile.ZipFile(OUT / "sources.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in sources:
            archive.write(ROOT / relative, relative)
    (OUT / "source").mkdir()
    with zipfile.ZipFile(OUT / "sources.zip") as archive:
        archive.extractall(OUT / "source")
    input_hashes = expected_inputs
    manifest = {
        "schema": 1,
        "purpose": "fixed external comparison; no candidate selection or promotion",
        "arms": list(ARMS),
        "cells": [list(cell) for cell in CELLS],
        "seeds": list(SEED_VALUES),
        "planned_fits": len(CELLS) * len(SEED_VALUES) * len(ARMS),
        "jobs_by_pair": [
            {"link": link, "m": m, "seed": seed, "order": arms}
            for link, m, seed, arms in jobs
        ],
        "inputs": input_hashes,
        "baseline_rows": baseline_rows,
        "validation_manifest_sha256": sha(validation_dir / "manifest.json"),
        "validation_summary_sha256": sha(validation_dir / "summary.json"),
        "validation_audit_sha256": sha(validation_audit_path),
        "selection_summary_sha256": sha(BASE_OUT / "selection/summary.json"),
        "validation_source_hashes": validation_manifest["sources"],
        "sources": sources,
        "source_archive_sha256": sha(OUT / "sources.zip"),
        "r_environment": r_environment,
        "rscript": R_SCRIPT,
        "r_method": 'MAVE::mave.compute(method="MEANMAVE", max.dim=m, screen=d)',
        "r_library_version": "1.3.12",
        "python": sys.version,
        "timeout_seconds": 120,
        "memory_limit_mib": 2048,
        "workers": 1,
        "blas_threads": 1,
        "edr": "||P_est-P_true||_F/sqrt(2m); recovered iff EDR<=0.1",
        "edr_difference_orientation": "two_sided - MEANMAVE; positive favors MEANMAVE",
        "time_ratio_orientation": "MEANMAVE / two_sided; >1 means two_sided faster",
        "bootstrap": {
            "resampling_unit": "10 seed clusters; each has all six cells",
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "confidence": 0.95,
        },
        "randomized_method_order_seed": 1_286_026,
        "protocol_sha256": sha(OUT / "PROTOCOL.md"),
        "r_environment_overrides": {
            key: r_env[key]
            for key in (*THREAD_ENV, "OMP_DYNAMIC", "MKL_DYNAMIC", "R_LIBS_USER")
            if key in r_env
        },
    }
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, default=default).encode()
    ).hexdigest()
    write_json(OUT / "manifest.json", manifest)
    return manifest


def worker(arm: str, link: str, m: int, seed: int):
    input_file = input_path(link, m, seed)
    baseline_file = baseline_path(link, m, seed)
    baseline = json.loads(baseline_file.read_text())
    with np.load(input_file) as saved:
        x, y, truth = saved["x"], saved["y"], saved["truth"]
    expected_seed = int(baseline["method_seed"])
    if expected_seed != method_seed(seed):
        raise RuntimeError("saved method_seed differs from expected deterministic seed")

    traced_on_start = tracemalloc.is_tracing()
    if traced_on_start:
        tracemalloc.stop()
    tracing_before_fit = tracemalloc.is_tracing()
    r_result = None
    r_error = None
    fit_profile = None
    initial_basis = None
    initial_sha256 = None
    selected_iteration = None
    stop_reason = None
    stages = None
    method_metadata = {}

    if arm == "two_sided":
        geometry_study.IndexProfiler = TimeOnlyProfiler
        fit = geometry_study.fit(x, y, truth, seed, "two_sided")
        basis = np.asarray(fit["basis"])
        initial_basis = np.asarray(fit["initial_basis"])
        initial_sha256 = fit["initial_sha256"]
        selected_iteration = int(fit["selected_iteration"])
        stop_reason = fit["stop_reason"]
        stages = fit["stages"]
        fit_profile = fit["profile"]
        fit_seconds = float(fit["fit_seconds"])
        method_metadata = {
            "method_seed": int(fit["method_seed"]),
            "first_weights_sha256": fit["first_weights_sha256"],
            "first_directions_sha256": fit["first_directions_sha256"],
            "first_moments_sha256": fit["first_moments_sha256"],
            "first_tau": float(fit["first_tau"]),
            "stage_iterations": [int(stage["iterations"]) for stage in stages],
            "stage_convergence": [bool(stage["converged"]) for stage in stages],
        }
    elif arm == "MEANMAVE":
        with tempfile.TemporaryDirectory(prefix="two-sided-meanmave-") as tmp:
            work = Path(tmp)
            x_path, y_path = work / "x.csv", work / "y.csv"
            basis_path, meta_path = work / "basis.csv", work / "meta.tsv"
            log_path = work / "R-worker.log"
            np.savetxt(x_path, x, delimiter=",", fmt="%.17g")
            np.savetxt(y_path, y, delimiter=",", fmt="%.17g")
            command = [
                R_SCRIPT,
                "--vanilla",
                str(OUT / "source/experiments/r_sdr/worker.R"),
                R_METHOD,
                str(x_path),
                str(y_path),
                str(basis_path),
                str(meta_path),
                str(expected_seed),
                str(m),
                "20",
                "1e-5",
            ]
            r_result = run_process(
                command,
                {**os.environ, **json.loads((OUT / "r_env.json").read_text())},
                log_path,
                timeout=120,
                memory_mib=2048,
                cwd=OUT / "source",
            )
            if r_result["status"] != "ok":
                r_error = (
                    f"MEANMAVE {r_result['status']}: {log_path.read_text()[-4000:]}"
                )
            else:
                basis = np.loadtxt(basis_path, delimiter=",", ndmin=2)
                meta = read_worker_meta(R_METHOD, meta_path)
                fit_seconds = float(meta["fit_seconds"])
                method_metadata = {
                    "r_metadata": meta,
                    "package_convergence": "package_not_exposed",
                    "package_iterations": "package_not_exposed",
                }
    else:
        raise ValueError(f"unknown arm: {arm}")

    tracing_after_fit = tracemalloc.is_tracing()
    if tracing_after_fit:
        tracemalloc.stop()
        raise RuntimeError("allocation tracing unexpectedly active during fit")
    if r_error is not None:
        return {
            "link": link,
            "m": m,
            "seed": seed,
            "arm": arm,
            "status": r_result["status"],
            "input_path": display_path(input_file),
            "input_sha256": sha(input_file),
            "reference_row_path": display_path(baseline_file),
            "reference_row_sha256": sha(baseline_file),
            "method_seed": expected_seed,
            "r_process": r_result,
            "error": r_error,
            "tracemalloc_tracing_at_worker_start": traced_on_start,
            "tracemalloc_tracing_before_fit": tracing_before_fit,
            "tracemalloc_tracing_after_fit": tracing_after_fit,
        }
    basis = np.asarray(basis, dtype=float)
    if basis.shape != (x.shape[1], m) or not np.isfinite(basis).all():
        raise ValueError(f"invalid basis: shape={basis.shape}")
    if not np.allclose(basis.T @ basis, np.eye(m), atol=1e-10, rtol=0):
        raise ValueError("returned basis is not orthonormal")

    raw_path = OUT / "raw" / f"{link}-m{m}-{seed}-{arm}.npz"
    payload = {"basis": basis}
    if initial_basis is not None:
        payload["initial_basis"] = initial_basis
    temporary_raw_path = raw_path.with_name(f".{raw_path.name}.tmp.npz")
    np.savez_compressed(temporary_raw_path, **payload)
    temporary_raw_path.replace(raw_path)
    edr = float(geometry_study.metrics(basis, truth)["projector_distance"])
    saved_basis = np.asarray(baseline["basis"], dtype=float)
    saved_basis_edr = float(
        geometry_study.metrics(saved_basis, truth)["projector_distance"]
    )
    projector_distance_to_saved = float(
        geometry_study.metrics(basis, saved_basis)["projector_distance"]
    )
    output = {
        "link": link,
        "m": m,
        "seed": seed,
        "arm": arm,
        "status": "ok",
        "fingerprint": json.loads((OUT / "manifest.json").read_text())["fingerprint"],
        "input_path": display_path(input_file),
        "input_sha256": sha(input_file),
        "reference_row_path": display_path(baseline_file),
        "reference_row_sha256": sha(baseline_file),
        "reference_edr_saved": float(baseline["edr"]),
        "reference_edr_recomputed": saved_basis_edr,
        "reference_self_edr_error": abs(saved_basis_edr - float(baseline["edr"])),
        "fit_seconds": fit_seconds,
        "edr": edr,
        "recovered": bool(edr <= 0.1),
        "basis_path": display_path(raw_path),
        "basis_sha256": sha(raw_path),
        "basis_array_sha256": array_sha(basis),
        "projector_distance_to_saved_two_sided": projector_distance_to_saved,
        "orthogonality_error": float(np.linalg.norm(basis.T @ basis - np.eye(m))),
        "method_seed": expected_seed,
        "initial_basis_sha256": initial_sha256,
        "selected_iteration": selected_iteration,
        "stop_reason": stop_reason,
        "stage_iterations": method_metadata.get("stage_iterations"),
        "stage_convergence": method_metadata.get("stage_convergence"),
        "first_weights_sha256": method_metadata.get("first_weights_sha256"),
        "first_directions_sha256": method_metadata.get("first_directions_sha256"),
        "first_moments_sha256": method_metadata.get("first_moments_sha256"),
        "first_tau": method_metadata.get("first_tau"),
        "profile": fit_profile,
        "diagnostics": method_metadata,
        "r_process": r_result,
        "tracemalloc_tracing_at_worker_start": traced_on_start,
        "tracemalloc_tracing_before_fit": tracing_before_fit,
        "tracemalloc_tracing_after_fit": tracing_after_fit,
        "data_sha256": array_sha(x, y, truth),
    }
    if arm == "two_sided":
        output.update(
            same_method_seed=expected_seed == int(baseline["method_seed"]),
            same_initial_basis_sha256=initial_sha256 == baseline["initial_sha256"],
            same_selected_iteration=selected_iteration
            == int(baseline["selected_iteration"]),
            same_stage_iterations=method_metadata["stage_iterations"]
            == [int(stage["iterations"]) for stage in baseline["stages"]],
            same_stage_convergence=method_metadata["stage_convergence"]
            == [bool(stage["converged"]) for stage in baseline["stages"]],
            same_first_weights_sha256=method_metadata["first_weights_sha256"]
            == baseline["first_weights_sha256"],
            same_first_directions_sha256=method_metadata["first_directions_sha256"]
            == baseline["first_directions_sha256"],
            same_first_moments_sha256=method_metadata["first_moments_sha256"]
            == baseline["first_moments_sha256"],
        )
    return output


def summarize(rows):
    good = [row for row in rows if row.get("status") == "ok"]
    lookup = {(row["link"], row["m"], row["seed"], row["arm"]): row for row in rows}
    pairs = []
    for link, m in CELLS:
        for seed in SEED_VALUES:
            two_sided = lookup.get((link, m, seed, "two_sided"))
            meanmave = lookup.get((link, m, seed, "MEANMAVE"))
            if not two_sided or not meanmave:
                continue
            if two_sided.get("status") != "ok" or meanmave.get("status") != "ok":
                continue
            pairs.append(
                {
                    "link": link,
                    "m": m,
                    "seed": seed,
                    "delta_edr_two_sided_minus_meanmave": two_sided["edr"]
                    - meanmave["edr"],
                    "meanmave_over_two_sided_fit_time": meanmave["fit_seconds"]
                    / two_sided["fit_seconds"],
                    "meanmave_over_two_sided_process_time": meanmave["process"][
                        "process_seconds"
                    ]
                    / two_sided["process"]["process_seconds"],
                    "same_input": two_sided["input_sha256"] == meanmave["input_sha256"],
                    "two_sided_replay_projector_distance": two_sided[
                        "projector_distance_to_saved_two_sided"
                    ],
                }
            )
    cell_rows = []
    for link, m in CELLS:
        cell_pairs = [p for p in pairs if p["link"] == link and p["m"] == m]
        cell_runs = {
            arm: [
                lookup[(link, m, seed, arm)]
                for seed in SEED_VALUES
                if lookup.get((link, m, seed, arm), {}).get("status") == "ok"
            ]
            for arm in ARMS
        }
        cell_rows.append(
            {
                "link": link,
                "m": m,
                "complete_pairs": len(cell_pairs),
                "mean_edr": {
                    arm: float(np.mean([row["edr"] for row in cell_runs[arm]]))
                    if cell_runs[arm]
                    else None
                    for arm in ARMS
                },
                "recovered": {
                    arm: sum(bool(row["recovered"]) for row in cell_runs[arm])
                    for arm in ARMS
                },
                "successful_fits": {arm: len(cell_runs[arm]) for arm in ARMS},
                "fit_seconds_median": {
                    arm: float(
                        np.median([row["fit_seconds"] for row in cell_runs[arm]])
                    )
                    if cell_runs[arm]
                    else None
                    for arm in ARMS
                },
                "process_seconds_median": {
                    arm: float(
                        np.median(
                            [
                                row["process"]["process_seconds"]
                                for row in cell_runs[arm]
                            ]
                        )
                    )
                    if cell_runs[arm]
                    else None
                    for arm in ARMS
                },
                "mean_delta_edr": float(
                    np.mean(
                        [p["delta_edr_two_sided_minus_meanmave"] for p in cell_pairs]
                    )
                )
                if cell_pairs
                else None,
                "median_meanmave_over_two_sided_fit_time": float(
                    np.median(
                        [p["meanmave_over_two_sided_fit_time"] for p in cell_pairs]
                    )
                )
                if cell_pairs
                else None,
            }
        )
    seed_cluster = {}
    for seed in SEED_VALUES:
        values = [
            pair["delta_edr_two_sided_minus_meanmave"]
            for pair in pairs
            if pair["seed"] == seed
        ]
        if len(values) == len(CELLS):
            seed_cluster[str(seed)] = float(np.mean(values))
    overall_mean = float(np.mean(list(seed_cluster.values()))) if seed_cluster else None
    ci = None
    if seed_cluster:
        means = np.asarray(list(seed_cluster.values()), dtype=float)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        samples = means[
            rng.integers(0, len(means), (BOOTSTRAP_DRAWS, len(means)))
        ].mean(axis=1)
        ci = np.quantile(samples, [0.025, 0.975]).tolist()
    return {
        "planned_fits": len(CELLS) * len(SEED_VALUES) * len(ARMS),
        "recorded_fits": len(rows),
        "successful_fits": len(good),
        "failed_fits": len(rows) - len(good),
        "complete_pairs": len(pairs),
        "complete_seed_clusters": len(seed_cluster),
        "expected_seed_clusters": len(SEED_VALUES),
        "pairing_ok": all(pair["same_input"] for pair in pairs),
        "cell_summary": cell_rows,
        "seed_cluster_mean_delta_edr": seed_cluster,
        "aggregate_mean_delta_edr_two_sided_minus_meanmave": overall_mean,
        "aggregate_seed_cluster_bootstrap_95_ci": ci,
        "meanmave_more_accurate_if_delta_positive": True,
        "meanmave_over_two_sided_time_ratio_above_one_means_two_sided_faster": True,
        "pairs": pairs,
    }


def run():
    if (OUT / "manifest.json").exists():
        manifest = json.loads((OUT / "manifest.json").read_text())
        verify_frozen(manifest)
        if any((OUT / "fits").iterdir()):
            raise FileExistsError("fit outputs already exist; refusing to rerun")
    else:
        manifest = freeze()
        write_json(OUT / "r_env.json", manifest["r_environment_overrides"])
    jobs = [
        (link, m, seed, arm)
        for item in manifest["jobs_by_pair"]
        for link, m, seed, arm in (
            (item["link"], item["m"], item["seed"], selected_arm)
            for selected_arm in item["order"]
        )
    ]
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        BLIS_NUM_THREADS="1",
    )
    started = time.perf_counter()
    rows = []
    for index, (link, m, seed, arm) in enumerate(jobs, 1):
        verify_frozen(manifest)
        name = f"{link}-m{m}-{seed}-{arm}"
        log_path = OUT / "fits" / f"{name}.log"
        worker_started = time.perf_counter()
        proc = run_process(
            [
                sys.executable,
                "-m",
                "experiments.two_sided_meanmave",
                "worker",
                arm,
                link,
                str(m),
                str(seed),
            ],
            env,
            log_path,
            timeout=120,
            memory_mib=2048,
            cwd=ROOT,
        )
        verify_frozen(manifest)
        if proc["status"] == "ok":
            try:
                row = json.loads(log_path.read_text().splitlines()[-1])
                if (row["link"], row["m"], row["seed"], row["arm"]) != (
                    link,
                    m,
                    seed,
                    arm,
                ):
                    raise ValueError("worker result key mismatch")
            except (KeyError, OSError, ValueError) as error:
                row = {
                    "link": link,
                    "m": m,
                    "seed": seed,
                    "arm": arm,
                    "status": "invalid_output",
                    "error": repr(error),
                }
        else:
            row = {
                "link": link,
                "m": m,
                "seed": seed,
                "arm": arm,
                "status": proc["status"],
                "error": log_path.read_text()[-4000:],
            }
        row["process"] = proc
        row["outer_wall_seconds"] = time.perf_counter() - worker_started
        row["fingerprint"] = manifest["fingerprint"]
        write_json(OUT / "fits" / f"{name}.json", row)
        rows.append(row)
        print(
            f"{index}/{len(jobs)} {link} m={m} seed={seed} {arm}: {row['status']}",
            flush=True,
        )
    summary = summarize(rows)
    summary.update(
        wall_seconds=time.perf_counter() - started,
        manifest_sha256=sha(OUT / "manifest.json"),
    )
    write_json(OUT / "summary.json", summary)
    return summary


def worker_cli(args):
    arm, link, m, seed = args
    result = worker(arm, link, int(m), int(seed))
    print(json.dumps(result, default=default, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "worker", "analyze"))
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.action == "freeze":
        manifest = freeze()
        write_json(OUT / "r_env.json", manifest["r_environment_overrides"])
        print(
            json.dumps(
                {
                    "out": str(OUT),
                    "planned_fits": manifest["planned_fits"],
                    "fingerprint": manifest["fingerprint"],
                    "manifest_sha256": sha(OUT / "manifest.json"),
                    "source_archive_sha256": manifest["source_archive_sha256"],
                }
            )
        )
    elif args.action == "run":
        run()
    elif args.action == "worker":
        worker_cli(args.arguments)
    else:
        rows = [
            json.loads(path.read_text())
            for path in sorted((OUT / "fits").glob("*.json"))
        ]
        write_json(OUT / "summary.json", summarize(rows))


if __name__ == "__main__":
    main()
