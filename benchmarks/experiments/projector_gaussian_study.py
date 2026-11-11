"""Frozen, resumeless benchmark; Luna's independent audit gates later phases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy

from ADP import ADP_Config
from ADP.core.ADP_Config import epanechnikov
from ADP.engine.common.index_fit import fit_index
from ADP.solver.grassman import solve as projected_solve
from experiments.mave_grassmann_grid import arrays, case
from experiments.projector_gaussian import digest, fit, gaussian_kernel
from experiments.r_sdr.runner import run_process
from experiments.two_sided_meanmave import TimeOnlyProfiler, default, method_seed

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/results/verification/weak-localization-20261028"
ARMS = ("projected", "full", "two_sided", "compact_projector", "gaussian_projector")
SEEDS = {
    "selection": tuple(range(2841000, 2841020)),
    "validation": tuple(range(2842000, 2842030)),
    "stress": tuple(range(2843000, 2843010)),
}


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=default, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cases(phase):
    if phase != "stress":
        return [
            {**case(25, m, 5000, 0.3, link), "scenario": "standard"}
            for link in ("additive", "quadratic", "sine")
            for m in (3, 5)
        ]
    return [
        {**case(d, m, n, 0.3, link), "name": name, "scenario": scenario}
        for name, scenario, d, m, n, link in (
            ("additive-d50", "standard", 50, 5, 5000, "additive"),
            ("additive-n1000", "standard", 25, 5, 1000, "additive"),
            ("additive-weaker", "weaker", 25, 5, 5000, "additive"),
            ("quartic", "quartic", 25, 3, 5000, "quartic"),
            ("uniform", "uniform", 25, 5, 5000, "additive"),
            ("correlated", "correlated", 25, 5, 5000, "additive"),
        )
    ]


def generate(spec, seed):
    if spec["scenario"] == "standard":
        return arrays(spec, seed)
    x, _, truth = arrays({**spec, "link": "additive"}, seed)
    if spec["scenario"] == "uniform":
        rng = np.random.default_rng(np.random.SeedSequence([seed, spec["d"], 2841028]))
        x = rng.uniform(-np.sqrt(3), np.sqrt(3), size=x.shape)
    elif spec["scenario"] == "correlated":
        indices = np.arange(spec["d"])
        cov = 0.5 ** np.abs(indices[:, None] - indices[None, :])
        x = x @ np.linalg.cholesky(cov).T
    z = x @ truth
    if spec["scenario"] == "quartic":
        raw = (
            z[:, 0]
            + 0.5 * (z[:, 1] ** 2 - 1)
            + 0.3 * (z[:, 2] ** 4 - 6 * z[:, 2] ** 2 + 3)
        )
    else:
        coefficient = 0.15 if spec["scenario"] == "weaker" else 0.35
        raw = (
            np.sin(z[:, 0])
            + 0.5 * z[:, 1] ** 2
            + coefficient * np.tanh(z[:, 2:]).sum(axis=1)
        )
    noise_stream = np.random.SeedSequence([seed, spec["d"], 20261025]).spawn(3)[2]
    noise = np.random.default_rng(noise_stream).normal(size=5000)[: spec["n"]]
    return x, (raw - raw.mean()) / raw.std() + spec["noise"] * noise, truth


def schedule(phase):
    result = []
    for spec in cases(phase):
        for seed in SEEDS[phase]:
            case_id = f"{spec['name']}-{seed}"
            result.extend(
                {"id": f"{case_id}-{arm}", "case_id": case_id, "arm": arm}
                for arm in ARMS
            )
    np.random.default_rng(2841028).shuffle(result)
    return result


def source_paths():
    return sorted(
        {
            *ROOT.joinpath("ADP").rglob("*.py"),
            *ROOT.joinpath("experiments").glob("*.py"),
            *ROOT.joinpath("benchmarks/experiments/r_sdr").glob("*.py"),
            *ROOT.joinpath("tests").glob("test_projector_gaussian*.py"),
            OUT / "PROTOCOL.md",
            OUT / "R1_AUDIT.md",
            OUT / "LUNA_PILOT_ANALYSIS.md",
            OUT / "audit_results.py",
        }
    )


def require_previous_pass(phase, output=OUT):
    previous = {"validation": "selection", "stress": "validation"}.get(phase)
    if previous is None:
        return
    path = output / previous / "independent_audit.json"
    if not path.exists():
        raise RuntimeError(f"previous independent audit missing: {path}")
    audit = json.loads(path.read_text())
    if not audit.get("integrity_pass") or not audit.get("gates_pass"):
        raise RuntimeError(f"previous independent audit did not pass: {previous}")


def verify_sources(manifest):
    for relative, expected in manifest["sources"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f"source drift: {relative}")


def freeze(phase):
    require_previous_pass(phase)
    directory = OUT / phase
    directory.mkdir(exist_ok=False)
    (directory / "inputs").mkdir()
    (directory / "fits").mkdir()
    (directory / "logs").mkdir()
    sources = {str(path.relative_to(ROOT)): sha(path) for path in source_paths()}
    if phase != "selection":
        selection = json.loads((OUT / "selection/manifest.json").read_text())
        if sources != selection["sources"]:
            raise RuntimeError("selection source map changed before held-out phase")
    with zipfile.ZipFile(
        directory / "sources.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for relative in sources:
            archive.write(ROOT / relative, relative)
    inputs = {}
    for spec in cases(phase):
        for seed in SEEDS[phase]:
            case_id = f"{spec['name']}-{seed}"
            x, y, truth = generate(spec, seed)
            path = directory / "inputs" / f"{case_id}.npz"
            np.savez_compressed(path, X=x, y=y, truth=truth)
            inputs[case_id] = {
                "path": str(path.relative_to(directory)),
                "sha256": sha(path),
                "array_sha256": digest(x, y, truth),
                "case": spec,
                "seed": seed,
            }
    manifest = {
        "phase": phase,
        "seeds": SEEDS[phase],
        "sources": sources,
        "inputs": inputs,
        "jobs": schedule(phase),
        "source_archive_sha256": sha(directory / "sources.zip"),
        "created_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "workers": 3,
        "blas_threads": 1,
        "timeout_seconds": 180,
        "rss_limit_mib": 2048,
    }
    write(directory / "manifest.json", manifest)
    verify_sources(manifest)
    return manifest


def metric(basis, truth):
    singular = np.linalg.svd(basis.T @ truth, compute_uv=False)
    sine2 = np.maximum(0.0, 1 - np.minimum(singular, 1.0) ** 2)
    return float(np.sqrt(sine2.mean())), float(np.sqrt(sine2.max()))


def fit_arm(x, y, m, seed, arm):
    cfg = ADP_Config(
        seed=method_seed(seed),
        N_J=min(len(x), 300),
        N_lin=max(50, 2 * x.shape[1]),
        N_phi=max(m + 1, min(x.shape[1], 12)),
        N_loc=20,
    )
    if arm == "projected":

        def solve(index, stats):
            return projected_solve(index, stats.U, stats.I, mass=stats.mass)

        return fit_index(
            x,
            y,
            config=cfg,
            mode="multi",
            index_dim=m,
            solve=solve,
            profiler=TimeOnlyProfiler(),
            trace_indices=True,
        )
    return fit(
        x,
        y,
        m=m,
        config=cfg,
        kernel=gaussian_kernel if arm == "gaussian_projector" else epanechnikov,
        floor=1.0 if arm in ("compact_projector", "gaussian_projector") else 0.0,
        solver="lbfgs" if arm == "full" else "two_sided",
    )


def worker(phase, job_id):
    directory = OUT / phase
    manifest = json.loads((directory / "manifest.json").read_text())
    verify_sources(manifest)
    job = next(j for j in manifest["jobs"] if j["id"] == job_id)
    item = manifest["inputs"][job["case_id"]]
    input_path = directory / item["path"]
    if sha(input_path) != item["sha256"]:
        raise RuntimeError("input file drift")
    with np.load(input_path) as saved:
        x, y, truth = saved["X"], saved["y"], saved["truth"]
    row = {
        **job,
        "seed": item["seed"],
        "case": item["case"],
        "data_sha256": digest(x, y, truth),
    }
    started = time.perf_counter()
    try:
        result = fit_arm(x, y, item["case"]["m"], item["seed"], job["arm"])
        elapsed = time.perf_counter() - started
        metadata = result.metadata
        stages = np.array([t.pop("basis") for t in metadata["trace"]])
        basis, initial = result.index.T, result.initial_index.T
        raw_path = directory / "fits" / f"{job_id}.npz"
        np.savez_compressed(raw_path, basis=basis, initial=initial, stages=stages)
        edr, max_sin = metric(basis, truth)
        row.update(
            status="ok",
            fit_seconds=elapsed,
            edr=edr,
            max_sin=max_sin,
            initial_edr=metric(initial, truth)[0],
            stage_edr=[metric(b, truth)[0] for b in stages],
            stage_max_sin=[metric(b, truth)[1] for b in stages],
            initial_sha256=digest(initial),
            basis_sha256=digest(basis),
            selected_iteration=metadata["selected_iteration"],
            outer_stop_reason=metadata["stop_reason"],
            metadata=metadata,
            raw_path=str(raw_path.relative_to(directory)),
            raw_sha256=sha(raw_path),
            worker_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
        )
    except Exception as exc:  # noqa: BLE001 -- retain every failed benchmark attempt
        row.update(
            status="error", error=repr(exc), fit_seconds=time.perf_counter() - started
        )
    write(directory / "fits" / f"{job_id}.json", row)


def run_one(phase, job, env):
    directory = OUT / phase
    manifest = json.loads((directory / "manifest.json").read_text())
    item = manifest["inputs"][job["case_id"]]
    tick = time.perf_counter()
    result = run_process(
        [
            sys.executable,
            "-m",
            "experiments.projector_gaussian_study",
            "worker",
            phase,
            job["id"],
        ],
        env,
        directory / "logs" / f"{job['id']}.log",
        timeout=180,
        memory_mib=2048,
        cwd=ROOT,
    )
    path = directory / "fits" / f"{job['id']}.json"
    row = (
        json.loads(path.read_text())
        if path.exists()
        else {
            **job,
            "status": "error",
            "seed": item["seed"],
            "case": item["case"],
            "data_sha256": item["array_sha256"],
        }
    )
    if result["status"] != "ok":
        row.update(status=result["status"], error=f"worker exit={result['returncode']}")
    row.update(
        process_seconds=time.perf_counter() - tick,
        process_peak_rss_mib=max(
            result["peak_rss_mib"] or 0.0, row.get("worker_peak_rss_mib", 0.0)
        ),
    )
    write(path, row)
    return row


def status(**values):
    write(
        OUT / "STATUS.json",
        dict(pid=os.getpid(), updated_utc=datetime.now(UTC).isoformat(), **values),
    )


def campaign():
    # An exclusive marker prevents accidental duplicate campaigns after detaching.
    with (OUT / "CAMPAIGN_STARTED.json").open("x") as stream:
        json.dump(
            {"pid": os.getpid(), "started_utc": datetime.now(UTC).isoformat()}, stream
        )
    env = os.environ.copy()
    env.update(
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    try:
        for phase in ("selection", "validation", "stress"):
            status(
                state="preparing",
                phase=phase,
                completed=0,
                planned=len(schedule(phase)),
            )
            manifest = freeze(phase)
            failures = completed = 0
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(run_one, phase, job, env)
                    for job in manifest["jobs"]
                ]
                for future in as_completed(futures):
                    row = future.result()
                    completed += 1
                    failures += row["status"] != "ok"
                    status(
                        state="running",
                        phase=phase,
                        completed=completed,
                        planned=len(futures),
                        failures=failures,
                        last_job=row["id"],
                    )
                    print(
                        f"{phase}: {completed}/{len(futures)}; failures={failures}; {row['id']}",
                        flush=True,
                    )
            verify_sources(manifest)
            status(
                state="auditing", phase=phase, completed=completed, planned=completed
            )
            audit_run = subprocess.run(
                [sys.executable, str(OUT / "audit_results.py"), str(OUT / phase)],
                cwd=ROOT,
                env=env,
                check=False,
            )
            audit_path = OUT / phase / "independent_audit.json"
            audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
            if audit_run.returncode or not audit.get("integrity_pass"):
                status(
                    state="audit_failed",
                    phase=phase,
                    completed=completed,
                    planned=completed,
                )
                return
            if phase != "stress" and not audit.get("gates_pass"):
                status(
                    state="gate_failed",
                    phase=phase,
                    completed=completed,
                    planned=completed,
                )
                return
        status(
            state="completed",
            phase="stress",
            message="Results await Luna interpretation; no production promotion",
        )
    except Exception as exc:
        status(state="runner_error", error=repr(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("campaign", "worker"))
    parser.add_argument("phase", nargs="?", choices=tuple(SEEDS))
    parser.add_argument("job_id", nargs="?")
    args = parser.parse_args()
    if args.command == "campaign":
        campaign()
    else:
        worker(args.phase, args.job_id)


if __name__ == "__main__":
    main()
