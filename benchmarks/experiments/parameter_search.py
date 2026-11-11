"""Reproducible multi-index parameter search; public estimator is unchanged."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "benchmarks/results/verification/parameter-search-20261025"
SEEDS = {"explore": list(range(792100, 792110))}
VARIABLES = ("N_J", "N_loc", "h_min", "N_phi", "direction_mode", "redraw_directions")


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def scenarios():
    return [
        dict(name="easy", n=1000, d=10, m=2, noise=.1, link="additive"),
        dict(name="base", n=1000, d=25, m=2, noise=.3, link="additive"),
        dict(name="m3", n=1000, d=25, m=3, noise=.3, link="additive"),
        dict(name="m5", n=1000, d=25, m=5, noise=.3, link="additive"),
        dict(name="noisy", n=1000, d=25, m=2, noise=1., link="additive"),
        dict(name="larger", n=2000, d=25, m=3, noise=.3, link="additive"),
        dict(name="high_d", n=1000, d=50, m=3, noise=.3, link="additive"),
        dict(name="balanced_m5", n=1000, d=25, m=5, noise=.3, link="sine"),
    ]


def base_config(case):
    from ADP import ADP_Config
    defaults = ADP_Config()
    return {name: getattr(defaults, name)
            for name in (*VARIABLES, "N_lin", "index_init", "select_step")}


def levels(case):
    return dict(N_J=[case["n"] // 10 * i for i in range(1, 11)],
                N_loc=list(range(10, 101, 10)),
                h_min=[round(.2 * i, 1) for i in range(1, 11)],
                N_phi=list(range(10, 101, 10)),
                direction_mode=["auto", "localized"],
                redraw_directions=[False, True])


def configurations(case):
    base, grid = base_config(case), levels(case)
    result = [base]
    for field, values in grid.items():
        result.extend(dict(base, **{field: value}) for value in values)
    return list({key(arm): arm for arm in result}.values())


def geometry(basis, truth):
    residual = basis - truth @ (truth.T @ basis)
    edr = float(np.linalg.norm(residual) / np.sqrt(truth.shape[1]))
    max_sine = min(1., float(np.linalg.norm(residual, ord=2)))
    return dict(edr=edr, trace_score=1 - edr**2, max_sine=max_sine,
                max_angle_deg=float(np.degrees(np.arcsin(max_sine))))


def freeze(out):
    path = out / "manifest.json"
    if path.exists():
        return json.loads(path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    files = sorted([*ROOT.joinpath("ADP").rglob("*.py"),
                    *ROOT.joinpath("experiments").rglob("*.py"),
                    ROOT / "benchmarks/experiments/PARAMETER_SEARCH.md"])
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in files}
    cases = scenarios()
    manifest = dict(cases=cases, seeds=SEEDS, protocol_version=3,
                    grids={}, axes={}, sources=hashes, python=sys.version,
                    platform=platform.platform(), cpu_count=os.cpu_count(),
                    numpy=np.__version__, solver=dict(max_steps=20, tol=1e-6),
                    recovery_threshold=.1, timeout_seconds=180,
                    timing="sequential fresh processes, one BLAS thread; full fit and process time",
                    analysis_model="gpt-6-luna", analysis_status="pending user follow-up",
                    experiment="one factor at a time; other ADP_Config fields retain public defaults",
                    data_generator="experiments.mave_grassmann_grid.arrays, nested X5000/basis5",
                    failure_penalty=1., created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    for c in cases:
        configs = configurations(c)
        manifest["grids"][c["name"]] = {
            "base" if a == base_config(c) else key(a): a for a in configs}
        base = base_config(c)
        manifest["axes"][c["name"]] = {
            field: [dict(value=value,
                         arm="base" if dict(base, **{field: value}) == base
                         else key(dict(base, **{field: value}))) for value in values]
            for field, values in levels(c).items()}
    manifest["fingerprint"] = key(manifest)
    with zipfile.ZipFile(out / "sources.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            archive.write(p, p.relative_to(ROOT))
    with zipfile.ZipFile(out / "sources.zip") as archive:
        archive.extractall(out / "source")
    dump(path, manifest)
    for name, args in (("git-status.txt", ["status", "--short"]),
                       ("git-diff.patch", ["diff", "HEAD"]),
                       ("git-head.txt", ["rev-parse", "HEAD"])):
        command = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        (out / name).write_text(command.stdout + command.stderr)
    return manifest


def verify_snapshot(out, manifest):
    for relative, sha in manifest["sources"].items():
        assert hashlib.sha256((out / "source" / relative).read_bytes()).hexdigest() == sha


def prepare_input(out, phase, case, seed):
    from experiments.mave_grassmann_grid import arrays
    path = out / phase / "inputs" / f'{case["name"]}-{seed}.npz'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        x, y, truth = arrays(case, seed)
        np.savez_compressed(path, x=x, y=y, truth=truth)
    return path


def worker(job_path):
    from ADP import ADP_Config, ADP_multi_index, ADP_solver
    from ADP.solver.grassman import solve as solve_grassman
    job = json.loads(Path(job_path).read_text())
    row = {k: job[k] for k in ("phase", "case", "seed", "arm", "fingerprint", "config")}
    with np.load(job["input"]) as data:
        x, y, truth = (data[k] for k in ("x", "y", "truth"))
    row["data_sha256"] = hashlib.sha256(x.tobytes() + y.tobytes() + truth.tobytes()).hexdigest()
    method_seed = int(np.random.SeedSequence([job["seed"], 1]).generate_state(1)[0]) % (2**31 - 1)
    started = time.perf_counter()
    try:
        model = ADP_multi_index(
            truth.shape[1], ADP_Config(seed=method_seed, **job["config"]),
            ADP_solver(solve_grassman, max_steps=20, tol=1e-6),
        )
        model.fit(x, y)
        elapsed = time.perf_counter() - started
        row.update(status="ok", fit_seconds=elapsed,
                   **geometry(model.basis_, truth),
                   initial_edr=geometry(model.result_.beta_init, truth)["edr"],
                   basis=model.basis_.tolist(), initial_basis=model.result_.beta_init.tolist(),
                   initial_sha256=hashlib.sha256(model.result_.beta_init.tobytes()).hexdigest(),
                   orthogonality_error=float(np.linalg.norm(model.basis_.T @ model.basis_ - np.eye(truth.shape[1]))),
                   outer_iterations=len(model.trace_), stop_reason=model.result_.stop_reason,
                   selected_iteration=model.effective_parameters_["selected_iteration"],
                   effective_N_J=model.effective_parameters_["N_J"],
                   effective_N_phi=model.effective_parameters_["N_phi"],
                   effective_N_lin=model.effective_parameters_["N_lin"],
                   effective_direction_mode=model.effective_parameters_["direction_mode"],
                   profile=model.profile_)
        row["trace"] = [dict(h=t["h"], alpha=t["factor"], mean_mass=t["mean_mass"],
                             err=t["err"], loss=t["solver"]["loss"],
                             converged=bool(t["solver"]["converged"]),
                             solver_stop=t["solver"]["stop_reason"],
                             accepted_steps=t["solver"]["accepted_steps"],
                             local_rank_loss=t["solver"]["local_rank_loss"],
                             stage_edr=geometry(t["basis"], truth)["edr"])
                        for t in model.trace_]
    except Exception:
        row.update(status="error", edr=None, fit_seconds=time.perf_counter() - started,
                   error=traceback.format_exc())
    row["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    dump(job["result"], row)


def launch(job, out, env):
    path = Path(job["result"])
    if path.exists():
        row = json.loads(path.read_text())
        assert row["fingerprint"] == job["fingerprint"]
        assert row["config"] == job["config"]
        return row
    job_path = path.with_suffix(".job.json")
    dump(job_path, job)
    started = time.perf_counter()
    try:
        command = subprocess.run([sys.executable, "-m", "experiments.parameter_search",
                                  "worker", "--job", str(job_path)],
                                 cwd=out / "source", env=env, capture_output=True, text=True,
                                 timeout=180)
        if command.returncode != 0 or not path.exists():
            raise RuntimeError(command.stdout + command.stderr)
        row = json.loads(path.read_text())
    except Exception:
        row = {k: job[k] for k in ("phase", "case", "seed", "arm", "fingerprint", "config")}
        row.update(status="process_error", edr=None, fit_seconds=None, error=traceback.format_exc())
    row["process_seconds"] = time.perf_counter() - started
    dump(path, row)
    return row


def phase_jobs(out, manifest, phase, arms):
    jobs = []
    for case in manifest["cases"]:
        name = case["name"]
        for seed in manifest["seeds"][phase]:
            data = prepare_input(out, phase, case, seed)
            for arm in arms[name]:
                folder = out / phase / "fits"
                folder.mkdir(parents=True, exist_ok=True)
                jobs.append(dict(phase=phase, case=name, seed=seed, arm=arm,
                                 config=manifest["grids"][name][arm],
                                 input=str(data), fingerprint=manifest["fingerprint"],
                                 result=str(folder / f"{name}-{seed}-{arm}.json")))
    np.random.default_rng(20261025).shuffle(jobs)
    schedule = out / phase / "schedule.json"
    if schedule.exists():
        assert json.loads(schedule.read_text()) == jobs, "schedule drift"
    else:
        dump(schedule, jobs)
    return jobs


def execute(out, manifest, phase, arms, workers):
    jobs = phase_jobs(out, manifest, phase, arms)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OMP_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", MPLCONFIGDIR="/tmp/adp-search-mpl",
               PYTHONPATH=str(out / "source"))
    started = time.perf_counter()
    rows = []
    print(json.dumps(dict(phase=phase, planned=len(jobs), workers=workers)), flush=True)
    dump(out / "state.json", dict(phase=phase, completed=0, planned=len(jobs),
                                 failures=0, workers=workers, wall_seconds=0.))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(launch, job, out, env) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 10 == 0 or len(rows) == len(jobs):
                state = dict(phase=phase, completed=len(rows), planned=len(jobs),
                             failures=sum(r["status"] != "ok" for r in rows),
                             wall_seconds=time.perf_counter() - started)
                dump(out / "state.json", state)
                print(json.dumps(state), flush=True)
    dump(out / phase / "rows.json", rows)
    return rows


def run(out, workers):
    manifest = freeze(out)
    verify_snapshot(out, manifest)
    names = [c["name"] for c in manifest["cases"]]
    rows = execute(out, manifest, "explore",
                   {n: list(manifest["grids"][n]) for n in names}, workers)
    dump(out / "complete.json", dict(fingerprint=manifest["fingerprint"], completed=len(rows),
                                     failures=sum(r["status"] != "ok" for r in rows),
                                     finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "worker", "dry-run"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "worker":
        worker(args.job)
    elif args.command == "dry-run":
        print(json.dumps({c["name"]: dict(case=c, configs=len(configurations(c)),
                                         explore_fits=len(SEEDS["explore"]) * len(configurations(c)))
                          for c in scenarios()}, indent=2))
    else:
        run(args.out.resolve(), args.workers)


if __name__ == "__main__":
    main()
