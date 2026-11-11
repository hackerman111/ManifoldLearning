"""Frozen paired recovery study. Run as a module; no production modifications."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from experiments.r_sdr.design import metrics
from experiments.r_sdr.protocol import ROOT, digest, environment, source_files
from experiments.r_sdr.runner import read_worker_meta, run_process

OUT = ROOT / "benchmarks/results/verification/mave-grassmann-grid-20261025"
BASE_SEED = 651000
G, M, C = "Grassmann", "MEANMAVE", "MAVE-reference"


def case(d, m, n, noise, link="additive", block="core"):
    return dict(name=f"{link}-d{d}-m{m}-n{n}-s{noise:g}", d=d, m=m,
                n=n, noise=noise, link=link, block=block)


def cases():
    result = [case(d, m, n, s) for d in (10, 25, 50)
              for m in (1, 2, 3, 5) for n in (300, 1000, 3000)
              for s in (0., .3, 1.)]
    result += [case(100, m, n, s, block="large") for m in (2, 5)
               for n in (1000, 5000) for s in (.3, 1.)]
    result += [case(25, m, 5000, .3, block="large") for m in (2, 3, 5)]
    result += [case(25, m, 1000, s, link, "link")
               for link in ("sine", "quadratic") for m in (2, 3, 5)
               for s in (0., .3, 1.)]
    assert len(result) == len({c["name"] for c in result}) == 137
    return result


def arrays(c, seed):
    # Nested n/m and common random numbers across noise, links and methods.
    streams = np.random.SeedSequence([seed, c["d"], 20261025]).spawn(3)
    xr, br, er = (np.random.default_rng(s) for s in streams)
    x = xr.normal(size=(5000, c["d"]))[:c["n"]].copy()
    truth = np.linalg.qr(br.normal(size=(c["d"], 5)))[0][:, :c["m"]].copy()
    eps = er.normal(size=5000)[:c["n"]]
    z = x @ truth
    if c["link"] == "additive":
        raw = np.sin(z[:, 0])
        if c["m"] >= 2:
            raw += .5 * z[:, 1] ** 2
        raw += .35 * np.tanh(z[:, 2:]).sum(axis=1)
    elif c["link"] == "sine":
        raw = np.sin(z).sum(axis=1)
    elif c["link"] == "quadratic":
        raw = (z ** 2).sum(axis=1)
    else:
        raise ValueError(c["link"])
    y = (raw - raw.mean()) / raw.std() + c["noise"] * eps
    return x, y, truth


def settings(c, arm):
    config = dict(N_lin=max(50, 2*c["d"]),
                  N_J=min(c["n"], max(300, math.ceil(c["n"]/20))),
                  N_phi=max(c["m"]+1, min(c["d"], 12)), N_loc=20,
                  index_init="local", select_step="best")
    solver = dict(max_steps=20, tol=1e-6, relative_gn=True,
                  quadratic_trials=3, constant_centers=True,
                  smooth_max_steps=None, local_ridge=0.)
    if arm == "Grassmann-steps5":
        solver["max_steps"] = 5
    elif arm == "Grassmann-steps40":
        solver["max_steps"] = 40
    elif arm == "Grassmann-loc40":
        config["N_loc"] = 40
    elif arm == "Grassmann-phi_d":
        config["N_phi"] = c["d"]
    return config, solver


def schedule(phase):
    if phase == "pilot":
        return [(case(10, 2, 300, .3), 650900, [G, M, C]),
                (case(25, 5, 1000, .3), 650900, [G, M, C]),
                (case(25, 5, 5000, .3), 650900, [G, M]),
                (case(100, 5, 5000, 1.), 650900, [G, M])]
    jobs = []
    for c in cases():
        arms = [G, M]
        sensitivity = (c["d"] == 25 and c["n"] == 1000
                       and c["m"] in (2, 3, 5) and c["noise"] in (.3, 1.)
                       and c["link"] == "additive")
        if sensitivity:
            arms += ["Grassmann-steps5", "Grassmann-steps40",
                     "Grassmann-loc40", "Grassmann-phi_d"]
        if (c["d"] in (10, 25) and c["n"] in (300, 1000)
                and c["m"] in (2, 5) and c["noise"] == .3
                and c["link"] == "additive"):
            arms += [C]
        anchor = (c["d"] == 25 and c["n"] == 1000 and c["m"] in (2, 3, 5)
                  and c["noise"] in (.3, 1.) and c["link"] == "additive")
        for seed in range(BASE_SEED, BASE_SEED + (30 if anchor else 10)):
            jobs.append((c, seed, arms if seed < BASE_SEED+10 else [G, M]))
    np.random.default_rng(20261025).shuffle(jobs)
    assert sum(len(a) for _, _, a in jobs) == 3300
    return jobs


def freeze(phase):
    folder = OUT / phase
    info, env = environment("Rscript", 1)
    files = sorted(set(source_files() + [Path(__file__).resolve()]))
    protocol = dict(phase=phase, jobs=schedule(phase), environment=info,
                    sources={str(p.relative_to(ROOT)): digest(p) for p in files},
                    timeout_seconds=180, reference_timeout_seconds=120,
                    memory_limit_mib=3072, workers=8,
                    r_iterations=20, r_tolerance=1e-5,
                    seed_scheme="SeedSequence([seed,d,20261025]).spawn(3): X5000,basis5,eps5000",
                    quality="||P_hat-P_true||F/sqrt(2m); recovery <= 0.1",
                    settings={c["name"]: {a: settings(c, a) for a in
                              [G, "Grassmann-steps5", "Grassmann-steps40",
                               "Grassmann-loc40", "Grassmann-phi_d"]} for c in cases()})
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    manifest = dict(fingerprint=fingerprint, protocol=protocol)
    if (folder / "manifest.json").exists():
        assert json.loads((folder / "manifest.json").read_text()) == manifest, "protocol drift"
    else:
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
        with zipfile.ZipFile(folder / "sources.zip", "x", zipfile.ZIP_DEFLATED) as archive:
            for p in files:
                archive.write(p, p.relative_to(ROOT))
        with zipfile.ZipFile(folder / "sources.zip") as archive:
            archive.extractall(folder / "source")
        for name, args in (("git-status.txt", ["status", "--short"]),
                           ("git-diff.patch", ["diff", "HEAD"]),
                           ("git-head.txt", ["rev-parse", "HEAD"])):
            p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
            (folder / name).write_text(p.stdout + p.stderr)
    for rel, sha in protocol["sources"].items():
        assert digest(folder / "source" / rel) == sha, rel
    env["PYTHONPATH"] = str(folder / "source")
    (folder / "inputs").mkdir(exist_ok=True)
    (folder / "fits").mkdir(exist_ok=True)
    return folder, manifest, env


def run_pair(task, folder, manifest, env):
    c, seed, arms = task
    stem = f'{c["name"]}-{seed}'
    todo = []
    for arm in arms:
        p = folder / "fits" / f"{stem}-{arm}" / "result.json"
        if p.exists():
            row = json.loads(p.read_text())
            assert row["fingerprint"] == manifest["fingerprint"]
        else:
            todo.append(arm)
    if not todo:
        return []
    x, y, truth = arrays(c, seed)
    data_sha = hashlib.sha256(x.tobytes()+y.tobytes()+truth.tobytes()).hexdigest()
    npz = folder / "inputs" / f"{stem}.npz"
    if npz.exists():
        with np.load(npz) as saved:
            assert all(np.array_equal(a, saved[k]) for a, k in
                       zip((x, y, truth), ("x", "y", "truth"), strict=True))
    else:
        np.savez_compressed(npz, x=x, y=y, truth=truth)
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(data_sha[:8], 16)]))
    rng.shuffle(todo)
    method_seed = int(np.random.SeedSequence([seed, 1]).generate_state(1)[0]) % (2**31-1)
    rows = []
    with tempfile.TemporaryDirectory(prefix="mave-grid-") as scratch:
        scratch = Path(scratch)
        xp, yp = scratch / "x.csv", scratch / "y.csv"
        np.savetxt(xp, x, delimiter=",", fmt="%.17g")
        np.savetxt(yp, y, delimiter=",", fmt="%.17g")
        hashes = dict(x_csv_sha256=digest(xp), y_csv_sha256=digest(yp))
        for arm in todo:
            directory = folder / "fits" / f"{stem}-{arm}"
            directory.mkdir(exist_ok=True)
            bp, mp = directory / "basis.csv", directory / "meta.txt"
            method = "ADP-Grassmann-combined" if arm.startswith(G) else (
                "MAVE-package" if arm == M else "MAVE")
            if arm.startswith(G):
                config, solver = settings(c, arm)
                job = dict(method=method, x=str(xp), y=str(yp), basis=str(bp),
                           meta=str(mp), m=c["m"], seed=method_seed,
                           config=config, solver=solver)
                jp = scratch / "job.json"
                jp.write_text(json.dumps(job))
                command = [sys.executable, "-m", "experiments.r_sdr.worker", str(jp)]
            else:
                command = ["Rscript", "--vanilla", str(folder / "source/experiments/r_sdr/worker.R"),
                           method, str(xp), str(yp), str(bp), str(mp), str(method_seed),
                           str(c["m"]), "20", "1e-5"]
            result = run_process(command, env, directory / "worker.log",
                                 timeout=120 if arm == C else 180,
                                 memory_mib=3072, cwd=folder / "source")
            row = dict(**c, case=c["name"], seed=seed, method=arm,
                       method_seed=method_seed, fingerprint=manifest["fingerprint"],
                       data_sha256=data_sha, **hashes, **result)
            if result["status"] == "ok":
                try:
                    row.update(read_worker_meta(method, mp))
                    basis = np.loadtxt(bp, delimiter=",", ndmin=2)
                    row.update(metrics(basis, truth))
                    row["recovered"] = row["projector_distance"] <= .1
                    row["basis_sha256"] = digest(bp)
                except Exception as exc:
                    row.update(status="invalid_output", error=repr(exc))
            path = directory / "result.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(row, indent=2, allow_nan=False))
            temporary.replace(path)
            rows.append(row)
    return rows


def run(phase):
    folder, manifest, env = freeze(phase)
    jobs = schedule(phase)
    started = time.perf_counter()
    print(f"{phase}: {len(jobs)} datasets, {sum(len(a) for _, _, a in jobs)} fits", flush=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_pair, j, folder, manifest, env) for j in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            rows = future.result()
            failures = [f'{r["method"]}:{r["status"]}' for r in rows if r["status"] != "ok"]
            if i % 10 == 0 or failures or phase == "pilot":
                print(f"{i}/{len(jobs)} datasets; elapsed={time.perf_counter()-started:.1f}s; {failures}", flush=True)
    rows = [json.loads(p.read_text()) for p in sorted((folder / "fits").glob("*/result.json"))]
    (folder / "runs.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    (folder / "execution.json").write_text(json.dumps(dict(
        invocation_wall_seconds=time.perf_counter()-started, planned_fits=sum(len(a) for _, _, a in jobs),
        recorded_fits=len(rows), failures=sum(r["status"] != "ok" for r in rows)), indent=2))
    print((folder / "execution.json").read_text(), flush=True)


def self_check():
    a, b = case(25, 3, 300, .3), case(25, 5, 1000, 1.)
    xa, ya, ba = arrays(a, BASE_SEED)
    xb, _, bb = arrays(b, BASE_SEED)
    np.testing.assert_array_equal(xa, xb[:300])
    np.testing.assert_array_equal(ba, bb[:, :3])
    np.testing.assert_allclose(ba.T @ ba, np.eye(3), atol=1e-14)
    x0, y0, b0 = arrays({**a, "noise": 0.}, BASE_SEED)
    np.testing.assert_array_equal(x0, xa)
    np.testing.assert_array_equal(b0, ba)
    assert abs(y0.std()-1) < 1e-14 and abs(y0.mean()) < 1e-14
    assert not np.array_equal(ya, y0)
    assert metrics(ba, ba)["projector_distance"] < 1e-14
    jobs = schedule("main")
    keys = [(c["name"], s, arm) for c, s, arms in jobs for arm in arms]
    assert len(set(keys)) == len(keys) == 3300
    for c, _, arms in jobs:
        for arm in arms:
            config, _ = settings(c, arm)
            assert config["N_J"] * config["N_loc"] >= c["n"]
    print("self-check: nested inputs, truth, noise, metric, schedule, coverage OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("phase", choices=("self-check", "pilot", "main"))
    arg = parser.parse_args().phase
    self_check() if arg == "self-check" else run(arg)
