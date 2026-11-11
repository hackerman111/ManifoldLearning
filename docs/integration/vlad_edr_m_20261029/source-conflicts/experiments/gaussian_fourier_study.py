"""Frozen paired Gaussian Fourier/ADP study, with isolated timed workers."""

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
import traceback
from pathlib import Path

import numpy as np
from scipy.linalg import subspace_angles

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "verification/gaussian-fourier-20261025"
SEEDS = {
    "pilot": range(520100, 520103),
    "selection": range(521100, 521110),
    "validation": range(522100, 522110),
}
FUNCTIONS = ("original", "sine", "quadratic", "product")
WIDTHS = (1.0, 2.0, 4.0)


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _digest(*arrays):
    return hashlib.sha256(
        b"".join(np.ascontiguousarray(a).tobytes() for a in arrays)
    ).hexdigest()


def _inputs(seed):
    streams = np.random.SeedSequence([seed, 20261025]).spawn(3)
    x = np.random.default_rng(streams[0]).normal(size=(5000, 25))
    basis = np.linalg.qr(np.random.default_rng(streams[1]).normal(size=(25, 5)))[0]
    noise = np.random.default_rng(streams[2]).normal(size=len(x))
    return x, basis, noise


def _response(x, basis, noise, function):
    # Population scaling preserves iid observations required by the U-statistic.
    z = x @ basis
    m = z.shape[1]
    sine_variance = (1 - np.exp(-2)) / 2
    if function == "original":
        signal = (
            np.sin(z[:, 0])
            + 0.5 * (z[:, 1] ** 2 - 1)
            + 0.35 * np.tanh(z[:, 2:]).sum(axis=1)
        )
        variance = sine_variance + 0.5 + 0.35**2 * (m - 2) * 0.3942944904
    elif function == "sine":
        signal, variance = np.sin(z).sum(axis=1), m * sine_variance
    elif function == "quadratic":
        signal, variance = (z * z).sum(axis=1) - m, 2 * m
    elif function == "product":
        signal, variance = np.prod(z, axis=1), 1.0
    else:
        raise ValueError(function)
    return signal / np.sqrt(variance) + 0.3 * noise


def worker(folder, job_id):
    manifest = json.loads((folder / "manifest.json").read_text())
    function, m, seed, arm = manifest["jobs"][job_id]
    row = {"function": function, "m": m, "seed": seed, "arm": arm, "status": "error"}
    try:
        with np.load(folder / "inputs" / f"{seed}.npz", allow_pickle=False) as saved:
            x, truth, noise = saved["X"], saved["basis"][:, :m], saved["noise"]
        y = _response(x, truth, noise, function)
        row["data_sha256"] = _digest(x, y, truth)
        started = time.perf_counter()
        if arm == "base":
            from ADP import ADP_Config, ADP_multi_index

            model = ADP_multi_index(
                m, ADP_Config(seed=seed, N_J=300, N_phi=12, N_loc=20, N_lin=50)
            ).fit(x, y)
            basis = model.basis_
            diagnostics = {
                "stop_reason": model.result_.stop_reason,
                "outer_iterations": len(model.trace_),
            }
        else:
            from experiments.gaussian_fourier import fit

            model = fit(
                x,
                y,
                m,
                seed=seed,
                n_frequencies=256,
                frequency_scale=float(arm.split("-")[1]),
            )
            basis, diagnostics = model.basis, model.diagnostics
        elapsed = time.perf_counter() - started
        # Truth is used only after the estimator call has completed.
        angles = subspace_angles(basis, truth)
        row.update(
            status="ok",
            edr=float(np.sqrt(np.mean(np.sin(angles) ** 2))),
            max_sin_angle=float(np.sin(angles).max()),
            fit_seconds=elapsed,
            basis=basis.tolist(),
            diagnostics=diagnostics,
        )
    except Exception:  # noqa: BLE001 - failed fits must remain in the experiment.
        row["error"] = traceback.format_exc()
    row["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    _write(folder / "fits" / f"{job_id}.json", row)


def evaluate_gate(rows, expected_count):
    """Apply predeclared gates to complete paired outcomes; failures never drop out."""
    keys = [(r["function"], r["m"], r["seed"], r["arm"]) for r in rows]
    complete = (
        len(rows) == expected_count
        and len(set(keys)) == expected_count
        and all(r["status"] == "ok" for r in rows)
    )
    by_case = {}
    for row in rows:
        by_case.setdefault((row["function"], row["m"], row["seed"]), {})[row["arm"]] = (
            row
        )
    pairs = []
    for key, arms in by_case.items():
        candidates = [r for arm, r in arms.items() if arm != "base"]
        if (
            "base" not in arms
            or len(candidates) != 1
            or any(r["status"] != "ok" for r in arms.values())
        ):
            complete = False
            continue
        pairs.append((key, arms["base"], candidates[0]))
    if not complete or not pairs:
        return {
            "pass": False,
            "checks": {"all_attempts_succeeded": False},
            "assigned": expected_count,
            "recorded": len(rows),
        }
    primary = [
        (key, base, candidate)
        for key, base, candidate in pairs
        if key[0] == "original" and key[1] in (3, 5)
    ]
    if not primary:
        raise ValueError("no primary original m=3,5 pairs")
    seed_logs = {}
    deltas = {}
    for key, base, candidate in pairs:
        deltas.setdefault(key[:2], []).append(candidate["edr"] - base["edr"])
    for key, base, candidate in primary:
        seed_logs.setdefault(key[2], []).append(
            np.log(max(candidate["edr"], 1e-12) / max(base["edr"], 1e-12))
        )
    log_means = np.array([np.mean(v) for _, v in sorted(seed_logs.items())])
    rng = np.random.default_rng(20261025)
    bootstrap = np.exp(
        log_means[rng.integers(len(log_means), size=(10000, len(log_means)))].mean(
            axis=1
        )
    )
    ratio = float(np.exp(log_means.mean()))
    interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    max_fit = max(c["fit_seconds"] / b["fit_seconds"] for _, b, c in pairs)
    max_process = max(c["process_seconds"] / b["process_seconds"] for _, b, c in pairs)
    recovered_base = sum(b["edr"] <= 0.1 for _, b, _ in primary)
    recovered_candidate = sum(c["edr"] <= 0.1 for _, _, c in primary)
    checks = {
        "all_attempts_succeeded": complete,
        "primary_edr_ratio_le_0_90": ratio <= 0.90,
        "primary_edr_ci_upper_lt_1": interval[1] < 1,
        "original_m2_regression_le_0_02": np.mean(deltas.get(("original", 2), [np.inf]))
        <= 0.02,
        "other_case_regression_le_0_05": all(
            np.mean(v) <= 0.05 for key, v in deltas.items() if key[0] != "original"
        ),
        "primary_recovery_count_nondecreasing": recovered_candidate >= recovered_base,
        "fit_time_le_3x": max_fit <= 3,
        "process_time_le_3x": max_process <= 3,
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": {k: bool(v) for k, v in checks.items()},
        "primary_edr_ratio": ratio,
        "primary_edr_ratio_ci95": interval,
        "max_fit_time_ratio": max_fit,
        "max_process_time_ratio": max_process,
        "primary_recovered_base": recovered_base,
        "primary_recovered_candidate": recovered_candidate,
        "pairs": len(pairs),
    }


def _source_files():
    return [
        *sorted((ROOT / "ADP").rglob("*.py")),
        ROOT / "experiments/__init__.py",
        ROOT / "experiments/gaussian_fourier.py",
        Path(__file__),
        ROOT / "tests/test_gaussian_fourier.py",
        OUT / "DESIGN.md",
        OUT / "AUDIT.md",
    ]


def freeze(phase):
    if phase == "pilot":
        arms = ["base", *(f"fourier-{w:g}" for w in WIDTHS)]
    else:
        choice = json.loads((OUT / "pilot/chosen.json").read_text())
        if choice["arm"] is None:
            raise ValueError("pilot supplied no valid candidate")
        if (
            phase == "validation"
            and not json.loads((OUT / "selection/summary.json").read_text())["gate"][
                "pass"
            ]
        ):
            raise ValueError("selection failed; validation remains untouched")
        arms = ["base", choice["arm"]]
    folder = OUT / phase
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "inputs").mkdir()
    (folder / "fits").mkdir()
    sources = {}
    for source in _source_files():
        relative = source.relative_to(ROOT)
        content = source.read_bytes()
        sources[str(relative)] = hashlib.sha256(content).hexdigest()
        target = folder / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    if phase != "pilot":
        previous = json.loads((OUT / "pilot/manifest.json").read_text())["sources"]
        if sources != previous:
            raise ValueError("sources changed since pilot; refuse to mix estimators")
    input_hashes = {}
    for seed in SEEDS[phase]:
        x, basis, noise = _inputs(seed)
        np.savez_compressed(
            folder / "inputs" / f"{seed}.npz", X=x, basis=basis, noise=noise
        )
        input_hashes[str(seed)] = _digest(x, basis, noise)
    jobs = [
        (function, m, seed, arm)
        for function in FUNCTIONS
        for m in (2, 3, 5)
        for seed in SEEDS[phase]
        for arm in arms
    ]
    np.random.default_rng(20261025).shuffle(jobs)
    _write(
        folder / "manifest.json",
        {
            "phase": phase,
            "jobs": jobs,
            "arms": arms,
            "sources": sources,
            "inputs": input_hashes,
            "n": 5000,
            "d": 25,
            "threads": 1,
            "timeout_seconds": 60,
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "design": "../DESIGN.md",
        },
    )
    return folder


def analyze(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    rows = [
        json.loads(path.read_text())
        for path in sorted((folder / "fits").glob("*.json"), key=lambda p: int(p.stem))
    ]
    expected = {tuple(job) for job in manifest["jobs"]}
    actual = [(r["function"], r["m"], r["seed"], r["arm"]) for r in rows]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("recorded jobs differ from the frozen manifest")
    for relative, digest in manifest["sources"].items():
        if (
            hashlib.sha256((folder / "source" / relative).read_bytes()).hexdigest()
            != digest
        ):
            raise ValueError(f"source snapshot changed: {relative}")
    for seed in {r["seed"] for r in rows}:
        with np.load(folder / "inputs" / f"{seed}.npz", allow_pickle=False) as saved:
            x, basis, noise = saved["X"], saved["basis"], saved["noise"]
        if _digest(x, basis, noise) != manifest["inputs"][str(seed)]:
            raise ValueError("frozen input changed")
        for row in (r for r in rows if r["seed"] == seed and r["status"] == "ok"):
            truth = basis[:, : row["m"]]
            y = _response(x, truth, noise, row["function"])
            estimate = np.asarray(row["basis"])
            distance = np.linalg.norm(
                estimate @ estimate.T - truth @ truth.T
            ) / np.sqrt(2 * row["m"])
            if (
                row["data_sha256"] != _digest(x, y, truth)
                or abs(distance - row["edr"]) > 1e-10
            ):
                raise ValueError("data fingerprint or EDR audit failed")
            if not np.allclose(estimate.T @ estimate, np.eye(row["m"]), atol=1e-10):
                raise ValueError("nonorthonormal fitted basis")
            if not 0 < row["fit_seconds"] <= row["process_seconds"]:
                raise ValueError("invalid fit/process timing")
    table = []
    for function in FUNCTIONS:
        for m in (2, 3, 5):
            for arm in manifest["arms"]:
                assigned = [
                    r
                    for r in rows
                    if (r["function"], r["m"], r["arm"]) == (function, m, arm)
                ]
                valid = [r for r in assigned if r["status"] == "ok"]
                table.append(
                    {
                        "function": function,
                        "m": m,
                        "arm": arm,
                        "attempts": len(assigned),
                        "successes": len(valid),
                        "mean_edr": float(np.mean([r["edr"] for r in valid]))
                        if valid
                        else None,
                        "recovered": sum(r["edr"] <= 0.1 for r in valid),
                        "median_fit_seconds": float(
                            np.median([r["fit_seconds"] for r in valid])
                        )
                        if valid
                        else None,
                        "median_process_seconds": float(
                            np.median([r["process_seconds"] for r in assigned])
                        )
                        if assigned
                        else None,
                        "median_peak_rss_mib": float(
                            np.median([r["peak_rss_mib"] for r in valid])
                        )
                        if valid
                        else None,
                    }
                )
    summary = {
        "phase": manifest["phase"],
        "expected": len(manifest["jobs"]),
        "recorded": len(rows),
        "successes": sum(r["status"] == "ok" for r in rows),
        "table": table,
    }
    if manifest["phase"] == "pilot":
        scores = {}
        expected = 2 * len(SEEDS["pilot"])
        for arm in manifest["arms"][1:]:
            primary = [
                r
                for r in rows
                if r["arm"] == arm and r["function"] == "original" and r["m"] in (3, 5)
            ]
            scores[arm] = (
                float(np.mean([np.log(max(r["edr"], 1e-12)) for r in primary]))
                if len(primary) == expected
                and all(r["status"] == "ok" for r in primary)
                else None
            )
        eligible = {k: v for k, v in scores.items() if v is not None}
        _write(
            folder / "chosen.json",
            {
                "arm": min(eligible, key=eligible.get) if eligible else None,
                "primary_mean_log_edr": scores,
            },
        )
    else:
        summary["gate"] = evaluate_gate(rows, len(manifest["jobs"]))
    _write(folder / "summary.json", summary)
    return summary


def run_phase(phase):
    folder = freeze(phase)
    manifest = json.loads((folder / "manifest.json").read_text())
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONPATH=str(folder / "source"),
    )
    for i, job in enumerate(manifest["jobs"]):
        started = time.perf_counter()
        command = [
            sys.executable,
            "-m",
            "experiments.gaussian_fourier_study",
            "worker",
            "--folder",
            str(folder),
            "--job",
            str(i),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=folder / "source",
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            process_seconds = time.perf_counter() - started
            (folder / "fits" / f"{i}.log").write_text(
                completed.stdout + completed.stderr
            )
            row_path = folder / "fits" / f"{i}.json"
            row = (
                json.loads(row_path.read_text())
                if row_path.exists()
                else {
                    "function": job[0],
                    "m": job[1],
                    "seed": job[2],
                    "arm": job[3],
                    "status": "worker_failed",
                    "error": completed.stderr,
                }
            )
            row.update(process_seconds=process_seconds, returncode=completed.returncode)
            if completed.returncode:
                row["status"] = "worker_failed"
        except subprocess.TimeoutExpired:
            row = {
                "function": job[0],
                "m": job[1],
                "seed": job[2],
                "arm": job[3],
                "status": "timeout",
                "process_seconds": time.perf_counter() - started,
            }
        _write(folder / "fits" / f"{i}.json", row)
        if (i + 1) % 12 == 0:
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "completed": i + 1,
                        "assigned": len(manifest["jobs"]),
                    }
                ),
                flush=True,
            )
    summary = analyze(folder)
    print(
        json.dumps(
            summary.get(
                "gate",
                {
                    "phase": phase,
                    "successes": summary["successes"],
                    "assigned": summary["expected"],
                },
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("pilot", "selection", "validation", "analyze", "worker")
    )
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--job", type=int)
    args = parser.parse_args()
    if args.phase == "worker":
        worker(args.folder, args.job)
    elif args.phase == "analyze":
        analyze(args.folder)
    else:
        run_phase(args.phase)


if __name__ == "__main__":
    main()
