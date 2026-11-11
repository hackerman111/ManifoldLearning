"""Frozen paired EDR screen for the cheapest chart metric approximations."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
import zipfile
from types import SimpleNamespace

import numpy as np

from experiments import full_ridge_batched as backend
from experiments import full_ridge_fast_study as harness
from experiments import light_metric

ROOT = harness.ROOT
OUT = ROOT / "benchmarks/results/verification/light-metric-edr-20261026"
CASES = tuple(itertools.product(("additive", "quadratic", "sine"), (3, 5)))
SEEDS = {
    "selection": tuple(range(1262000, 1262010)),
    "validation": tuple(range(1262100, 1262110)),
}
CANDIDATES = light_metric.ARMS[1:]
write, sha = harness.write, harness.sha


def fit(x, y, truth, seed, arm):
    # The existing harness dispatches its direct arm through this worker-local binding.
    harness.fast = SimpleNamespace(
        calculate_statistics=backend.calculate_statistics,
        solve=lambda index, stats: light_metric.solve(index, stats, arm),
    )
    row = harness.fit(x, y, truth, seed, "direct")
    return {**row, "arm": arm, "status": "ok"}


def freeze(phase, arms):
    harness.OUT, harness.CASES, harness.SEEDS, harness.ARMS = (
        OUT,
        CASES,
        SEEDS,
        tuple(arms),
    )
    directory, manifest = harness.freeze(phase)
    extras = [
        ROOT / name
        for name in (
            "tests/test_light_metric.py",
            "tests/test_light_metric_study.py",
            "benchmarks/results/verification/light-metric-edr-20261026/PROOF.md",
        )
    ]
    with zipfile.ZipFile(
        directory / "sources.zip", "a", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in extras:
            manifest["sources"][str(path.relative_to(ROOT))] = sha(path)
            archive.write(path, path.relative_to(ROOT))
    manifest["archive_sha256"] = sha(directory / "sources.zip")
    manifest["arms"] = arms
    manifest["settings"] = {
        str(m): harness.settings(harness.case(25, m, 5000, 0.3), "Grassmann")[0]
        for m in (3, 5)
    }
    manifest["bootstrap"] = {
        "replicates": 50000,
        "rng_seed": 1262026,
        "upper_quantile": 1 - 0.05 / 3,
        "unit": "seed cluster containing all six cells",
    }
    del manifest["fingerprint"]
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    write(directory / "manifest.json", manifest)
    return directory, manifest


def verify(directory, manifest):
    for name, expected in manifest["sources"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f"frozen source drift: {name}")
    if sha(directory / "sources.zip") != manifest["archive_sha256"]:
        raise RuntimeError("source archive drift")
    for name, expected in manifest["inputs"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f"frozen input drift: {name}")


def summarize(pairs, control, candidate, planned):
    deltas = np.array([p["delta_edr"] for p in pairs])
    clusters = [
        np.mean([p["delta_edr"] for p in pairs if p["seed"] == seed])
        for seed in sorted({p["seed"] for p in pairs})
    ]
    if len(pairs) == planned:
        rng = np.random.default_rng(1262026)
        bootstrap = np.mean(
            rng.choice(clusters, size=(50000, len(clusters)), replace=True), axis=1
        )
        upper = float(np.quantile(bootstrap, 1 - 0.05 / 3))
        interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    else:
        upper, interval = None, None
    cells = {
        f"{link}-m{m}": float(
            np.mean(
                [p["delta_edr"] for p in pairs if p["link"] == link and p["m"] == m]
            )
        )
        for link, m in CASES
        if any(p["link"] == link and p["m"] == m for p in pairs)
    }

    def fraction(rows):
        stages = [s for r in rows for s in r.get("stages", [])]
        return sum(s["converged"] for s in stages) / len(stages) if stages else 0.0

    ratios = {
        key: float(np.median([p[key] for p in pairs])) if pairs else None
        for key in ("fit_ratio", "process_ratio", "rss_ratio")
    }
    recovered = [
        sum(r["edr"] <= 0.1 for r in rows if r["status"] == "ok")
        for rows in (control, candidate)
    ]
    c0, c1 = fraction(control), fraction(candidate)
    gates = {
        "complete": len(pairs) == planned
        and len(control) == planned
        and len(candidate) == planned,
        "pairing": all(p["paired"] for p in pairs),
        "mean_edr": bool(len(deltas)) and float(deltas.mean()) <= -0.005,
        "cluster_upper": upper is not None and upper < 0,
        "cell_mean_edr": len(cells) == len(CASES) and max(cells.values()) <= 0.02,
        "worst_pair_edr": bool(len(deltas)) and float(deltas.max()) <= 0.08,
        "recovery": recovered[1] >= recovered[0],
        "convergence": c1 >= c0 - 0.05,
        "fit_time": ratios["fit_ratio"] is not None and ratios["fit_ratio"] <= 1.25,
        "process_time": ratios["process_ratio"] is not None
        and ratios["process_ratio"] <= 1.25,
        "rss": ratios["rss_ratio"] is not None and ratios["rss_ratio"] <= 1.10,
    }
    return {
        "pairs_count": len(pairs),
        "planned_pairs": planned,
        "mean_edr_control": float(
            np.mean([r["edr"] for r in control if r["status"] == "ok"])
        )
        if pairs
        else None,
        "mean_edr_candidate": float(
            np.mean([r["edr"] for r in candidate if r["status"] == "ok"])
        )
        if pairs
        else None,
        "mean_delta_edr": float(deltas.mean()) if len(deltas) else None,
        "cluster_ci95": interval,
        "familywise_upper": upper,
        "cell_mean_deltas": cells,
        "worst_pair": max(pairs, key=lambda p: p["delta_edr"]) if pairs else None,
        "recovery_control": recovered[0],
        "recovery_candidate": recovered[1],
        "convergence_control": c0,
        "convergence_candidate": c1,
        **ratios,
        "gates": gates,
        "pairs": pairs,
    }


def analyze(phase):
    directory = OUT / phase
    manifest = json.loads((directory / "manifest.json").read_text())
    verify(directory, manifest)
    rows = [
        json.loads(p.read_text()) for p in sorted((directory / "fits").glob("*.json"))
    ]
    expected = {tuple(job) for job in manifest["jobs"]}
    keys = [(r["link"], r["m"], r["seed"], r["arm"]) for r in rows]
    schedule_ok = len(set(keys)) == len(keys) and set(keys) == expected
    row_checks, max_edr_error = {}, 0.0
    for row in rows:
        if row["fingerprint"] != manifest["fingerprint"]:
            raise RuntimeError("row fingerprint mismatch")
        if row["status"] != "ok":
            row_checks[(row["link"], row["m"], row["seed"], row["arm"])] = (
                False,
                False,
            )
            continue
        basis = np.asarray(row["basis"])
        with np.load(
            directory / "inputs" / f"{row['link']}-m{row['m']}-{row['seed']}.npz"
        ) as saved:
            truth = saved["truth"]
        orthogonal = np.linalg.norm(basis.T @ basis - np.eye(row["m"]))
        geometry_ok = bool(np.isfinite(basis).all() and orthogonal <= 1e-10)
        actual = np.linalg.norm(basis @ basis.T - truth @ truth.T) / np.sqrt(
            2 * row["m"]
        )
        max_edr_error = max(max_edr_error, abs(float(actual) - row["edr"]))
        geometry_ok &= abs(float(actual) - row["edr"]) <= 1e-10
        numerical_ok = True
        for stage in row["stages"]:
            values = stage["profile_history"]
            numerical_ok &= stage["normal_residual_max"] <= 1e-8 and all(
                b <= a + 1e-10 * max(1, abs(a)) for a, b in itertools.pairwise(values)
            )
        row_checks[(row["link"], row["m"], row["seed"], row["arm"])] = (
            geometry_ok,
            numerical_ok,
        )
    lookup = dict(zip(keys, rows))
    planned = len(SEEDS[phase]) * len(CASES)
    summaries = {}
    for arm in manifest["arms"]:
        if arm == "identity":
            continue
        control, candidate, pairs = [], [], []
        for link, m, seed in itertools.product(
            ("additive", "quadratic", "sine"), (3, 5), SEEDS[phase]
        ):
            old, new = (
                lookup.get((link, m, seed, "identity")),
                lookup.get((link, m, seed, arm)),
            )
            if old is not None:
                control.append(old)
            if new is not None:
                candidate.append(new)
            if (
                old is None
                or new is None
                or old["status"] != "ok"
                or new["status"] != "ok"
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
                    "rss_ratio": new["process_peak_rss_mib"]
                    / old["process_peak_rss_mib"],
                    "paired": all(
                        old[k] == new[k]
                        for k in (
                            "data_sha256",
                            "initial_sha256",
                            "first_weights_sha256",
                            "first_directions_sha256",
                        )
                    ),
                }
            )
        summary = summarize(pairs, control, candidate, planned)
        checks = [
            row_checks[(r["link"], r["m"], r["seed"], r["arm"])]
            for r in control + candidate
        ]
        summary["gates"].update(
            schedule=schedule_ok,
            geometry=bool(checks) and all(c[0] for c in checks),
            numerical=bool(checks) and all(c[1] for c in checks),
        )
        summary["passed"] = all(summary["gates"].values())
        summaries[arm] = summary
    result = {
        "phase": phase,
        "planned": len(expected),
        "recorded": len(rows),
        "status_counts": {
            s: sum(r["status"] == s for r in rows) for s in {r["status"] for r in rows}
        },
        "max_recomputed_edr_error": max_edr_error,
        "candidates": summaries,
        "passed_arms": [a for a, s in summaries.items() if s["passed"]],
    }
    write(directory / "summary.json", result)
    print(
        json.dumps(
            {
                "phase": phase,
                "recorded": len(rows),
                "passed_arms": result["passed_arms"],
                "results": {
                    a: {
                        k: v
                        for k, v in s.items()
                        if k not in ("pairs", "worst_pair", "cell_mean_deltas")
                    }
                    for a, s in summaries.items()
                },
            },
            default=harness.default,
        ),
        flush=True,
    )
    return result


def run(phase):
    arms = list(light_metric.ARMS)
    if phase == "validation":
        selected = json.loads((OUT / "selection/summary.json").read_text())
        audit = json.loads((OUT / "selection/independent_audit.json").read_text())
        if (
            not selected["passed_arms"]
            or not audit["integrity_ok"]
            or audit["agreed_passed_arms"] != selected["passed_arms"]
        ):
            raise RuntimeError(
                "no independently audited passing selection; validation untouched"
            )
        selected_manifest = json.loads((OUT / "selection/manifest.json").read_text())
        verify(OUT / "selection", selected_manifest)
        if audit.get("manifest_sha256") != sha(
            OUT / "selection/manifest.json"
        ) or audit.get("summary_sha256") != sha(OUT / "selection/summary.json"):
            raise RuntimeError("stale audit; validation remains untouched")
        arms = ["identity", *selected["passed_arms"]]
    directory, manifest = freeze(phase, arms)
    if (
        phase == "validation"
        and manifest["sources"]
        != json.loads((OUT / "selection/manifest.json").read_text())["sources"]
    ):
        raise RuntimeError("source changed after selection")
    env = dict(
        os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    started = time.perf_counter()
    for i, (link, m, seed, arm) in enumerate(manifest["jobs"], 1):
        verify(directory, manifest)
        name = f"{link}-m{m}-{seed}-{arm}"
        log = directory / "fits" / f"{name}.log"
        process = harness.run_process(
            [
                sys.executable,
                "-m",
                "experiments.light_metric_study",
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
        if i % 12 == 0 or row["status"] != "ok":
            print(
                f"{phase}: {i}/{len(manifest['jobs'])}, {time.perf_counter() - started:.1f}s, {row['status']}",
                flush=True,
            )
    write(
        directory / "execution.json",
        {"wall_seconds": time.perf_counter() - started, "recorded": i},
    )
    return analyze(phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pilot", "run", "worker", "analyze"))
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.action == "worker":
        phase, link, m, seed, arm = args.arguments
        with np.load(OUT / phase / "inputs" / f"{link}-m{m}-{seed}.npz") as saved:
            result = fit(saved["x"], saved["y"], saved["truth"], int(seed), arm)
        print(json.dumps(result, default=harness.default, allow_nan=False))
    elif args.action == "pilot":
        x, y, truth = harness.arrays(
            harness.case(25, 5, 5000, 0.3, "additive"), 1262900
        )
        rows = [fit(x, y, truth, 1262900, arm) for arm in light_metric.ARMS]
        write(OUT / "pilot.json", rows)
        print(
            json.dumps(
                [
                    {
                        k: r[k]
                        for k in ("arm", "status", "fit_seconds", "orthogonality_error")
                    }
                    for r in rows
                ]
            )
        )
    elif args.action == "run":
        run(args.arguments[0])
    else:
        analyze(args.arguments[0])


if __name__ == "__main__":
    main()
