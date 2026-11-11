from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from experiments.r_sdr.design import Case, catalog, generate, metrics
from experiments.r_sdr.protocol import (
    ROOT,
    digest,
    ensure_manifest,
    environment,
    source_files,
)
from experiments.r_sdr.runner import adp_settings, read_worker_meta, run_process

HERE = ROOT / "benchmarks/experiments/grassman_accuracy_2026_10_22"
METHODS = (
    "baseline",
    "budget10",
    "relative10",
    "smooth_relative",
    "stable_jacobian",
    "diagonal",
    "gn_quadratic",
    "ridge6",
    "ridge4",
    "ridge2",
    "krylov10",
    "krylov20",
    "lora1_10",
    "lora_k10",
    "lora_k20",
    "projected10",
    "projected20",
    "svd1_10",
    "svd_k10",
    "svd_k20",
    "enriched10",
    "enriched20",
    "svd_greedy10",
    "MEANMAVE",
)
TIMEOUT, MEMORY_MIB = 300.0, 4096.0


def _catalog_cases() -> dict[str, Case]:
    cases: dict[str, Case] = {}
    for case in (*catalog("full"), *catalog("multi-rank")):
        if case.name in cases and cases[case.name] != case:
            raise ValueError(f"catalog case name collision: {case.name}")
        cases[case.name] = case
    return cases


def _snapshot(output: Path, sources: list[Path]) -> None:
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    with zipfile.ZipFile(output / "sources.zip", "x", zipfile.ZIP_DEFLATED) as zf:
        for path in sources:
            zf.write(path, path.relative_to(ROOT))
    with zipfile.ZipFile(output / "sources.zip") as zf:
        zf.extractall(output / "source")
    for rel, sha in hashes.items():
        if digest(output / "source" / rel) != sha:
            raise ValueError(f"snapshot hash mismatch: {rel}")


def _fit_adp(job: dict) -> None:
    from experiments.grassman_accuracy_2026_10_22.candidates import solver_for
    from experiments.r_sdr import worker

    worker.solver_for = lambda _method: solver_for(job["method"])
    worker.run(job)


def _summaries(
    rows: list[dict], cases: list[Case], methods: list[str], seeds: list[int]
) -> dict:
    lookup = {(r["case"], r["seed"], r["method"]): r for r in rows}
    result = {
        "planned_rows": len(cases) * len(methods) * len(seeds),
        "rows_recorded": len(rows),
        "methods": {},
    }
    for method in (m for m in methods if m != "MEANMAVE"):
        by_case = {}
        for case in cases:
            pairs = [
                (
                    lookup.get((case.name, s, method)),
                    lookup.get((case.name, s, "MEANMAVE")),
                )
                for s in seeds
            ]
            valid = [
                (a, b)
                for a, b in pairs
                if a
                and b
                and a.get("status") == b.get("status") == "ok"
                and "projector_distance" in a
                and "projector_distance" in b
            ]
            ratios = np.array(
                [
                    a["projector_distance"] / max(b["projector_distance"], 1e-15)
                    for a, b in valid
                ]
            )
            times = np.array([a["fit_seconds"] / b["fit_seconds"] for a, b in valid])
            rng = np.random.default_rng(
                np.random.SeedSequence([sum(map(ord, method)), case.d, len(seeds)])
            )
            boot = (
                np.exp(
                    np.log(ratios)[
                        rng.integers(0, len(ratios), (2000, len(ratios)))
                    ].mean(axis=1)
                )
                if len(ratios)
                else np.array([])
            )
            stops = [
                r.get("stop_reason")
                for r in rows
                if r["case"] == case.name and r["method"] == method
            ]
            by_case[case.name] = {
                "planned_pairs": len(seeds),
                "valid_pairs": len(valid),
                "candidate_failures": sum(
                    not r or r.get("status") != "ok" for r, _ in pairs
                ),
                "meanmave_failures": sum(
                    not r or r.get("status") != "ok" for _, r in pairs
                ),
                "edr_ratio_geomean": float(np.exp(np.mean(np.log(ratios))))
                if len(ratios)
                else None,
                "edr_ratio_bootstrap95": np.quantile(boot, [0.025, 0.975]).tolist()
                if len(boot)
                else None,
                "fit_time_ratio_median": float(np.median(times))
                if len(times)
                else None,
                "fit_time_ratio_p90": float(np.quantile(times, 0.9))
                if len(times)
                else None,
                "candidate_edr_median": float(
                    np.median([a["projector_distance"] for a, _ in valid])
                )
                if valid
                else None,
                "meanmave_edr_median": float(
                    np.median([b["projector_distance"] for _, b in valid])
                )
                if valid
                else None,
                "stops": {s: stops.count(s) for s in set(stops)},
            }
        result["methods"][method] = by_case
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--cases")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.job:
        _fit_adp(json.loads(args.job.read_text()))
        return 0
    if not args.output or not args.cases or args.seed < 0 or args.runs < 1:
        parser.error("--output, --cases and valid --seed/--runs are required")
    methods = args.methods.split(",")
    if (
        len(set(methods)) != len(methods)
        or "MEANMAVE" not in methods
        or set(methods) - set(METHODS)
    ):
        parser.error(
            f"methods must be unique members of {METHODS} and include MEANMAVE"
        )
    all_cases = _catalog_cases()
    names = args.cases.split(",")
    if not names or len(set(names)) != len(names) or set(names) - all_cases.keys():
        parser.error("cases must be unique names from the full and multi-rank catalogs")
    cases = [all_cases[n] for n in names]
    if not (HERE / "candidates.py").is_file():
        parser.error("candidate implementation is not present: candidates.py")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    rscript = shutil.which("Rscript")
    if not rscript:
        parser.error("Rscript not found")
    env_info, env = environment(rscript, 1)
    sources = sorted(set(source_files()) | set(HERE.glob("*.py")))
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    protocol = {
        "schema": 1,
        "cases": [asdict(c) for c in cases],
        "methods": methods,
        "seed": args.seed,
        "runs": args.runs,
        "threads": 1,
        "timeout": TIMEOUT,
        "memory_mib": MEMORY_MIB,
        "rscript": rscript,
        "environment": env_info,
        "adp_config": {c.name: adp_settings(c, {}) for c in cases},
        "solver_config": {
            m: "candidates.solver_for; frozen defaults"
            for m in methods
            if m != "MEANMAVE"
        },
        "sources": source_hashes,
        "quality": "projector_distance=||P_est-P_true||_F/sqrt(2m)",
    }
    ensure_manifest(output, protocol, resume=False)
    frozen = output / "source"
    _snapshot(output, sources)
    env["PYTHONPATH"] = str(frozen)
    (output / "inputs").mkdir()
    (output / "fits").mkdir()
    rows = []
    for case in cases:
        for seed in range(args.seed, args.seed + args.runs):
            x, y, truth = generate(case, seed)
            stem = f"{case.name}-{seed}"
            np.savez_compressed(
                output / "inputs" / f"{stem}.npz", x=x, y=y, truth=truth
            )
            x_path, y_path = (
                output / "inputs" / f"{stem}-x.csv",
                output / "inputs" / f"{stem}-y.csv",
            )
            np.savetxt(x_path, x, delimiter=",", fmt="%.17g")
            np.savetxt(y_path, y, delimiter=",", fmt="%.17g")
            hashes = {
                "npz": digest(output / "inputs" / f"{stem}.npz"),
                "x": digest(x_path),
                "y": digest(y_path),
                "truth": hashlib.sha256(truth.tobytes()).hexdigest(),
            }
            order = list(methods)
            np.random.default_rng(seed + int(hashes["npz"][:8], 16)).shuffle(order)
            method_seed = int(
                np.random.SeedSequence([seed, 1]).generate_state(1)[0]
            ) % (2**31 - 1)
            for pos, method in enumerate(order):
                fit_dir = output / "fits" / f"{stem}-{method}"
                fit_dir.mkdir()
                basis, meta_path, log = (
                    fit_dir / "basis.csv",
                    fit_dir / "meta.json",
                    fit_dir / "worker.log",
                )
                if method == "MEANMAVE":
                    command = [
                        rscript,
                        "--vanilla",
                        str(frozen / "experiments/r_sdr/worker.R"),
                        "MAVE-package",
                        str(x_path),
                        str(y_path),
                        str(basis),
                        str(fit_dir / "meta.tsv"),
                        str(method_seed),
                        str(case.m),
                        "25",
                        "1e-5",
                    ]
                    meta_path = fit_dir / "meta.tsv"
                else:
                    job = {
                        "method": method,
                        "x": str(x_path),
                        "y": str(y_path),
                        "basis": str(basis),
                        "meta": str(meta_path),
                        "seed": method_seed,
                        "m": case.m,
                        "config": adp_settings(case, {}),
                        "solver": {},
                    }
                    job_path = fit_dir / "job.json"
                    job_path.write_text(json.dumps(job))
                    command = [
                        sys.executable,
                        "-m",
                        "experiments.grassman_accuracy_2026_10_22.benchmark",
                        "--job",
                        str(job_path),
                    ]
                timing = run_process(
                    command,
                    env,
                    log,
                    timeout=TIMEOUT,
                    memory_mib=MEMORY_MIB,
                    cwd=frozen,
                )
                row = {
                    "case": case.name,
                    "seed": seed,
                    "method": method,
                    "method_seed": method_seed,
                    "order": pos,
                    "n": case.n,
                    "d": case.d,
                    "m": case.m,
                    "input_hashes": hashes,
                    "status": timing["status"],
                    **timing,
                    "log": str(log.relative_to(output)),
                }
                try:
                    meta = read_worker_meta(
                        "MAVE-package" if method == "MEANMAVE" else "ADP-Grassmann",
                        meta_path,
                    )
                    row.update(meta)
                    estimate = np.loadtxt(basis, delimiter=",", ndmin=2)
                    row.update(metrics(estimate, truth))
                except Exception as exc:
                    if row["status"] == "ok":
                        row["status"] = "error"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
                with (output / "runs.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
    report = _summaries(
        rows, cases, methods, list(range(args.seed, args.seed + args.runs))
    )
    (output / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
