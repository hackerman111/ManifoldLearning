"""Замороженное парное исследование; python -m experiments.e_means_adp.study."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import scipy
from experiments.r_sdr.design import Case, catalog, generate, metrics
from experiments.r_sdr.runner import run_process

from ADP import ADP_Config, ADP_multi_index, ADP_solver
from ADP.solver import grassman as g

from .candidate import BATCHES, EPOCHS, FINAL_STEPS, MiniBatchSolver

ROOT = Path(__file__).resolve().parents[3]
METHODS = ["original20", "full6", "mini", "mini_ema"]
SELECTION = list(range(1131400, 1131410))
VALIDATION = list(range(1132400, 1132410))
THREAD_ENV = dict.fromkeys(
    (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ),
    "1",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    def convert(x):
        return x.tolist() if isinstance(x, np.ndarray) else x.item()

    path.write_text(
        json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n"
    )


def cases() -> list[Case]:
    return [replace(c, n=800) for c in catalog("seven-families")] + [
        Case("scale-additive-d50", 2000, 50, 3)
    ]


def worker(job: dict) -> None:
    data = np.load(job["data"])
    method = job["method"]
    solver = (
        MiniBatchSolver(job["mini_seed"], ema=method == "mini_ema")
        if method in {"mini", "mini_ema"}
        else g.solve
    )
    options = {"max_steps": 6} if method == "full6" else {}
    model = ADP_multi_index(
        data["truth"].shape[1],
        ADP_Config(**job["config"]),
        ADP_solver(solver, **options),
    )
    started = time.perf_counter()
    model.fit(data["X"], data["y"])
    assert model.basis_ is not None
    elapsed = time.perf_counter() - started
    out = Path(job["output"])
    np.savez(
        out / "basis.npz",
        basis=model.basis_,
        initial=model.result_.beta_init,
        coefficients=model.coefficients_,
        eigenvalues=model.eigenvalues_,
    )
    diagnostics = model.solver_diagnostics_
    measured = metrics(model.basis_, data["truth"])
    histories = [d.get("profile_history", ()) for d in diagnostics]
    increases = [
        (float(h[-1]) - float(h[0])) / max(1, abs(float(h[0])))
        for h in histories
        if len(h)
    ]
    row = {
        "fit_seconds": elapsed,
        "worker_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "edr": measured["projector_distance"],
        **{k: v for k, v in measured.items() if k != "projector_distance"},
        "all_inner_certified": all(d["converged"] for d in diagnostics),
        "n_inner_certified": sum(d["converged"] is True for d in diagnostics),
        "n_inner_total": len(diagnostics),
        "stop_reason": model.result_.stop_reason,
        "selected_iteration": model.effective_parameters_["selected_iteration"],
        "full_guard_max_increase": max(increases, default=0.0),
    }
    write_json(out / "result.json", row)
    write_json(out / "trace.json", model.effective_parameters_)
    write_json(out / "profile.json", model.profile_)


def evaluate(rows: list[dict], protocol: dict) -> dict:
    """Предзаданные gates; неполные/неуспешные пары не исчезают из знаменателя."""
    expected = len(protocol["cases"]) * len(protocol["seeds"])
    lookup = {(r["case"], r["seed"], r["method"]): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("duplicate result rows")
    names = [c["name"] for c in protocol["cases"]]
    allowed = {
        (name, seed, method)
        for name in names
        for seed in protocol["seeds"]
        for method in protocol["methods"]
    }
    if not lookup.keys() <= allowed:
        raise ValueError("foreign result row")
    result: dict = {"expected_blocks": expected, "candidates": {}, "summary": {}}
    for method in protocol["methods"]:
        entries = [r for r in rows if r["method"] == method]
        valid = [r for r in entries if r["status"] == "ok"]
        result["summary"][method] = {
            "valid": len(valid),
            "failures": len(entries) - len(valid),
            "missing": expected - len(entries),
            "mean_edr": float(np.mean([r["edr"] for r in valid])) if valid else None,
            "recovery_le_0.3": sum(r["edr"] <= 0.3 for r in valid),
            "inner_certified": sum(r["n_inner_certified"] for r in valid),
            "inner_total": sum(r["n_inner_total"] for r in valid),
        }
        if method == "original20":
            continue
        pairs = []
        for name in names:
            for seed in protocol["seeds"]:
                a = lookup.get((name, seed, "original20"))
                b = lookup.get((name, seed, method))
                if a and b and a["status"] == b["status"] == "ok":
                    pairs.append((a, b))
        paired = len(pairs) == expected
        d = {
            "pass": False,
            "pairs": len(pairs),
            "failures": len(entries) - len(valid),
            "missing": expected - len(entries),
        }
        if pairs:
            delta = np.array([b["edr"] - a["edr"] for a, b in pairs])
            by_seed = [
                np.mean([b["edr"] - a["edr"] for a, b in pairs if a["seed"] == seed])
                for seed in protocol["seeds"]
                if any(a["seed"] == seed for a, _ in pairs)
            ]
            rng = np.random.default_rng(1133400)
            boot = rng.choice(by_seed, size=(10000, len(by_seed))).mean(axis=1)
            upper = float(np.quantile(boot, 0.95))
            time_ratio = float(
                np.median([b["fit_seconds"] / a["fit_seconds"] for a, b in pairs])
            )
            rss_ratio = float(
                np.median([b["peak_rss_mib"] / a["peak_rss_mib"] for a, b in pairs])
            )
            case_deltas = {
                name: float(
                    np.mean(
                        [b["edr"] - a["edr"] for a, b in pairs if a["case"] == name]
                    )
                )
                for name in names
                if any(a["case"] == name for a, _ in pairs)
            }
            gates = {
                "complete": paired,
                "orthogonality": all(
                    b["orthogonality_error"] <= 1e-8 for _, b in pairs
                ),
                "full_profile_guard": all(
                    b["full_guard_max_increase"] <= 1e-12 for _, b in pairs
                ),
                "worst_edr_regression": float(delta.max()) <= 0.05,
                "per_case_mean_regression": max(case_deltas.values()) <= 0.02,
                "rss": rss_ratio <= 1.10,
                "speed_or_quality": (upper <= 0.01 and time_ratio <= 0.90)
                or (upper <= -0.01 and time_ratio <= 1.10),
            }
            d.update(
                mean_delta=float(delta.mean()),
                upper95_mean_delta=upper,
                worst_delta=float(delta.max()),
                median_time_ratio=time_ratio,
                median_rss_ratio=rss_ratio,
                case_mean_deltas=case_deltas,
                gates=gates,
            )
            d["pass"] = all(gates.values())
        result["candidates"][method] = d
    return result


def freeze(root: Path) -> dict:
    if root.exists():
        raise FileExistsError(f"output already exists: {root}")
    root.mkdir(parents=True)
    source = root / "source"
    files = sorted((ROOT / "ADP").rglob("*.py"))
    files += sorted((ROOT / "benchmarks/experiments/e_means_adp").glob("*.py"))
    files += sorted((ROOT / "benchmarks/experiments/r_sdr").glob("*.py"))
    files += [ROOT / "experiments.py"]
    hashes = {}
    for path in files:
        rel = path.relative_to(ROOT)
        target = source / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(rel)] = digest(target)
    buf = io.StringIO()
    with redirect_stdout(buf):
        np.show_config()
    write_json(
        root / "environment.json",
        {
            "python": sys.version,
            "executable": sys.executable,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "load_average": os.getloadavg(),
            "numpy_config": buf.getvalue(),
            "threads": {k: os.environ.get(k) for k in THREAD_ENV},
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "git_dirty": subprocess.check_output(
                ["git", "status", "--short"], cwd=ROOT, text=True
            ),
        },
    )
    protocol = {
        "created_unix": time.time(),
        "cases": [asdict(c) for c in cases()],
        "selection_seeds": SELECTION,
        "validation_seeds": VALIDATION,
        "methods": METHODS,
        "sources_sha256": hashes,
        "epochs": EPOCHS,
        "batches": BATCHES,
        "final_steps": FINAL_STEPS,
        "batch_mass_scale": "J/b; absolute smooth-GN damping unchanged",
        "ema": "projector factor, peak .5, floor .05, t/(M-1), initial mass1e-10",
        "kernel": "max(1-u^2,0), u=(alpha^2 residual2+sum lambda*z^2)/h^2",
        "local_ridge": 0.0,
        "initializer_ridge": 1e-8,
        "dtype": "float64",
        "timeout_seconds": 120,
        "memory_mib": 2048,
        "adp": (
            "N_J=min(n,200) for d<=20;min(n,300) for d>20, "
            "N_phi12,N_loc20,N_lin2d; other public defaults"
        ),
        "gates": (
            "complete80pairs;orth<=1e-8;computed profile increase<=1e-12;"
            "worst EDR delta<=.05;each case mean delta<=.02;"
            "median RSS ratio<=1.10;either upper95meanEDRdelta<=.01 and "
            "median time ratio<=.90 OR upper95delta<=-.01 and time ratio<=1.10"
        ),
        "bootstrap": "10000 seed-cluster resamples; one-sided95 percentile, RNG1133400",
        "promotion": "none; no universal EDR/convergence guarantee",
    }
    write_json(root / "PROTOCOL.json", protocol)
    return protocol


def stage(
    root: Path, protocol: dict, name: str, seeds: list[int], methods: list[str]
) -> dict:
    output = root / name
    output.mkdir()
    spec = {"cases": protocol["cases"], "seeds": seeds, "methods": methods}
    write_json(output / "manifest.json", spec)
    rows = []
    hashes = {}
    env = {**os.environ, **THREAD_ENV, "PYTHONPATH": str(root / "source")}
    stage_cases = [Case(**c) for c in spec["cases"]]
    # Freeze inputs before outcomes; validation data remains absent.
    for case in stage_cases:
        for seed in seeds:
            block = output / f"{case.name}-{seed}"
            block.mkdir()
            X, y, truth = generate(case, seed)
            data = block / "data.npz"
            np.savez(data, X=X, y=y, truth=truth)
            hashes[str(data.relative_to(root))] = digest(data)
    write_json(output / "data_hashes.json", hashes)
    total = len(stage_cases) * len(seeds) * len(methods)
    for case in stage_cases:
        for seed in seeds:
            block = output / f"{case.name}-{seed}"
            data = block / "data.npz"
            config_seed = int(np.random.SeedSequence([seed, 4101]).generate_state(1)[0])
            mini_seed = int(np.random.SeedSequence([seed, 4102]).generate_state(1)[0])
            order = np.random.default_rng(
                np.random.SeedSequence([seed, 4103])
            ).permutation(methods)
            for method in order.tolist():
                fit = block / method
                fit.mkdir()
                job = {
                    "data": str(data),
                    "output": str(fit),
                    "method": method,
                    "mini_seed": mini_seed,
                    "config": {
                        "seed": config_seed,
                        "N_J": min(case.n, 200 if case.d <= 20 else 300),
                        "N_phi": 12,
                        "N_loc": 20,
                        "N_lin": 2 * case.d,
                    },
                }
                write_json(fit / "job.json", job)
                row = {
                    "case": case.name,
                    "seed": seed,
                    "method": method,
                    "data_sha256": hashes[str(data.relative_to(root))],
                }
                row.update(
                    run_process(
                        [
                            sys.executable,
                            "-m",
                            "experiments.e_means_adp.study",
                            "worker",
                            str(fit / "job.json"),
                        ],
                        env,
                        fit / "process.log",
                        cwd=root / "source",
                        timeout=protocol["timeout_seconds"],
                        memory_mib=protocol["memory_mib"],
                    )
                )
                if row["status"] == "ok":
                    try:
                        row.update(json.loads((fit / "result.json").read_text()))
                        row["peak_rss_mib"] = max(
                            row["peak_rss_mib"] or 0, row["worker_peak_rss_mib"]
                        )
                    except (OSError, ValueError) as error:
                        row.update(status="invalid_result", error=str(error))
                rows.append(row)
                with (output / "runs.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(
                f"{name} {case.name} seed{seed}: {len(rows)}/{total}",
                flush=True,
            )
    evaluated = evaluate(rows, spec)
    write_json(output / "gates.json", evaluated)
    return evaluated


def campaign(root: Path) -> None:
    from .audit import audit_stage

    protocol = freeze(root)
    print("Source/protocol frozen before selection outcomes", flush=True)
    selected = stage(root, protocol, "selection", SELECTION, METHODS)
    audit = audit_stage(root, "selection")
    write_json(root / "selection/audit.json", audit)
    if not audit["pass"]:
        raise RuntimeError("selection read-back audit failed; validation is blocked")
    eligible = [m for m in ("mini", "mini_ema") if selected["candidates"][m]["pass"]]
    validation_passed: list[str] = []
    if eligible:
        validation = stage(
            root, protocol, "validation", VALIDATION, ["original20", "full6", *eligible]
        )
        audit = audit_stage(root, "validation")
        write_json(root / "validation/audit.json", audit)
        if not audit["pass"]:
            raise RuntimeError("validation read-back audit failed")
        validation_passed = [m for m in eligible if validation["candidates"][m]["pass"]]
    write_json(
        root / "completion.json",
        {
            "execution_complete": True,
            "selection_eligible": eligible,
            "validation_passed": validation_passed,
            "validation_failed": [m for m in eligible if m not in validation_passed],
            "validated_success": bool(validation_passed),
            "validation": "run" if eligible else "untouched_selection_failed",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "worker", "audit"])
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.action == "worker":
        worker(json.loads(args.path.read_text()))
    elif args.action == "run":
        campaign(args.path.resolve())
    else:
        from .audit import audit_stage

        for name in ("selection", "validation"):
            if (args.path / name).exists():
                print(json.dumps(audit_stage(args.path.resolve(), name), indent=2))


if __name__ == "__main__":
    main()
