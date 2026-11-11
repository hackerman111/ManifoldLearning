"""Парное исследование initialization/solver без изменения public ADP defaults."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy

from ADP.core.ADP_Config import ADP_Config
from ADP.engine.common import index_fit

from .data import _generate_data, _make_seed_bundle
from .diagnostic import _code_fingerprint
from .inverse_moment_init import initialize_basis_inverse_moments
from .models import Build, ExperimentPoint
from .multiv2_quality import _baseline
from .runner import _fit, _git, _numpy_config, _threadpool_info

ROOT = Path("docs/experiments/index_recovery_2026-09-27")
VARIANTS = ("local", "local-cv", "sir-save", "cg", "pilot")


def make_protocol() -> dict:
    small = replace(_baseline("d10"), solver_max_steps=80)
    large = replace(_baseline("n1000"), solver_max_steps=80)
    cases = {
        "single_d10_quadratic": replace(
            small, mode="single", index_dim=1, link="quadratic", direction_mode="auto"
        ),
        "single_d100_quadratic": replace(
            large, mode="single", index_dim=1, link="quadratic", direction_mode="auto"
        ),
        "single_d100_square": replace(
            large, mode="single", index_dim=1, link="square", direction_mode="auto"
        ),
        "multi_d10_additive": small,
        "multi_d100_additive": large,
        "multi_d100_multiplicative": replace(large, link="multi_multiplicative"),
    }
    return {
        "code_fingerprint": _code_fingerprint(),
        "commit": _git("rev-parse", "HEAD"),
        "dirty_status": _git("status", "--short"),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numpy_config": _numpy_config(),
        "threadpools": _threadpool_info(),
        "cases": {name: asdict(point) for name, point in cases.items()},
        "variants": list(VARIANTS),
        "selection_seeds": list(range(73000, 73004)),
        "validation_seeds": list(range(74000, 74010)),
        "quality_threshold": 0.95,
        "nonregression_tolerance": 1e-10,
        "selection_gain": {"quality_median": 0.01, "recovery_rate": 0.25},
        "max_time_ratio": 2.0,
        "max_rss_ratio": 1.5,
        "fit_timeout_seconds": 180,
        "phase_budget_seconds": 2400,
        "solver_tol": 1e-6,
        "inverse_moment_slices": 10,
        "threads": 1,
        "kernel": "max(1-(distance2/h**2)**2,0)",
        "mass": "normalized moments plus external mass",
        "pairing": (
            "same X/Y/truth and model seed; independent spawned centers/init/directions"
        ),
        "rss": "fresh worker process ru_maxrss, includes imports and data",
        "convergence": "native solver flag; CG uses one certified step, HPAO two",
        "scope": "Gaussian inputs, tau=0.4; no universal recovery guarantee",
    }


def worker(protocol: dict, case: str, variant: str, seed: int) -> dict:
    point = ExperimentPoint(**protocol["cases"][case])
    seeds = _make_seed_bundle("index-recovery-" + case, point, seed)
    data = _generate_data("index-recovery-" + case, point, seeds, seed)
    data_hash = hashlib.sha256(data.X.tobytes() + data.Y.tobytes()).hexdigest()
    row = {
        "case": case,
        "variant": variant,
        "seed": seed,
        "seed_bundle": asdict(seeds),
        "model_seed": seeds.init,
        "data_sha256": data_hash,
        "error": None,
    }
    init_diagnostics = {}
    original = index_fit._initial_index

    def initializer(mode, d, index_dim, config, X, Y, centers, distance2, n_lin, rng):
        started = time.perf_counter()
        if variant == "sir-save":
            basis, diagnostics = initialize_basis_inverse_moments(
                X,
                Y,
                index_dim,
                slices=protocol["inverse_moment_slices"],
                seed=config.seed,
            )
            init_diagnostics.update(diagnostics)
            result = (basis[:, 0] if mode == "single" else basis.T, np.empty(0))
        else:
            result = original(
                mode, d, index_dim, config, X, Y, centers, distance2, n_lin, rng
            )
        init_diagnostics["wall_seconds"] = time.perf_counter() - started
        return result

    point = replace(
        point, index_init=variant if variant in {"local-cv", "pilot"} else "local"
    )
    build = Build(
        variant,
        ADP_Config(),
        solver_max_steps=80,
        solver="cg" if variant == "cg" else "lsmr",
    )
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with patch.object(index_fit, "_initial_index", initializer):
                row.update(
                    _fit(build, point, data, seeds.init, protocol["quality_threshold"])
                )
            for field in (
                "trace",
                "solver_diagnostics",
                "effective_config",
                "initial_eigenvalues",
            ):
                row[field] = json.loads(row[field])
            row["stationary"] = (
                max(
                    row["solver_diagnostics"][key]
                    for key in (
                        "riemannian_gradient",
                        "local_gradient",
                        "orthogonality",
                    )
                )
                < protocol["solver_tol"]
            )
        except Exception as error:
            # Experiments retain every numerical/library failure in the denominator.
            row.update(
                status="numerical_failure",
                error=f"{type(error).__name__}: {error}",
                quality=None,
                convergence_pass=False,
                quality_pass=False,
                recovered=False,
                stationary=False,
            )
        row["warnings"] = [
            f"{item.category.__name__}: {item.message}" for item in caught
        ]
    row["spent_seconds"] = time.perf_counter() - started
    row["peak_process_rss_mib"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    )
    row["initializer"] = init_diagnostics
    return row


def summarize(protocol: dict, rows: list[dict], phase: str) -> dict:
    seeds = protocol[f"{phase}_seeds"]
    seen = {(r["case"], r["variant"], r["seed"]): r for r in rows}
    if len(seen) != len(rows):
        raise ValueError("duplicate fits")
    groups = {}
    for case, variant in sorted({(r["case"], r["variant"]) for r in rows}):
        group = [
            seen[(case, variant, seed)]
            for seed in seeds
            if (case, variant, seed) in seen
        ]
        paired = []
        for row in group:
            base = seen.get((case, "local", row["seed"]))
            if base is None:
                raise ValueError("missing paired baseline")
            if (
                row.get("data_sha256")
                and base.get("data_sha256")
                and row["data_sha256"] != base["data_sha256"]
            ):
                raise ValueError("unpaired data")
            paired.append((row, base))
        valid = [
            (r, b)
            for r, b in paired
            if r.get("quality") is not None and b.get("quality") is not None
        ]
        deltas = [r["quality"] - b["quality"] for r, b in valid]
        failures = sum(r.get("quality") is None for r in group)
        recovery = sum(r.get("recovered") is True for r in group)
        converged = sum(r.get("convergence_pass") is True for r in group)
        baseline_recovery = sum(b.get("recovered") is True for _, b in paired)
        baseline_convergence = sum(b.get("convergence_pass") is True for _, b in paired)
        median_delta = float(np.median(deltas)) if deltas else None
        time_ratio = float(
            np.median([r["spent_seconds"] / b["spent_seconds"] for r, b in paired])
        )
        rss_ratios = [
            r["peak_process_rss_mib"] / b["peak_process_rss_mib"]
            for r, b in paired
            if r.get("peak_process_rss_mib") and b.get("peak_process_rss_mib")
        ]
        rss_ratio = float(np.median(rss_ratios)) if rss_ratios else None
        gates = {
            "complete_pairs": len(valid) == len(seeds) == len(group),
            "quality_nonregression": bool(deltas)
            and min(deltas) >= -protocol["nonregression_tolerance"],
            "recovery_nonregression": recovery >= baseline_recovery,
            "no_failures": failures == 0,
            "gain": median_delta is not None
            and (
                median_delta >= protocol["selection_gain"]["quality_median"]
                or (recovery - baseline_recovery) / len(seeds)
                >= protocol["selection_gain"]["recovery_rate"]
            )
            if phase == "selection"
            else median_delta is not None
            and (median_delta > 1e-10 or recovery > baseline_recovery),
            "time": time_ratio <= protocol["max_time_ratio"],
            "rss": rss_ratio is not None and rss_ratio <= protocol["max_rss_ratio"],
        }
        groups[f"{case}/{variant}"] = {
            "fits": len(group),
            "quality_median": float(
                np.median([r["quality"] for r in group if r.get("quality") is not None])
            )
            if valid
            else None,
            "initial_quality_median": float(
                np.median(
                    [
                        r["initial_quality"]
                        for r in group
                        if r.get("initial_quality") is not None
                    ]
                )
            )
            if valid
            else None,
            "quality_passes": sum(r.get("quality_pass") is True for r in group),
            "converged": converged,
            "stationary": sum(r.get("stationary") is True for r in group),
            "recovered": recovery,
            "failures": failures,
            "paired_quality_deltas": deltas,
            "paired_quality_delta_median": median_delta,
            "quality_regressions": sum(
                value < -protocol["nonregression_tolerance"] for value in deltas
            ),
            "convergence_nonregression_debug": (
                converged >= baseline_convergence
            ),
            "time_median_seconds": float(
                np.median([r["spent_seconds"] for r in group])
            ),
            "time_ratio_median": time_ratio,
            "rss_ratio_median": rss_ratio,
            "gates": gates,
            "passed": variant != "local" and all(gates.values()),
        }
    return {"phase": phase, "fits": len(rows), "groups": groups}


def run_phase(root: Path, phase: str) -> None:
    protocol_path = root / "protocol.json"
    if not protocol_path.exists():
        if phase != "selection":
            raise ValueError("selection must precede validation")
        root.mkdir(parents=True, exist_ok=True)
        protocol_path.write_text(json.dumps(make_protocol(), indent=2) + "\n")
    protocol = json.loads(protocol_path.read_text())
    if protocol["code_fingerprint"] != _code_fingerprint():
        raise ValueError("source fingerprint changed; use a new study directory")
    if phase == "selection":
        candidates = {
            case: [v for v in VARIANTS if point["mode"] == "multi" or v != "pilot"]
            for case, point in protocol["cases"].items()
        }
    else:
        selection = json.loads((root / "selection_summary.json").read_text())
        if not selection["complete"]:
            raise ValueError("incomplete selection; validation prohibited")
        candidates = {}
        for key, result in selection["groups"].items():
            if result["passed"]:
                case, variant = key.split("/")
                candidates.setdefault(case, ["local"]).append(variant)
        if not candidates:
            raise ValueError("no selection candidate passed; validation prohibited")
    output = root / f"{phase}.jsonl"
    if output.exists():
        raise ValueError("phase output already exists; refusing to overwrite fits")
    rows = []
    started = time.perf_counter()
    complete = True
    with output.open("x") as stream:
        for case, variants in candidates.items():
            for seed in protocol[f"{phase}_seeds"]:
                for variant in variants:
                    if (
                        time.perf_counter() - started
                        >= protocol["phase_budget_seconds"]
                    ):
                        complete = False
                        break
                    command = [
                        sys.executable,
                        "-m",
                        "experiments.index_recovery",
                        "worker",
                        "--output",
                        str(root),
                        "--case",
                        case,
                        "--variant",
                        variant,
                        "--seed",
                        str(seed),
                    ]
                    job_start = time.perf_counter()
                    try:
                        process = subprocess.run(
                            command,
                            capture_output=True,
                            text=True,
                            timeout=protocol["fit_timeout_seconds"],
                            check=True,
                            env={
                                **os.environ,
                                "OPENBLAS_NUM_THREADS": "1",
                                "OMP_NUM_THREADS": "1",
                                "MKL_NUM_THREADS": "1",
                            },
                        )
                        row = json.loads(process.stdout)
                    except (subprocess.SubprocessError, json.JSONDecodeError) as error:
                        row = {
                            "case": case,
                            "variant": variant,
                            "seed": seed,
                            "status": "numerical_failure",
                            "error": str(error),
                            "spent_seconds": time.perf_counter() - job_start,
                        }
                    rows.append(row)
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    print(
                        json.dumps(
                            {
                                k: row.get(k)
                                for k in (
                                    "case",
                                    "variant",
                                    "seed",
                                    "quality",
                                    "recovered",
                                    "spent_seconds",
                                    "error",
                                )
                            }
                        ),
                        flush=True,
                    )
                if not complete:
                    break
            if not complete:
                break
    summary = summarize(protocol, rows, phase)
    summary.update(complete=complete, spent_seconds=time.perf_counter() - started)
    (root / f"{phase}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("selection", "validation", "worker"))
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--case")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.phase == "worker":
        protocol = json.loads((args.output / "protocol.json").read_text())
        print(json.dumps(worker(protocol, args.case, args.variant, args.seed)))
    else:
        run_phase(args.output, args.phase)


if __name__ == "__main__":
    main()
