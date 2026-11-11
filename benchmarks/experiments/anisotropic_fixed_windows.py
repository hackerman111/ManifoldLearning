"""Deterministic fixed-window diagnostics for anisotropic geometry operators."""

from __future__ import annotations

import hashlib
import json
import time
import zipfile
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments import anisotropic_metrics as geometry
from experiments import full_ridge_batched as ridge
from experiments.coupled_penalties import Profile
from experiments.r_sdr.design import metrics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/results/verification/anisotropic-metrics-20261026/fixed"
ARMS = (*geometry.ARMS, "lbfgs")
REPEATS = 3
SEED = 20261026


def digest(*values):
    return hashlib.sha256(
        b"".join(np.ascontiguousarray(value).tobytes() for value in values)
    ).hexdigest()


def default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def synthetic(seed: int, windows: int, rank_mode: str):
    if windows not in (40, 80) or rank_mode not in ("spd", "mixed"):
        raise ValueError("fixed suite expects J=40/80 and spd/mixed covariances")
    rng = np.random.default_rng(seed + windows + (0 if rank_mode == "spd" else 1000))
    d, m = 25, 5
    truth = np.linalg.qr(rng.normal(size=(d, m)))[0]
    frame = np.linalg.qr(np.column_stack((truth, rng.normal(size=(d, d - m)))))[0]
    complement = frame[:, m:]
    initial = geometry.polar(truth + complement @ (0.02 * rng.normal(size=(d - m, m))))
    covariance, cross, mass, ranks = [], [], rng.uniform(0.5, 1.5, size=windows), []
    ideal_coefficients = rng.normal(size=(windows, m))
    for j in range(windows):
        q = np.linalg.qr(rng.normal(size=(d, d)))[0]
        if rank_mode == "spd":
            rank = d
        elif j == windows - 1:
            rank = 0
        elif j % 2 == 0:
            rank = 19
        else:
            rank = d
        values = np.zeros(d)
        if rank:
            values[:rank] = np.exp(rng.uniform(-0.8, 0.8, size=rank))
        c = (q * values) @ q.T
        coefficient = rng.normal(size=m)
        perturbation = 0.15 * rng.normal(size=d)
        h = c @ (truth @ coefficient + perturbation)
        covariance.append(c)
        cross.append(h)
        ranks.append(rank)
    covariance, cross, mass = (
        np.asarray(covariance),
        np.asarray(cross),
        np.asarray(mass),
    )
    ideal_cross = (covariance @ truth @ ideal_coefficients[..., None]).squeeze(-1)
    mass /= mass.sum()
    tensor_frame = np.linalg.qr(rng.normal(size=(d, d)))[0]
    tensor = tensor_frame @ np.diag(np.geomspace(0.2, 2.0, d)) @ tensor_frame.T
    stats = SimpleNamespace(U=covariance, I=cross, mass=mass)
    return {
        "seed": seed,
        "windows": windows,
        "rank_mode": rank_mode,
        "truth": truth,
        "initial": initial,
        "covariance": covariance,
        "cross": cross,
        "ideal_cross": ideal_cross,
        "mass": mass,
        "ranks": np.asarray(ranks),
        "kernel_tensor": tensor,
        "stats": stats,
    }


def ideal_state(basis, problem):
    """Zero-ridge profiled state that safely assigns zero-design rows zero g."""
    p, b = problem, basis
    design = p.covariance @ b
    blocks = b.T @ design
    rhs = p.cross @ b
    active = np.any(p.covariance != 0, axis=(1, 2))
    coefficients = np.zeros_like(rhs)
    if np.any(active):
        values = np.linalg.eigvalsh(blocks[active])
        if np.any(values[:, 0] <= 1e-12 * values[:, -1]):
            raise ValueError("ideal tau=0 snapshot has a local rank-drop")
        coefficients[active] = np.linalg.solve(
            blocks[active], rhs[active, :, None]
        ).squeeze(-1)
    residual = (design @ coefficients[..., None]).squeeze(-1) - p.cross
    value = float(p.energy - np.sum(p.pi[:, None] * rhs * coefficients))
    gradient = 2 * residual.T @ (p.pi[:, None] * coefficients)
    normal = (blocks[active] @ coefficients[active, :, None]).squeeze(-1) - rhs[active]
    normal_error = (
        float(np.linalg.norm(normal) / max(np.linalg.norm(rhs[active]), 1e-300))
        if np.any(active)
        else 0.0
    )
    return Profile(
        value, gradient, coefficients, {"ideal_tau_zero": True}, 0, normal_error
    )


def fixed_operator_diagnostics(case):
    b, stats = case["initial"], case["stats"]

    def snapshot(basis, local_stats, *, tau_zero):
        problem = ridge.build(local_stats, basis)
        if tau_zero:
            problem.tau = 0.0
            problem.reference_problem = None
            state = ideal_state(basis, problem)
        else:
            state = ridge.profile(basis, problem)
        operator = geometry.TangentGeometry(basis, problem, state)
        matrices, spectra = {}, {}
        for kind in ("schur", "varpro", "hessian"):
            matrix = operator.matrix(kind)
            matrices[kind] = matrix
            values = np.linalg.eigvalsh(matrix)
            spectra[kind] = {
                "eigenvalues": values,
                "min": float(values[0]),
                "max": float(values[-1]),
                "negative_count": int(np.count_nonzero(values < -1e-10)),
                "near_zero_count": int(np.count_nonzero(np.abs(values) <= 1e-10)),
            }
        return problem, state, operator, spectra, matrices

    ridge_problem, ridge_state, _, ridge_spectra, _ = snapshot(b, stats, tau_zero=False)
    p, state, operator, spectra, _ = snapshot(b, stats, tau_zero=True)
    rhs = geometry.horizontal(b, state.gradient)
    dense_cg = {}
    for kind in ("schur", "varpro"):
        metric = geometry.Metric(b, p, state, kind)
        damping = metric.damping
        tick = time.perf_counter()
        dense, dense_info = operator.solve(rhs, kind, damping=damping, method="dense")
        dense_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        cg, cg_info = operator.solve(rhs, kind, damping=damping, method="cg")
        cg_seconds = time.perf_counter() - tick
        dense_cg[kind] = {
            "damping": damping,
            "dense_seconds": dense_seconds,
            "cg_seconds": cg_seconds,
            "relative_step_difference": float(
                np.linalg.norm(dense - cg) / max(np.linalg.norm(dense), 1e-300)
            ),
            "dense_residual": dense_info["relative_residual"],
            "cg_residual": cg_info["relative_residual"],
            "cg_iterations": cg_info["cg_iterations"],
        }
    direction = geometry.horizontal(
        b, np.random.default_rng(case["seed"]).normal(size=b.shape)
    )
    epsilon = 2e-5
    fp = ideal_state(geometry.polar(b + epsilon * direction), p).value
    f0 = state.value
    fm = ideal_state(geometry.polar(b - epsilon * direction), p).value
    numeric_hessian = (fp - 2 * f0 + fm) / epsilon**2
    operator_hessian = 2 * float(
        np.sum(direction * operator.apply(direction, "hessian"))
    )
    ideal_stats = SimpleNamespace(
        U=case["covariance"], I=case["ideal_cross"], mass=case["mass"]
    )
    ideal_problem, ideal_state_at_truth, _, ideal_spectra, ideal_matrices = snapshot(
        case["truth"], ideal_stats, tau_zero=True
    )
    ideal_residual = (
        ideal_problem.covariance
        @ case["truth"]
        @ ideal_state_at_truth.coefficients[..., None]
    ).squeeze(-1) - ideal_problem.cross
    sc_scale = max(float(np.linalg.norm(ideal_matrices["schur"])), 1e-300)
    ideal_equalities = {
        "sc_vs_vp_relative": float(
            np.linalg.norm(ideal_matrices["schur"] - ideal_matrices["varpro"])
            / sc_scale
        ),
        "sc_vs_half_hessian_relative": float(
            np.linalg.norm(ideal_matrices["schur"] - ideal_matrices["hessian"])
            / sc_scale
        ),
        "max_abs_moment_residual": float(
            np.max(np.linalg.norm(ideal_residual, axis=1))
        ),
        "profile_normal_residual": ideal_state_at_truth.normal_residual,
    }
    return {
        "ridge_snapshot": {
            "tau": ridge_problem.tau,
            "profile_value": ridge_state.value,
            "normal_residual": ridge_state.normal_residual,
            "spectra": ridge_spectra,
        },
        "tau0_nonzero_residual_snapshot": {
            "tau": 0.0,
            "profile_value": state.value,
            "normal_residual": state.normal_residual,
            "spectra": spectra,
        },
        "tau0_ideal_zero_residual_snapshot": {
            "tau": ideal_problem.tau,
            "profile_value": ideal_state_at_truth.value,
            "normal_residual": ideal_state_at_truth.normal_residual,
            "spectra": ideal_spectra,
            "operator_equalities": ideal_equalities,
        },
        "dense_vs_cg": dense_cg,
        "exact_hessian_directional_fd": {
            "finite_difference": numeric_hessian,
            "operator": operator_hessian,
            "absolute_difference": abs(numeric_hessian - operator_hessian),
        },
        "initial_edr": metrics(b, case["truth"])["projector_distance"],
    }


def run_case(case, repeats=REPEATS):
    stats, initial, tensor = case["stats"], case["initial"], case["kernel_tensor"]
    rows = []
    for repeat in range(repeats):
        arms = ARMS if repeat % 2 == 0 else tuple(reversed(ARMS))
        for arm in arms:
            started = time.perf_counter()
            if arm == "lbfgs":
                result = ridge.solve(initial.T, stats)
            else:
                result = geometry.solve(
                    initial.T,
                    stats,
                    arm,
                    max_steps=20,
                    tol=1e-6,
                    metric_tensor=tensor if arm == "kernel" else None,
                )
            elapsed = time.perf_counter() - started
            basis = result.index.T
            diagnostics = result.diagnostics
            history = diagnostics["profile_history"]
            rows.append(
                {
                    "arm": arm,
                    "repeat": repeat,
                    "elapsed_build_and_solve_seconds": elapsed,
                    "initial_profile_objective": history[0],
                    "final_profile_objective": diagnostics["profile_objective"],
                    "standard_gradient_norm": diagnostics["riemannian_gradient"],
                    "iterations": diagnostics["iterations"],
                    "profile_evaluations": diagnostics["profile_evaluations"],
                    "monotone_history": all(
                        new <= old + 1e-10 * max(1.0, abs(old))
                        for old, new in pairwise(history)
                    ),
                    "initial_edr": metrics(initial, case["truth"])[
                        "projector_distance"
                    ],
                    "final_edr": metrics(basis, case["truth"])["projector_distance"],
                    "basis": basis,
                    "diagnostics": diagnostics,
                }
            )
    medians = {
        arm: float(
            np.median(
                [
                    row["elapsed_build_and_solve_seconds"]
                    for row in rows
                    if row["arm"] == arm
                ]
            )
        )
        for arm in ARMS
    }
    common_rows = [row for row in rows if row["arm"] != "lbfgs"]
    return {
        "rows": rows,
        "median_seconds": medians,
        "checks": {
            "monotone_history": bool(common_rows)
            and all(row["monotone_history"] for row in common_rows),
            "common_budget": bool(common_rows)
            and all(
                row["iterations"] <= 20 and row["profile_evaluations"] <= 401
                for row in common_rows
            ),
            "finite_outputs": bool(common_rows)
            and all(
                np.isfinite(row["final_profile_objective"])
                and np.isfinite(row["standard_gradient_norm"])
                and np.isfinite(row["final_edr"])
                for row in rows
            ),
        },
    }


def source_files():
    return [
        ROOT / relative
        for relative in (
            "ADP/engine/common/index_fit.py",
            "ADP/engine/common/weights.py",
            "benchmarks/experiments/anisotropic_metrics.py",
            "benchmarks/experiments/anisotropic_geometry_study.py",
            "benchmarks/experiments/anisotropic_fixed_windows.py",
            "benchmarks/experiments/coupled_penalties.py",
            "benchmarks/experiments/full_ridge_fast.py",
            "benchmarks/experiments/full_ridge_batched.py",
            "tests/test_anisotropic_metrics.py",
            "tests/test_anisotropic_geometry_study.py",
            "benchmarks/results/verification/anisotropic-metrics-20261026/PLAN.md",
            "benchmarks/results/verification/anisotropic-metrics-20261026/PROTOCOL.md",
            "benchmarks/results/verification/anisotropic-metrics-20261026/PROOF.md",
            "benchmarks/results/verification/anisotropic-metrics-20261026/R1_AUDIT.md",
            "benchmarks/results/verification/anisotropic-metrics-20261026/formula_checks.json",
            "benchmarks/results/verification/anisotropic-metrics-20261026/audit_results.py",
        )
    ]


def verify_sources(manifest):
    for relative, expected in manifest["sources"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"frozen fixed-window source drift: {relative}")
    if (
        hashlib.sha256((OUT / "sources.zip").read_bytes()).hexdigest()
        != manifest["archive_sha256"]
    ):
        raise RuntimeError("fixed-window source archive drift")
    for name, details in manifest["cases"].items():
        path = OUT / f"{name}.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != details["sha256"]:
            raise RuntimeError(f"fixed-window input drift: {name}")


def run():
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    audit_path = OUT.parent / "R1_AUDIT.md"
    if not audit_path.exists() or "Status: PASS" not in audit_path.read_text():
        raise RuntimeError("independent R1 review must pass before fixed-window run")
    OUT.mkdir(parents=True)
    cases, inputs, results = {}, {}, {}
    for rank_mode in ("spd", "mixed"):
        for windows in (40, 80):
            case = synthetic(SEED, windows, rank_mode)
            name = f"{rank_mode}-j{windows}"
            cases[name] = case
            input_path = OUT / f"{name}.npz"
            np.savez_compressed(
                input_path,
                truth=case["truth"],
                initial=case["initial"],
                covariance=case["covariance"],
                cross=case["cross"],
                ideal_cross=case["ideal_cross"],
                mass=case["mass"],
                ranks=case["ranks"],
                kernel_tensor=case["kernel_tensor"],
            )
            inputs[name] = {
                "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "rank_histogram": {
                    str(rank): int(np.count_nonzero(case["ranks"] == rank))
                    for rank in np.unique(case["ranks"])
                },
                "moments_sha256": digest(
                    case["covariance"], case["cross"], case["mass"]
                ),
                "ideal_moments_sha256": digest(
                    case["covariance"], case["ideal_cross"], case["mass"]
                ),
            }
    sources = source_files()
    source_hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    with zipfile.ZipFile(OUT / "sources.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, path.relative_to(ROOT))
    manifest = {
        "purpose": "fixed-window algebra and unpaired microtiming, not ADP recovery evidence",
        "seed": SEED,
        "dimensions": {"d": 25, "m": 5},
        "cases": inputs,
        "arms": list(ARMS),
        "repeats": REPEATS,
        "timing_scope": "inclusive build and solve; same saved input and initial basis; unpaired medians",
        "sources": source_hashes,
        "archive_sha256": hashlib.sha256(
            (OUT / "sources.zip").read_bytes()
        ).hexdigest(),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    start = time.perf_counter()
    for name, case in cases.items():
        verify_sources(manifest)
        operator_results = fixed_operator_diagnostics(case)
        ridge_results = run_case(case)
        solve_checks = [check for check in operator_results["dense_vs_cg"].values()]
        all_checks = {
            **ridge_results["checks"],
            "dense_cg_residual": bool(solve_checks)
            and all(
                check["dense_residual"] <= 1e-8 and check["cg_residual"] <= 1e-8
                for check in solve_checks
            ),
            "dense_cg_step_parity": bool(solve_checks)
            and all(
                check["relative_step_difference"] <= 1e-8 for check in solve_checks
            ),
            "schur_psd": operator_results["tau0_nonzero_residual_snapshot"]["spectra"][
                "schur"
            ]["min"]
            >= -1e-8,
            "varpro_psd": operator_results["tau0_nonzero_residual_snapshot"]["spectra"][
                "varpro"
            ]["min"]
            >= -1e-8,
            "ridge_schur_psd": operator_results["ridge_snapshot"]["spectra"]["schur"][
                "min"
            ]
            >= -1e-8,
            "ridge_varpro_psd": operator_results["ridge_snapshot"]["spectra"]["varpro"][
                "min"
            ]
            >= -1e-8,
            "ideal_zero_residual_equality": max(
                operator_results["tau0_ideal_zero_residual_snapshot"][
                    "operator_equalities"
                ][key]
                for key in (
                    "sc_vs_vp_relative",
                    "sc_vs_half_hessian_relative",
                    "max_abs_moment_residual",
                    "profile_normal_residual",
                )
            )
            <= 1e-8,
        }
        results[name] = {
            "operators_tau0": operator_results,
            "ridge_geometry": ridge_results,
            "checks": all_checks,
            "passed": all(all_checks.values()),
        }
    manifest["wall_seconds"] = time.perf_counter() - start
    manifest["results"] = results
    (OUT / "results.json").write_text(
        json.dumps(manifest, indent=2, default=default, allow_nan=False) + "\n"
    )
    report = [
        "# Fixed-window geometry diagnostics",
        "",
        "These deterministic fixed-window runs are operator/algebra checks and descriptive microtimings. They are not adaptive ADP recovery evidence and no speed gate is applied.",
        "",
        f"Seed {SEED}; d=25,m=5; all arms use the same C,c,pi and incoming basis in each case. Each common-solver arm uses at most 20 steps; timings include build and solve and are unpaired medians over {REPEATS} alternating repeats.",
        "",
        "| Case | Ranks | Checks | ridge SC min | tau=0 noisy SC/VP/H min | tau=0 ideal SC/VP/H relative equality | Hessian FD error | Dense/CG step diff | Median seconds A/C/D/SC/VP/kernel/LBFGS |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, result in results.items():
        ops = result["operators_tau0"]
        ridge_spectrum = ops["ridge_snapshot"]["spectra"]
        noisy_spectrum = ops["tau0_nonzero_residual_snapshot"]["spectra"]
        ideal_equality = ops["tau0_ideal_zero_residual_snapshot"]["operator_equalities"]
        meds = result["ridge_geometry"]["median_seconds"]
        order = ["euclidean", "left", "two_sided", "schur", "varpro", "kernel", "lbfgs"]
        timings = "/".join(f"{meds[arm]:.5f}" for arm in order)
        max_parity = max(
            value["relative_step_difference"] for value in ops["dense_vs_cg"].values()
        )
        report.append(
            f"| {name} | {manifest['cases'][name]['rank_histogram']} | {'pass' if result['passed'] else 'fail'} | {ridge_spectrum['schur']['min']:.4g} | {noisy_spectrum['schur']['min']:.4g}/{noisy_spectrum['varpro']['min']:.4g}/{noisy_spectrum['hessian']['min']:.4g} | {ideal_equality['sc_vs_vp_relative']:.3g}/{ideal_equality['sc_vs_half_hessian_relative']:.3g} | {ops['exact_hessian_directional_fd']['absolute_difference']:.4g} | {max_parity:.4g} | {timings} |"
        )
    report.extend(
        [
            "",
            "`results.json` retains all input hashes, rank histograms, raw fitted bases, profile trajectories, stage diagnostics, spectra, and timing repeats. Finite-difference Hessian errors are diagnostics; SC/VP PSD, solve residual/parity, monotonicity, and common budgets are the fixed-window checks.",
            "",
        ]
    )
    (OUT / "REPORT.md").write_text("\n".join(report))
    print(
        json.dumps(
            {"cases": inputs, "wall_seconds": manifest["wall_seconds"]}, indent=2
        )
    )


if __name__ == "__main__":
    run()
