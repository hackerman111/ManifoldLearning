"""Изолированный CPU-прототип и парный эксперимент H1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

# Ограничение одного output-тензора прототипа; U потребует ещё столько же.
_MAX_DIRECTION_BYTES = 256 * 2**20


def orthogonal_directions(
    rng: np.random.Generator,
    J: int,
    P: int,
    d: int,
    *,
    compact_full: bool = False,
) -> np.ndarray:
    """Вернуть Haar-блоки `(J,P,d)` или их точный Gram-фактор.

    При ``compact_full=True`` полные `d`-блоки заменяются на
    ``sqrt(k) I_d``. Тогда число строк равно `d + P % d`, если `P>=d`;
    эти строки не являются единичными random directions, но сохраняют
    всю квадратичную цель исходного H1-скетча над вещественными числами.
    """
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value <= 0
        for value in (J, P, d)
    ):
        raise ValueError("J, P and d must be positive integers")
    full, partial = divmod(P, d)
    rows = d + partial if compact_full and full else P
    if J * rows * d * np.dtype(float).itemsize > _MAX_DIRECTION_BYTES:
        raise MemoryError("H1 direction tensor exceeds the 256 MiB prototype cap")

    directions = np.empty((J, rows, d))
    if compact_full and full:
        directions[:, :d] = np.sqrt(full) * np.eye(d)
        blocks = ((d, partial),) if partial else ()
    else:
        blocks = tuple((start, min(d, P - start)) for start in range(0, P, d))

    for j in range(J):
        for start, width in blocks:
            gaussian = rng.standard_normal((d, width))
            q, r = np.linalg.qr(gaussian, mode="reduced")
            q *= np.where(np.diag(r) >= 0, 1.0, -1.0)
            directions[j, start : start + width] = q.T
    if not np.all(np.isfinite(directions)):
        raise RuntimeError("H1 generated non-finite directions")
    return directions


@dataclass(frozen=True)
class _ManifoldM2Point:
    """Экспериментальная m=2 точка вне m=1 валидатора radial CLI."""

    n: int = 240
    d: int = 4
    index_dim: int = 2
    sigma_eps: float = 0.05
    N_loc: int = 60
    N_lin: int = 120
    N_J: int = 24
    N_phi: int = 12
    N_manifold: int = 8
    sync_steps: int = 1
    lambda_manifold: float = 0.5
    h_min_factor: float = 10.0


def _point(name: str):
    from dataclasses import replace

    from .diagnostic import _base_point
    from .multiv2_quality import _baseline

    if name == "multi_d10":
        return _baseline("d10")
    if name == "multi_d100":
        return _baseline("n1000")
    if name == "manifold_m1":
        return replace(
            _base_point("manifold"),
            N_loc=40,
            N_lin=80,
            N_manifold=8,
        )
    if name == "manifold_m2":
        return _ManifoldM2Point()
    raise ValueError(f"unknown H1 case: {name}")


def _actual_rows(P: int, d: int) -> int:
    return d + P % d if d <= P else P


def _data(name: str, point, seed: int):
    from .data import _generate_data, _GeneratedData, _make_seed_bundle

    streams = _make_seed_bundle("proof-first-h1-" + name, cast(Any, point), seed)
    if name == "manifold_m2":
        X = np.random.default_rng(streams.features).normal(size=(point.n, point.d))
        radius = np.linalg.norm(X[:, 1:3], axis=1)
        if np.any(radius <= np.finfo(float).eps):
            raise RuntimeError("undefined m2 radial direction")
        basis = np.zeros((point.n, point.d, 2))
        basis[:, 0, 0] = 1.0
        basis[:, 1:3, 1] = X[:, 1:3] / radius[:, None]
        signal = np.sin(X[:, 0]) + 0.5 * np.square(radius)
        signal = (signal - signal.mean()) / signal.std()
        Y = signal + point.sigma_eps * np.random.default_rng(streams.noise).normal(
            size=point.n
        )
        return _GeneratedData(X, Y, basis), streams
    else:
        return _generate_data("proof-first-h1-" + name, point, streams, seed), streams


def _single(name: str, variant: str, seed: int) -> dict[str, object]:
    import json
    import resource
    import tracemalloc
    from contextlib import ExitStack
    from dataclasses import replace
    from time import perf_counter
    from unittest.mock import patch

    from ADP.core.ADP_Config import ADP_Config
    from ADP.core.manifold.ADP_Manifold import ADP_Manifold
    from ADP.engine.common import index_fit
    from ADP.engine.common.calculus import generate_isotropic_proj

    from .models import Build, ExperimentPoint
    from .runner import _fit, _local_subspace_metrics

    base = _point(name)
    assert base.N_phi is not None
    nominal = base.N_phi
    actual = _actual_rows(nominal, base.d)
    if variant not in {"iid", "h1", "iid_budget"}:
        raise ValueError("unknown H1 variant")
    if variant == "iid_budget" and actual == nominal:
        raise ValueError("same-row control duplicates the baseline")
    point = replace(base, N_phi=actual) if variant == "iid_budget" else base
    data, streams = _data(name, base, seed)

    def directions(rng: np.random.Generator, J: int, P: int, d: int) -> np.ndarray:
        if variant == "h1":
            return orthogonal_directions(rng, J, P, d, compact_full=True)
        return np.sqrt(nominal / actual) * generate_isotropic_proj(rng, J, P, d)

    started = perf_counter()
    try:
        with ExitStack() as stack:
            if variant != "iid":
                stack.enter_context(
                    patch.object(index_fit, "generate_isotropic_proj", directions)
                )
                stack.enter_context(
                    patch.object(
                        ADP_Manifold, "_random_directions", staticmethod(directions)
                    )
                )
            if name.startswith("manifold"):
                assert point.N_loc is not None
                assert point.sync_steps is not None
                assert point.lambda_manifold is not None
                assert point.h_min_factor is not None
                tracemalloc.start()
                try:
                    model = ADP_Manifold(
                        point.index_dim,
                        N_loc=point.N_loc,
                        N_lin=point.N_lin,
                        N_J=point.N_J,
                        N_phi=point.N_phi,
                        N_manifold=point.N_manifold,
                        lambda_manifold=point.lambda_manifold,
                        sync_steps=point.sync_steps,
                        h_min=point.h_min_factor / np.sqrt(point.n),
                        seed=streams.init,
                        cg_tol=1e-6,
                        estimator="manifold",
                        scale_boundary="stop",
                    ).fit(data.X, data.Y)
                    traced_peak = tracemalloc.get_traced_memory()[1] / 2**20
                finally:
                    tracemalloc.stop()
                true_projectors = data.beta[model.center_indices_].swapaxes(1, 2)
                quality, worst_local_error, _ = _local_subspace_metrics(
                    true_projectors, model.projectors_
                )
                result = {
                    "quality": quality,
                    "quality_pass": quality <= 0.2,
                    "whole_manifold_max_local_projector_distance": worst_local_error,
                    "whole_manifold_pass": worst_local_error <= 0.2,
                    "recovered": quality <= 0.2 and worst_local_error <= 0.2,
                    "convergence_pass": None,
                    "stop_reason": model.stop_reason_,
                    "solver_diagnostics": json.dumps(
                        {
                            "linear_relative_residual_max": max(
                                float(entry["linear_relative_residual_max"])
                                for entry in model.trace_
                            )
                        }
                    ),
                    "max_stage_traced_peak_mib": traced_peak,
                }
            else:
                result = _fit(
                    Build("proof-first-h1", ADP_Config()),
                    cast(ExperimentPoint, point),
                    data,
                    streams.init,
                    0.95,
                )
        error = None
    except (ValueError, RuntimeError, np.linalg.LinAlgError, MemoryError) as exc:
        result = None
        error = f"{type(exc).__name__}: {exc}"
    elapsed = perf_counter() - started
    rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**10
    row: dict[str, object] = {
        "case": name,
        "variant": variant,
        "seed": seed,
        "model_seed": streams.init,
        "nominal_P": nominal,
        "actual_rows": actual if variant != "iid" else nominal,
        "error": error,
        "fit_seconds": elapsed,
        "process_peak_rss_mib": rss_mib,
    }
    if result is not None:
        diagnostics = json.loads(str(result["solver_diagnostics"]))
        residual = diagnostics.get("linear_relative_residual_max")
        stop = result["stop_reason"]
        row.update(
            quality=result["quality"],
            convergence=bool(result["convergence_pass"])
            if name.startswith("multi")
            else None,
            completion=(
                stop in {"h_min", "function_mass_boundary", "manifold_mass_boundary"}
                and isinstance(residual, (int, float))
                and residual <= 1e-5
            )
            if name.startswith("manifold")
            else None,
            whole_manifold_max_local_projector_distance=result.get(
                "whole_manifold_max_local_projector_distance"
            ),
            whole_manifold_pass=result.get("whole_manifold_pass"),
            recovered=bool(result["recovered"]),
            stop_reason=stop,
            inner_residual=residual,
            traced_peak_mib=result["max_stage_traced_peak_mib"],
        )
    return row


def _summarize(rows: list[dict[str, object]], expected_pairs: int) -> dict[str, object]:
    import statistics

    by_case: dict[str, dict[str, object]] = {}
    for name in ("multi_d10", "multi_d100", "manifold_m1", "manifold_m2"):
        subset = [row for row in rows if row["case"] == name]
        if not subset:
            continue
        paired = {}
        for seed in sorted({cast(int, row["seed"]) for row in subset}):
            group = {str(row["variant"]): row for row in subset if row["seed"] == seed}
            if {"iid", "h1"} <= group.keys():
                paired[seed] = group
        reasons = []
        time_ratios = []
        rss_ratios = []
        quality_deltas = []
        baseline_recovery = candidate_recovery = 0
        for seed, group in paired.items():
            base, candidate = group["iid"], group["h1"]
            baseline_recovery += bool(base.get("recovered"))
            candidate_recovery += bool(candidate.get("recovered"))
            if candidate["error"] is not None and base["error"] is None:
                reasons.append(f"seed {seed}: new numerical failure")
            if base["error"] is not None or candidate["error"] is not None:
                continue
            time_ratios.append(
                cast(float, candidate["fit_seconds"]) / cast(float, base["fit_seconds"])
            )
            rss_ratios.append(
                cast(float, candidate["process_peak_rss_mib"])
                / cast(float, base["process_peak_rss_mib"])
            )
            sign = 1.0 if name.startswith("multi") else -1.0
            delta = sign * (
                cast(float, candidate["quality"]) - cast(float, base["quality"])
            )
            quality_deltas.append(delta)
            if delta < -1e-10:
                reasons.append(f"seed {seed}: quality decreased by {-delta:.6g}")
            if base.get("recovered") and not candidate.get("recovered"):
                reasons.append(f"seed {seed}: lost recovery")
        if len(paired) != expected_pairs or len(time_ratios) != expected_pairs:
            reasons.append(f"{expected_pairs} complete paired fits required")
        median_time = statistics.median(time_ratios) if time_ratios else None
        median_rss = statistics.median(rss_ratios) if rss_ratios else None
        speed_gate = (
            not reasons
            and median_time is not None
            and median_time <= 0.8
            and median_rss is not None
            and median_rss <= 1.05
        )
        recovery_gate = (
            not reasons
            and candidate_recovery
            >= baseline_recovery + max(2, (expected_pairs + 2) // 3)
            and median_time is not None
            and median_time <= 1.1
            and median_rss is not None
            and median_rss <= 1.1
        )
        if not (speed_gate or recovery_gate):
            reasons.append("neither prespecified cost/recovery gate passed")
        if time_ratios:
            rng = np.random.default_rng(70000)
            samples = rng.integers(0, len(time_ratios), (2000, len(time_ratios)))
            time_ci = np.quantile(
                np.median(np.asarray(time_ratios)[samples], axis=1), [0.025, 0.975]
            ).tolist()
            rss_ci = np.quantile(
                np.median(np.asarray(rss_ratios)[samples], axis=1), [0.025, 0.975]
            ).tolist()
        else:
            time_ci = rss_ci = None
        by_case[name] = {
            "pairs": len(paired),
            "baseline_recovered": baseline_recovery,
            "candidate_recovered": candidate_recovery,
            "median_time_ratio": median_time,
            "median_rss_ratio": median_rss,
            "median_time_ratio_bootstrap95": time_ci,
            "median_rss_ratio_bootstrap95": rss_ci,
            "quality_deltas": quality_deltas,
            "speed_gate": speed_gate,
            "recovery_gate": recovery_gate,
            "passed": speed_gate or recovery_gate,
            "reasons": reasons,
        }
    return {"cases": by_case, "any_passed": any(v["passed"] for v in by_case.values())}


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    import os
    import subprocess
    import sys
    from dataclasses import asdict
    from pathlib import Path

    from .diagnostic import _code_fingerprint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("single", "selection", "validation"), required=True
    )
    parser.add_argument(
        "--case", choices=("multi_d10", "multi_d100", "manifold_m1", "manifold_m2")
    )
    parser.add_argument("--variant", choices=("iid", "h1", "iid_budget"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--selection-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "single":
        if args.case is None or args.variant is None or args.seed is None:
            parser.error("single requires --case, --variant and --seed")
        print(json.dumps(_single(args.case, args.variant, args.seed), sort_keys=True))
        return 0

    cases = ("multi_d10", "multi_d100", "manifold_m1", "manifold_m2")
    seeds = (
        tuple(range(61000, 61006))
        if args.stage == "selection"
        else tuple(range(62000, 62020))
    )
    if args.stage == "validation":
        if args.case is None:
            parser.error("validation requires --case selected from R3")
        if args.selection_dir is None:
            parser.error("validation requires --selection-dir")
        selection_manifest = json.loads(
            (args.selection_dir / "manifest.json").read_text()
        )
        selection_summary = json.loads(
            (args.selection_dir / "summary.json").read_text()
        )
        selected = selection_summary["cases"].get(args.case)
        if (
            selection_manifest["stage"] != "selection"
            or not selected
            or not selected["passed"]
        ):
            parser.error("selected case did not pass R3 gate")
        if selection_manifest["code_fingerprint"] != _code_fingerprint():
            parser.error("code changed after selection")
        cases = (args.case,)
    specs = []
    for name in cases:
        point = _point(name)
        assert point.N_phi is not None
        variants = ("iid", "h1")
        if (
            _actual_rows(point.N_phi, point.d) < point.N_phi
            and args.stage == "selection"
        ):
            variants += ("iid_budget",)
        for seed in seeds:
            order = variants if seed % 2 == 0 else variants[::-1]
            specs.extend((name, variant, seed) for variant in order)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "fits": len(specs),
                    "cases": cases,
                    "seeds": seeds,
                }
            )
        )
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required")
    if args.output_dir.exists():
        parser.error("output directory already exists")
    args.output_dir.mkdir(parents=True)
    manifest = {
        "stage": args.stage,
        "code_fingerprint": _code_fingerprint(),
        "commit": subprocess.check_output(
            ["rtk", "git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["rtk", "git", "status", "--short"], text=True
        ),
        "numpy": np.__version__,
        "python": sys.version,
        "threads": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "cases": {name: asdict(_point(name)) for name in cases},
        "seeds": seeds,
        "variants": (
            "iid nominal P; H1 compact Gram; iid_budget same rows "
            "rescaled to nominal expected Gram"
        ),
        "gate": (
            "paired fits; no new failure/lost geometric recovery; convergence "
            "and completion are diagnostics only; "
            "per-fit quality tolerance 1e-10; speed median wall<=0.8 "
            "and RSS<=1.05 or recovery +ceil(pairs/3) with wall/RSS<=1.1"
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    rows = []
    with (args.output_dir / "runs.jsonl").open("w", encoding="utf-8") as stream:
        for name, variant, seed in specs:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.proof_first_h1",
                    "--stage",
                    "single",
                    "--case",
                    name,
                    "--variant",
                    variant,
                    "--seed",
                    str(seed),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode:
                row = {
                    "case": name,
                    "variant": variant,
                    "seed": seed,
                    "error": completed.stderr[-2000:],
                    "fit_seconds": None,
                }
            else:
                row = json.loads(completed.stdout)
            rows.append(row)
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            print(
                f"{name} {variant} {seed}: {row.get('error') or row.get('quality')}",
                flush=True,
            )
    summary = _summarize(rows, len(seeds))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
