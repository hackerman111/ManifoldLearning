"""S1: capture paired Multi v2 inner problems and profile baseline HPAO."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator

from ADP.core.ADP_Config import ADP_Config
from ADP.solver import LSMR

from .data import _generate_data, _make_seed_bundle
from .models import Build
from .multi_solver_certificate import evaluate
from .multiv2_quality import make_case
from .runner import _arguments, _compact_json, _config_spec, _effective_config

_SOURCE_FILES = (
    "ADP/solver/LSMR.py",
    "ADP/engine/common/index_fit.py",
    "ADP/engine/common/statistic.py",
    "ADP/cli/main.py",
    "experiments/data.py",
    "experiments/multiv2_quality.py",
    "experiments/multi_solver_certificate.py",
    "experiments/multi_solver_search.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["rtk", "git", *args], text=True).strip()


def _environment() -> dict[str, object]:
    return {
        "commit": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--short"),
        "source_sha256": {name: _sha256(Path(name)) for name in _SOURCE_FILES},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas_threads": {
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "dtype": "float64",
    }


def _capture_fit(name: str, seed: int, output: Path) -> dict[str, object]:
    cli_main = importlib.import_module("ADP.cli.main")
    args_for_case = argparse.Namespace(
        inner_steps=None, outer_steps=None, factor=None, candidate_file=None
    )
    experiment, _ = make_case(name, "inner", args_for_case)
    point = experiment.full[2]  # Замороженный baseline cap 80.
    seeds = _make_seed_bundle(
        experiment.selector,
        point,
        seed,
        common_random_fields=experiment.common_random_fields,
    )
    generated = _generate_data(experiment.selector, point, seeds, seed)
    config = _effective_config(ADP_Config(), point, seeds.init)
    args = _arguments(Build("ADP", ADP_Config()), config, point)
    if args.solver != "lsmr" or args.solver_max_steps != 80:
        raise RuntimeError("S1 requires LSMR with cap 80")

    original_solve = cli_main.solve_lsmr
    original_refit = LSMR._local_refit
    original_correction = LSMR._global_correction
    original_operator = LSMR._linear_operator
    calls: list[dict[str, object]] = []
    active: dict[str, float] | None = None

    def timed_refit(*values):
        started = perf_counter()
        result = original_refit(*values)
        if active is not None:
            active["local_refit_sec"] += perf_counter() - started
            active["local_refit_calls"] += 1
        return result

    def timed_operator(*values):
        operator = original_operator(*values)

        def forward(vector):
            started = perf_counter()
            result = operator.matvec(vector)
            if active is not None:
                active["forward_sec"] += perf_counter() - started
                active["forward_calls"] += 1
            return result

        def adjoint(vector):
            started = perf_counter()
            result = operator.rmatvec(vector)
            if active is not None:
                active["adjoint_sec"] += perf_counter() - started
                active["adjoint_calls"] += 1
            return result

        return LinearOperator(
            operator.shape, matvec=forward, rmatvec=adjoint, dtype=float
        )

    def timed_correction(*values):
        started = perf_counter()
        result = original_correction(*values)
        if active is not None:
            active["correction_sec"] += perf_counter() - started
            active["correction_calls"] += 1
        return result

    def capture(index, U, I, **settings):
        nonlocal active
        outer = len(calls)
        frozen = None
        if outer in (0, 2):
            filename = f"{name}-seed{seed}-outer{outer}.npz"
            frozen = output / filename
            if frozen.exists():
                raise FileExistsError(frozen)
            capture_started = perf_counter()
            np.savez(frozen, B=np.asarray(index), U=U, I=I, mass=settings["mass"])
            frozen.chmod(0o444)
            capture_sec = perf_counter() - capture_started
        else:
            capture_sec = 0.0
        active = dict.fromkeys(
            (
                "local_refit_sec",
                "local_refit_calls",
                "correction_sec",
                "correction_calls",
                "forward_sec",
                "forward_calls",
                "adjoint_sec",
                "adjoint_calls",
            ),
            0.0,
        )
        started = perf_counter()
        try:
            result = original_solve(index, U, I, **settings)
        finally:
            elapsed = perf_counter() - started
        row: dict[str, object] = {
            "outer": outer,
            "solver_wall_sec": elapsed,
            "capture_sec": capture_sec,
            "profile": active,
            "diagnostics": result.diagnostics,
        }
        if frozen is not None:
            with np.load(frozen, allow_pickle=False) as arrays:
                initial = evaluate(
                    arrays["B"], arrays["U"], arrays["I"], arrays["mass"]
                )
            final = evaluate(result.index, U, I, settings["mass"])
            for field in (
                "objective",
                "riemannian_gradient",
                "local_gradient",
                "orthogonality",
            ):
                expected = result.diagnostics["loss" if field == "objective" else field]
                observed = getattr(final, field)
                if not np.isclose(observed, expected, rtol=1e-8, atol=1e-10):
                    raise RuntimeError(
                        "independent certificate mismatch: "
                        f"{name} {seed} {outer} {field}"
                    )
            row.update(
                file=frozen.name,
                sha256=_sha256(frozen),
                initial=asdict(initial),
                final=asdict(final),
            )
        calls.append(row)
        active = None
        return result

    cli_main.solve_lsmr = capture
    LSMR._local_refit = timed_refit
    LSMR._global_correction = timed_correction
    LSMR._linear_operator = timed_operator
    try:
        fitted, _, profile, metadata = cli_main._run(
            args, data=(generated.X, generated.Y, generated.beta)
        )
    finally:
        cli_main.solve_lsmr = original_solve
        LSMR._local_refit = original_refit
        LSMR._global_correction = original_correction
        LSMR._linear_operator = original_operator
    if len(calls) != 3 or sum("file" in call for call in calls) != 2:
        raise RuntimeError(
            f"expected three outer calls and two frozen tasks: {name} {seed}"
        )
    return {
        "point": name,
        "seed": seed,
        "seed_bundle": asdict(seeds),
        "point_config": asdict(point),
        "effective_config": _config_spec(config),
        "solver_settings": {
            "solver": args.solver,
            "max_steps": args.solver_max_steps,
            "tol": args.solver_tol,
            "theta": args.theta,
            "lambda_prox": config.lambda_penalty,
            "trust_radius": args.trust_radius,
            "lsmr_maxiter": args.lsmr_maxiter,
        },
        "profile": profile,
        "calls": calls,
        "trace": json.loads(_compact_json(metadata["trace"])),
        "final_B": fitted.tolist(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args(argv)
    budget = {
        "points": ("d10", "n1000"),
        "seeds": (1000, 1001, 1002),
        "fits": 6,
        "outer_solver_calls": 18,
        "frozen_tasks": 12,
        "baseline_cap": 80,
        "execution_timeout_sec": 900,
    }
    if options.dry_run:
        print(json.dumps(budget, indent=2))
        return 0
    options.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"budget": budget, "environment": _environment(), "fits": []}
    path = options.output_dir / "manifest.json"
    for name in budget["points"]:
        for seed in budget["seeds"]:
            manifest["fits"].append(_capture_fit(name, seed, options.output_dir))
            path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
            print(f"captured {name} seed {seed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
