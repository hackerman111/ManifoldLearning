"""Capture and compare both solvers at seed-2 frozen outer call 106.

Run from the repository root:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python -m experiments.grassman_optim_2026_10_01.capture_call106

The capture fits stop immediately before solving call 106. They are diagnostic
runs, not timings. Every captured input is then solved by both implementations.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import fields
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from ADP import ADP_multi_index, ADP_solver
from ADP.solver import grassman, grassman_optim
from benchmarks.grassman_benchmark import PARAMS, _make_problem

OUT = Path(__file__).resolve().parent / "call106_diagnostic"
CALL = 106
SEED = 2
OPTIONS: dict[str, object] = {
    "method": "core_gn",
    "angle_backend": "schur",
    "max_steps": 5,
    "tol": 1e-6,
    "energy_tol": 0.05,
    "local_ridge": 0.0,
    "max_angle": 0.5,
    "workspace_bytes": 16 * 1024**2,
}


class _CaptureCall(Exception):
    pass


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(repr(array.shape).encode())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _callable_name(value: Any) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _capture(
    module_name: str,
    module: Any,
    X: np.ndarray,
    y: np.ndarray,
    truth: np.ndarray,
    config: Any,
) -> Path:
    save_path = OUT / f"seed2_call106_{module_name}.npz"
    if save_path.exists():
        raise FileExistsError(f"refusing to overwrite {save_path}")
    solve = module.solve
    original_method = partial(
        solve,
        **{**OPTIONS, "lambda_prox": float(config.lambda_penalty)},
    )
    calls = 0

    def capture_method(index_init, U, I, *, mass=None, **kwargs):
        nonlocal calls
        calls += 1
        if calls == CALL:
            arrays = {
                "index_init": np.array(index_init, copy=True),
                "U": np.array(U, copy=True),
                "I": np.array(I, copy=True),
                "mass": None if mass is None else np.array(mass, copy=True),
                "truth_basis": np.array(truth, copy=True),
            }
            np.savez(save_path, **arrays)
            raise _CaptureCall
        return original_method(index_init, U, I, mass=mass, **kwargs)

    model_solver = ADP_solver(capture_method)
    model = ADP_multi_index(int(PARAMS["m"]), config, model_solver)
    try:
        model.fit(X, y)
    except _CaptureCall:
        pass
    else:
        raise RuntimeError(f"fit ended before solver call {CALL}")
    with np.load(save_path, allow_pickle=False) as arrays:
        row = {
            "module": module_name,
            "seed": SEED,
            "call_one_based": CALL,
            "snapshot": str(save_path.relative_to(OUT.parent)),
            "snapshot_sha256": _hash(save_path),
            "array_sha256": {key: _array_hash(arrays[key]) for key in arrays.files},
            "shapes": {key: list(arrays[key].shape) for key in arrays.files},
            "dtype": str(arrays["U"].dtype),
        }
    (OUT / f"seed2_call106_{module_name}.json").write_text(
        json.dumps(row, indent=2) + "\n", encoding="utf-8"
    )
    return save_path


def _input_summary(
    module: Any, B: np.ndarray, U: np.ndarray, I: np.ndarray, mass: np.ndarray
) -> dict[str, object]:
    Y = module._normalize_index(B).T
    M = module._project(U, Y) if hasattr(module, "_project") else U @ Y
    state = module._profile(M, I, mass, 0.0)
    grad = module._gradient(Y, U, state, mass)
    return {
        "basis_orthogonality_error": float(np.linalg.norm(B @ B.T - np.eye(len(B)))),
        "profile_objective": state.value,
        "local_rank_min": int(np.min(state.ranks)),
        "local_rank_max": int(np.max(state.ranks)),
        "local_rank_loss_count": int(np.count_nonzero(state.ranks < len(B))),
        "profile_smooth": state.smooth,
        "horizontal_gradient_norm": float(np.linalg.norm(grad)),
        "coefficients_norm": float(np.linalg.norm(state.coefficients)),
        "residual_norm": float(np.linalg.norm(state.residual)),
    }


def _solve_snapshot(
    module: Any, B: np.ndarray, U: np.ndarray, I: np.ndarray, mass: np.ndarray
) -> dict[str, object]:
    result = module.solve(
        B,
        U,
        I,
        mass=mass,
        lambda_prox=0.05,
        **OPTIONS,
    )
    basis = np.asarray(result.index)
    M = (
        module._project(np.ascontiguousarray(U), module._normalize_index(basis).T)
        if hasattr(module, "_project")
        else U @ module._normalize_index(basis).T
    )
    final_state = module._profile(M, I, mass, 0.0)
    return {
        "basis": basis,
        "profile_objective": final_state.value,
        "iterations": result.diagnostics["iterations"],
        "stop_reason": result.diagnostics["stop_reason"],
        "converged": result.diagnostics["converged"],
        "diagnostics": _json(result.diagnostics),
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    X, y, truth, config = _make_problem(SEED)
    captured = {
        "grassman": _capture("grassman", grassman, X, y, truth, config),
        "grassman_optim": _capture(
            "grassman_optim", grassman_optim, X, y, truth, config
        ),
    }
    inputs: dict[str, dict[str, np.ndarray]] = {}
    input_summaries = {}
    for name, path in captured.items():
        with np.load(path, allow_pickle=False) as arrays:
            inputs[name] = {key: arrays[key] for key in arrays.files}
        input_summaries[name] = {}
        for solver_name, module in (
            ("grassman", grassman),
            ("grassman_optim", grassman_optim),
        ):
            input_summaries[name][solver_name] = _input_summary(
                module,
                inputs[name]["index_init"],
                np.ascontiguousarray(inputs[name]["U"]),
                inputs[name]["I"],
                inputs[name]["mass"],
            )

    cross_solve = {}
    for input_name, arrays in inputs.items():
        cross_solve[input_name] = {}
        results = {}
        for solver_name, module in (
            ("grassman", grassman),
            ("grassman_optim", grassman_optim),
        ):
            result = _solve_snapshot(
                module,
                arrays["index_init"],
                np.ascontiguousarray(arrays["U"]),
                arrays["I"],
                arrays["mass"],
            )
            results[solver_name] = result
            cross_solve[input_name][solver_name] = {
                key: value for key, value in result.items() if key != "basis"
            }
        left = results["grassman"]["basis"]
        right = results["grassman_optim"]["basis"]
        cross_solve[input_name]["final_projector_frobenius_difference"] = float(
            np.linalg.norm(left.T @ left - right.T @ right, ord="fro")
        )

    b0, b1 = inputs["grassman"], inputs["grassman_optim"]
    basis_left, basis_right = b0["index_init"], b1["index_init"]
    U_left, U_right = b0["U"], b1["U"]
    I_left, I_right = b0["I"], b1["I"]
    mass_left, mass_right = b0["mass"], b1["mass"]
    input_difference = {
        "basis_projector_frobenius": float(
            np.linalg.norm(
                basis_left.T @ basis_left - basis_right.T @ basis_right,
                ord="fro",
            )
        ),
        "U_relative_frobenius": float(
            np.linalg.norm(U_left - U_right) / max(1.0, np.linalg.norm(U_left))
        ),
        "I_relative_norm": float(
            np.linalg.norm(I_left - I_right) / max(1.0, np.linalg.norm(I_left))
        ),
        "mass_relative_norm": float(
            np.linalg.norm(mass_left - mass_right) / max(1.0, np.linalg.norm(mass_left))
        ),
    }
    metadata = {
        "diagnostic_only": True,
        "capture_command": (
            "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
            "python -m experiments.grassman_optim_2026_10_01.capture_call106"
        ),
        "diagnostic_script_sha256": _hash(Path(__file__).resolve()),
        "capture_call_one_based": CALL,
        "seed": SEED,
        "lambda_prox": float(config.lambda_penalty),
        "solver_options": OPTIONS,
        "effective_config": {
            field.name: getattr(config, field.name)
            if not callable(getattr(config, field.name))
            else _callable_name(getattr(config, field.name))
            for field in fields(config)
        },
        "data_sha256": {
            "X": _array_hash(X),
            "y": _array_hash(y),
            "truth_basis": _array_hash(truth),
        },
        "solver_sha256": {
            "grassman": _hash(Path("ADP/solver/grassman.py")),
            "grassman_optim": _hash(Path("ADP/solver/grassman_optim.py")),
        },
        "source_sha256": {
            path: _hash(Path(path))
            for path in (
                "experiments/grassman_2026_10_01/benchmark/frozen_ablation/frozen_protocol.json",
                "benchmarks/grassman_benchmark.py",
                "experiments/data.py",
                "experiments/models.py",
                "experiments/runner.py",
                "ADP/core/ADP_Config.py",
            )
        },
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip(),
        "git_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                check=True,
                text=True,
            ).stdout.strip()
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "thread_env": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "input_profiles": input_summaries,
        "input_differences": input_difference,
        "cross_solve": cross_solve,
        "interpretation": (
            "The frozen cross-solves isolate within-solver differences on each "
            "captured input. Differences between captured inputs quantify "
            "outer-trajectory divergence by call 106."
        ),
    }
    (OUT / "summary.json").write_text(
        json.dumps(_json(metadata), indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            _json(
                {
                    "input_differences": input_difference,
                    "input_profiles": input_summaries,
                    "cross_solve": cross_solve,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
