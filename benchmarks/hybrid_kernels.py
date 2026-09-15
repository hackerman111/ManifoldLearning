from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    """Изолированные ядра HYBRID; одинаковые данные для двух деревьев кода."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.output.exists() or any(
        args.output.parent.glob(f"{args.output.stem}_*.npy")
    ):
        parser.error("output prefix already exists; choose a new --output path")
    if args.reference_root is not None:
        if not (args.reference_root / "ADP" / "__init__.py").is_file():
            parser.error("--reference-root must contain the reference ADP package")
        sys.path.insert(0, str(args.reference_root.resolve()))
    import ADP
    from ADP.solver.HYBRID import RidgeWorkspace, solve
    from ADP.solver.LSMR import _index_distance, _local_refit
    from benchmarks.localization import environment, measure

    source_root = Path(ADP.__file__).parent
    rows = []
    for seed in args.seeds:
        for name, J, P, d, m in (
            ("refit2", 500, 20, 100, 2),
            ("refit10", 500, 20, 100, 10),
            ("refit_batched", 500, 200, 100, 10),
            ("subspace", 1, 1, 1000, 10),
            ("actions", 500, 20, 1000, 10),
            ("solver10", 100, 20, 100, 10),
            ("solver1000", 100, 20, 1000, 10),
            ("ridge100", 100, 20, 100, 10),
            ("ridge1000", 100, 20, 1000, 10),
        ):
            if args.cases and name not in args.cases:
                continue
            streams = np.random.SeedSequence(seed).spawn(3)
            rng, index_rng, rhs_rng = map(np.random.default_rng, streams)
            U = rng.normal(size=(J, P, d))
            I = rhs_rng.normal(size=(J, P))
            index = np.linalg.qr(index_rng.normal(size=(d, m)))[0].T
            row = {"case": name, "seed": seed, "shape_J_P_d_m": [J, P, d, m]}
            if name.startswith("refit"):
                expected = np.array(
                    [
                        np.linalg.lstsq(local, rhs, rcond=None)[0]
                        for local, rhs in zip(U @ index.T, I, strict=True)
                    ]
                )
                actual = _local_refit(I, U, index)[0]
                row["relative_error"] = float(
                    np.linalg.norm(actual - expected) / np.linalg.norm(expected)
                )
                row["measurement"] = measure(
                    lambda I=I, U=U, index=index: float(
                        np.linalg.norm(_local_refit(I, U, index)[0])
                    ),
                    args.repeats,
                )
            elif name == "subspace":
                prior = np.linalg.qr(index_rng.normal(size=(d, m)))[0].T
                expected = np.linalg.norm(index.T @ index - prior.T @ prior) / np.sqrt(
                    2
                )
                row["absolute_error"] = abs(_index_distance(index, prior) - expected)
                row["measurement"] = measure(
                    lambda index=index, prior=prior: _index_distance(index, prior),
                    args.repeats,
                )
            elif name.startswith("ridge"):
                coefficients = rhs_rng.normal(size=(J, m))
                mass = np.geomspace(0.01, 100, J)
                U /= np.sqrt(d)
                inputs = (U, I, index, coefficients, mass)

                def trials(inputs=inputs, m=m):
                    workspace = RidgeWorkspace(*inputs, max_unknowns=0)
                    for power in range(24):
                        ridge = 0.05 * 2**power
                        if workspace.rejects_trust(ridge, 0.15 * np.sqrt(m), 2000):
                            continue
                        result = workspace.correction(ridge, 1e-6, 2000)
                        if result[3] <= 0.1 and np.linalg.norm(
                            result[0]
                        ) <= 0.15 * np.sqrt(m):
                            return workspace, result, ridge
                    raise RuntimeError("ridge trial budget exhausted")

                row["settings"] = {
                    "lambda_grid": "0.05 * 2**arange(24)",
                    "radius": 0.15 * np.sqrt(m),
                    "tol": 1e-6,
                    "maxiter": 2000,
                    "mass": "geomspace(0.01,100,J)",
                    "U_scale": "1/sqrt(d)",
                }
                try:
                    workspace, result, ridge = trials()
                    row["measurement"] = measure(
                        lambda trials=trials: float(np.linalg.norm(trials()[1][0])),
                        args.repeats,
                    )
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    continue
                row.update(
                    ridge=ridge,
                    normal_residual_ratio=result[3],
                    iterations=workspace.iterations,
                    screened_trials=workspace.screened_trials,
                    forward_calls=getattr(
                        workspace,
                        "forward_calls",
                        workspace.iterations
                        + workspace.screening_matvecs
                        + workspace.calls,
                    ),
                    adjoint_calls=getattr(
                        workspace,
                        "adjoint_calls",
                        workspace.iterations
                        + workspace.screening_matvecs
                        + 2 * workspace.calls
                        + 1,
                    ),
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                path = args.output.with_name(f"{args.output.stem}_{name}_{seed}.npy")
                with path.open("xb") as stream:
                    np.save(stream, result[0])
            elif name.startswith("solver"):
                truth = np.linalg.qr(index_rng.normal(size=(d, m)))[0].T
                coefficients = rhs_rng.normal(size=(J, m))
                I = (U @ (coefficients @ truth)[..., None]).squeeze(-1) + 0.1 * I
                mass = np.geomspace(0.01, 100, J)
                settings = {
                    "mass": mass,
                    "lambda_prox": 0.05,
                    "max_steps": 1,
                    "lsmr_maxiter": 200,
                    "tol": 1e-6,
                }
                row["purpose"] = (
                    "bounded availability probe; excluded from timing aggregation"
                )
                row["settings"] = settings | {"mass": "geomspace(0.01,100,J)"}
                try:
                    result = solve(index, U, I, **settings)
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    continue
                row["diagnostics"] = result.diagnostics
                row["projector_distance"] = float(
                    np.square(result.index - (result.index @ truth.T) @ truth).sum()
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                path = args.output.with_name(f"{args.output.stem}_{name}_{seed}.npy")
                with path.open("xb") as stream:
                    np.save(stream, result.index)
            else:
                coefficients = rhs_rng.normal(size=(J, m))
                mass = np.geomspace(0.01, 100, J)
                inputs = (U, I, index, coefficients, mass)
                workspace = RidgeWorkspace(*inputs, max_bytes=0)
                vector, data = index_rng.normal(size=m * d), rhs_rng.normal(size=J * P)
                expected_forward = (
                    np.sqrt(mass)[:, None]
                    * (U @ (coefficients @ vector.reshape(m, d))[..., None]).squeeze(-1)
                ).ravel()
                pulled = (
                    U.swapaxes(1, 2)
                    @ (np.sqrt(mass)[:, None, None] * data.reshape(J, P, 1))
                ).squeeze(-1)
                expected_adjoint = (coefficients.T @ pulled).ravel()
                error = max(
                    np.linalg.norm(workspace.matvec(vector) - expected_forward)
                    / np.linalg.norm(expected_forward),
                    np.linalg.norm(workspace.rmatvec(data) - expected_adjoint)
                    / np.linalg.norm(expected_adjoint),
                )
                row["relative_error"] = float(error)
                row["workspace_bytes_excluding_inputs"] = sum(
                    v.nbytes
                    for v in vars(workspace).values()
                    if isinstance(v, np.ndarray)
                    and not any(np.shares_memory(v, x) for x in inputs)
                )
                row["measurement"] = measure(
                    lambda workspace=workspace, vector=vector, data=data: float(
                        np.linalg.norm(workspace.matvec(vector))
                        + np.linalg.norm(workspace.rmatvec(data))
                    ),
                    args.repeats,
                )
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("case", "seed")}), flush=True)

    report = {
        **environment(),
        "source_root": str(source_root),
        "source_sha256": {
            str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source_root.rglob("*.py"))
        },
        "dtype": "float64",
        "repeats": args.repeats,
        "seed_scheme": "SeedSequence(seed).spawn(3): U, indices, I/coefficients",
        "measurement": (
            "warmup; untraced wall-clock; separate tracemalloc peak, "
            "excludes existing inputs/workspace and some native allocations"
        ),
        "formula": (
            "local minimum-norm U_j B.T L_j=I_j; "
            "A(B)_j=sqrt(mass_j) U_j B.T L_j; projector distance"
        ),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
