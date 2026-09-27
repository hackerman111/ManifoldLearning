"""Замороженная проверка manifold при m=1,2,3 и меняющейся геометрии."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_info, threadpool_limits

from ADP import ADP_Manifold

from .manifold_recovery_probe import _version
from .runner import _local_subspace_metrics


class DiagnosticManifold(ADP_Manifold):
    """Снимки private hooks без изменения production-вычислений или доступа к truth."""

    debug: dict[str, float | int]

    def _build_manifold_graph(
        self,
        centers: np.ndarray,
        projectors: np.ndarray | None,
        eigenvalues: np.ndarray | None,
        h: float,
        alpha: float,
    ) -> csr_matrix:
        graph = super()._build_manifold_graph(
            centers, projectors, eigenvalues, h, alpha
        )
        if not hasattr(self, "debug"):
            self.debug = {"initial_graph_degree_min": int(np.diff(graph.indptr).min())}
        self.debug["last_graph_degree_min"] = int(np.diff(graph.indptr).min())
        return graph

    def _calculate_statistics(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        centers: np.ndarray,
        directions: np.ndarray,
        projectors: np.ndarray | None,
        eigenvalues: np.ndarray | None,
        h: float,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        result = super()._calculate_statistics(
            X, Y, centers, directions, projectors, eigenvalues, h, alpha
        )
        self.debug.update(
            last_mass_min=float(result[2].min()), last_n_eff_min=float(result[3].min())
        )
        return result

    def _one_step(
        self,
        I: np.ndarray,
        U: np.ndarray,
        mass: np.ndarray,
        graph: csr_matrix,
        projectors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
        if not hasattr(self, "initial_basis"):
            self.initial_basis = projectors.copy()
        result = super()._one_step(I, U, mass, graph, projectors)
        self.last_basis = result[0].copy()
        self.debug["successful_updates"] = (
            int(self.debug.get("successful_updates", 0)) + 1
        )
        self.debug["last_linear_residual_max"] = result[2][
            "linear_relative_residual_max"
        ]
        return result


def geometry(
    X: np.ndarray, rotation: np.ndarray, m: int, curvature: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """f=sum(z_k²)/2, z_k=u_k+c*u_{m+k}²/2, u=X@rotation.

    Возвращает отклик, ортонормированный row-basis Dz и точный gradient f.
    Dz имеет ранг m всюду благодаря единичному блоку первых m координат.
    Матрицы (n,d,d) не строятся; промежуточный Jacobian имеет (n,m,d).
    """
    u = X @ rotation
    z = u[:, :m] + 0.5 * curvature * np.square(u[:, m : 2 * m])
    jacobian = np.zeros((len(X), m, X.shape[1]))
    indices = np.arange(m)
    jacobian[:, indices, indices] = 1.0
    jacobian[:, indices, m + indices] = curvature * u[:, m : 2 * m]
    jacobian = jacobian @ rotation.T
    q, _ = np.linalg.qr(jacobian.swapaxes(1, 2), mode="reduced")
    gradient = np.einsum("nm,nmd->nd", z, jacobian)
    return 0.5 * np.sum(np.square(z), axis=1), q.swapaxes(1, 2), gradient


def make_data(
    seed: int, n: int, d: int, m: int, curvature: float, noise: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Раздельные потоки; X, rotation, noise и queries общие по профилям seed."""
    streams = np.random.SeedSequence(seed).spawn(5)
    X = np.random.default_rng(streams[0]).normal(size=(n, d))
    rotation, _ = np.linalg.qr(np.random.default_rng(streams[1]).normal(size=(d, d)))
    response, truth, _ = geometry(X, rotation, m, curvature)
    scale = float(response.std())
    Y = (response - response.mean()) / scale
    Y += noise * np.random.default_rng(streams[2]).normal(size=n)
    queries = np.random.default_rng(streams[3]).normal(size=(512, d))
    _, query_truth, _ = geometry(queries, rotation, m, curvature)
    model_seed = int(streams[4].generate_state(1)[0])
    return X, Y, truth, queries, query_truth, model_seed


def self_check() -> None:
    """Независимые finite differences и проверки геометрической метрики."""
    rng = np.random.default_rng(77)
    X = rng.normal(size=(12, 8))
    rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    for m in (1, 2, 3):
        for curvature in (0.0, 0.35, 0.8):
            _, basis, gradient = geometry(X, rotation, m, curvature)
            numerical = np.empty_like(X)
            for k in range(X.shape[1]):
                delta = np.zeros_like(X)
                delta[:, k] = 1e-5
                plus = geometry(X + delta, rotation, m, curvature)[0]
                minus = geometry(X - delta, rotation, m, curvature)[0]
                numerical[:, k] = (plus - minus) / 2e-5
            np.testing.assert_allclose(gradient, numerical, rtol=1e-8, atol=1e-8)
            projected = np.einsum("nmd,nd->nm", basis, gradient)
            reconstructed = np.einsum("nmd,nm->nd", basis, projected)
            np.testing.assert_allclose(gradient, reconstructed, atol=1e-12)
            change, _ = np.linalg.qr(rng.normal(size=(m, m)))
            assert _local_subspace_metrics(basis, change @ basis)[1] < 1e-7
            # Независимый dense projector reference только на маленьком наборе.
            other = geometry(X + 0.3, rotation, m, curvature)[1]
            distances = [
                np.linalg.norm(a.T @ a - b.T @ b, ord=2)
                for a, b in zip(basis, other, strict=True)
            ]
            np.testing.assert_allclose(
                max(distances), _local_subspace_metrics(basis, other)[1], atol=1e-7
            )
    first = make_data(77, 64, 8, 3, 0.35, 0.1)
    repeat = make_data(77, 64, 8, 3, 0.35, 0.1)
    for a, b in zip(first, repeat, strict=True):
        np.testing.assert_array_equal(a, b)
    noiseless = make_data(77, 64, 8, 3, 0.35, 0.0)
    for k in (0, 2, 3, 4, 5):
        np.testing.assert_array_equal(first[k], noiseless[k])
    assert not np.array_equal(first[1], noiseless[1])
    print("geometry/reference self-check passed", flush=True)


def run_case(
    case: dict[str, Any], n: int, d: int, boundary: Literal["raise", "stop"]
) -> dict[str, Any]:
    X, Y, truth, queries, query_truth, model_seed = make_data(
        case["seed"], n, d, case["m"], case["curvature"], case["noise"]
    )
    broad = case["support"] == "broad"
    model = DiagnosticManifold(
        case["m"],
        estimator="manifold",
        N_loc=300 if broad else 80,
        N_lin=300 if broad else 200,
        N_J=40,
        N_phi=40,
        N_manifold=30 if broad else 10,
        sync_steps=3,
        lambda_manifold=0.5,
        cg_tol=1e-6,
        solver="cg",
        seed=model_seed,
        scale_boundary=boundary,
    )
    row: dict[str, Any] = {
        **case,
        "model_seed": model_seed,
        "error": None,
        "recovered": False,
        "center_recovered": False,
        "query_recovered": False,
    }
    started = time.perf_counter()
    try:
        model.fit(X, Y)
        center_rms, center_max, _ = _local_subspace_metrics(
            truth[model.center_indices_], model.projectors_
        )
        # Ближайший chart выбирается по правилу публичного transform.
        nearest = model._nearest_center_indices(model._prepare_queries(queries))
        query_rms, query_max, _ = _local_subspace_metrics(
            query_truth, model.projectors_[nearest]
        )
        # Oracle chart показывает дискретизационную ошибку даже при точных центрах.
        oracle_rms, oracle_max, _ = _local_subspace_metrics(
            query_truth, truth[model.center_indices_][nearest]
        )
        row.update(
            center_rms=center_rms,
            center_max=center_max,
            query_rms=query_rms,
            query_max=query_max,
            oracle_query_rms=oracle_rms,
            oracle_query_max=oracle_max,
            center_recovered=center_rms <= 0.2 and center_max <= 0.2,
            query_recovered=query_rms <= 0.2 and query_max <= 0.2,
            stop_reason=model.stop_reason_,
            n_scales=model.n_scales_,
            effective_config=model.effective_config_,
            linear_residual_max=max(
                float(t["linear_relative_residual_max"]) for t in model.trace_
            ),
            trace=model.trace_,
        )
        row["recovered"] = row["center_recovered"] and row["query_recovered"]
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["support_diagnostics"] = getattr(model, "debug", {})
    center_stream = np.random.SeedSequence(model_seed).spawn(2)[0]
    indices = np.random.default_rng(center_stream).choice(n, size=40, replace=False)
    if hasattr(model, "initial_basis"):
        row["initial_rms"], row["initial_max"], _ = _local_subspace_metrics(
            truth[indices], model.initial_basis
        )
    if hasattr(model, "last_basis"):
        row["partial_center_rms"], row["partial_center_max"], _ = (
            _local_subspace_metrics(truth[indices], model.last_basis)
        )
    row["fit_seconds"] = time.perf_counter() - started
    row["process_peak_rss_mib"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    )
    return row


def summarize(
    rows: list[dict[str, Any]], cases: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result = []
    for support, m in ((s, m) for s in ("local", "broad") for m in (1, 2, 3)):
        for curvature in (0.0, 0.35, 0.8):
            for noise in (0.0, 0.1):

                def matches(
                    r: dict[str, Any],
                    key: tuple[str, int, float, float] = (support, m, curvature, noise),
                ) -> bool:
                    return (r["support"], r["m"], r["curvature"], r["noise"]) == key

                selected = [r for r in rows if matches(r)]
                entry = {
                    "support": support,
                    "m": m,
                    "curvature": curvature,
                    "noise": noise,
                    "planned": sum(matches(r) for r in cases),
                    "finished": len(selected),
                    "numerical_failures": sum(r["error"] is not None for r in selected),
                    **{
                        key: sum(bool(r[key]) for r in selected)
                        for key in ("recovered", "center_recovered", "query_recovered")
                    },
                }
                for metric in (
                    "initial_rms",
                    "initial_max",
                    "center_rms",
                    "center_max",
                    "query_rms",
                    "query_max",
                    "oracle_query_max",
                    "fit_seconds",
                ):
                    values = [r[metric] for r in selected if metric in r]
                    entry[metric + "_median"] = (
                        float(np.median(values)) if values else None
                    )
                result.append(entry)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "main"), default="main")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--budget-seconds", type=float, default=600)
    parser.add_argument("--scale-boundary", choices=("raise", "stop"), default="raise")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if (
        args.out is None
        or not math.isfinite(args.budget_seconds)
        or args.budget_seconds <= 0
    ):
        parser.error("--out и положительный --budget-seconds обязательны")
    # Новый каталог предотвращает перезапись прежнего замороженного запуска.
    args.out.mkdir(parents=True, exist_ok=False)
    seeds = [80000] if args.profile == "smoke" else list(range(81000, 81005))
    cases = [
        {"seed": seed, "support": support, "m": m, "curvature": c, "noise": noise}
        for seed in seeds
        for support in ("local", "broad")
        for m in (1, 2, 3)
        for c in (0.0, 0.35, 0.8)
        for noise in (0.0, 0.1)
    ]
    # Фиксированная перестановка распределяет бюджет между всеми профилями.
    np.random.default_rng(20260928).shuffle(cases)
    rows: list[dict[str, Any]] = []
    with threadpool_limits(limits=1):
        version = _version()
        hashes = version["source_sha256"]
        assert isinstance(hashes, dict)
        hashes[__file__] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        hashes["experiments/runner.py"] = hashlib.sha256(
            Path("experiments/runner.py").read_bytes()
        ).hexdigest()
        manifest = {
            "profile": args.profile,
            "cases": cases,
            "n": 600,
            "d": 8,
            "query_count": 512,
            "dtype": "float64",
            "version": version,
            "threads": threadpool_info(),
            "budget_seconds": args.budget_seconds,
            "model": {
                "estimator": "manifold",
                "solver": "cg",
                "N_loc": "local:80, broad:300",
                "N_lin": "local:200, broad:300",
                "N_J": 40,
                "N_phi": 40,
                "N_manifold": "local:10, broad:30",
                "sync_steps": 3,
                "lambda_manifold": 0.5,
                "cg_tol": 1e-6,
                "cg_maxiter": None,
                "a": "2**(1/m)",
                "h_min": "3*mean(std(X))/sqrt(n)",
                "scale_boundary": args.scale_boundary,
                "batch_size": 32,
            },
            "geometry": "u=X@Q; z_k=u_k+c*u_(m+k)^2/2; f=sum(z_k^2)/2; truth=row(Dz)",
            "response": "standardized noiseless f + noise*N(0,1)",
            "seed_scheme": "SeedSequence(seed).spawn(5): X, Q, noise, queries, model",
            "kernel": "max(1-(distance_squared/h^2)^2,0)",
            "directions": "independent normalized Gaussian, refreshed each scale",
            "centers": "uniform without replacement, production seed stream",
            "ridge": "no local linear ridge; production rank guards unchanged",
            "anisotropy": "production structure-adaptive alpha mass search",
            "recovery": "center AND query RMS and maximum principal sine <=0.2",
            "convergence": "debug only; no outer convergence certificate",
            "rss": (
                "Linux ru_maxrss cumulative process high-water mark; not a per-fit peak"
            ),
            "limitations": (
                "specified generating geometry; scalar response does not "
                "uniquely identify it; finite queries do not prove "
                "continuum recovery"
            ),
        }
        (args.out / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        (args.out / "summary.json").write_text(
            json.dumps(
                {
                    "complete": False,
                    "planned": len(cases),
                    "finished": 0,
                    "cases": summarize([], cases),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        started = time.perf_counter()
        with (args.out / "runs.jsonl").open("w", encoding="utf-8") as stream:
            for case in cases:
                if time.perf_counter() - started >= args.budget_seconds:
                    break
                row = run_case(case, 600, 8, args.scale_boundary)
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                print(
                    f"{len(rows)}/{len(cases)} {case} centers={row.get('center_rms')} "
                    f"max={row.get('center_max')} recovered={row['recovered']} "
                    f"error={row['error']}",
                    flush=True,
                )
                summary = {
                    "complete": len(rows) == len(cases),
                    "planned": len(cases),
                    "finished": len(rows),
                    "cases": summarize(rows, cases),
                }
                (args.out / "summary.json").write_text(
                    json.dumps(summary, indent=2), encoding="utf-8"
                )


if __name__ == "__main__":
    main()
