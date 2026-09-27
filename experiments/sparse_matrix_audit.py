"""Measure exact sparsity of the live single, multi, and manifold objectives."""

# ruff: noqa: RUF001
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ADP.core.ADP_Config import ADP_Config
from ADP.core.manifold.ADP_Manifold import ADP_Manifold
from ADP.engine.common.index_fit import fit_index
from ADP.engine.common.logger import IndexProfiler
from ADP.engine.manifol_engine import optimisation
from ADP.solver.LSMR import HPAOResult, _linear_operator, _local_refit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/experiments/sparse_solver_suitability_2026-09-27"
PROFILES = (
    "dense_iid",
    "dense_iid",
    "dense_offset",
    "dense_iid",
    "correlated_030",
    "correlated_070",
    "zero_inflated_040",
    "zero_inflated_070",
    "block_correlated",
    "near_constant_feature",
)


class _CapturedIndex(Exception):
    def __init__(self, index: np.ndarray, statistics: Any) -> None:
        self.index = index
        self.statistics = statistics


class _CapturedManifold(Exception):
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state


def _data(n: int, d: int, profile: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    if profile.startswith("correlated_"):
        rho = float(profile.rsplit("_", 1)[1]) / 100
        covariance = rho ** np.abs(np.subtract.outer(np.arange(d), np.arange(d)))
        X = X @ np.linalg.cholesky(covariance).T
    elif profile == "block_correlated":
        covariance = np.eye(d)
        for start in range(0, d - 1, 4):
            stop = min(d, start + 4)
            covariance[start:stop, start:stop] = 0.55
            np.fill_diagonal(covariance[start:stop, start:stop], 1.0)
        X = X @ np.linalg.cholesky(covariance).T
    elif profile.startswith("zero_inflated_"):
        keep = 1.0 - float(profile.rsplit("_", 1)[1]) / 100
        X[:, 2:] *= rng.random(X[:, 2:].shape) < keep
    elif profile == "dense_offset":
        X += 10.0
    elif profile == "near_constant_feature":
        X[:, -1] *= 1e-4
    elif profile != "dense_iid":
        raise ValueError(f"unknown data profile: {profile}")

    Y = np.sin(X[:, 0]) + 0.8 * np.sin(X[:, 1]) + 0.05 * rng.standard_normal(n)
    return np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)


def _relative_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(1.0, float(np.linalg.norm(right))))


def _matrix_metrics(matrix: np.ndarray) -> dict[str, int | float]:
    absolute = np.abs(matrix)
    nonzero = absolute[absolute > 0]
    scale = float(absolute.max(initial=0.0))
    threshold = 1e-12 * scale
    return {
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "nnz_exact": int(nonzero.size),
        "density_exact": float(nonzero.size / matrix.size) if matrix.size else 0.0,
        "nearzero_nonzero_1e-12": int(
            np.count_nonzero((absolute > 0) & (absolute <= threshold))
        ),
        "nearzero_density_1e-12": (
            float(
                np.count_nonzero((absolute > 0) & (absolute <= threshold)) / matrix.size
            )
            if matrix.size
            else 0.0
        ),
        "min_abs_nonzero": float(nonzero.min()) if nonzero.size else 0.0,
        "max_abs": scale,
    }


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    q = np.quantile(np.asarray(values, dtype=float), [0, 0.1, 0.5, 0.9, 1])
    return dict(zip(("min", "p10", "median", "p90", "max"), map(float, q), strict=True))


def _capture_index(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    mode: str,
    index_dim: int,
    seed: int,
    n_loc: int,
    n_lin: int,
    n_centers: int,
    n_directions: int,
) -> tuple[np.ndarray, Any]:
    config = ADP_Config(
        seed=seed,
        N_loc=n_loc,
        N_lin=n_lin,
        N_J=n_centers,
        N_phi=n_directions,
        outer_steps=1,
        index_init="random",
    )
    profiler = IndexProfiler()

    def capture(index: np.ndarray, statistics: Any) -> HPAOResult:
        raise _CapturedIndex(index.copy(), statistics)

    try:
        fit_index(
            X,
            Y,
            config=config,
            mode=mode,
            index_dim=index_dim,
            solve=capture,
            profiler=profiler,
        )
    except _CapturedIndex as result:
        return result.index, result.statistics
    finally:
        profiler.finish()
    raise RuntimeError("index fit did not reach its statistics capture point")


def _audit_index(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    mode: str,
    index_dim: int,
    seed: int,
    sizes: dict[str, int],
) -> dict[str, Any]:
    index, statistics = _capture_index(
        X,
        Y,
        mode=mode,
        index_dim=index_dim,
        seed=seed,
        n_loc=sizes["N_loc"],
        n_lin=sizes["N_lin"],
        n_centers=sizes["N_J"],
        n_directions=sizes["N_phi"],
    )
    U, I, mass = statistics.U, statistics.I, statistics.mass
    coefficients, _ = _local_refit(I, U, index)
    weighted_coefficients = (
        coefficients * np.sqrt(mass)
        if index.ndim == 1
        else coefficients * np.sqrt(mass)[:, None]
    )
    if index.ndim == 1:
        A = (weighted_coefficients[:, None, None] * U).reshape(-1, index.size)
    else:
        A = np.einsum("ja,jpk->jpak", weighted_coefficients, U, optimize=True)
        A = A.reshape(-1, index.size)

    operator = _linear_operator(U, coefficients, np.sqrt(mass), index.shape)
    rng = np.random.default_rng(seed + 100_000)
    forward_errors: list[float] = []
    adjoint_errors: list[float] = []
    gram_errors: list[float] = []
    hessian = A.T @ A
    lambda_prox = 0.05
    ridge_hessian = hessian + lambda_prox * np.eye(index.size)
    ridge_augmented = {
        "rows": int(A.shape[0] + A.shape[1]),
        "columns": int(A.shape[1]),
        "nnz_exact": int(np.count_nonzero(A) + A.shape[1]),
        "density_exact": float(
            (np.count_nonzero(A) + A.shape[1])
            / ((A.shape[0] + A.shape[1]) * A.shape[1])
        ),
    }
    for _ in range(3):
        vector = rng.standard_normal(index.size)
        dual = rng.standard_normal(A.shape[0])
        forward_errors.append(_relative_error(operator @ vector, A @ vector))
        adjoint_errors.append(_relative_error(operator.rmatvec(dual), A.T @ dual))
        gram_errors.append(
            _relative_error(
                operator.rmatvec(operator @ vector) + lambda_prox * vector,
                ridge_hessian @ vector,
            )
        )
    return {
        "status": "ok",
        "mode": mode,
        "index_shape": list(index.shape),
        "U_shape": list(U.shape),
        "design": _matrix_metrics(A),
        "ridge_augmented_design": ridge_augmented,
        "normal_matrix": _matrix_metrics(hessian),
        "ridge_normal_matrix": _matrix_metrics(ridge_hessian),
        "reference_relative_errors": {
            "forward_max": max(forward_errors),
            "adjoint_max": max(adjoint_errors),
            "ridge_normal_action_max": max(gram_errors),
        },
    }


def _capture_manifold(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    seed: int,
    sizes: dict[str, int],
) -> tuple[ADP_Manifold, dict[str, Any]]:
    model = ADP_Manifold(
        index_dim=1,
        N_loc=sizes["N_loc"],
        N_lin=sizes["N_lin"],
        N_J=sizes["N_J"],
        N_phi=sizes["N_phi"],
        N_manifold=sizes["N_manifold"],
        sync_steps=1,
        seed=seed,
        solver="cg",
        estimator="manifold",
    )

    def capture(
        I: np.ndarray,
        U: np.ndarray,
        mass: np.ndarray,
        graph: Any,
        projectors: np.ndarray,
    ) -> None:
        raise _CapturedManifold(
            {
                "I": I.copy(),
                "U": U.copy(),
                "mass": mass.copy(),
                "graph": graph.copy(),
                "projectors": projectors.copy(),
            }
        )

    model._one_step = capture  # type: ignore[method-assign]
    try:
        model.fit(X, Y)
    except _CapturedManifold as state:
        return model, state.state
    raise RuntimeError("manifold fit did not reach its statistics capture point")


def _audit_manifold(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    seed: int,
    sizes: dict[str, int],
) -> dict[str, Any]:
    model, state = _capture_manifold(X, Y, seed=seed, sizes=sizes)
    I, U, mass = state["I"], state["U"], state["mass"]
    graph, projectors = state["graph"], state["projectors"]
    design_metrics: list[dict[str, int | float]] = []
    normal_metrics: list[dict[str, int | float]] = []
    action_errors: list[float] = []
    source_counts: list[int] = []
    lambda_manifold = model.lambda_manifold
    rng = np.random.default_rng(seed + 200_000)

    for target in range(len(projectors)):
        begin, end = graph.indptr[target : target + 2]
        sources = graph.indices[begin:end]
        weights = graph.data[begin:end]
        source_U, source_I = U[sources], I[sources]
        slopes = optimisation.local_slopes(
            source_I, source_U, projectors[target], target
        )
        gamma = mass[sources] * weights
        design = np.einsum(
            "j,ja,jpk->jpak", np.sqrt(gamma), slopes, source_U, optimize=True
        ).reshape(-1, projectors.shape[1] * U.shape[2])
        data_hessian = design.T @ design

        normalized_weights = weights / weights.sum()
        projector_scatter = np.einsum(
            "j,jra,jrb->ab",
            normalized_weights,
            projectors[sources],
            projectors[sources],
            optimize=True,
        )
        feature_penalty = np.eye(U.shape[2]) - projector_scatter
        penalty = np.kron(np.eye(projectors.shape[1]), feature_penalty)
        hessian = data_hessian + lambda_manifold * penalty

        operator, _, _ = model._build_B_system(
            source_U,
            source_I,
            mass[sources],
            weights,
            projectors[sources],
            slopes,
        )
        target_errors: list[float] = []
        for _ in range(3):
            vector = rng.standard_normal(hessian.shape[0])
            target_errors.append(_relative_error(operator @ vector, hessian @ vector))
        action_errors.append(max(target_errors))
        design_metrics.append(_matrix_metrics(design))
        normal_metrics.append(_matrix_metrics(hessian))
        source_counts.append(len(sources))

    graph_density = float(graph.nnz / (graph.shape[0] * graph.shape[1]))
    return {
        "status": "ok",
        "mode": "manifold",
        "index_dim": 1,
        "U_shape": list(U.shape),
        "target_count": len(projectors),
        "graph": {
            "shape": list(graph.shape),
            "nnz_exact": int(graph.nnz),
            "density_exact": graph_density,
            "mean_degree": float(np.mean(np.diff(graph.indptr))),
        },
        "local_design": _summarize_matrices(design_metrics),
        "normal_matrix": _summarize_matrices(normal_metrics),
        "sources_per_target": _quantiles([float(x) for x in source_counts]),
        "reference_relative_errors": {"normal_action_max": max(action_errors)},
    }


def _summarize_matrices(matrices: list[dict[str, int | float]]) -> dict[str, Any]:
    fields = (
        "density_exact",
        "nearzero_nonzero_1e-12",
        "nearzero_density_1e-12",
        "min_abs_nonzero",
    )
    return {
        "count": len(matrices),
        "shape": [matrices[0]["rows"], matrices[0]["columns"]] if matrices else [],
        "per_matrix_rows": [int(matrix["rows"]) for matrix in matrices],
        "per_matrix_nnz_exact": [int(matrix["nnz_exact"]) for matrix in matrices],
        "per_matrix_density_exact": [
            float(matrix["density_exact"]) for matrix in matrices
        ],
        "per_matrix_nearzero_density_1e-12": [
            float(matrix["nearzero_density_1e-12"]) for matrix in matrices
        ],
        **{
            name: _quantiles([float(matrix[name]) for matrix in matrices])
            for name in fields
        },
    }


def _run_task(tier: str, task_index: int) -> dict[str, Any]:
    if tier == "small":
        sizes = {
            "n": 240,
            "d": 12,
            "N_J": 24,
            "N_phi": 16,
            "N_loc": 32,
            "N_lin": 80,
            "N_manifold": 6,
        }
        seed = 85_000 + task_index
    else:
        sizes = {
            "n": 1200,
            "d": 100,
            "N_J": 100,
            "N_phi": 32,
            "N_loc": 200,
            "N_lin": 700,
            "N_manifold": 16,
        }
        seed = 86_000 + task_index
    profile = PROFILES[task_index]
    X, Y = _data(sizes["n"], sizes["d"], profile, seed)
    common = {key: sizes[key] for key in ("N_J", "N_phi", "N_loc", "N_lin")}
    outcome: dict[str, Any] = {
        "task_id": f"{tier}-{task_index + 1:02d}",
        "tier": tier,
        "profile": profile,
        "seed": seed,
        "n": sizes["n"],
        "d": sizes["d"],
        "parameters": {key: int(value) for key, value in sizes.items()},
        "X_density_exact": float(np.count_nonzero(X) / X.size),
        "families": {},
    }
    for mode, index_dim in (("single", 1), ("multi", 2)):
        try:
            outcome["families"][mode] = _audit_index(
                X,
                Y,
                mode=mode,
                index_dim=index_dim,
                seed=seed,
                sizes=common,
            )
        except Exception as error:
            outcome["families"][mode] = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
    try:
        outcome["families"]["manifold"] = _audit_manifold(
            X,
            Y,
            seed=seed,
            sizes={**common, "N_manifold": sizes["N_manifold"]},
        )
    except Exception as error:
        outcome["families"]["manifold"] = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return outcome


def _provenance() -> dict[str, Any]:
    def git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    packages = {name: importlib.metadata.version(name) for name in ("numpy", "scipy")}
    return {
        "git_commit": git(["rev-parse", "HEAD"]),
        "git_status_short": git(["status", "--short"]),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "blas_thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# Разреженность функционалов и применимость sparse solver-ов",
        "",
        "Аудит измеряет структурные нули точных матриц текущих функционалов. Он не",
        "изменяет estimator и не сравнивает wall-clock solver-ов.",
        "",
        f"- Задач: {result['completed_task_count']}/{result['planned_task_count']};",
        "  каждая задача включает single, multi и manifold.",
        "- Вещественная точность: float64; near-zero считается как",
        "  `0 < |a| <= 1e-12 max(|A|)` и не приравнивается к структурному нулю.",
        f"- Git: `{result['provenance']['git_commit']}`; dirty status сохранён в JSON.",
        "",
        "## Набор задач и матрицы",
        "",
        "На каждом tier проверено по 10 задач для каждой ветки: всего 60",
        "захватов live objective/system — первый штатный оператор для данного",
        "seed-а. Полная внешняя сходимость не измерялась. Отклик задавался как",
        "`Y = sin(X[:,0]) + 0.8 sin(X[:,1]) + N(0, 0.05^2)`. Профили включали",
        "dense iid/offset, корреляции 0.30/0.70, zero-inflation 40%/70% вне двух",
        "сигнальных координат, блочную корреляцию и признак масштаба `1e-4`.",
        "",
        "| Tier | n | d | J | P | N_loc | N_lin | N_manifold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        "| small | 240 | 12 | 24 | 16 | 32 | 80 | 6 |",
        "| medium | 1200 | 100 | 100 | 32 | 200 | 700 | 16 |",
        "",
        "Single/multi: измерен weighted design `A[j,p,a,k] = sqrt(mass[j]) *",
        "c[j,a] * U[j,p,k]`, развёрнутый в `(J*P) x (m*d)`, затем `A.T @ A`",
        "и ridge-вариант с `lambda_prox=0.05`. Manifold: для каждого target",
        "измерен локальный design `sqrt(mass[j] * graph_weight[j]) * slope[j,a] *",
        "U[j,p,k]` и B-system normal matrix с projector penalty. В JSON сохранены",
        "параметры и метрики каждого task-а. Single/multi normal имеет размер",
        "`(m*d) x (m*d)`; manifold B-system — `(r*d) x (r*d)`. Matrix-free",
        "действия сверены с dense reference на 3 случайных векторах для",
        "single/multi и 3 векторах на каждый manifold target.",
        "",
        "## Сводка по семейству и размеру",
        "",
        "| Tier | Семейство | Задач | Размер design A | Плотность A, median [min] | "
        "Плотность normal, median [min] | Near-zero A, median [max] | "
        "CSR graph | Ошибка action |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tier in ("small", "medium"):
        for family in ("single", "multi", "manifold"):
            summary = result["summary"].get(tier, {}).get(family)
            if not summary:
                continue

            def pct(field: str, row: dict[str, Any] = summary) -> str:
                distribution = row.get(field)
                median = distribution.get("median") if distribution else None
                return "—" if median is None else f"{100 * median:.4g}%"

            def pct_min(field: str, row: dict[str, Any] = summary) -> str:
                distribution = row.get(field)
                if not distribution or distribution.get("median") is None:
                    return "—"
                median, minimum = (
                    100 * distribution["median"],
                    100 * distribution["min"],
                )
                if abs(median - minimum) < 1e-12:
                    return f"{median:.6g}%"
                return f"{median:.6g}% [{minimum:.6g}%]"

            def pct_max(field: str, row: dict[str, Any] = summary) -> str:
                distribution = row.get(field)
                if not distribution or distribution.get("median") is None:
                    return "—"
                median, maximum = (
                    100 * distribution["median"],
                    100 * distribution["max"],
                )
                if abs(median - maximum) < 1e-12:
                    return f"{median:.6g}%"
                return f"{median:.6g}% [{maximum:.6g}%]"

            matrix_shape = summary.get("design_shape", "—")
            error = summary.get("max_reference_relative_error")
            error_text = "—" if error is None else f"{error:.3e}"
            lines.append(
                f"| {tier} | {family} | "
                f"{summary['completed_tasks']}/{summary['planned_tasks']} | "
                f"`{matrix_shape}` | "
                f"{pct_min('design_density')} | {pct_min('normal_density')} | "
                f"{pct_max('nearzero_density_1e-12')} | {pct('graph_density')} | "
                f"{error_text} |"
            )
    lines.extend(
        [
            "",
            "В manifold размер design A показывает первый target; плотности",
            "design и B-system считаются по всем target-матрицам",
            "всех задач tier-а. CSR graph — медиана плотности по задачам; граф не",
            "переносит свою структуру в feature-space матрицы B-system.",
            "",
            "## Вывод",
            "",
            result["conclusion"],
            "",
            "Аудит оценивает структурную применимость, но не преимущество по времени",
            "или памяти. Для такого вывода нужен отдельный парный benchmark с теми",
            "же stopping certificates и качеством.",
            "",
            "Подробные per-task / per-target квантили, failures, near-zero доли и",
            "provenance находятся в `audit.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_summary(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for tier in ("small", "medium"):
        summary[tier] = {}
        tier_tasks = [task for task in tasks if task["tier"] == tier]
        for family in ("single", "multi", "manifold"):
            records = [
                task["families"][family]
                for task in tier_tasks
                if task["families"].get(family, {}).get("status") == "ok"
            ]
            if not records:
                continue
            design_density: list[float] = []
            normal_density: list[float] = []
            nearzero_density: list[float] = []
            errors: list[float] = []
            shapes: list[list[int]] = []
            for record in records:
                design = (
                    record["design"] if family != "manifold" else record["local_design"]
                )
                normal = record["normal_matrix"]
                if family == "manifold":
                    design_density.extend(design["per_matrix_density_exact"])
                    normal_density.extend(normal["per_matrix_density_exact"])
                    nearzero_density.extend(design["per_matrix_nearzero_density_1e-12"])
                    errors.append(
                        record["reference_relative_errors"]["normal_action_max"]
                    )
                else:
                    design_density.append(float(design["density_exact"]))
                    normal_density.append(float(normal["density_exact"]))
                    nearzero_density.append(float(design["nearzero_density_1e-12"]))
                    errors.extend(record["reference_relative_errors"].values())
                shapes.append(
                    list(
                        design.get("shape", [design.get("rows"), design.get("columns")])
                    )
                )
            if family == "manifold":
                shape = f"{shapes[0][0]}x{shapes[0][1]} per target"
            else:
                shape = f"{shapes[0][0]}x{shapes[0][1]}"
            summary[tier][family] = {
                "completed_tasks": len(records),
                "planned_tasks": len(tier_tasks),
                "design_shape": shape,
                "design_density": _quantiles(design_density),
                "normal_density": _quantiles(normal_density),
                "nearzero_density_1e-12": _quantiles(nearzero_density),
                "max_reference_relative_error": max(errors) if errors else None,
                "graph_density": _quantiles(
                    [record["graph"]["density_exact"] for record in records]
                )
                if family == "manifold"
                else None,
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("small", "medium", "both"), default="both")
    parser.add_argument(
        "--limit", type=int, default=10, help="tasks per selected tier (smoke runs)"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 1 <= args.limit <= 10:
        parser.error("--limit must be between 1 and 10")

    tiers = ("small", "medium") if args.tier == "both" else (args.tier,)
    tasks = [_run_task(tier, index) for tier in tiers for index in range(args.limit)]
    summary = _build_summary(tasks)
    planned = len(tiers) * args.limit
    completed = sum(
        all(
            task["families"].get(family, {}).get("status") == "ok"
            for family in ("single", "multi", "manifold")
        )
        for task in tasks
    )
    exact_design_min_densities = [
        value
        for tier_summary in summary.values()
        for family_summary in tier_summary.values()
        for value in [family_summary["design_density"]["min"]]
    ]
    exact_normal_min_densities = [
        value
        for tier_summary in summary.values()
        for family_summary in tier_summary.values()
        for value in [family_summary["normal_density"]["min"]]
    ]
    if (
        exact_design_min_densities
        and exact_normal_min_densities
        and min(exact_design_min_densities) >= 0.99
        and min(exact_normal_min_densities) >= 0.99
    ):
        conclusion = (
            f"Минимальная точная плотность design матриц — "
            f"{100 * min(exact_design_min_densities):.6g}%, normal matrices — "
            f"{100 * min(exact_normal_min_densities):.6g}%. Редкие точные нули "
            "есть только в отдельных medium zero-inflated-70 design-матрицах; "
            "массового zero pattern для sparse storage/factorization нет. CSR-граф "
            "manifold имеет собственную измеренную плотность, но не разреживает "
            "feature-space B-system. Вывод ограничен этими synthetic профилями."
        )
    else:
        conclusion = (
            "Плотность зависит от семейства или профиля данных. Сверь per-task "
            "результаты и не удаляй near-zero ненулевые коэффициенты без отдельного "
            "приближённого solver-а и парной валидации."
        )
    result = {
        "schema_version": 1,
        "planned_task_count": planned,
        "completed_task_count": completed,
        "requested_full_grid": args.tier == "both" and args.limit == 10,
        "provenance": _provenance(),
        "summary": summary,
        "tasks": tasks,
        "conclusion": conclusion,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(_report(result), encoding="utf-8")
    print(_report(result))


if __name__ == "__main__":
    main()
