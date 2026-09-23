"""Общий single/multi outer-цикл для моделей и CLI; HPAO statistics/solver."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from ...core.ADP_Config import ADP_Config
from ...core.ADP_Statistic import ADP_Statistics
from ...solver.LSMR import HPAOResult
from . import utils
from .calculus import (
    calculate_alpha_k,
    calculate_rho_k,
    generate_isotropic_proj,
    generate_multi_proj,
    generate_proj,
    pairwise_distance2,
    search_bandwidth,
)
from .initialize import (
    initialize_basis_local_with_spectrum,
    initialize_basis_pilot,
    initialize_basis_random,
)
from .logger import IndexProfiler
from .statistic import calculate_statistics
from .weights import calculate_multi_weight, calculate_weight


@dataclass(frozen=True, slots=True)
class IndexFitResult:
    """Результат outer-цикла; multi index хранится строками (m,d)."""

    initial_index: np.ndarray
    index: np.ndarray
    coefficients: np.ndarray
    eigenvalues: np.ndarray
    metadata: dict[str, object]


def validate_sizes(
    config: ADP_Config, mode: str, n: int, d: int, index_dim: int
) -> tuple[int, int, int]:
    """Проверить размеры и training split до построения статистик."""
    if mode not in {"single", "multi"}:
        raise ValueError("mode must be single or multi")
    if n < 2:
        raise ValueError("n must be at least two")
    if config.index_init != "random" and n <= d + 1:
        raise ValueError("n must exceed d + 1 unless index_init is random")
    if mode == "single" and index_dim != 1:
        raise ValueError("index_dim must equal 1 in single mode")
    if mode == "multi" and not 1 <= index_dim < d:
        raise ValueError("index_dim must lie between 1 and d - 1 in multi mode")
    if mode == "single" and config.index_init == "pilot":
        raise ValueError("pilot initialization is multi-index only")
    n_lin = config.N_lin or 2 * d
    n_centers = config.N_J or n
    n_directions = config.N_phi or min(config.N_loc, d)
    utils._check_model_sizes(
        n,
        d,
        config.N_loc,
        n_lin,
        n_centers,
        config.index_init,
    )
    if config.training_set == "exclude_centers":
        training_size = n - n_centers
        if training_size < 2:
            raise ValueError("excluding center indices leaves fewer than two rows")
        if config.N_loc > training_size:
            raise ValueError("N_loc cannot exceed the training-set size")
        if config.index_init in {"local", "local-cv"} and n_lin > training_size:
            raise ValueError("N_lin cannot exceed the training-set size")
    if mode == "multi" and n_directions <= index_dim:
        raise ValueError("N_phi must exceed index_dim in multi mode")
    return n_lin, n_centers, n_directions


def _initial_index(
    mode: str,
    d: int,
    index_dim: int,
    config: ADP_Config,
    X: np.ndarray,
    Y: np.ndarray,
    centers: np.ndarray,
    distance2: np.ndarray,
    n_lin: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Построить исходный индекс и спектр по текущему протоколу CLI."""
    if config.index_init == "random":
        basis = initialize_basis_random(rng, d, index_dim)
        spectrum = np.empty(0)
    elif config.index_init == "pilot":
        basis = initialize_basis_pilot(
            X,
            Y,
            index_dim,
            seed=config.seed,
        )
        spectrum = np.empty(0)
    else:
        basis, spectrum = initialize_basis_local_with_spectrum(
            X,
            Y,
            centers,
            distance2,
            n_lin,
            config.kernel,
            config.local_ridge,
            index_dim,
            mass_weighted=config.estimator == "new",
            **({"ridge_selection": "loo"} if config.index_init == "local-cv" else {}),
        )
    index = basis[:, 0] if mode == "single" else basis.T
    return index, spectrum


def canonical_index(
    mode: str,
    result: HPAOResult,
    prior: np.ndarray,
    *,
    mass: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Проверить индекс и канонизировать multi basis через малый фактор."""
    index = np.array(result.index, dtype=float, copy=True)
    if mode == "single":
        norm = np.linalg.norm(index)
        if index.shape != prior.shape or not np.isfinite(norm) or norm == 0:
            raise RuntimeError("single-index solver returned an invalid index")
        index /= norm
        if np.dot(index, prior) < 0:
            index = -index
        return index, np.ones(1)

    coefficients = np.asarray(result.coefficients, dtype=float)
    if index.shape != prior.shape or coefficients.shape[1:] != (index.shape[0],):
        raise RuntimeError("multi-index solver returned incompatible shapes")
    if not all(np.all(np.isfinite(value)) for value in (index, coefficients)):
        raise RuntimeError("multi-index solver returned non-finite values")
    if not np.allclose(
        index @ index.T,
        np.eye(index.shape[0]),
        rtol=1e-8,
        atol=1e-10,
    ):
        raise RuntimeError("multi-index solver returned a non-orthonormal index")

    if mass is None:
        weighted_coefficients = coefficients
    else:
        # ESTIMATOR: новая версия использует G=sum_j mass_j l_j l_j^T.
        mass = np.asarray(mass, dtype=float)
        if mass.shape != (len(coefficients),):
            raise RuntimeError("multi-index mass has incompatible shape")
        if not np.all(np.isfinite(mass)) or np.any(mass <= 0):
            raise RuntimeError("multi-index mass must be finite and positive")
        weighted_coefficients = np.sqrt(mass)[:, None] * coefficients

    # EXACT при заданных mass: SVD малого фактора без структурной d x d матрицы.
    _, singular_values, right_vectors = np.linalg.svd(
        weighted_coefficients,
        full_matrices=False,
    )
    values = np.square(singular_values)
    threshold = (
        np.finfo(float).eps
        * max(weighted_coefficients.shape)
        * (singular_values[0] if len(singular_values) else 0.0)
    )
    if len(values) < index.shape[0] or singular_values[index.shape[0] - 1] <= threshold:
        raise RuntimeError("local coefficients do not identify a multi-index")
    index = right_vectors @ index
    signs = np.sign(index[np.arange(len(index)), np.argmax(np.abs(index), axis=1)])
    index *= np.where(signs == 0, 1.0, signs)[:, None]
    return index, values / values[0]


def fit_index(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    config: ADP_Config,
    mode: str,
    index_dim: int,
    solve: Callable[[np.ndarray, ADP_Statistics], HPAOResult],
    profiler: IndexProfiler,
    progress: Callable[[dict], None] | None = None,
    quality: Callable[[np.ndarray], float] | None = None,
    displacement_scale: float | None = None,
    trace_indices: bool = False,
    statistics_function: Callable[..., ADP_Statistics] | None = None,
) -> IndexFitResult:
    """Выполнить текущий estimator; truth используется только внешней метрикой.

    X=(n,d), Y=(n,); возвращаются single (d,) либо multi (m,d).
    ESTIMATOR: перенос legacy-классов на этот явно выбранный HPAO-путь
    меняет local refit, нормировку и seed scheme; CLI-формулы сохранены.
    """
    X, Y = utils._prepare_xy(X, Y, require_overdetermined=config.index_init != "random")
    if statistics_function is None:
        statistics_function = calculate_statistics
    gpu_statistics = None
    if config.gpu:
        from ...gpu import require_cupy

        require_cupy()  # Ошибка CUDA до дорогой CPU-инициализации.
    n, d = X.shape
    n_lin, n_centers, n_directions = validate_sizes(config, mode, n, d, index_dim)
    if config.h_min is None:
        raise ValueError("h_min must be configured for the current index engine")
    center_seed, init_seed, direction_seed = np.random.SeedSequence(config.seed).spawn(
        3
    )
    with profiler.stage("initialization"):
        center_rng = np.random.default_rng(center_seed)
        center_indices = (
            np.arange(n)
            if n_centers == n
            else center_rng.choice(n, size=n_centers, replace=False)
        )
        centers = X[center_indices].copy()
        requested_displacement_scale = displacement_scale
        displacement_scale = 0.0
        if config.center_displacement > 0:
            # ESTIMATOR/data protocol: TeX centers x_j = X_i + nu sigma_X z_j.
            requested_scale = requested_displacement_scale
            displacement_scale = (
                float(requested_scale)
                if requested_scale is not None
                else float(np.sqrt(np.mean(np.var(X, axis=0))))
            )
            if not np.isfinite(displacement_scale) or displacement_scale <= 0:
                raise ValueError("center displacement scale must be positive")
            centers += (
                config.center_displacement
                * displacement_scale
                * center_rng.standard_normal(centers.shape)
            )

        if config.training_set == "exclude_centers":
            # ESTIMATOR/data protocol: testing indices не участвуют в fit.
            training_mask = np.ones(n, dtype=bool)
            training_mask[center_indices] = False
            training_X = X[training_mask]
            training_Y = Y[training_mask]
        else:
            training_X = X
            training_Y = Y
        test_Y = Y[center_indices]

        distance2 = pairwise_distance2(training_X, centers)
        index, initial_eigenvalues = _initial_index(
            mode,
            d,
            index_dim,
            config,
            training_X,
            training_Y,
            centers,
            distance2,
            n_lin,
            np.random.default_rng(init_seed),
        )
        if config.gpu:
            from .gpu_statistics import GPUStatistics

            gpu_statistics = GPUStatistics(
                training_X, training_Y, keep_on_device=config.gpu_solver
            )
        initial_index = index.copy()
        initial_quality = quality(index) if quality is not None else None

    with profiler.stage("bandwidth"):
        h = search_bandwidth(
            distance2,
            config.N_loc,
            config.kernel,
            lower=config.h_min,
        )

    direction_rng = np.random.default_rng(direction_seed)
    factor = 1.0
    eigenvalues = np.ones(index_dim)
    diagnostics: dict[str, object] = {}
    trace: list[dict[str, object]] = []
    best_error = float("inf")
    best_iteration = 0
    best_index = index.copy()
    best_eigenvalues = eigenvalues.copy()
    best_diagnostics: dict[str, object] = {}
    best_coefficients = np.empty(0)
    normalized = config.estimator == "new"
    direction_mode = config.direction_mode
    if direction_mode == "auto":
        direction_mode = "isotropic" if normalized and mode == "multi" else "localized"
    stop_reason = "h_min"
    outer_iteration = 0
    fixed_directions: np.ndarray | None = None

    while True:
        with profiler.stage("directions"):
            if fixed_directions is None:
                if direction_mode == "isotropic":
                    directions = generate_isotropic_proj(
                        direction_rng,
                        n_centers,
                        n_directions,
                        d,
                    )
                elif mode == "single":
                    directions = generate_proj(
                        direction_rng,
                        n_centers,
                        n_directions,
                        index,
                        factor,
                    )
                else:
                    directions = generate_multi_proj(
                        direction_rng,
                        n_centers,
                        n_directions,
                        index.T,
                        eigenvalues,
                        factor,
                    )
                if not config.redraw_directions:
                    # ESTIMATOR: fixed sketch переиспользует направления
                    # первого шага.
                    fixed_directions = directions
            else:
                directions = fixed_directions

        with profiler.stage("statistics"):
            effective_tensor = (
                "orthogonal"
                if mode == "multi" and outer_iteration == 0
                else config.multi_tensor
            )
            if mode == "single":
                weights = calculate_weight(
                    training_X,
                    centers,
                    index,
                    h,
                    factor,
                    config.kernel,
                    block_size=config.batch_size,
                    distance2=distance2,
                    estimator=config.estimator,
                )
            else:
                weights = calculate_multi_weight(
                    training_X,
                    centers,
                    index.T,
                    eigenvalues,
                    h,
                    factor,
                    config.kernel,
                    block_size=config.batch_size,
                    distance2=distance2,
                    tensor=effective_tensor,
                )
            if gpu_statistics is None:
                statistics = statistics_function(
                    training_X,
                    training_Y,
                    weights,
                    directions,
                    batch_size=config.batch_size,
                    normalized=normalized,
                )
            else:
                statistics = gpu_statistics.calculate(
                    weights,
                    directions,
                    batch_size=config.batch_size,
                    normalized=normalized,
                )

        with profiler.stage("solver"):
            result = solve(index, statistics)
            expected_coefficients = (
                (n_centers,) if mode == "single" else (n_centers, index_dim)
            )
            if result.coefficients.shape != expected_coefficients or not np.all(
                np.isfinite(result.coefficients)
            ):
                raise RuntimeError("solver returned invalid local coefficients")
            index, eigenvalues = canonical_index(
                mode,
                result,
                index,
                mass=statistics.mass if normalized else None,
            )
            if index.ndim == 1:
                coefficients = result.coefficients * float(result.index @ index)
            else:
                # EXACT: поворот coefficients сохраняет prediction после смены basis.
                rotation = index @ result.index.T
                coefficients = result.coefficients @ rotation.T
            diagnostics = dict(result.diagnostics)

        outer_iteration += 1
        fit_error = float(np.sum(np.square(test_Y - statistics.S)))
        step_quality = quality(index) if quality is not None else None
        trace.append(
            {
                "iteration": outer_iteration - 1,
                "h": float(h),
                "factor_name": "rho" if mode == "single" else "alpha",
                "factor": float(factor),
                "h_over_factor": (float(h / factor) if factor > 0 else float("inf")),
                "err": fit_error,
                "quality": step_quality,
                "eigenvalues": tuple(float(value) for value in eigenvalues),
                "tensor": effective_tensor if mode == "multi" else None,
                "solver": diagnostics,
            }
        )
        if trace_indices:
            row = trace[-1]
            row.update(diagnostics)
            row["k"] = outer_iteration - 1
            row["rho" if mode == "single" else "alpha"] = float(factor)
            row["beta" if mode == "single" else "basis"] = (
                index.copy() if mode == "single" else index.T.copy()
            )
            row["mean_mass"] = float(np.mean(statistics.mass))
            if mode == "single":
                row["cosine_initial"] = float(abs(initial_index @ index))
            else:
                row["distance_initial"] = float(
                    np.linalg.norm(index - (index @ initial_index.T) @ initial_index)
                    / np.sqrt(index_dim)
                )
        if progress is not None:
            progress(deepcopy(trace[-1]))
        if fit_error < best_error:
            best_error = fit_error
            best_iteration = outer_iteration - 1
            best_index = index.copy()
            best_eigenvalues = eigenvalues.copy()
            best_diagnostics = diagnostics
            best_coefficients = coefficients.copy()

        del statistics
        if config.outer_steps is not None and outer_iteration >= config.outer_steps:
            stop_reason = "outer_steps"
            break
        next_h = h / config.a
        if next_h < config.h_min:
            break

        with profiler.stage("update"):
            if mode == "single":
                next_factor = calculate_rho_k(
                    training_X,
                    centers,
                    index,
                    next_h,
                    config.N_loc,
                    config.kernel,
                    distance2=distance2,
                    estimator=config.estimator,
                )
            else:
                next_factor = calculate_alpha_k(
                    training_X,
                    centers,
                    index.T,
                    eigenvalues,
                    next_h,
                    config.N_loc,
                    config.kernel,
                    distance2=distance2,
                    tensor=config.multi_tensor,
                    block_size=config.batch_size,
                )
            if next_factor is None:
                stop_reason = "local_mass_limit"
                break
            factor = next_factor
            h = float(next_h)

    last_diagnostics = diagnostics
    if trace_indices:
        trace[-1]["stop_reason"] = stop_reason
    selected_iteration = outer_iteration - 1
    if config.select_step == "best":
        index = best_index
        eigenvalues = best_eigenvalues
        diagnostics = best_diagnostics
        coefficients = best_coefficients
        selected_iteration = best_iteration

    metadata: dict[str, object] = {
        "gpu": bool(config.gpu),
        "gpu_solver": bool(config.gpu_solver),
        "N_lin": n_lin,
        "N_J": n_centers,
        "N_phi": n_directions,
        "outer_iterations": outer_iteration,
        "stop_reason": stop_reason,
        "estimator": config.estimator,
        "direction_mode": direction_mode,
        "multi_tensor": config.multi_tensor,
        "selection": config.select_step,
        "selected_iteration": selected_iteration,
        "selected_error": (
            best_error
            if config.select_step == "best"
            else trace[selected_iteration]["err"]
        ),
        "selected_eigenvalues": tuple(float(value) for value in eigenvalues),
        "initial_quality": initial_quality,
        "last_quality": trace[-1]["quality"],
        "initial_eigenvalues": tuple(
            float(value) for value in initial_eigenvalues[: index_dim + 1]
        ),
        "training_set": config.training_set,
        "training_size": len(training_X),
        "center_displacement": config.center_displacement,
        "center_displacement_scale": displacement_scale,
        "redraw_directions": config.redraw_directions,
        "trace": trace,
        "diagnostics": diagnostics,
        "last_diagnostics": last_diagnostics,
    }
    return IndexFitResult(initial_index, index, coefficients, eigenvalues, metadata)
