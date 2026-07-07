import numpy as np
from algorithm.step0 import (
    ChooseH0,
    ChooseJ,
    ComputeWeight,
    GenerateDirectionsForCenters,
    Kernel,
    NormVector,
    PrepareADPInitialState,
    SquaredDistancesGram,
    _as_feature_matrix,
    _local_rng,
)
from algorithm.stepk import (
    AlternatingProjectionMinimization,
    CosineSimilarity,
    EstimateLocalGradients,
    LocalLinearGradient,
    ProjectionStatistics,
    StandardizeFeatures,
    _as_response_vector,
)
from diagnostics.trace import CreateTrace, SaveADPDiagnostics, TraceStep
from runtime.monitoring import CreateRuntimeMonitor, RuntimeStage, RuntimeSummary


def AverageDerivativeProcedure(
    X,
    Y,
    n_J=None,
    n_min=10,
    seed=None,
    center_sigma=0.0,
    kernel=Kernel,
    h0=None,
    ridge=1e-6,
    standardize=True,
    n_directions=None,
    return_state=True,
    trace=None,
    trace_enabled=False,
    trace_store_arrays=False,
    make_plots=False,
    plot_dir="adp_trace_plots",
    runtime_monitor=None,
    show_progress=False,
    log_runtime=False,
    runtime_log_path=None,
    use_rich=False,
    outer_steps=4,
    inner_steps=5,
    bandwidth_decay=np.sqrt(2.0),
    h_min=1e-2,
    lambda_penalty=None,
    cg_tol=1e-8,
    cg_maxiter=250,
):
    """
    Оркестратор полного учебного пайплайна Average Derivative Procedure.
    """
    X = _as_feature_matrix(X)
    Y = _as_response_vector(Y, X.shape[0])

    if ridge < 0:
        raise ValueError("ridge должен быть неотрицательным")

    if outer_steps <= 0:
        raise ValueError("outer_steps должно быть положительным")

    if inner_steps <= 0:
        raise ValueError("inner_steps должно быть положительным")

    if bandwidth_decay <= 1:
        raise ValueError("bandwidth_decay должен быть больше 1")

    if h_min <= 0:
        raise ValueError("h_min должен быть положительным")

    if lambda_penalty is None:
        lambda_penalty = max(1.0, float(n_min))

    if lambda_penalty < 0:
        raise ValueError("lambda_penalty должен быть неотрицательным")

    local_rng = _local_rng(seed)

    # --- Runtime-monitoring: progress bar, logging, timing ---
    if runtime_monitor is None and (show_progress or log_runtime or use_rich):
        runtime_monitor = CreateRuntimeMonitor(
            enabled=True,
            use_tqdm=show_progress,
            use_rich=use_rich,
            log_runtime=log_runtime,
            log_path=runtime_log_path,
        )

    with RuntimeStage(runtime_monitor, "total_pipeline", n=X.shape[0], d=X.shape[1]):
        # --- Трассировка: входные данные и настройки ---
        if trace is None and trace_enabled:
            trace = CreateTrace(enabled=True, store_arrays=trace_store_arrays)

        TraceStep(
            trace,
            "input",
            X=X,
            Y=Y,
            n_J=n_J,
            n_min=n_min,
            center_sigma=center_sigma,
            h0=h0,
            ridge=ridge,
            standardize=standardize,
            n_directions=n_directions,
            outer_steps=outer_steps,
            inner_steps=inner_steps,
            bandwidth_decay=bandwidth_decay,
            h_min=h_min,
            lambda_penalty=lambda_penalty,
        )

        with RuntimeStage(runtime_monitor, "standardize"):
            if standardize:
                X_work, x_mean, x_scale = StandardizeFeatures(X)
            else:
                X_work = X.copy()
                x_mean = np.zeros(X.shape[1])
                x_scale = np.ones(X.shape[1])

        # --- Трассировка: стандартизация признаков ---
        TraceStep(
            trace,
            "standardize",
            X_work=X_work,
            x_mean=x_mean,
            x_scale=x_scale,
        )

        if n_directions is None:
            n_directions = min(20, X.shape[1])

        with RuntimeStage(runtime_monitor, "step0_centers"):
            J, centers_work = ChooseJ(
                X_work,
                n_J=n_J,
                seed=local_rng,
                sigma_x=center_sigma,
                replace=False,
            )
            directions = GenerateDirectionsForCenters(
                centers_work,
                n_directions=n_directions,
                seed=local_rng,
            )

        # --- Трассировка: выбор центров и направлений ---
        TraceStep(
            trace,
            "step0_centers",
            J=J,
            centers_work=centers_work,
            directions=directions,
        )

        with RuntimeStage(runtime_monitor, "step0_weights"):
            if h0 is None:
                h0, weights = ChooseH0(
                    X_work,
                    centers_work,
                    n_min=n_min,
                    kernel=kernel,
                    return_weights=True,
                )
            else:
                if h0 <= 0:
                    raise ValueError("h0 должен быть положительным")
                weights = ComputeWeight(X_work, centers_work, h0, kernel=kernel)

        # --- Трассировка: h0 и веса ядра ---
        TraceStep(
            trace,
            "step0_weights",
            h0=h0,
            weights=weights,
            weight_sums=weights.sum(axis=1),
        )

        beta_seed = X_work.T @ (Y - Y.mean())
        if np.linalg.norm(beta_seed) == 0:
            beta_seed = local_rng.normal(size=X_work.shape[1])
        beta_work = NormVector(beta_seed)
        history = []
        h_history = [h0]
        fcl = np.ones(centers_work.shape[0])
        projection_state = None
        h = h0
        current_weights = weights
        current_directions = directions
        isotropic_squared_distances = SquaredDistancesGram(X_work, centers_work)

        with RuntimeStage(runtime_monitor, "stepk_projection_outer"):
            for outer in range(int(outer_steps)):
                if outer > 0:
                    h = max(h / bandwidth_decay, h_min)
                    projected_X = X_work @ beta_work
                    projected_centers = centers_work @ beta_work
                    projected_squared_distances = (
                        projected_centers[:, None] - projected_X[None, :]
                    ) ** 2
                    current_weights = kernel(
                        (isotropic_squared_distances + projected_squared_distances) / h**2
                    )
                    raw_directions = local_rng.normal(
                        size=(centers_work.shape[0], n_directions, X_work.shape[1])
                    )
                    raw_directions += 2.0 * beta_work[None, None, :]
                    current_directions = raw_directions / np.maximum(
                        np.linalg.norm(raw_directions, axis=2, keepdims=True),
                        np.finfo(float).eps,
                    )
                    h_history.append(h)

                projection_state = ProjectionStatistics(
                    X_work,
                    Y,
                    current_weights,
                    current_directions,
                    trace=trace,
                )
                beta_work, fcl, inner_history = AlternatingProjectionMinimization(
                    projection_state["Ima"],
                    projection_state["U"],
                    beta_work,
                    lambda_penalty=lambda_penalty,
                    ridge=ridge,
                    n_inner=inner_steps,
                    cg_tol=cg_tol,
                    cg_maxiter=cg_maxiter,
                )
                history.extend(inner_history)

                TraceStep(
                    trace,
                    "stepk_projection_outer",
                    outer=outer,
                    h=h,
                    beta_work=beta_work,
                    fcl=fcl,
                    weight_sums=current_weights.sum(axis=1),
                )

                if h <= h_min:
                    break

        local_gradients_work = np.repeat(beta_work.reshape(1, -1), centers_work.shape[0], axis=0)

        with RuntimeStage(runtime_monitor, "final_estimate"):
            local_gradients = local_gradients_work / x_scale
            average_gradient = local_gradients.mean(axis=0)
            beta = NormVector(average_gradient)
            centers = centers_work * x_scale + x_mean

        # --- Трассировка: финальная оценка направления ---
        TraceStep(
            trace,
            "final_result",
            local_gradients=local_gradients,
            average_gradient=average_gradient,
            beta=beta,
            centers=centers,
        )

        result = {
            "beta": beta,
            "direction": beta,
            "average_gradient": average_gradient,
            "local_gradients": local_gradients,
            "local_gradients_work": local_gradients_work,
            "h0": h0,
            "weights": weights,
            "final_weights": current_weights,
            "J": J,
            "x_j": centers,
            "x_j_work": centers_work,
            "directions": current_directions,
            "initial_directions": directions,
            "x_mean": x_mean,
            "x_scale": x_scale,
            "standardize": standardize,
            "algorithm": "projection_average_derivative",
            "history": history,
            "h": h,
            "h_history": h_history,
            "fcl": fcl,
            "projection_state": projection_state,
        }

        if trace is not None:
            result["trace"] = trace

        if return_state:
            result["state"] = {
                "X_work": X_work,
                "Y": Y,
                "kernel": kernel,
                "ridge": ridge,
                "n_min": n_min,
                "n_J": centers_work.shape[0],
                "n_directions": n_directions,
                "outer_steps": outer_steps,
                "inner_steps": inner_steps,
                "lambda_penalty": lambda_penalty,
            }

    if runtime_monitor is not None:
        result["runtime"] = {
            "events": runtime_monitor.get("events", []),
            "summary": RuntimeSummary(runtime_monitor),
        }

    if make_plots:
        result["diagnostics"] = SaveADPDiagnostics(result, output_dir=plot_dir)

    return result


def RunADP(*args, **kwargs):
    """
    Короткий alias для полного пайплайна ADP.
    """
    return AverageDerivativeProcedure(*args, **kwargs)


def FitADP(*args, **kwargs):
    """
    Alias для вызова ADP как fit-процедуры.
    """
    return AverageDerivativeProcedure(*args, **kwargs)


def AlteringOptimisation(*args, **kwargs):
    """
    Совместимое имя для запуска готового пайплайна.
    """
    return AverageDerivativeProcedure(*args, **kwargs)
