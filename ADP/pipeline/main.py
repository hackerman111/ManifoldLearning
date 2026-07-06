import numpy as np
from algorithm.step0 import (
    ChooseH0,
    ChooseJ,
    ComputeWeight,
    GenerateDirectionsForCenters,
    Kernel,
    NormVector,
    PrepareADPInitialState,
    _as_feature_matrix,
    _local_rng,
)
from algorithm.stepk import (
    CosineSimilarity,
    EstimateLocalGradients,
    LocalLinearGradient,
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
):
    """
    Оркестратор полного учебного пайплайна Average Derivative Procedure.
    """
    X = _as_feature_matrix(X)
    Y = _as_response_vector(Y, X.shape[0])

    if ridge < 0:
        raise ValueError("ridge должен быть неотрицательным")

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
            n_directions = X.shape[1]

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

        with RuntimeStage(runtime_monitor, "stepk_local_gradients"):
            local_gradients_work = EstimateLocalGradients(
                X=X_work,
                Y=Y,
                x_j=centers_work,
                h=h0,
                kernel=kernel,
                ridge=ridge,
                weights=weights,
                trace=trace,
                runtime_monitor=runtime_monitor,
            )

        # --- Трассировка: локальные градиенты в рабочем пространстве ---
        TraceStep(
            trace,
            "stepk_local_gradients",
            local_gradients_work=local_gradients_work,
            local_gradient_norms=np.linalg.norm(local_gradients_work, axis=1),
        )

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
            "J": J,
            "x_j": centers,
            "x_j_work": centers_work,
            "directions": directions,
            "x_mean": x_mean,
            "x_scale": x_scale,
            "standardize": standardize,
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
