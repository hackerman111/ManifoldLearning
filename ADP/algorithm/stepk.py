import numpy as np

from runtime.monitoring import IterateWithProgress
from diagnostics.trace import TraceStep
from algorithm.step0 import (
    ComputeWeight,
    Kernel,
    NormVector,
    _as_feature_matrix,
)


def _as_response_vector(Y, n):
    Y = np.asarray(Y, dtype=float)

    if Y.ndim != 1:
        raise ValueError("Y должен быть одномерным вектором длины n")

    if Y.shape[0] != n:
        raise ValueError("X и Y должны содержать одинаковое число наблюдений")

    return Y


def CalculateRho(h, step=0, factor=2.0, rho_min=1e-3):
    """
    Считает простой параметр локализации rho для k-го шага.
    """
    if h <= 0:
        raise ValueError("h должен быть положительным")

    if step < 0:
        raise ValueError("step должен быть неотрицательным")

    if factor <= 1:
        raise ValueError("factor должен быть больше 1")

    if rho_min <= 0:
        raise ValueError("rho_min должен быть положительным")

    return max(h / factor**step, rho_min)


def StandardizeFeatures(X):
    """
    Центрирует и масштабирует признаки.
    """
    X = _as_feature_matrix(X)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)

    return (X - mean) / scale, mean, scale


def LocalLinearGradient(X, Y, center, h, kernel=Kernel, ridge=1e-6, weights=None):
    """
    Оценивает локальный градиент m(x) = E[Y | X=x] около одного центра.
    """
    X = _as_feature_matrix(X)
    Y = _as_response_vector(Y, X.shape[0])
    center = np.asarray(center, dtype=float)

    if center.shape != (X.shape[1],):
        raise ValueError("center должен иметь длину d")

    if ridge < 0:
        raise ValueError("ridge должен быть неотрицательным")

    shifted = X - center

    if weights is None:
        weights = ComputeWeight(X, center, h, kernel=kernel)[0]
    else:
        weights = np.asarray(weights, dtype=float)

    if weights.shape != (X.shape[0],):
        raise ValueError("weights должен иметь длину n")

    design = np.column_stack([np.ones(X.shape[0]), shifted])
    weighted_design = design * weights[:, None]
    lhs = design.T @ weighted_design
    rhs = weighted_design.T @ Y

    penalty = np.diag(np.r_[0.0, np.full(X.shape[1], ridge)])

    try:
        coefficients = np.linalg.solve(lhs + penalty, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(lhs + penalty, rhs, rcond=None)[0]

    return coefficients[1:]


def EstimateLocalGradients(
    X,
    Y,
    x_j,
    h,
    kernel=Kernel,
    ridge=1e-6,
    weights=None,
    trace=None,
    runtime_monitor=None,
):
    """
    Считает локальные градиенты во всех центрах x_j.
    """
    X = _as_feature_matrix(X)
    Y = _as_response_vector(Y, X.shape[0])
    x_j = np.asarray(x_j, dtype=float)

    if x_j.ndim == 1:
        x_j = x_j.reshape(1, -1)

    x_j = _as_feature_matrix(x_j, name="x_j")

    if X.shape[1] != x_j.shape[1]:
        raise ValueError("X и x_j должны иметь одинаковое число признаков")

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (x_j.shape[0], X.shape[0]):
            raise ValueError("weights должен иметь форму n_J x n")

    # --- Трассировка stepk: вход локальных регрессий ---
    TraceStep(
        trace,
        "stepk_input",
        X=X,
        Y=Y,
        x_j=x_j,
        h=h,
        weights=weights,
        ridge=ridge,
    )

    gradients = []

    iterator = IterateWithProgress(
        enumerate(x_j),
        monitor=runtime_monitor,
        total=x_j.shape[0],
        description="local gradients",
    )

    for j, center in iterator:
        current_weights = None if weights is None else weights[j]
        gradient = LocalLinearGradient(
            X=X,
            Y=Y,
            center=center,
            h=h,
            kernel=kernel,
            ridge=ridge,
            weights=current_weights,
        )
        gradients.append(gradient)

    gradients = np.array(gradients)

    # --- Трассировка stepk: локальные градиенты ---
    TraceStep(
        trace,
        "stepk_output",
        local_gradients=gradients,
        local_gradient_norms=np.linalg.norm(gradients, axis=1),
    )

    return gradients


def CosineSimilarity(first_vector, second_vector, absolute=True):
    """
    Косинус угла между двумя направлениями.
    """
    first_unit = NormVector(first_vector)
    second_unit = NormVector(second_vector)
    cosine = float(first_unit @ second_unit)

    if absolute:
        return abs(cosine)

    return cosine
