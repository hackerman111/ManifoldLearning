import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

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


def ProjectionStatistics(X, Y, weights, directions, trace=None):
    """
    Считает проекционные статистики Ima_{j,phi}, S_{j,phi}, U_{j,phi}.

    Xbar_j берется как нормированное взвешенное среднее, а не как сырая сумма.
    """
    X = _as_feature_matrix(X)
    Y = _as_response_vector(Y, X.shape[0])
    weights = np.asarray(weights, dtype=float)
    directions = np.asarray(directions, dtype=float)

    if weights.ndim != 2:
        raise ValueError("weights должен быть матрицей n_J x n")

    if weights.shape[1] != X.shape[0]:
        raise ValueError("weights должен иметь форму n_J x n")

    if directions.ndim != 3:
        raise ValueError("directions должен иметь форму n_J x n_directions x d")

    if directions.shape[0] != weights.shape[0] or directions.shape[2] != X.shape[1]:
        raise ValueError("directions должен согласовываться с weights и X")

    n_J, n_directions, d = directions.shape
    effective_counts = weights.sum(axis=1)
    safe_counts = np.maximum(effective_counts, np.finfo(float).eps)
    S = np.empty((n_J, n_directions))
    Ima = np.empty((n_J, n_directions))
    U = np.empty((n_J, n_directions, d))

    for j in range(n_J):
        w = weights[j]
        xbar = (w @ X) / safe_counts[j]
        centered = X - xbar
        projected = centered @ directions[j].T
        weighted_projected = projected * w[:, None]
        S[j] = weighted_projected.sum(axis=0)
        Ima[j] = (w * Y) @ projected
        U[j] = weighted_projected.T @ centered

    TraceStep(
        trace,
        "projection_statistics",
        effective_counts=effective_counts,
        S=S,
        Ima=Ima,
        U=U,
    )

    return {
        "N": effective_counts,
        "S": S,
        "Ima": Ima,
        "U": U,
    }


def UpdateFcl(Ima, U, beta):
    """
    Закрытая форма fcl_j = <Ima_j, U_j beta> / ||U_j beta||^2.
    """
    Ima = np.asarray(Ima, dtype=float)
    U = np.asarray(U, dtype=float)
    beta = NormVector(beta)

    projected_beta = np.einsum("jsd,d->js", U, beta)
    numerator = np.einsum("js,js->j", Ima, projected_beta)
    denominator = np.einsum("js,js->j", projected_beta, projected_beta)
    denominator = np.maximum(denominator, np.finfo(float).tiny)

    return numerator / denominator


def UpdateBetaCG(
    Ima,
    U,
    fcl,
    beta_prev,
    lambda_penalty=1.0,
    ridge=1e-10,
    tol=1e-8,
    maxiter=250,
):
    """
    Решает beta-обновление matrix-free через CG без формирования d x d матрицы.
    """
    Ima = np.asarray(Ima, dtype=float)
    U = np.asarray(U, dtype=float)
    fcl = np.asarray(fcl, dtype=float)
    beta_prev = NormVector(beta_prev)

    if lambda_penalty < 0:
        raise ValueError("lambda_penalty должен быть неотрицательным")

    d = U.shape[2]
    regularization = float(lambda_penalty) + float(ridge)

    def matvec(vector):
        projected = np.einsum("jsd,d->js", U, vector)
        result = np.einsum("j,js,jsd->d", fcl**2, projected, U)
        result += regularization * vector
        return result

    rhs = np.einsum("j,js,jsd->d", fcl, Ima, U)
    rhs += float(lambda_penalty) * beta_prev

    operator = LinearOperator((d, d), matvec=matvec, dtype=float)
    beta, info = cg(operator, rhs, x0=beta_prev, rtol=tol, atol=0.0, maxiter=maxiter)

    if info < 0 or not np.all(np.isfinite(beta)) or np.linalg.norm(beta) == 0:
        return beta_prev.copy()

    return beta


def AlternatingProjectionMinimization(
    Ima,
    U,
    beta0,
    lambda_penalty=1.0,
    ridge=1e-10,
    n_inner=5,
    cg_tol=1e-8,
    cg_maxiter=250,
):
    """
    Чередует закрытое обновление fcl_j и matrix-free CG-обновление beta.
    """
    beta = NormVector(beta0)
    fcl = np.ones(U.shape[0])
    history = []

    for _ in range(max(1, int(n_inner))):
        old_beta = beta.copy()
        fcl = UpdateFcl(Ima, U, beta)
        beta = UpdateBetaCG(
            Ima,
            U,
            fcl,
            old_beta,
            lambda_penalty=lambda_penalty,
            ridge=ridge,
            tol=cg_tol,
            maxiter=cg_maxiter,
        )
        beta = NormVector(beta)
        history.append(beta.copy())

    return beta, fcl, history


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
