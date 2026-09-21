"""Инициализация single- и multi-index базисов."""

# ruff: noqa: RUF002

from collections.abc import Callable
from importlib import import_module

import numpy as np

from . import utils
from .calculus import search_bandwidth


def initialize_beta_local(
    X: np.ndarray,
    Y: np.ndarray,
    centers: np.ndarray,
    distance2: np.ndarray,
    N_lin: int,
    kernel: Callable,
    local_ridge: float,
    *,
    mass_weighted: bool = False,
    ridge_selection: str = "fixed",
) -> np.ndarray:
    """Получить один локальный gradient-вектор из local initializer."""
    return initialize_basis_local(
        X,
        Y,
        centers,
        distance2,
        N_lin,
        kernel,
        local_ridge,
        1,
        mass_weighted=mass_weighted,
        ridge_selection=ridge_selection,
    )[:, 0]


def initialize_basis_local(
    X: np.ndarray,
    Y: np.ndarray,
    centers: np.ndarray,
    distance2: np.ndarray,
    N_lin: int,
    kernel: Callable,
    local_ridge: float,
    index_dim: int,
    *,
    mass_weighted: bool = False,
    ridge_selection: str = "fixed",
) -> np.ndarray:
    """Построить ``index_dim`` локальных направлений из weighted gradients."""
    basis, _ = initialize_basis_local_with_spectrum(
        X,
        Y,
        centers,
        distance2,
        N_lin,
        kernel,
        local_ridge,
        index_dim,
        mass_weighted=mass_weighted,
        ridge_selection=ridge_selection,
    )
    return basis


def initialize_basis_local_with_spectrum(
    X: np.ndarray,
    Y: np.ndarray,
    centers: np.ndarray,
    distance2: np.ndarray,
    N_lin: int,
    kernel: Callable,
    local_ridge: float,
    index_dim: int,
    *,
    mass_weighted: bool = False,
    ridge_selection: str = "fixed",
) -> tuple[np.ndarray, np.ndarray]:
    """Вернуть локальную gradient-PCA базу и спектр ``J_EDR``.

    Для каждого центра решается локальная ridge-регрессия отклика на
    центрированные признаки, после чего SVD матрицы градиентов извлекает
    главные EDR-направления. Выходы имеют формы ``(d,m)`` и ``(m,)``.
    """
    X, Y = utils._prepare_xy(X, Y, require_overdetermined=False)
    X, centers = utils._prepare_pairwise(X, centers)
    distance2 = utils._prepare_distance2(distance2, centers, len(X))
    utils.require(
        np.isfinite(local_ridge) and local_ridge >= 0,
        "local_ridge must be finite and nonnegative",
    )
    utils.require(
        not isinstance(index_dim, bool) and isinstance(index_dim, (int, np.integer)),
        "index_dim must be an integer",
        TypeError,
    )
    utils.require(
        1 <= index_dim <= X.shape[1],
        "index_dim must lie between 1 and d",
    )
    utils.require(
        isinstance(mass_weighted, bool), "mass_weighted must be boolean", TypeError
    )
    utils.require(
        ridge_selection in {"fixed", "loo"},
        "ridge_selection must be 'fixed' or 'loo'",
    )

    h_lin = search_bandwidth(
        distance2,
        N_lin,
        kernel,
        lower=np.finfo(float).eps,
    )
    local_mass = np.empty(len(centers))
    n, d = X.shape
    gradients = np.empty((len(centers), d))
    rcond = np.finfo(float).eps * max(n + d, d + 1)
    ridge_rows = None

    for j, center in enumerate(centers):
        # EXACT: каждая строка используется только один раз; нет W=(J,n).
        weights = np.asarray(kernel(distance2[j : j + 1] / h_lin**2))
        utils.require(
            weights.shape == (1, n),
            "initialization kernel must preserve the input shape",
        )
        weights = weights[0]
        utils.require(
            np.all(np.isfinite(weights)) and not np.any(weights < 0),
            "initialization weights must be finite and nonnegative",
        )
        local_mass[j] = weights.sum()
        utils.require(
            np.isfinite(local_mass[j]) and local_mass[j] > 0,
            "local initialization contains an empty neighborhood",
            RuntimeError,
        )
        support = weights != 0
        local_X = X[support]
        local_Y = Y[support]
        root_weight = np.sqrt(weights[support])
        gradient = None
        # При k>=d augmented lstsq дешевле полного U из SVD (см. local_ridge).
        # Для LOO этот U нужен при любом k, чтобы вычислить диагональ hat matrix.
        if len(local_X) < d or ridge_selection == "loo":
            gradient = _centered_ridge_gradient(
                local_X - center,
                local_Y,
                root_weight,
                local_ridge,
                rcond,
                ridge_selection=ridge_selection,
            )
        if gradient is not None:
            gradients[j] = gradient
            continue
        # Численно усечённая исходная задача зависит от координат intercept.
        # При небезопасной нижней границе ранга сохраняем её буквально.
        if ridge_rows is None:
            ridge_rows = np.zeros((d, d + 1))
            ridge_rows[:, 1:] = np.sqrt(local_ridge) * np.eye(d)
        design = np.column_stack((np.ones(len(local_X)), local_X - center))
        augmented_design = np.vstack((design * root_weight[:, None], ridge_rows))
        augmented_Y = np.concatenate((local_Y * root_weight, np.zeros(d)))
        # EXACT: нулевые строки не влияют на LS. Сохраняем исходный cutoff
        # rcond=None для (n+d,d+1), чтобы screening не менял численный ранг.
        gradients[j] = np.linalg.lstsq(
            augmented_design,
            augmented_Y,
            rcond=rcond,
        )[0][1:]

    return _principal_gradient_basis_with_spectrum(
        gradients,
        index_dim,
        mass=local_mass if mass_weighted else None,
    )


def _centered_ridge_gradient(
    delta: np.ndarray,
    Y: np.ndarray,
    root_weight: np.ndarray,
    ridge: float,
    rcond: float,
    *,
    ridge_selection: str = "fixed",
) -> np.ndarray | None:
    """Вычислить устойчивый локальный ridge-gradient через SVD.

    Intercept устраняется weighted-центрированием, а коэффициенты получают
    фильтр ``s/(s²+ridge)`` без normal equations.

    NUMERICAL: исключить свободный intercept до ridge-SVD.

    delta=(k,d). После weighted centering A=diag(sqrt(w))*(delta-mean),
    g=V diag(s/(s²+ridge)) U.T y. Нет normal matrix или ridge rows (d,d).
    None означает, что исходный rank cutoff требует augmented reference.
    """
    if ridge <= 0 and ridge_selection == "fixed":
        return None
    mass = float(root_weight @ root_weight)
    mean = (np.square(root_weight) / mass) @ delta
    A = (delta - mean) * root_weight[:, None]
    # A_aug = A_centered_aug @ T, ||T||,||T^-1|| <= 1+||mean||.
    # Эти границы гарантируют отсутствие отсечения исходным rcond.
    change_bound = 1.0 + float(np.linalg.norm(mean))
    lower = min(np.sqrt(mass), np.sqrt(ridge)) / change_bound
    upper = (np.sqrt(mass) + np.linalg.norm(A) + np.sqrt(ridge)) * change_bound
    if lower <= rcond * upper and ridge_selection == "fixed":
        return None
    y = Y - Y[0]
    y -= (np.square(root_weight) / mass) @ y
    y *= root_weight
    left, values, right = np.linalg.svd(A, full_matrices=False)
    projected_y = left.T @ y
    if ridge_selection == "loo":
        ridge = _select_loo_ridge(left, values, projected_y, y, root_weight, ridge)
    denominator = np.hypot(values, np.sqrt(ridge))
    factors = (values / denominator) / denominator
    gradient = right.T @ (factors * projected_y)
    utils.require(
        np.all(np.isfinite(gradient)),
        "local ridge initialization returned non-finite gradients",
        RuntimeError,
    )
    return gradient


def _select_loo_ridge(
    left: np.ndarray,
    values: np.ndarray,
    projected_y: np.ndarray,
    y: np.ndarray,
    root_weight: np.ndarray,
    minimum: float,
) -> float:
    """Выбрать ridge по weighted leave-one-out PRESS-критерию.

    Hat-diagonal вычисляется из SVD один раз, поэтому повторные факторизации
    для кандидатов не нужны. ``minimum`` задаёт нижнюю границу ridge.

    ESTIMATOR: weighted leave-one-out выбор ridge при фиксированных весах.

    H=ww.T/(w.Tw)+U diag(s²/(s²+lambda)) U.T, w=sqrt(weights).
    PRESS=sum((residual/(1-diag(H)))²), без k повторных факторизаций.
    Сетка относительных ridge 1e-6..1e2 проверяет восемь порядков масштаба;
    minimum — нижняя граница пользователя, включённая отдельным кандидатом.
    """
    scale = float(values[0] ** 2) if len(values) else 0.0
    candidates = np.unique(np.maximum(minimum, scale * np.logspace(-6, 2, 9)))
    candidates = np.unique(np.append(candidates, minimum))
    squared_left = np.square(left)
    intercept = np.square(root_weight) / (root_weight @ root_weight)
    best_loss = float("inf")
    best_ridge = None
    for ridge in candidates:
        denominator = np.hypot(values, np.sqrt(ridge))
        shrink = np.square(
            np.divide(
                values, denominator, out=np.zeros_like(values), where=denominator > 0
            )
        )
        residual = y - left @ (shrink * projected_y)
        remaining = 1.0 - intercept - squared_left @ shrink
        # LOO неопределён при leverage=1: такой кандидат не принимаем.
        if np.any(remaining <= np.sqrt(np.finfo(float).eps)):
            continue
        loss = float(np.sum(np.square(residual / remaining)))
        if np.isfinite(loss) and loss < best_loss:
            best_loss, best_ridge = loss, float(ridge)
    best_ridge = utils.require_not_none(
        best_ridge,
        "no local ridge candidate has a finite leave-one-out score",
        RuntimeError,
    )
    return best_ridge


def _principal_gradient_basis(
    gradients: np.ndarray,
    index_dim: int,
    *,
    mass: np.ndarray | None = None,
) -> np.ndarray:
    """Вернуть главные направления из mass-взвешенного gradient PCA.

    Функция делегирует SVD-варианту со спектром и возвращает первые
    ``index_dim`` ориентированных ортонормированных направлений формы ``(d,m)``.
    """
    basis, _ = _principal_gradient_basis_with_spectrum(
        gradients,
        index_dim,
        mass=mass,
    )
    return basis


def _principal_gradient_basis_with_spectrum(
    gradients: np.ndarray,
    index_dim: int,
    *,
    mass: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Разложить матрицу локальных градиентов и вернуть basis со спектром.

    SVD применяется к градиентам или к ``sqrt(mass) * gradients``; проверка
    сингулярных значений подтверждает идентифицируемость ``index_dim``.
    """
    values = gradients
    if mass is not None:
        utils.require(mass.shape == (len(gradients),), "mass must have shape (J,)")
        utils.require(
            np.all(np.isfinite(mass)) and not np.any(mass <= 0),
            "mass must be finite and positive",
        )
        # ESTIMATOR: SVD(sqrt(mass) * gradients) соответствует J_EDR из TeX.
        values = np.sqrt(mass)[:, None] * gradients

    _, singular_values, right_vectors = np.linalg.svd(values, full_matrices=False)
    threshold = (
        np.finfo(float).eps
        * max(values.shape)
        * (singular_values[0] if len(singular_values) else 0.0)
    )
    utils.require(
        len(singular_values) >= index_dim
        and singular_values[index_dim - 1] > threshold,
        "local gradients do not identify the requested index",
        RuntimeError,
    )

    spectrum = np.square(singular_values)
    return _orient_basis(right_vectors[:index_dim].T.copy()), spectrum


def initialize_basis_pilot(
    X: np.ndarray,
    Y: np.ndarray,
    index_dim: int,
    *,
    seed: int,
) -> np.ndarray:
    """Получить экспериментальный pilot-basis из небольшой MLP-регрессии."""
    try:
        MLPRegressor = import_module("sklearn.neural_network").MLPRegressor
    except ImportError as error:
        utils.raise_import_error("pilot initialization requires scikit-learn", error)

    X, Y = utils._prepare_xy(X, Y)
    utils.require(
        not isinstance(index_dim, bool) and isinstance(index_dim, (int, np.integer)),
        "index_dim must be an integer",
        TypeError,
    )
    utils.require(
        1 <= index_dim <= X.shape[1],
        "index_dim must lie between 1 and d",
    )
    utils.require(
        not isinstance(seed, bool) and isinstance(seed, (int, np.integer)),
        "seed must be an integer",
        TypeError,
    )
    utils.require(seed >= 0, "seed must be nonnegative")

    pilot = MLPRegressor(
        hidden_layer_sizes=(int(index_dim),),
        activation="tanh",
        solver="lbfgs",
        alpha=0.1,
        max_iter=1000,
        random_state=int(seed),
    ).fit(X, Y)
    weights = np.asarray(pilot.coefs_[0], dtype=float)
    utils.require(
        weights.shape == (X.shape[1], index_dim) and np.all(np.isfinite(weights)),
        "pilot initializer returned invalid weights",
        RuntimeError,
    )
    utils.require(
        np.linalg.matrix_rank(weights) == index_dim,
        "pilot initializer returned a rank-deficient basis",
        RuntimeError,
    )
    basis, _ = np.linalg.qr(weights, mode="reduced")
    return _orient_basis(basis)


def initialize_basis_random(
    rng: np.random.Generator,
    n_features: int,
    index_dim: int,
) -> np.ndarray:
    """Сгенерировать воспроизводимый случайный ортонормированный basis."""
    for name, value in (("n_features", n_features), ("index_dim", index_dim)):
        utils.require(
            not isinstance(value, bool) and isinstance(value, (int, np.integer)),
            f"{name} must be an integer",
            TypeError,
        )
    utils.require(
        1 <= index_dim <= n_features,
        "index_dim must lie between 1 and n_features",
    )

    basis, _ = np.linalg.qr(
        rng.standard_normal((n_features, index_dim)),
        mode="reduced",
    )
    return _orient_basis(basis)


def _orient_basis(basis: np.ndarray) -> np.ndarray:
    """Канонизировать знаки столбцов ортонормированного basis."""
    columns = np.arange(basis.shape[1])
    signs = np.sign(basis[np.argmax(np.abs(basis), axis=0), columns])
    basis *= np.where(signs == 0, 1.0, signs)
    return basis
