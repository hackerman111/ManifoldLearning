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
    """Вернуть локальную gradient-PCA базу и спектр ``J_EDR``."""
    X, Y = utils._prepare_xy(X, Y, require_overdetermined=False)
    X, centers = utils._prepare_pairwise(X, centers)
    distance2 = utils._prepare_distance2(distance2, centers, len(X))
    if not np.isfinite(local_ridge) or local_ridge < 0:
        raise ValueError("local_ridge must be finite and nonnegative")
    if isinstance(index_dim, bool) or not isinstance(index_dim, (int, np.integer)):
        raise TypeError("index_dim must be an integer")
    if not 1 <= index_dim <= X.shape[1]:
        raise ValueError("index_dim must lie between 1 and d")
    if not isinstance(mass_weighted, bool):
        raise TypeError("mass_weighted must be boolean")
    if ridge_selection not in {"fixed", "loo"}:
        raise ValueError("ridge_selection must be 'fixed' or 'loo'")

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
        if weights.shape != (1, n):
            raise ValueError("initialization kernel must preserve the input shape")
        weights = weights[0]
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("initialization weights must be finite and nonnegative")
        local_mass[j] = weights.sum()
        if not np.isfinite(local_mass[j]) or local_mass[j] <= 0:
            raise RuntimeError("local initialization contains an empty neighborhood")
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
    """NUMERICAL: исключить свободный intercept до ridge-SVD.

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
    if not np.all(np.isfinite(gradient)):
        raise RuntimeError("local ridge initialization returned non-finite gradients")
    return gradient


def _select_loo_ridge(
    left: np.ndarray,
    values: np.ndarray,
    projected_y: np.ndarray,
    y: np.ndarray,
    root_weight: np.ndarray,
    minimum: float,
) -> float:
    """ESTIMATOR: weighted leave-one-out выбор ridge при фиксированных весах.

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
    if best_ridge is None:
        raise RuntimeError("no local ridge candidate has a finite leave-one-out score")
    return best_ridge


def _principal_gradient_basis(
    gradients: np.ndarray,
    index_dim: int,
    *,
    mass: np.ndarray | None = None,
) -> np.ndarray:
    """Вернуть главные направления из малого mass-взвешенного gradient PCA."""
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
    values = gradients
    if mass is not None:
        if mass.shape != (len(gradients),):
            raise ValueError("mass must have shape (J,)")
        if not np.all(np.isfinite(mass)) or np.any(mass <= 0):
            raise ValueError("mass must be finite and positive")
        # ESTIMATOR: SVD(sqrt(mass) * gradients) соответствует J_EDR из TeX.
        values = np.sqrt(mass)[:, None] * gradients

    _, singular_values, right_vectors = np.linalg.svd(values, full_matrices=False)
    threshold = (
        np.finfo(float).eps
        * max(values.shape)
        * (singular_values[0] if len(singular_values) else 0.0)
    )
    if len(singular_values) < index_dim or singular_values[index_dim - 1] <= threshold:
        raise RuntimeError("local gradients do not identify the requested index")

    spectrum = np.square(singular_values)
    return _orient_basis(right_vectors[:index_dim].T.copy()), spectrum


def initialize_basis_pilot(
    X: np.ndarray,
    Y: np.ndarray,
    index_dim: int,
    *,
    seed: int,
) -> np.ndarray:
    try:
        MLPRegressor = import_module("sklearn.neural_network").MLPRegressor
    except ImportError as error:
        raise ImportError("pilot initialization requires scikit-learn") from error

    X, Y = utils._prepare_xy(X, Y)
    if isinstance(index_dim, bool) or not isinstance(index_dim, (int, np.integer)):
        raise TypeError("index_dim must be an integer")
    if not 1 <= index_dim <= X.shape[1]:
        raise ValueError("index_dim must lie between 1 and d")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be nonnegative")

    pilot = MLPRegressor(
        hidden_layer_sizes=(int(index_dim),),
        activation="tanh",
        solver="lbfgs",
        alpha=0.1,
        max_iter=1000,
        random_state=int(seed),
    ).fit(X, Y)
    weights = np.asarray(pilot.coefs_[0], dtype=float)
    if weights.shape != (X.shape[1], index_dim) or not np.all(np.isfinite(weights)):
        raise RuntimeError("pilot initializer returned invalid weights")
    if np.linalg.matrix_rank(weights) != index_dim:
        raise RuntimeError("pilot initializer returned a rank-deficient basis")
    basis, _ = np.linalg.qr(weights, mode="reduced")
    return _orient_basis(basis)


def initialize_basis_random(
    rng: np.random.Generator,
    n_features: int,
    index_dim: int,
) -> np.ndarray:
    for name, value in (("n_features", n_features), ("index_dim", index_dim)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
    if not 1 <= index_dim <= n_features:
        raise ValueError("index_dim must lie between 1 and n_features")

    basis, _ = np.linalg.qr(
        rng.standard_normal((n_features, index_dim)),
        mode="reduced",
    )
    return _orient_basis(basis)


def _orient_basis(basis: np.ndarray) -> np.ndarray:
    columns = np.arange(basis.shape[1])
    signs = np.sign(basis[np.argmax(np.abs(basis), axis=0), columns])
    basis *= np.where(signs == 0, 1.0, signs)
    return basis
