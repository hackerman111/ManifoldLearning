import numpy as np

from diagnostics.trace import TraceStep

rng = np.random.default_rng(seed=42)


def _local_rng(seed=None):
    """
    Возвращает генератор случайных чисел.
    """
    if isinstance(seed, np.random.Generator):
        return seed

    if seed is None:
        return rng

    return np.random.default_rng(seed)


def _as_feature_matrix(X, name="X"):
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"{name} должен быть матрицей размера n x d")

    if X.shape[0] == 0:
        raise ValueError(f"{name} должен содержать хотя бы одну точку")

    if X.shape[1] == 0:
        raise ValueError(f"{name} должен содержать хотя бы один признак")

    return X


def NormVector(x):
    """
    Нормирует вектор на евклидову длину.
    """
    x = np.asarray(x, dtype=float)
    norm = np.linalg.norm(x)

    if norm == 0:
        raise ValueError("Ошибка: вектор нулевой длины")

    return x / norm


def NormalizeRows(matrix):
    """
    Нормирует каждую строку матрицы.
    """
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2:
        raise ValueError("matrix должен быть двумерной матрицей")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)

    if np.any(norms == 0):
        raise ValueError("Нельзя нормировать строку нулевой длины")

    return matrix / norms


def ChooseJ(X, n_J=None, seed=None, sigma_x=0.0, replace=False):
    """
    Выбирает множество индексов J и соответствующие центры x_j.
    """
    X = _as_feature_matrix(X)
    n, d = X.shape

    if n_J is None:
        n_J = n

    if not isinstance(n_J, (int, np.integer)):
        raise TypeError("n_J должно быть целым числом")

    if n_J <= 0:
        raise ValueError("n_J должно быть положительным")

    if not replace and n_J > n:
        raise ValueError("n_J не может быть больше n при replace=False")

    if sigma_x < 0:
        raise ValueError("sigma_x должен быть неотрицательным")

    local_rng = _local_rng(seed)
    J = local_rng.choice(n, size=n_J, replace=replace)
    x_j = X[J].copy()

    if sigma_x > 0:
        x_j += sigma_x * local_rng.normal(size=(n_J, d))

    return J, x_j


def Kernel(t):
    """
    Ядро Эпанечникова из описания алгоритма: K(t) = (1 - t^2)_+.
    """
    t = np.asarray(t, dtype=float)
    return np.maximum(1.0 - t**2, 0.0)


def ComputeWeight(X, x_j, h, kernel=Kernel):
    """
    Считает матрицу весов w_{j,i} = K(||X_i - x_j||^2 / h^2).
    """
    X = _as_feature_matrix(X)
    x_j = np.asarray(x_j, dtype=float)

    if x_j.ndim == 1:
        x_j = x_j.reshape(1, -1)

    x_j = _as_feature_matrix(x_j, name="x_j")

    if X.shape[1] != x_j.shape[1]:
        raise ValueError("X и x_j должны иметь одинаковое число признаков")

    if h <= 0:
        raise ValueError("h должен быть положительным")

    differences = X[None, :, :] - x_j[:, None, :]
    squared_distances = np.sum(differences**2, axis=2)

    return kernel(squared_distances / h**2)


def ComputeWeigth(X, x_j, h, kernel=Kernel):
    """
    Обратная совместимость с прежним именем, где была опечатка.
    """
    return ComputeWeight(X, x_j, h, kernel=kernel)


def ChooseH0(
    X,
    x_j,
    n_min=10,
    kernel=Kernel,
    h_min=1e-8,
    h_max=None,
    tol=1e-6,
    max_iter=80,
    return_weights=False,
):
    """
    Выбирает начальную ширину окна h_0.

    h_0 берется как минимальное h, для которого

        mean_j sum_i K(||X_i - x_j||^2 / h^2) >= n_min.
    """
    X = _as_feature_matrix(X)
    x_j = np.asarray(x_j, dtype=float)

    if x_j.ndim == 1:
        x_j = x_j.reshape(1, -1)

    x_j = _as_feature_matrix(x_j, name="x_j")

    if X.shape[1] != x_j.shape[1]:
        raise ValueError("X и x_j должны иметь одинаковое число признаков")

    if n_min <= 0:
        raise ValueError("n_min должно быть положительным")

    if n_min > X.shape[0]:
        raise ValueError("n_min не может быть больше числа точек в X")

    if h_min <= 0:
        raise ValueError("h_min должен быть положительным")

    if tol <= 0:
        raise ValueError("tol должен быть положительным")

    # --- Оптимизация ChooseH0 ---
    # Бинарный поиск много раз проверяет разные h на одних и тех же X и x_j.
    # Поэтому расстояния считаем один раз, а затем меняем только масштаб h.
    differences = X[None, :, :] - x_j[:, None, :]
    squared_distances = np.sum(differences**2, axis=2)

    def weights_for_h(h):
        return kernel(squared_distances / h**2)

    def average_local_count(h):
        return weights_for_h(h).sum(axis=1).mean()

    low = h_min

    if h_max is None:
        high = h_min
        while average_local_count(high) < n_min:
            high *= 2.0
    else:
        if h_max <= h_min:
            raise ValueError("h_max должен быть больше h_min")
        high = h_max

    if average_local_count(high) < n_min:
        raise ValueError("На отрезке [h_min, h_max] условие для h_0 не выполняется")

    for _ in range(max_iter):
        middle = 0.5 * (low + high)

        if average_local_count(middle) >= n_min:
            high = middle
        else:
            low = middle

        if high - low <= tol * max(1.0, high):
            break

    h0 = high

    if return_weights:
        return h0, weights_for_h(h0)

    return h0


def GenerateDirection(d, n_directions=1, seed=None, dtype=float):
    """
    Генерирует случайные единичные направления в R^d.
    """
    if d <= 0:
        raise ValueError("d должно быть положительным")

    if n_directions <= 0:
        raise ValueError("n_directions должно быть положительным")

    local_rng = _local_rng(seed)
    directions = local_rng.normal(size=(n_directions, d))
    directions = NormalizeRows(directions).astype(dtype, copy=False)

    if n_directions == 1:
        return directions[0]

    return directions


def GenerateDirectionsForCenters(x_j, n_directions, seed=None, dtype=float):
    """
    Для каждого центра x_j генерирует набор направлений S_j.
    """
    x_j = np.asarray(x_j, dtype=float)

    if x_j.ndim == 1:
        x_j = x_j.reshape(1, -1)

    x_j = _as_feature_matrix(x_j, name="x_j")

    if n_directions <= 0:
        raise ValueError("n_directions должно быть положительным")

    local_rng = _local_rng(seed)
    raw = local_rng.normal(size=(x_j.shape[0], n_directions, x_j.shape[1]))
    flat = raw.reshape(-1, x_j.shape[1])
    normalized = NormalizeRows(flat).reshape(raw.shape)

    return normalized.astype(dtype, copy=False)


def PrepareADPInitialState(
    X,
    n_J=None,
    n_directions=None,
    n_min=10,
    seed=None,
    center_sigma=0.0,
    trace=None,
):
    """
    Готовит объекты шага 0: J, x_j, S_j, h_0 и начальные веса.
    """
    X = _as_feature_matrix(X)
    local_rng = _local_rng(seed)

    if n_directions is None:
        n_directions = X.shape[1]

    # --- Трассировка step0: входные параметры ---
    TraceStep(
        trace,
        "step0_input",
        X=X,
        n_J=n_J,
        n_directions=n_directions,
        n_min=n_min,
        center_sigma=center_sigma,
    )

    J, x_j = ChooseJ(X, n_J=n_J, seed=local_rng, sigma_x=center_sigma)
    directions = GenerateDirectionsForCenters(x_j, n_directions, seed=local_rng)
    h0, weights = ChooseH0(X, x_j, n_min=n_min, return_weights=True)

    # --- Трассировка step0: выбранные переменные ---
    TraceStep(
        trace,
        "step0_output",
        J=J,
        x_j=x_j,
        directions=directions,
        h0=h0,
        weights=weights,
        weight_sums=weights.sum(axis=1),
    )

    return {
        "J": J,
        "x_j": x_j,
        "directions": directions,
        "h0": h0,
        "weights": weights,
    }
