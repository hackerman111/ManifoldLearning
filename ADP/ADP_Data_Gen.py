import numpy as np

from ADP_step0 import NormVector


rng = np.random.default_rng(seed=42)


def GenerateX(
    n,
    d,
    data_type="normal",
    seed=None,
    dtype=float,
    sigma_x=1.0,
    corr=0.0,
    low=-1.0,
    high=1.0,
):
    """
    Генерирует матрицу признаков X размера n x d.

    data_type:
    normal - независимые нормальные признаки;
    uniform - равномерные признаки на [low, high];
    correlated_normal - нормальные признаки с общей компонентой;
    sphere - точки на сфере радиуса sigma_x;
    student - тяжелые хвосты;
    mixture - смесь двух нормальных облаков.
    """
    if n <= 0:
        raise ValueError("n должно быть положительным")

    if d <= 0:
        raise ValueError("d должно быть положительным")

    if sigma_x <= 0:
        raise ValueError("sigma_x должен быть положительным")

    dtype = np.dtype(dtype)
    local_rng = np.random.default_rng(seed) if seed is not None else rng
    data_type = str(data_type).lower()

    if data_type in ("normal", "gaussian"):
        X = local_rng.normal(scale=sigma_x, size=(n, d))
    elif data_type == "uniform":
        if high <= low:
            raise ValueError("high должен быть больше low")
        X = local_rng.uniform(low=low, high=high, size=(n, d))
    elif data_type in ("correlated", "correlated_normal"):
        if not 0 <= corr < 1:
            raise ValueError("corr должен лежать в [0, 1)")
        common = local_rng.normal(size=(n, 1))
        individual = local_rng.normal(size=(n, d))
        X = sigma_x * (corr * common + np.sqrt(1.0 - corr**2) * individual)
    elif data_type in ("sphere", "unit_sphere"):
        X = local_rng.normal(size=(n, d))
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        X = sigma_x * X / norms
    elif data_type in ("student", "student_t"):
        X = sigma_x * local_rng.standard_t(df=3, size=(n, d))
    elif data_type == "mixture":
        labels = local_rng.choice([-1.0, 1.0], size=(n, 1))
        X = sigma_x * labels + 0.5 * sigma_x * local_rng.normal(size=(n, d))
    else:
        raise ValueError(
            "Неизвестный data_type. Используй normal, uniform, "
            "correlated_normal, sphere, student или mixture"
        )

    return X.astype(dtype, copy=False)


def FunctionValue(index, function="sin", linear_weight=0.3):
    """
    Считает значение выбранной одномерной функции от индекса beta^T X.
    """
    index = np.asarray(index, dtype=float)

    if callable(function):
        values = function(index)
        return np.asarray(values, dtype=float)

    function = str(function).lower()

    if function == "linear":
        return index
    if function == "quadratic":
        return index**2
    if function == "cubic":
        return index**3
    if function == "sin":
        return np.sin(index) + linear_weight * index
    if function == "tanh":
        return np.tanh(index)
    if function == "exp":
        return np.exp(np.clip(index, -20.0, 20.0))
    if function == "step":
        return (index > 0).astype(float)
    if function in ("abs", "absolute"):
        return np.abs(index)

    raise ValueError(
        "Неизвестная function. Используй linear, quadratic, cubic, "
        "sin, tanh, exp, step, abs или передай callable"
    )


def GenerateNoise(n, noise_std=0.15, noise_type="normal", seed=None, dtype=float):
    """
    Генерирует шум для отклика Y.
    """
    if n <= 0:
        raise ValueError("n должно быть положительным")

    if noise_std < 0:
        raise ValueError("noise_std должен быть неотрицательным")

    dtype = np.dtype(dtype)
    local_rng = np.random.default_rng(seed) if seed is not None else rng
    noise_type = str(noise_type).lower()

    if noise_std == 0 or noise_type in ("none", "zero", "zeros"):
        noise = np.zeros(n)
    elif noise_type in ("normal", "gaussian"):
        noise = local_rng.normal(scale=noise_std, size=n)
    elif noise_type == "uniform":
        bound = np.sqrt(3.0) * noise_std
        noise = local_rng.uniform(low=-bound, high=bound, size=n)
    elif noise_type in ("student", "student_t"):
        noise = noise_std * local_rng.standard_t(df=3, size=n)
    else:
        raise ValueError(
            "Неизвестный noise_type. Используй normal, uniform, student или none"
        )

    return noise.astype(dtype, copy=False)


def MakeData(
    n,
    d,
    beta=None,
    f=None,
    function="sin",
    data_type="normal",
    noise_std=0.15,
    noise_type="normal",
    seed=42,
    dtype=float,
    sigma_x=1.0,
    corr=0.0,
    low=-1.0,
    high=1.0,
    linear_weight=0.3,
    return_info=False,
):
    """
    Генерирует данные single-index модели Y = f(beta^T X) + noise.
    """
    dtype = np.dtype(dtype)
    local_rng = np.random.default_rng(seed) if seed is not None else rng

    X = GenerateX(
        n=n,
        d=d,
        data_type=data_type,
        seed=local_rng,
        dtype=dtype,
        sigma_x=sigma_x,
        corr=corr,
        low=low,
        high=high,
    )

    if beta is None:
        beta = local_rng.normal(size=d)

    beta = NormVector(np.asarray(beta, dtype=float)).astype(dtype, copy=False)

    if beta.shape != (d,):
        raise ValueError("beta должен иметь длину d")

    if f is not None:
        function = f

    index = X @ beta
    signal = FunctionValue(index, function=function, linear_weight=linear_weight)
    noise = GenerateNoise(
        n=n,
        noise_std=noise_std,
        noise_type=noise_type,
        seed=local_rng,
        dtype=dtype,
    )
    Y = (signal + noise).astype(dtype, copy=False)

    if return_info:
        info = {
            "function": function if isinstance(function, str) else "callable",
            "data_type": data_type,
            "noise_type": noise_type,
            "noise_std": noise_std,
            "sigma_x": sigma_x,
            "corr": corr,
            "dtype": str(dtype),
        }
        return X, Y, beta, info

    return X, Y, beta
