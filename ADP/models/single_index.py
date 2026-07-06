import numpy as np
from dataclasses import asdict, dataclass

from data.generation import (
    FunctionValue as _FunctionValue,
    GenerateNoise as _GenerateNoise,
    GenerateX as _GenerateX,
    MakeData as _MakeData,
)
from runtime.monitoring import CreateRuntimeMonitor
from diagnostics.trace import CreateTrace, SaveADPDiagnostics
from algorithm.step0 import (
    ChooseH0 as _ChooseH0,
    ChooseJ as _ChooseJ,
    ComputeWeight as _ComputeWeight,
    ComputeWeigth as _ComputeWeigth,
    GenerateDirection as _GenerateDirection,
    GenerateDirectionsForCenters as _GenerateDirectionsForCenters,
    Kernel,
    NormVector as _NormVector,
    NormalizeRows as _NormalizeRows,
    PrepareADPInitialState as _PrepareADPInitialState,
    _as_feature_matrix,
)
from algorithm.stepk import (
    CalculateRho as _CalculateRho,
    CosineSimilarity as _CosineSimilarity,
    EstimateLocalGradients as _EstimateLocalGradients,
    LocalLinearGradient as _LocalLinearGradient,
    StandardizeFeatures as _StandardizeFeatures,
    _as_response_vector,
)
from pipeline.main import AverageDerivativeProcedure as _AverageDerivativeProcedure


@dataclass
class ADPDataConfig:
    """
    Параметры генерации данных single-index модели.
    """

    function: object = "sin"
    data_type: str = "normal"
    noise_std: float = 0.15
    noise_type: str = "normal"
    dtype: object = float
    sigma_x: float = 1.0
    corr: float = 0.0
    low: float = -1.0
    high: float = 1.0
    linear_weight: float = 0.3

    def to_kwargs(self):
        return asdict(self)


class ADP_single_index:
    """
    Единый класс для работы с single-index ADP.

    Класс не заменяет низкоуровневые функции из algorithm/step0.py и algorithm/stepk.py,
    а собирает их в удобный объектный интерфейс для экспериментов.
    """

    def __init__(
        self,
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
        trace_enabled=False,
        trace_store_arrays=False,
        make_plots=False,
        plot_dir="adp_trace_plots",
        show_progress=True,
        log_runtime=False,
        runtime_log_path=None,
        use_rich=False,
        data_config=None,
        function="sin",
        data_type="normal",
        noise_std=0.15,
        noise_type="normal",
        dtype=float,
        sigma_x=1.0,
        corr=0.0,
        low=-1.0,
        high=1.0,
        linear_weight=0.3,
    ):
        # --- Параметры алгоритма ADP ---
        self.n_J = n_J
        self.n_min = n_min
        self.seed = seed
        self.center_sigma = center_sigma
        self.kernel = kernel
        self.h0 = h0
        self.ridge = ridge
        self.standardize = standardize
        self.n_directions = n_directions
        self.return_state = return_state

        # --- Параметры трассировки, графиков и runtime-monitoring ---
        self.trace_enabled = trace_enabled
        self.trace_store_arrays = trace_store_arrays
        self.make_plots = make_plots
        self.plot_dir = plot_dir
        self.show_progress = show_progress
        self.log_runtime = log_runtime
        self.runtime_log_path = runtime_log_path
        self.use_rich = use_rich

        # --- Параметры генерации данных ---
        if data_config is None:
            data_config = ADPDataConfig(
                function=function,
                data_type=data_type,
                noise_std=noise_std,
                noise_type=noise_type,
                dtype=dtype,
                sigma_x=sigma_x,
                corr=corr,
                low=low,
                high=high,
                linear_weight=linear_weight,
            )
        elif not isinstance(data_config, ADPDataConfig):
            raise TypeError("data_config должен быть экземпляром ADPDataConfig")

        self.data_config = data_config
        self._sync_data_attrs_from_config()

        self._reset_fit_state()

    def _sync_data_attrs_from_config(self):
        for name, value in self.data_config.to_kwargs().items():
            setattr(self, name, value)

    def _sync_data_config_from_attrs(self):
        params = {
            name: getattr(self, name)
            for name in ADPDataConfig.__dataclass_fields__
        }
        self.data_config = ADPDataConfig(**params)

    def _reset_fit_state(self):
        self.result_ = None
        self.beta_ = None
        self.direction_ = None
        self.average_gradient_ = None
        self.local_gradients_ = None
        self.h0_ = None
        self.weights_ = None
        self.J_ = None
        self.x_j_ = None
        self.x_j_work_ = None
        self.directions_ = None
        self.x_mean_ = None
        self.x_scale_ = None
        self.trace_ = None
        self.runtime_ = None
        self.diagnostics_ = None
        self.n_features_in_ = None
        self.n_samples_fit_ = None

    def _algorithm_params(self):
        return {
            "n_J": self.n_J,
            "n_min": self.n_min,
            "seed": self.seed,
            "center_sigma": self.center_sigma,
            "kernel": self.kernel,
            "h0": self.h0,
            "ridge": self.ridge,
            "standardize": self.standardize,
            "n_directions": self.n_directions,
            "return_state": self.return_state,
            "trace_enabled": self.trace_enabled,
            "trace_store_arrays": self.trace_store_arrays,
            "make_plots": self.make_plots,
            "plot_dir": self.plot_dir,
            "show_progress": self.show_progress,
            "log_runtime": self.log_runtime,
            "runtime_log_path": self.runtime_log_path,
            "use_rich": self.use_rich,
        }

    def _data_params(self):
        return self.data_config.to_kwargs()

    def get_params(self):
        """
        Возвращает текущие параметры объекта.
        """
        params = self._algorithm_params()
        params.update(self._data_params())
        return params

    def set_params(self, **params):
        """
        Меняет параметры объекта и сбрасывает результат предыдущего fit.
        """
        unknown = [name for name in params if not hasattr(self, name)]

        if unknown:
            raise ValueError(f"Неизвестные параметры: {unknown}")

        for name, value in params.items():
            setattr(self, name, value)

        if "data_config" in params:
            if not isinstance(self.data_config, ADPDataConfig):
                raise TypeError("data_config должен быть экземпляром ADPDataConfig")
            self._sync_data_attrs_from_config()
        elif any(name in ADPDataConfig.__dataclass_fields__ for name in params):
            self._sync_data_config_from_attrs()

        self._reset_fit_state()
        return self

    def GenerateX(self, n, d, seed=None, **overrides):
        """
        Метод-обертка над генерацией матрицы X.
        """
        params = self._data_params()
        params.update(overrides)
        if seed is None:
            seed = self.seed

        return _GenerateX(
            n=n,
            d=d,
            data_type=params["data_type"],
            seed=seed,
            dtype=params["dtype"],
            sigma_x=params["sigma_x"],
            corr=params["corr"],
            low=params["low"],
            high=params["high"],
        )

    def FunctionValue(self, index, function=None, **overrides):
        """
        Метод-обертка над одномерной функцией отклика.
        """
        params = self._data_params()
        params.update(overrides)
        if function is None:
            function = params["function"]

        return _FunctionValue(
            index,
            function=function,
            linear_weight=params["linear_weight"],
        )

    def GenerateNoise(self, n, seed=None, **overrides):
        """
        Метод-обертка над генерацией шума.
        """
        params = self._data_params()
        params.update(overrides)
        if seed is None:
            seed = self.seed

        return _GenerateNoise(
            n=n,
            noise_std=params["noise_std"],
            noise_type=params["noise_type"],
            seed=seed,
            dtype=params["dtype"],
        )

    def MakeData(self, n, d, beta=None, f=None, return_info=False, **overrides):
        """
        Генерирует данные single-index модели с параметрами класса.
        """
        params = self._data_params()
        params.update(overrides)
        seed = params.pop("seed", self.seed)

        return _MakeData(
            n=n,
            d=d,
            beta=beta,
            f=f,
            seed=seed,
            return_info=return_info,
            **params,
        )

    def make_data(self, *args, **kwargs):
        """
        Pythonic alias для MakeData.
        """
        return self.MakeData(*args, **kwargs)

    def NormVector(self, x):
        return _NormVector(x)

    def NormalizeRows(self, matrix):
        return _NormalizeRows(matrix)

    def ChooseJ(self, *args, **kwargs):
        return _ChooseJ(*args, **kwargs)

    def ComputeWeight(self, *args, **kwargs):
        return _ComputeWeight(*args, **kwargs)

    def ComputeWeigth(self, *args, **kwargs):
        return _ComputeWeigth(*args, **kwargs)

    def ChooseH0(self, *args, **kwargs):
        return _ChooseH0(*args, **kwargs)

    def GenerateDirection(self, *args, **kwargs):
        return _GenerateDirection(*args, **kwargs)

    def GenerateDirectionsForCenters(self, *args, **kwargs):
        return _GenerateDirectionsForCenters(*args, **kwargs)

    def PrepareADPInitialState(self, *args, **kwargs):
        return _PrepareADPInitialState(*args, **kwargs)

    def CalculateRho(self, *args, **kwargs):
        return _CalculateRho(*args, **kwargs)

    def StandardizeFeatures(self, *args, **kwargs):
        return _StandardizeFeatures(*args, **kwargs)

    def LocalLinearGradient(self, *args, **kwargs):
        return _LocalLinearGradient(*args, **kwargs)

    def EstimateLocalGradients(self, *args, **kwargs):
        return _EstimateLocalGradients(*args, **kwargs)

    def CosineSimilarity(self, *args, **kwargs):
        return _CosineSimilarity(*args, **kwargs)

    def _make_trace_and_runtime(self, trace=None, runtime_monitor=None, params=None):
        if params is None:
            params = self._algorithm_params()

        if trace is None and params.get("trace_enabled", False):
            trace = CreateTrace(
                enabled=True,
                store_arrays=params.get("trace_store_arrays", False),
            )

        if runtime_monitor is None and (
            params.get("show_progress", False)
            or params.get("log_runtime", False)
            or params.get("use_rich", False)
        ):
            runtime_monitor = CreateRuntimeMonitor(
                enabled=True,
                use_tqdm=params.get("show_progress", False),
                use_rich=params.get("use_rich", False),
                log_runtime=params.get("log_runtime", False),
                log_path=params.get("runtime_log_path"),
            )

        return trace, runtime_monitor

    def AverageDerivativeProcedure(self, X, Y, trace=None, runtime_monitor=None, **overrides):
        """
        Метод-оркестратор полного ADP-пайплайна.
        """
        params = self._algorithm_params()
        params.update(overrides)
        trace, runtime_monitor = self._make_trace_and_runtime(
            trace=trace,
            runtime_monitor=runtime_monitor,
            params=params,
        )

        return _AverageDerivativeProcedure(
            X,
            Y,
            trace=trace,
            runtime_monitor=runtime_monitor,
            **params,
        )

    def RunADP(self, *args, **kwargs):
        return self.AverageDerivativeProcedure(*args, **kwargs)

    def FitADP(self, *args, **kwargs):
        return self.AverageDerivativeProcedure(*args, **kwargs)

    def AlteringOptimisation(self, *args, **kwargs):
        return self.AverageDerivativeProcedure(*args, **kwargs)

    def fit(self, X, Y, trace=None, runtime_monitor=None, **overrides):
        """
        Запускает полный ADP-пайплайн и сохраняет результат в атрибутах объекта.
        """
        X = _as_feature_matrix(X)
        Y = _as_response_vector(Y, X.shape[0])

        result = self.AverageDerivativeProcedure(
            X,
            Y,
            trace=trace,
            runtime_monitor=runtime_monitor,
            **overrides,
        )

        self.result_ = result
        self.beta_ = result["beta"]
        self.direction_ = result["direction"]
        self.average_gradient_ = result["average_gradient"]
        self.local_gradients_ = result["local_gradients"]
        self.h0_ = result["h0"]
        self.weights_ = result["weights"]
        self.J_ = result["J"]
        self.x_j_ = result["x_j"]
        self.x_j_work_ = result["x_j_work"]
        self.directions_ = result["directions"]
        self.x_mean_ = result["x_mean"]
        self.x_scale_ = result["x_scale"]
        self.trace_ = result.get("trace")
        self.runtime_ = result.get("runtime")
        self.diagnostics_ = result.get("diagnostics")
        self.n_features_in_ = X.shape[1]
        self.n_samples_fit_ = X.shape[0]

        return self

    def _check_is_fitted(self):
        if self.beta_ is None:
            raise ValueError("Сначала вызови fit(X, Y)")

    def predict_index(self, X, centered=False):
        """
        Возвращает одномерный индекс beta_hat^T X.

        centered=True вычитает среднее обучающей выборки перед проекцией.
        """
        self._check_is_fitted()
        X = _as_feature_matrix(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X должен иметь то же число признаков, что и при fit")

        if centered:
            X = X - self.x_mean_

        return X @ self.beta_

    def transform(self, X, centered=False):
        """
        Возвращает проекцию X на найденное направление как матрицу n x 1.
        """
        return self.predict_index(X, centered=centered).reshape(-1, 1)

    def fit_transform(self, X, Y, **fit_params):
        """
        Обучает модель и возвращает одномерную проекцию обучающей выборки.
        """
        self.fit(X, Y, **fit_params)
        return self.transform(X)

    def score_direction(self, true_direction, absolute=True):
        """
        Считает cos(beta_hat, true_direction).
        """
        self._check_is_fitted()
        return self.CosineSimilarity(self.beta_, true_direction, absolute=absolute)

    def save_diagnostics(self, output_dir=None):
        """
        Сохраняет графики и CSV trace для последнего результата fit.
        """
        self._check_is_fitted()

        if output_dir is None:
            output_dir = self.plot_dir

        self.diagnostics_ = SaveADPDiagnostics(self.result_, output_dir=output_dir)
        return self.diagnostics_

    def summary(self):
        """
        Возвращает краткую сводку по последнему запуску.
        """
        self._check_is_fitted()

        return {
            "n_samples": self.n_samples_fit_,
            "n_features": self.n_features_in_,
            "n_centers": None if self.J_ is None else int(self.J_.shape[0]),
            "h0": float(self.h0_),
            "beta_norm": float(np.linalg.norm(self.beta_)),
            "trace_steps": 0 if self.trace_ is None else len(self.trace_.get("steps", [])),
            "runtime": None if self.runtime_ is None else self.runtime_.get("summary"),
        }


ADPSingleIndex = ADP_single_index
