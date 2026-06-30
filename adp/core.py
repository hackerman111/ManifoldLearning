from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from scipy import linalg

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm является удобством, а не ядром.
    tqdm = None


KernelName = Literal["epanechnikov", "quartic", "gaussian"]
VariantName = Literal["new", "old"]
BackendName = Literal["numpy", "torch", "cupy", "auto"]


@dataclass(slots=True)
class ADPConfig:
    """Настройки ADP.

    Сейчас реализован одномерный EDR-вектор beta. Поле target_dim оставлено,
    чтобы позже расширить solver до multi-index варианта из manifold_new.tex.
    """

    n_centers: int | None = None
    n_directions: int = 10
    target_dim: int = 1
    min_neighbors: float = 10.0
    lambda_penalty: float | None = None
    outer_steps: int = 4
    inner_steps: int = 20
    tol: float = 1e-6
    bandwidth_decay: float = math.sqrt(2.0)
    anisotropy_min: float | None = None
    kernel: KernelName = "epanechnikov"
    backend: BackendName = "numpy"
    device: str | None = None
    dtype: str = "float64"
    center_noise_scale: float = 1.0
    renew_directions: bool = True
    chunk_size: int = 64
    ridge: float = 1e-10
    show_progress: bool = True
    random_state: int | None = None
    use_neighbor_index: bool = True

    def resolved_lambda(self) -> float:
        return float(self.min_neighbors if self.lambda_penalty is None else self.lambda_penalty)


@dataclass(slots=True)
class ADPData:
    X: np.ndarray
    y: np.ndarray
    beta: np.ndarray
    centers: np.ndarray
    directions: np.ndarray | None
    noise: np.ndarray
    link_name: str


@dataclass(slots=True)
class LocalStatistics:
    variant: VariantName
    imav: np.ndarray
    centers: np.ndarray
    h: float
    weights_mean: float
    directions: np.ndarray | None = None
    S: np.ndarray | None = None
    U: np.ndarray | None = None
    N: np.ndarray | None = None
    VP: np.ndarray | None = None
    anisotropy: float | None = None
    b: float | None = None


@dataclass(slots=True)
class TrainingStep:
    outer: int
    inner: int
    objective: float
    beta_delta: float
    h: float
    anisotropy: float | None
    elapsed: float


@dataclass(slots=True)
class ADPResult:
    beta: np.ndarray
    intercepts: np.ndarray
    slopes: np.ndarray
    statistics: LocalStatistics
    history: list[TrainingStep]
    progress: list[dict[str, Any]]
    objective: float
    backend: str
    timings: dict[str, float] = field(default_factory=dict)
    diagnostic_plots: dict[str, Path] = field(default_factory=dict)

    @property
    def projector(self) -> np.ndarray:
        """Ортогональный проектор на найденное одномерное EDR-направление."""

        beta = _unit_vector(self.beta)
        return np.outer(beta, beta)

    @property
    def basis(self) -> np.ndarray:
        """EDR-базис через eig/SVD, чтобы API совпадал с будущим multi-index."""

        values, vectors = linalg.eigh(self.projector)
        return vectors[:, np.argsort(values)[::-1][:1]].T


class _ArrayBackend:
    """Тонкая обертка над numpy/torch/cupy для тяжёлых локальных сумм.

    Для CPU путь полностью рабочий на numpy. Torch/CuPy подключаются лениво и
    используются в местах, где строятся веса и локальные суммы по всем точкам.
    """

    def __init__(self, name: BackendName, dtype: str = "float64", device: str | None = None):
        self.requested_name = name
        self.name = self._resolve_name(name)
        self.dtype_name = dtype
        self.device = device
        self.module: Any = None
        self.dtype: Any = None

        if self.name == "torch":
            import torch

            self.module = torch
            self.dtype = torch.float64 if dtype == "float64" else torch.float32
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self.name == "cupy":
            import cupy as cp

            self.module = cp
            self.dtype = cp.float64 if dtype == "float64" else cp.float32
        else:
            self.module = np
            self.dtype = np.float64 if dtype == "float64" else np.float32

    @staticmethod
    def _resolve_name(name: BackendName) -> str:
        if name == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    return "torch"
            except Exception:
                pass
            return "numpy"
        if name == "torch":
            try:
                import torch  # noqa: F401
            except Exception as exc:
                raise ImportError("backend='torch' требует установленный torch") from exc
            return "torch"
        if name == "cupy":
            try:
                import cupy  # noqa: F401
            except Exception as exc:
                raise ImportError("backend='cupy' требует установленный cupy") from exc
            return "cupy"
        return "numpy"

    def asarray(self, value: np.ndarray) -> Any:
        if self.name == "torch":
            return self.module.as_tensor(value, dtype=self.dtype, device=self.device)
        if self.name == "cupy":
            return self.module.asarray(value, dtype=self.dtype)
        return np.asarray(value, dtype=self.dtype)

    def to_numpy(self, value: Any) -> np.ndarray:
        if self.name == "torch":
            return value.detach().cpu().numpy()
        if self.name == "cupy":
            return self.module.asnumpy(value)
        return np.asarray(value)

    def kernel(self, q: Any, name: KernelName) -> Any:
        if self.name == "torch":
            torch = self.module
            if name == "gaussian":
                return torch.exp(-0.5 * q)
            if name == "quartic":
                return torch.clamp(1.0 - q * q, min=0.0)
            return torch.clamp(1.0 - q, min=0.0)

        xp = self.module
        if name == "gaussian":
            return xp.exp(-0.5 * q)
        if name == "quartic":
            return xp.maximum(1.0 - q * q, 0.0)
        return xp.maximum(1.0 - q, 0.0)

    def random_projection_sums(
        self,
        diff: np.ndarray,
        y: np.ndarray,
        directions: np.ndarray,
        q: np.ndarray,
        kernel: KernelName,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Суммы Ima, S, U для new-варианта со случайными направлениями."""

        if self.name == "torch":
            torch = self.module
            tdiff = self.asarray(diff)
            ty = self.asarray(y)
            tdirs = self.asarray(directions)
            tq = self.asarray(q)
            w = self.kernel(tq, kernel)
            proj = torch.einsum("cnd,cpd->cnp", tdiff, tdirs)
            imav = torch.einsum("n,cn,cnp->cp", ty, w, proj)
            s_vec = torch.einsum("cn,cnp->cp", w, proj)
            u_mat = torch.einsum("cnd,cn,cnp->cpd", tdiff, w, proj)
            return (
                self.to_numpy(imav),
                self.to_numpy(s_vec),
                self.to_numpy(u_mat),
                float(self.to_numpy(w.sum(dim=1)).mean()),
            )

        xp = self.module
        xdiff = self.asarray(diff)
        xy = self.asarray(y)
        xdirs = self.asarray(directions)
        xq = self.asarray(q)
        w = self.kernel(xq, kernel)
        proj = xp.einsum("cnd,cpd->cnp", xdiff, xdirs)
        imav = xp.einsum("n,cn,cnp->cp", xy, w, proj)
        s_vec = xp.einsum("cn,cnp->cp", w, proj)
        u_mat = xp.einsum("cnd,cn,cnp->cpd", xdiff, w, proj)
        return (
            self.to_numpy(imav),
            self.to_numpy(s_vec),
            self.to_numpy(u_mat),
            float(self.to_numpy(w.sum(axis=1)).mean()),
        )

    def full_moment_sums(
        self,
        diff: np.ndarray,
        y: np.ndarray,
        q: np.ndarray,
        kernel: KernelName,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Суммы Ima, N, S, VP для old-варианта без случайных проекций."""

        if self.name == "torch":
            torch = self.module
            tdiff = self.asarray(diff)
            ty = self.asarray(y)
            tq = self.asarray(q)
            w = self.kernel(tq, kernel)
            im0 = torch.einsum("n,cn->c", ty, w)
            im1 = torch.einsum("n,cn,cnd->cd", ty, w, tdiff)
            n_vec = w.sum(dim=1)
            s_vec = torch.einsum("cn,cnd->cd", w, tdiff)
            vp = torch.einsum("cn,cnd,cne->cde", w, tdiff, tdiff)
            imav = torch.cat([im0[:, None], im1], dim=1)
            return (
                self.to_numpy(imav),
                self.to_numpy(n_vec),
                self.to_numpy(s_vec),
                self.to_numpy(vp),
                float(self.to_numpy(n_vec).mean()),
            )

        xp = self.module
        xdiff = self.asarray(diff)
        xy = self.asarray(y)
        xq = self.asarray(q)
        w = self.kernel(xq, kernel)
        im0 = xp.einsum("n,cn->c", xy, w)
        im1 = xp.einsum("n,cn,cnd->cd", xy, w, xdiff)
        n_vec = w.sum(axis=1)
        s_vec = xp.einsum("cn,cnd->cd", w, xdiff)
        vp = xp.einsum("cn,cnd,cne->cde", w, xdiff, xdiff)
        imav = xp.concatenate([im0[:, None], im1], axis=1)
        return (
            self.to_numpy(imav),
            self.to_numpy(n_vec),
            self.to_numpy(s_vec),
            self.to_numpy(vp),
            float(self.to_numpy(n_vec).mean()),
        )


class _NeighborIndex:
    """Опциональный индекс соседей.

    Локальные суммы пока считаются по всем точкам, потому что компактная
    поддержка ядра и anisotropic tensor усложняют radius-запросы. Индекс нужен
    для быстрых оценок масштаба и будущей замены полного прохода на окрестности.
    """

    def __init__(self, enabled: bool = True):
        self.backend = "none"
        self.index: Any = None
        self.enabled = enabled

    def fit(self, X: np.ndarray) -> "_NeighborIndex":
        if not self.enabled:
            return self
        try:
            import faiss  # type: ignore

            index = faiss.IndexFlatL2(X.shape[1])
            index.add(np.asarray(X, dtype=np.float32))
            self.index = index
            self.backend = "faiss"
            return self
        except Exception:
            pass
        try:
            from sklearn.neighbors import NearestNeighbors

            index = NearestNeighbors(algorithm="auto")
            index.fit(X)
            self.index = index
            self.backend = "sklearn"
        except Exception:
            self.index = None
            self.backend = "none"
        return self

    def kth_distances(self, centers: np.ndarray, k: int) -> np.ndarray | None:
        if self.index is None or k <= 0:
            return None
        if self.backend == "faiss":
            distances, _ = self.index.search(np.asarray(centers, dtype=np.float32), k)
            return np.sqrt(np.maximum(distances[:, -1], 0.0))
        distances, _ = self.index.kneighbors(centers, n_neighbors=k)
        return distances[:, -1]


class ADP:
    """Фабрика Average Derivative Procedure."""

    variant: VariantName

    @classmethod
    def create(
        cls,
        variant: VariantName = "new",
        config: ADPConfig | None = None,
        **config_kwargs: Any,
    ) -> "ADPBase":
        if config is None:
            config = ADPConfig(**config_kwargs)
        elif config_kwargs:
            config = replace(config, **config_kwargs)

        if variant == "new":
            return RandomProjectionADP(config)
        if variant == "old":
            return FullMomentADP(config)
        raise ValueError("variant должен быть 'new' или 'old'")


class ADPBase(ADP):
    variant: VariantName = "new"

    def __init__(self, config: ADPConfig | None = None):
        self.config = config or ADPConfig()
        if self.config.target_dim != 1:
            raise NotImplementedError("Сейчас реализован target_dim=1; multi-index оставлен следующим слоем.")
        self.rng = np.random.default_rng(self.config.random_state)
        self.backend = _ArrayBackend(self.config.backend, self.config.dtype, self.config.device)
        self.result_: ADPResult | None = None
        self.data_: tuple[np.ndarray, np.ndarray] | None = None
        self.centers_: np.ndarray | None = None
        self.directions_: np.ndarray | None = None
        self.neighbor_index_: _NeighborIndex | None = None
        self.diagnostic_plots_: dict[str, Path] = {}

    def generate_data(
        self,
        n: int,
        d: int,
        *,
        n_centers: int | None = None,
        n_directions: int | None = None,
        beta: np.ndarray | None = None,
        noise: float = 0.1,
        sigma_x: float = 1.0,
        corr: float = 0.5,
        link: str | Callable[[np.ndarray], np.ndarray] = "quadratic",
    ) -> ADPData:
        """Сгенерировать single-index данные из TeX-описания."""

        if n <= 0 or d <= 0:
            raise ValueError("n и d должны быть положительными")
        if not 0 <= corr < 1:
            raise ValueError("corr должен лежать в [0, 1)")

        beta_vec = _unit_vector(beta if beta is not None else self.rng.normal(size=d))
        shared = self.rng.normal(size=d)
        individual = self.rng.normal(size=(n, d))
        X = sigma_x * (corr * shared[None, :] + (1.0 - corr) * individual)
        eps = self.rng.normal(scale=noise, size=n)
        link_fn, link_name = _link_function(link)
        y = link_fn(X @ beta_vec) + eps

        j_count = int(n_centers or self.config.n_centers or n)
        j_count = min(max(j_count, 1), n)
        selected = self.rng.choice(n, size=j_count, replace=False)
        centers = X[selected] + self.config.center_noise_scale * sigma_x * self.rng.normal(size=(j_count, d))

        directions = None
        if self.variant == "new":
            p_count = int(n_directions or self.config.n_directions)
            directions = self._sample_directions(j_count, p_count, d)

        return ADPData(X=X, y=y, beta=beta_vec, centers=centers, directions=directions, noise=eps, link_name=link_name)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        centers: np.ndarray | None = None,
        beta0: np.ndarray | None = None,
        directions: np.ndarray | None = None,
    ) -> ADPResult:
        X_arr = _as_2d_float(X, "X")
        y_arr = _as_1d_float(y, "y")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X и y имеют разные размеры по n")

        started = time.perf_counter()
        n, d = X_arr.shape
        centers_arr = _as_2d_float(centers, "centers") if centers is not None else self._choose_centers(X_arr)
        if centers_arr.shape[1] != d:
            raise ValueError("centers должны иметь ту же размерность d, что и X")

        beta_prev = _unit_vector(beta0 if beta0 is not None else self._initial_beta(X_arr, y_arr))
        lambda_penalty = self.config.resolved_lambda()
        neighbor_index = _NeighborIndex(self.config.use_neighbor_index).fit(X_arr)
        self.neighbor_index_ = neighbor_index

        h = self._select_isotropic_bandwidth(X_arr, centers_arr, neighbor_index)
        b_old = h
        directions_arr = self._prepare_directions(centers_arr, d, directions)

        history: list[TrainingStep] = []
        progress: list[dict[str, Any]] = []
        timings: dict[str, float] = {}
        intercepts = np.zeros(centers_arr.shape[0])
        slopes = np.ones(centers_arr.shape[0])
        statistics: LocalStatistics | None = None

        outer_iter = range(max(1, self.config.outer_steps))
        progress_bar = None
        if self.config.show_progress and tqdm is not None:
            progress_bar = tqdm(
                outer_iter,
                desc=f"ADP-{self.variant} {self.backend.name}",
                leave=False,
            )
            outer_iter = progress_bar

        try:
            for outer in outer_iter:
                step_started = time.perf_counter()
                if outer == 0:
                    anisotropy = None
                    b_value = None
                elif self.variant == "new":
                    h = max(h / self.config.bandwidth_decay, np.finfo(float).eps)
                    anisotropy = self._select_new_anisotropy(X_arr, centers_arr, h, beta_prev)
                    if self.config.renew_directions:
                        directions_arr = self._sample_directions(
                            centers_arr.shape[0],
                            self.config.n_directions,
                            d,
                            beta=beta_prev,
                            anisotropy=anisotropy,
                        )
                    b_value = None
                else:
                    b_old = max(b_old / self.config.bandwidth_decay, np.finfo(float).eps)
                    h = self._select_old_bandwidth(X_arr, centers_arr, beta_prev, b_old)
                    anisotropy = None
                    b_value = b_old

                stats_started = time.perf_counter()
                statistics = self._compute_statistics(X_arr, y_arr, centers_arr, h, beta_prev, directions_arr, anisotropy, b_value)
                timings["statistics"] = timings.get("statistics", 0.0) + time.perf_counter() - stats_started

                solve_started = time.perf_counter()
                beta_new, intercepts, slopes, inner_history = self._alternating_solve(
                    statistics,
                    beta_prev,
                    lambda_penalty,
                    outer,
                    step_started,
                )
                timings["solve"] = timings.get("solve", 0.0) + time.perf_counter() - solve_started
                history.extend(inner_history)
                beta_prev = beta_new

                if inner_history:
                    progress_record = self._progress_record(
                        stats=statistics,
                        step=inner_history[-1],
                        outer_index=outer,
                        outer_total=max(1, self.config.outer_steps),
                        inner_count=len(inner_history),
                        started=started,
                    )
                    progress.append(progress_record)
                    if progress_bar is not None and hasattr(progress_bar, "set_postfix"):
                        progress_bar.set_postfix(_format_progress_postfix(progress_record), refresh=True)

                if self.config.anisotropy_min is not None and self.variant == "new" and anisotropy is not None:
                    if anisotropy <= self.config.anisotropy_min:
                        break
        finally:
            if progress_bar is not None and hasattr(progress_bar, "close"):
                progress_bar.close()

        if statistics is None:
            raise RuntimeError("fit не смог вычислить локальные статистики")

        objective = history[-1].objective if history else self._objective(statistics, beta_prev, intercepts, slopes, beta_prev, lambda_penalty)
        if progress:
            progress[-1]["objective"] = float(objective)

        timings["total"] = time.perf_counter() - started
        result = ADPResult(
            beta=beta_prev,
            intercepts=intercepts,
            slopes=slopes,
            statistics=statistics,
            history=history,
            progress=progress,
            objective=float(objective),
            backend=self.backend.name,
            timings=timings,
        )
        self.result_ = result
        self.data_ = (X_arr, y_arr)
        self.centers_ = centers_arr
        self.directions_ = directions_arr
        self.diagnostic_plots_ = {}
        return result

    def score(self, beta_true: np.ndarray) -> dict[str, float]:
        """Метрики восстановления направления beta.

        Знак beta не идентифицируется, поэтому главная метрика - |cos|.
        """

        result = self._require_result()
        expected = _unit_vector(beta_true)
        estimated = _unit_vector(result.beta)
        cosine = float(np.clip(expected @ estimated, -1.0, 1.0))
        cosine_abs = abs(cosine)
        signed_l2 = min(np.linalg.norm(estimated - expected), np.linalg.norm(estimated + expected))
        return {
            "cosine": cosine,
            "cosine_abs": cosine_abs,
            "angle_deg": float(np.degrees(np.arccos(np.clip(cosine_abs, -1.0, 1.0)))),
            "signed_l2": float(signed_l2),
        }

    def summary(self) -> dict[str, Any]:
        result = self._require_result()
        return {
            "variant": self.variant,
            "backend": result.backend,
            "n_centers": int(result.statistics.centers.shape[0]),
            "n_directions": None if result.statistics.directions is None else int(result.statistics.directions.shape[1]),
            "h": float(result.statistics.h),
            "weights_mean": float(result.statistics.weights_mean),
            "objective": float(result.objective),
            "progress_last": dict(result.progress[-1]) if result.progress else None,
            "diagnostic_plots": {name: str(path) for name, path in self.diagnostic_plots_.items()},
            "timings": dict(result.timings),
        }

    def plot_history(
        self,
        ax: Any = None,
        *,
        save_path: str | Path | None = None,
        dpi: int = 150,
        close: bool = False,
    ) -> Any:
        """Нарисовать историю objective и при необходимости сохранить в файл."""

        result = self._require_result()
        _ensure_matplotlib_config_dir()
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        ax.plot([step.objective for step in result.history], marker="o")
        ax.set_xlabel("iteration")
        ax.set_ylabel("objective")
        ax.set_title(f"ADP {self.variant}")
        if save_path is not None:
            saved_path = _save_figure(ax.figure, save_path, dpi=dpi, close=close)
            self._remember_diagnostic_plot("history", saved_path)
        return ax

    def save_diagnostics(
        self,
        output_dir: str | Path,
        *,
        beta_true: np.ndarray | None = None,
        prefix: str = "adp",
        dpi: int = 150,
        close: bool = True,
    ) -> dict[str, Path]:
        """Автоматически построить и сохранить диагностические графики.

        Возвращает словарь ``имя_графика -> путь``. Метод не меняет результат
        обучения, только сохраняет удобные картинки для отчёта или отладки.
        """

        result = self._require_result()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        _ensure_matplotlib_config_dir()
        import matplotlib.pyplot as plt

        saved: dict[str, Path] = {}
        iterations = np.arange(len(result.history))

        fig, ax = plt.subplots()
        ax.plot(iterations, [step.objective for step in result.history], marker="o")
        ax.set_xlabel("iteration")
        ax.set_ylabel("objective")
        ax.set_title("ADP objective")
        saved["objective"] = _save_figure(fig, output_path / f"{prefix}_objective.png", dpi=dpi, close=close)

        fig, ax = plt.subplots()
        ax.plot(iterations, [step.beta_delta for step in result.history], marker="o")
        ax.set_xlabel("iteration")
        ax.set_ylabel("||beta_k - beta_{k-1}||")
        ax.set_title("ADP beta update")
        ax.set_yscale("log")
        saved["delta"] = _save_figure(fig, output_path / f"{prefix}_delta.png", dpi=dpi, close=close)

        outer = np.arange(1, len(result.progress) + 1)
        fig, ax = plt.subplots()
        ax.plot(outer, [record["h"] for record in result.progress], marker="o", label="h")
        if any("rho" in record for record in result.progress):
            ax.plot(outer, [record.get("rho", np.nan) for record in result.progress], marker="s", label="rho")
        if any("b" in record for record in result.progress):
            ax.plot(outer, [record.get("b", np.nan) for record in result.progress], marker="s", label="b")
        ax.set_xlabel("outer step")
        ax.set_ylabel("scale")
        ax.set_title("ADP localization scales")
        ax.legend()
        saved["bandwidth"] = _save_figure(fig, output_path / f"{prefix}_bandwidth.png", dpi=dpi, close=close)

        fig, ax = plt.subplots()
        ax.plot(outer, [record["weights"] for record in result.progress], marker="o")
        ax.axhline(self.config.min_neighbors, color="tab:red", linestyle="--", linewidth=1, label="min_neighbors")
        ax.set_xlabel("outer step")
        ax.set_ylabel("average local weight")
        ax.set_title("ADP local mass")
        ax.legend()
        saved["weights"] = _save_figure(fig, output_path / f"{prefix}_weights.png", dpi=dpi, close=close)

        if beta_true is not None:
            expected = _unit_vector(beta_true)
            estimated = _unit_vector(result.beta)
            if expected @ estimated < 0:
                estimated = -estimated
            x = np.arange(estimated.size)
            fig, ax = plt.subplots()
            width = 0.4
            ax.bar(x - width / 2, expected, width=width, label="true")
            ax.bar(x + width / 2, estimated, width=width, label="estimated")
            ax.set_xlabel("component")
            ax.set_ylabel("value")
            ax.set_title("ADP beta comparison")
            ax.legend()
            saved["beta_compare"] = _save_figure(fig, output_path / f"{prefix}_beta_compare.png", dpi=dpi, close=close)

        for name, path in saved.items():
            self._remember_diagnostic_plot(name, path)
        return saved

    def _remember_diagnostic_plot(self, name: str, path: Path) -> None:
        self.diagnostic_plots_[name] = path
        if self.result_ is not None:
            self.result_.diagnostic_plots[name] = path

    def _progress_record(
        self,
        *,
        stats: LocalStatistics,
        step: TrainingStep,
        outer_index: int,
        outer_total: int,
        inner_count: int,
        started: float,
    ) -> dict[str, Any]:
        """Единый снимок состояния для tqdm и программного анализа.

        Здесь храним сырые числа, а форматирование терминала делаем отдельно.
        Так проще строить графики/логи и не зависеть от текстового вывода.
        """

        record: dict[str, Any] = {
            "variant": self.variant,
            "backend": self.backend.name,
            "outer": outer_index + 1,
            "outer_total": outer_total,
            "inner": inner_count,
            "h": float(stats.h),
            "weights": float(stats.weights_mean),
            "objective": float(step.objective),
            "delta": float(step.beta_delta),
            "elapsed": float(time.perf_counter() - started),
        }
        if stats.anisotropy is not None:
            record["rho"] = float(stats.anisotropy)
        if stats.b is not None:
            record["b"] = float(stats.b)
        if stats.directions is not None:
            record["directions"] = int(stats.directions.shape[1])
        return record

    def _compute_statistics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        centers: np.ndarray,
        h: float,
        beta: np.ndarray,
        directions: np.ndarray | None,
        anisotropy: float | None,
        b_value: float | None,
    ) -> LocalStatistics:
        raise NotImplementedError

    def _alternating_solve(
        self,
        stats: LocalStatistics,
        beta_start: np.ndarray,
        lambda_penalty: float,
        outer: int,
        outer_started: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[TrainingStep]]:
        beta = _unit_vector(beta_start)
        prior = beta.copy()
        history: list[TrainingStep] = []
        intercepts = np.zeros(stats.centers.shape[0])
        slopes = np.ones(stats.centers.shape[0])
        last_objective = math.inf

        for inner in range(max(1, self.config.inner_steps)):
            old_beta = beta.copy()
            intercepts, slopes = self._solve_local_coefficients(stats, beta)
            beta = self._solve_beta(stats, intercepts, slopes, prior, lambda_penalty)

            norm = np.linalg.norm(beta)
            if norm > 0:
                beta = beta / norm
                slopes = slopes * norm

            objective = self._objective(stats, beta, intercepts, slopes, prior, lambda_penalty)
            beta_delta = float(np.linalg.norm(beta - old_beta))
            history.append(
                TrainingStep(
                    outer=outer,
                    inner=inner,
                    objective=float(objective),
                    beta_delta=beta_delta,
                    h=float(stats.h),
                    anisotropy=stats.anisotropy,
                    elapsed=time.perf_counter() - outer_started,
                )
            )
            if beta_delta < self.config.tol or abs(last_objective - objective) < self.config.tol:
                break
            last_objective = objective

        return _unit_vector(beta), intercepts, slopes, history

    def _solve_local_coefficients(self, stats: LocalStatistics, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _solve_beta(
        self,
        stats: LocalStatistics,
        intercepts: np.ndarray,
        slopes: np.ndarray,
        prior: np.ndarray,
        lambda_penalty: float,
    ) -> np.ndarray:
        raise NotImplementedError

    def _objective(
        self,
        stats: LocalStatistics,
        beta: np.ndarray,
        intercepts: np.ndarray,
        slopes: np.ndarray,
        prior: np.ndarray,
        lambda_penalty: float,
    ) -> float:
        raise NotImplementedError

    def _choose_centers(self, X: np.ndarray) -> np.ndarray:
        n, d = X.shape
        j_count = int(self.config.n_centers or n)
        j_count = min(max(j_count, 1), n)
        selected = self.rng.choice(n, size=j_count, replace=False)
        scale = float(np.std(X)) if np.std(X) > 0 else 1.0
        return X[selected] + self.config.center_noise_scale * scale * self.rng.normal(size=(j_count, d))

    def _prepare_directions(self, centers: np.ndarray, d: int, directions: np.ndarray | None) -> np.ndarray | None:
        if self.variant == "old":
            return None
        if directions is None:
            return self._sample_directions(centers.shape[0], self.config.n_directions, d)
        directions_arr = np.asarray(directions, dtype=float)
        expected = (centers.shape[0], self.config.n_directions, d)
        if directions_arr.shape != expected:
            raise ValueError(f"directions должны иметь форму {expected}, получено {directions_arr.shape}")
        return _normalize_rows(directions_arr)

    def _sample_directions(
        self,
        n_centers: int,
        n_directions: int,
        d: int,
        *,
        beta: np.ndarray | None = None,
        anisotropy: float | None = None,
    ) -> np.ndarray:
        z = self.rng.normal(size=(n_centers, n_directions, d))
        if beta is not None and anisotropy is not None:
            # Семплирование из N(0, rho^2 I + beta beta^T); общий множитель h не
            # нужен, потому что дальше направления нормируются.
            beta_unit = _unit_vector(beta)
            z = float(anisotropy) * z + self.rng.normal(size=(n_centers, n_directions, 1)) * beta_unit
        return _normalize_rows(z)

    def _initial_beta(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_centered = X - X.mean(axis=0, keepdims=True)
        y_centered = y - y.mean()
        try:
            beta, *_ = linalg.lstsq(x_centered, y_centered)
        except Exception:
            beta = x_centered.T @ y_centered
        if np.linalg.norm(beta) < np.finfo(float).eps:
            beta = np.zeros(X.shape[1])
            beta[0] = 1.0
        return _unit_vector(beta)

    def _select_isotropic_bandwidth(self, X: np.ndarray, centers: np.ndarray, index: _NeighborIndex | None = None) -> float:
        diff_norm2 = _pairwise_norm2(X, centers)
        high_hint = None
        if index is not None:
            k = min(max(1, int(math.ceil(self.config.min_neighbors))), X.shape[0])
            kth = index.kth_distances(centers, k)
            if kth is not None and np.all(np.isfinite(kth)):
                high_hint = float(np.nanmedian(kth))
        return self._binary_search_scale(lambda h: _average_kernel_weight(diff_norm2 / (h * h), self.config.kernel), high_hint)

    def _select_new_anisotropy(self, X: np.ndarray, centers: np.ndarray, h: float, beta: np.ndarray) -> float:
        norm2 = _pairwise_norm2(X, centers)
        proj2 = _pairwise_projection2(X, centers, beta)

        def avg_for(rho: float) -> float:
            q = (rho * rho * norm2 + proj2) / (h * h)
            return _average_kernel_weight(q, self.config.kernel)

        if avg_for(1.0) >= self.config.min_neighbors:
            return 1.0
        if avg_for(0.0) < self.config.min_neighbors:
            return 0.0
        low, high = 0.0, 1.0
        # Берём максимально возможную rho: это самая локальная окрестность при
        # заданном h, которая всё ещё содержит достаточно точек.
        for _ in range(50):
            mid = (low + high) / 2.0
            if avg_for(mid) >= self.config.min_neighbors:
                low = mid
            else:
                high = mid
        return float(low)

    def _select_old_bandwidth(self, X: np.ndarray, centers: np.ndarray, beta: np.ndarray, b_value: float) -> float:
        norm2 = _pairwise_norm2(X, centers)
        proj2 = _pairwise_projection2(X, centers, beta)

        def avg_for(h: float) -> float:
            q = norm2 / (h * h) + proj2 / (b_value * b_value)
            return _average_kernel_weight(q, self.config.kernel)

        return self._binary_search_scale(avg_for, b_value)

    def _binary_search_scale(self, avg_fn: Callable[[float], float], high_hint: float | None = None) -> float:
        target = float(self.config.min_neighbors)
        low = np.finfo(float).eps
        high = max(float(high_hint or 1.0), low * 2.0)
        for _ in range(80):
            if avg_fn(high) >= target:
                break
            high *= 2.0
        for _ in range(70):
            mid = (low + high) / 2.0
            if avg_fn(mid) >= target:
                high = mid
            else:
                low = mid
        return float(high)

    def _require_result(self) -> ADPResult:
        if self.result_ is None:
            raise RuntimeError("Сначала вызовите fit(...)")
        return self.result_


class RandomProjectionADP(ADPBase):
    """ADP из manifold_new.tex: локальные суммы по случайным направлениям."""

    variant: VariantName = "new"

    def _compute_statistics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        centers: np.ndarray,
        h: float,
        beta: np.ndarray,
        directions: np.ndarray | None,
        anisotropy: float | None,
        b_value: float | None,
    ) -> LocalStatistics:
        if directions is None:
            raise ValueError("new-вариант требует directions")

        J, P, d = directions.shape
        imav = np.zeros((J, P))
        s_all = np.zeros((J, P))
        u_all = np.zeros((J, P, d))
        weight_means: list[float] = []
        rho = 1.0 if anisotropy is None else float(anisotropy)

        for start in range(0, J, self.config.chunk_size):
            stop = min(start + self.config.chunk_size, J)
            diff = X[None, :, :] - centers[start:stop, None, :]
            norm2 = np.einsum("cnd,cnd->cn", diff, diff)
            if anisotropy is None:
                q = norm2 / (h * h)
            else:
                proj2 = np.square(np.einsum("cnd,d->cn", diff, beta))
                q = (rho * rho * norm2 + proj2) / (h * h)
            chunk_imav, chunk_s, chunk_u, weight_mean = self.backend.random_projection_sums(
                diff,
                y,
                directions[start:stop],
                q,
                self.config.kernel,
            )
            imav[start:stop] = chunk_imav
            s_all[start:stop] = chunk_s
            u_all[start:stop] = chunk_u
            weight_means.append(weight_mean)

        return LocalStatistics(
            variant="new",
            imav=imav,
            centers=centers,
            h=h,
            weights_mean=float(np.mean(weight_means)),
            directions=directions,
            S=s_all,
            U=u_all,
            anisotropy=anisotropy,
        )

    def _solve_local_coefficients(self, stats: LocalStatistics, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if stats.S is None or stats.U is None:
            raise ValueError("Некорректные статистики new-варианта")
        J = stats.S.shape[0]
        intercepts = np.zeros(J)
        slopes = np.zeros(J)
        for j in range(J):
            col0 = stats.S[j]
            col1 = stats.U[j] @ beta
            design = np.column_stack([col0, col1])
            lhs = design.T @ design + self.config.ridge * np.eye(2)
            rhs = design.T @ stats.imav[j]
            intercepts[j], slopes[j] = _safe_solve(lhs, rhs)
        return intercepts, slopes

    def _solve_beta(
        self,
        stats: LocalStatistics,
        intercepts: np.ndarray,
        slopes: np.ndarray,
        prior: np.ndarray,
        lambda_penalty: float,
    ) -> np.ndarray:
        if stats.S is None or stats.U is None:
            raise ValueError("Некорректные статистики new-варианта")
        d = stats.U.shape[2]
        lhs = lambda_penalty * np.eye(d)
        rhs = lambda_penalty * prior
        for j, slope in enumerate(slopes):
            Uj = stats.U[j]
            residual = stats.imav[j] - intercepts[j] * stats.S[j]
            lhs += slope * slope * (Uj.T @ Uj)
            rhs += slope * (Uj.T @ residual)
        return _safe_solve(lhs + self.config.ridge * np.eye(d), rhs)

    def _objective(
        self,
        stats: LocalStatistics,
        beta: np.ndarray,
        intercepts: np.ndarray,
        slopes: np.ndarray,
        prior: np.ndarray,
        lambda_penalty: float,
    ) -> float:
        if stats.S is None or stats.U is None:
            raise ValueError("Некорректные статистики new-варианта")
        total = 0.0
        for j, slope in enumerate(slopes):
            pred = intercepts[j] * stats.S[j] + slope * (stats.U[j] @ beta)
            total += float(np.sum((stats.imav[j] - pred) ** 2))
        total += lambda_penalty * float(np.sum((beta - prior) ** 2))
        return total


class FullMomentADP(ADPBase):
    """ADP из manifold_old.tex: полная матрица локальных моментов, без Sph."""

    variant: VariantName = "old"

    def _compute_statistics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        centers: np.ndarray,
        h: float,
        beta: np.ndarray,
        directions: np.ndarray | None,
        anisotropy: float | None,
        b_value: float | None,
    ) -> LocalStatistics:
        J, d = centers.shape
        imav = np.zeros((J, d + 1))
        n_all = np.zeros(J)
        s_all = np.zeros((J, d))
        vp_all = np.zeros((J, d, d))
        weight_means: list[float] = []

        for start in range(0, J, self.config.chunk_size):
            stop = min(start + self.config.chunk_size, J)
            diff = X[None, :, :] - centers[start:stop, None, :]
            norm2 = np.einsum("cnd,cnd->cn", diff, diff)
            if b_value is None:
                q = norm2 / (h * h)
            else:
                proj2 = np.square(np.einsum("cnd,d->cn", diff, beta))
                q = norm2 / (h * h) + proj2 / (b_value * b_value)
            chunk_imav, chunk_n, chunk_s, chunk_vp, weight_mean = self.backend.full_moment_sums(
                diff,
                y,
                q,
                self.config.kernel,
            )
            imav[start:stop] = chunk_imav
            n_all[start:stop] = chunk_n
            s_all[start:stop] = chunk_s
            vp_all[start:stop] = chunk_vp
            weight_means.append(weight_mean)

        return LocalStatistics(
            variant="old",
            imav=imav,
            centers=centers,
            h=h,
            weights_mean=float(np.mean(weight_means)),
            N=n_all,
            S=s_all,
            VP=vp_all,
            b=b_value,
        )

    def _solve_local_coefficients(self, stats: LocalStatistics, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if stats.N is None or stats.S is None or stats.VP is None:
            raise ValueError("Некорректные статистики old-варианта")
        J = stats.N.shape[0]
        intercepts = np.zeros(J)
        slopes = np.zeros(J)
        for j in range(J):
            col0 = np.concatenate([[stats.N[j]], stats.S[j]])
            col1 = np.concatenate([[stats.S[j] @ beta], stats.VP[j] @ beta])
            design = np.column_stack([col0, col1])
            lhs = design.T @ design + self.config.ridge * np.eye(2)
            rhs = design.T @ stats.imav[j]
            intercepts[j], slopes[j] = _safe_solve(lhs, rhs)
        return intercepts, slopes

    def _solve_beta(
        self,
        stats: LocalStatistics,
        intercepts: np.ndarray,
        slopes: np.ndarray,
        prior: np.ndarray,
        lambda_penalty: float,
    ) -> np.ndarray:
        if stats.N is None or stats.S is None or stats.VP is None:
            raise ValueError("Некорректные статистики old-варианта")
        d = stats.S.shape[1]
        lhs = lambda_penalty * np.eye(d)
        rhs = lambda_penalty * prior
        for j, slope in enumerate(slopes):
            sj = stats.S[j]
            vpj = stats.VP[j]
            im0 = stats.imav[j, 0]
            im1 = stats.imav[j, 1:]
            lhs += slope * slope * (np.outer(sj, sj) + vpj.T @ vpj)
            rhs += slope * sj * (im0 - intercepts[j] * stats.N[j])
            rhs += slope * (vpj.T @ (im1 - intercepts[j] * sj))
        return _safe_solve(lhs + self.config.ridge * np.eye(d), rhs)

    def _objective(
        self,
        stats: LocalStatistics,
        beta: np.ndarray,
        intercepts: np.ndarray,
        slopes: np.ndarray,
        prior: np.ndarray,
        lambda_penalty: float,
    ) -> float:
        if stats.N is None or stats.S is None or stats.VP is None:
            raise ValueError("Некорректные статистики old-варианта")
        total = 0.0
        for j, slope in enumerate(slopes):
            pred0 = intercepts[j] * stats.N[j] + slope * (stats.S[j] @ beta)
            pred1 = intercepts[j] * stats.S[j] + slope * (stats.VP[j] @ beta)
            residual = stats.imav[j] - np.concatenate([[pred0], pred1])
            total += float(np.sum(residual**2))
        total += lambda_penalty * float(np.sum((beta - prior) ** 2))
        return total


def _as_2d_float(value: np.ndarray | None, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} не должен быть None")
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} должен быть двумерным массивом")
    return arr


def _as_1d_float(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} должен быть одномерным массивом")
    return arr


def _unit_vector(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    norm = np.linalg.norm(arr)
    if norm < np.finfo(float).eps:
        raise ValueError("Нельзя нормировать нулевой вектор")
    return arr / norm


def _normalize_rows(value: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(value, axis=-1, keepdims=True)
    norms = np.maximum(norms, np.finfo(float).eps)
    return value / norms


def _pairwise_norm2(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    diff = X[None, :, :] - centers[:, None, :]
    return np.einsum("jnd,jnd->jn", diff, diff)


def _pairwise_projection2(X: np.ndarray, centers: np.ndarray, beta: np.ndarray) -> np.ndarray:
    diff = X[None, :, :] - centers[:, None, :]
    return np.square(np.einsum("jnd,d->jn", diff, beta))


def _kernel_np(q: np.ndarray, name: KernelName) -> np.ndarray:
    if name == "gaussian":
        return np.exp(-0.5 * q)
    if name == "quartic":
        return np.maximum(1.0 - q * q, 0.0)
    return np.maximum(1.0 - q, 0.0)


def _average_kernel_weight(q: np.ndarray, name: KernelName) -> float:
    return float(_kernel_np(q, name).sum(axis=1).mean())


def _save_figure(fig: Any, path: str | Path, *, dpi: int = 150, close: bool = False) -> Path:
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    if close:
        _ensure_matplotlib_config_dir()
        import matplotlib.pyplot as plt

        plt.close(fig)
    return save_path


def _ensure_matplotlib_config_dir() -> None:
    import os

    if "MPLCONFIGDIR" in os.environ:
        return
    config_dir = Path("/tmp") / "adp_matplotlib"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(config_dir)


def _format_progress_postfix(record: dict[str, Any]) -> dict[str, Any]:
    postfix: dict[str, Any] = {
        "variant": record["variant"],
        "backend": record["backend"],
        "outer": f"{record['outer']}/{record['outer_total']}",
        "inner": record["inner"],
        "h": _format_float(record["h"]),
        "weights": _format_float(record["weights"]),
        "objective": _format_float(record["objective"]),
        "delta": _format_float(record["delta"]),
        "elapsed": f"{record['elapsed']:.1f}s",
    }
    if "rho" in record:
        postfix["rho"] = _format_float(record["rho"])
    if "b" in record:
        postfix["b"] = _format_float(record["b"])
    if "directions" in record:
        postfix["dirs"] = record["directions"]
    return postfix


def _format_float(value: float) -> str:
    number = float(value)
    if number == 0:
        return "0"
    if not np.isfinite(number):
        return str(number)
    if abs(number) >= 1e4 or abs(number) < 1e-3:
        return f"{number:.2e}"
    return f"{number:.4g}"


def _safe_solve(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return linalg.solve(lhs, rhs, assume_a="pos")
    except Exception:
        return linalg.lstsq(lhs, rhs)[0]


def _link_function(link: str | Callable[[np.ndarray], np.ndarray]) -> tuple[Callable[[np.ndarray], np.ndarray], str]:
    if callable(link):
        return link, getattr(link, "__name__", "callable")
    if link == "linear":
        return lambda z: z, "linear"
    if link == "sin":
        return np.sin, "sin"
    if link == "quadratic":
        return lambda z: z**2, "quadratic"
    if link == "tanh":
        return np.tanh, "tanh"
    raise ValueError("link должен быть callable или одним из: linear, sin, quadratic, tanh")
