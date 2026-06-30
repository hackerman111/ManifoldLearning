from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from .core import ADP, ADPConfig, ADPData


BenchmarkMethod = Literal[
    "adp_new",
    "adp_old",
    "statsmodels_sir",
    "statsmodels_save",
    "statsmodels_phd",
    "sklearn_pls",
]


@dataclass(slots=True)
class BenchmarkScenario:
    """Один воспроизводимый сценарий для проверки EDR-методов."""

    name: str
    n: int
    d: int
    link: str = "linear"
    noise: float = 0.05
    corr: float = 0.5
    sigma_x: float = 1.0
    n_centers: int | None = None
    n_directions: int = 8
    min_neighbors: float = 10.0
    outer_steps: int = 3
    inner_steps: int = 8
    trials: int = 3

    def adp_config(self, *, random_state: int, show_progress: bool, backend: str = "numpy") -> ADPConfig:
        return ADPConfig(
            n_centers=self.n_centers or min(self.n, max(20, self.n // 4)),
            n_directions=self.n_directions,
            min_neighbors=self.min_neighbors,
            outer_steps=self.outer_steps,
            inner_steps=self.inner_steps,
            backend=backend,  # type: ignore[arg-type]
            show_progress=show_progress,
            random_state=random_state,
        )


def default_scenarios(*, quick: bool = False) -> list[BenchmarkScenario]:
    """Набор сценариев, закрывающий несколько типичных режимов.

    quick=True нужен для smoke-прогона. Полный режим полезнее для реальной
    оценки эффективности, но занимает заметно больше времени.
    """

    trials = 1 if quick else 5
    scale = 0.55 if quick else 1.0

    def n(value: int) -> int:
        return max(80, int(value * scale))

    return [
        BenchmarkScenario(
            name="linear_low_noise",
            n=n(240),
            d=8,
            link="linear",
            noise=0.03,
            corr=0.2,
            n_centers=n(60),
            n_directions=8,
            min_neighbors=8,
            outer_steps=2 if quick else 4,
            inner_steps=5 if quick else 10,
            trials=trials,
        ),
        BenchmarkScenario(
            name="sin_correlated",
            n=n(280),
            d=10,
            link="sin",
            noise=0.08,
            corr=0.55,
            n_centers=n(70),
            n_directions=10,
            min_neighbors=10,
            outer_steps=2 if quick else 4,
            inner_steps=5 if quick else 10,
            trials=trials,
        ),
        BenchmarkScenario(
            name="quadratic_symmetric",
            n=n(320),
            d=10,
            link="quadratic",
            noise=0.05,
            corr=0.35,
            n_centers=n(80),
            n_directions=12,
            min_neighbors=10,
            outer_steps=2 if quick else 5,
            inner_steps=5 if quick else 12,
            trials=trials,
        ),
        BenchmarkScenario(
            name="dimension_stress",
            n=n(360),
            d=18,
            link="linear",
            noise=0.05,
            corr=0.4,
            n_centers=n(90),
            n_directions=12,
            min_neighbors=12,
            outer_steps=2 if quick else 4,
            inner_steps=5 if quick else 10,
            trials=trials,
        ),
    ]


def run_benchmark_suite(
    scenarios: Iterable[BenchmarkScenario] | None = None,
    *,
    methods: Iterable[BenchmarkMethod] = ("adp_new", "adp_old", "statsmodels_sir", "statsmodels_save", "statsmodels_phd", "sklearn_pls"),
    random_state: int = 0,
    show_progress: bool = False,
    adp_backend: str = "numpy",
) -> pd.DataFrame:
    """Запустить набор сценариев и вернуть таблицу результатов."""

    scenario_list = list(scenarios) if scenarios is not None else default_scenarios()
    method_list = list(methods)
    rows: list[dict[str, Any]] = []

    seed_seq = np.random.SeedSequence(random_state)
    scenario_seeds = seed_seq.spawn(sum(scenario.trials for scenario in scenario_list))
    seed_index = 0

    for scenario in scenario_list:
        for trial in range(scenario.trials):
            trial_seed = int(scenario_seeds[seed_index].generate_state(1)[0])
            seed_index += 1
            data = _make_data(scenario, trial_seed)
            for method_index, method in enumerate(method_list):
                method_seed = trial_seed + 10_000 + method_index
                rows.append(_run_method(method, scenario, data, trial, method_seed, show_progress, adp_backend))

    return pd.DataFrame(rows)


def save_benchmark_report(frame: pd.DataFrame, output_dir: str | Path, *, prefix: str = "adp_benchmark", dpi: int = 150) -> dict[str, Path]:
    """Сохранить CSV и обзорные графики качества/времени."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    csv_path = output_path / f"{prefix}.csv"
    frame.to_csv(csv_path, index=False)
    saved["csv"] = csv_path

    _ensure_matplotlib_config_dir()
    import matplotlib.pyplot as plt

    quality = frame.groupby(["scenario", "method"], as_index=False)["cosine_abs"].mean()
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * quality["scenario"].nunique()), 4.5))
    _plot_grouped_bars(ax, quality, value="cosine_abs", ylabel="mean |cos(beta, beta_hat)|", title="EDR recovery quality")
    saved["quality_plot"] = _save_figure(fig, output_path / f"{prefix}_quality.png", dpi=dpi)

    timings = frame.groupby(["scenario", "method"], as_index=False)["fit_time_sec"].mean()
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * timings["scenario"].nunique()), 4.5))
    _plot_grouped_bars(ax, timings, value="fit_time_sec", ylabel="mean fit time, sec", title="EDR fit time")
    saved["time_plot"] = _save_figure(fig, output_path / f"{prefix}_time.png", dpi=dpi)

    return saved


def benchmark_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Сводная таблица по качеству и времени."""

    return (
        frame.groupby(["scenario", "method"])
        .agg(
            cosine_abs_mean=("cosine_abs", "mean"),
            cosine_abs_std=("cosine_abs", "std"),
            angle_deg_mean=("angle_deg", "mean"),
            fit_time_sec_mean=("fit_time_sec", "mean"),
            failures=("failed", "sum"),
        )
        .reset_index()
    )


def _make_data(scenario: BenchmarkScenario, seed: int) -> ADPData:
    generator = ADP.create(
        "new",
        ADPConfig(
            n_centers=scenario.n_centers or min(scenario.n, max(20, scenario.n // 4)),
            n_directions=scenario.n_directions,
            show_progress=False,
            random_state=seed,
        ),
    )
    return generator.generate_data(
        n=scenario.n,
        d=scenario.d,
        n_centers=scenario.n_centers,
        n_directions=scenario.n_directions,
        noise=scenario.noise,
        sigma_x=scenario.sigma_x,
        corr=scenario.corr,
        link=scenario.link,
    )


def _run_method(
    method: BenchmarkMethod,
    scenario: BenchmarkScenario,
    data: ADPData,
    trial: int,
    seed: int,
    show_progress: bool,
    adp_backend: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    failed = False
    error = ""
    objective = np.nan
    beta_hat: np.ndarray

    try:
        if method == "adp_new":
            model = ADP.create("new", scenario.adp_config(random_state=seed, show_progress=show_progress, backend=adp_backend))
            result = model.fit(data.X, data.y, centers=data.centers)
            beta_hat = result.beta
            objective = result.objective
        elif method == "adp_old":
            model = ADP.create("old", scenario.adp_config(random_state=seed, show_progress=show_progress, backend=adp_backend))
            result = model.fit(data.X, data.y, centers=data.centers)
            beta_hat = result.beta
            objective = result.objective
        elif method == "statsmodels_sir":
            beta_hat = _fit_statsmodels_dimred(data.X, data.y, "sir")
        elif method == "statsmodels_save":
            beta_hat = _fit_statsmodels_dimred(data.X, data.y, "save")
        elif method == "statsmodels_phd":
            beta_hat = _fit_statsmodels_dimred(data.X, data.y, "phd")
        elif method == "sklearn_pls":
            beta_hat = _fit_sklearn_pls(data.X, data.y)
        else:
            raise ValueError(f"Неизвестный метод benchmark: {method}")
    except Exception as exc:
        failed = True
        error = f"{type(exc).__name__}: {exc}"
        beta_hat = np.full_like(data.beta, np.nan)

    fit_time = time.perf_counter() - started
    metrics = _direction_metrics(beta_hat, data.beta)
    return {
        "scenario": scenario.name,
        "trial": trial,
        "method": method,
        "n": scenario.n,
        "d": scenario.d,
        "link": scenario.link,
        "noise": scenario.noise,
        "corr": scenario.corr,
        "cosine": metrics["cosine"],
        "cosine_abs": metrics["cosine_abs"],
        "angle_deg": metrics["angle_deg"],
        "signed_l2": metrics["signed_l2"],
        "fit_time_sec": fit_time,
        "objective": objective,
        "failed": failed,
        "error": error,
    }


def _fit_statsmodels_dimred(X: np.ndarray, y: np.ndarray, kind: str) -> np.ndarray:
    from statsmodels.regression.dimred import PHD, SAVE, SIR

    cls = {"sir": SIR, "save": SAVE, "phd": PHD}[kind]
    if kind == "sir":
        result = cls(y, X).fit(slice_n=min(20, max(4, X.shape[0] // 8)))
    else:
        result = cls(y, X).fit()
    params = np.asarray(result.params, dtype=float)
    if params.ndim == 1:
        return params
    return params[:, 0]


def _fit_sklearn_pls(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    from sklearn.cross_decomposition import PLSRegression

    model = PLSRegression(n_components=1, scale=True)
    model.fit(X, y.reshape(-1, 1))
    return np.asarray(model.x_weights_[:, 0], dtype=float)


def _direction_metrics(beta_hat: np.ndarray, beta_true: np.ndarray) -> dict[str, float]:
    if not np.all(np.isfinite(beta_hat)):
        return {"cosine": np.nan, "cosine_abs": np.nan, "angle_deg": np.nan, "signed_l2": np.nan}
    estimated = _unit_vector(beta_hat)
    expected = _unit_vector(beta_true)
    cosine = float(np.clip(expected @ estimated, -1.0, 1.0))
    cosine_abs = abs(cosine)
    return {
        "cosine": cosine,
        "cosine_abs": cosine_abs,
        "angle_deg": float(np.degrees(np.arccos(np.clip(cosine_abs, -1.0, 1.0)))),
        "signed_l2": float(min(np.linalg.norm(estimated - expected), np.linalg.norm(estimated + expected))),
    }


def _unit_vector(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm < np.finfo(float).eps:
        return np.full_like(vector, np.nan)
    return vector / norm


def _plot_grouped_bars(ax: Any, frame: pd.DataFrame, *, value: str, ylabel: str, title: str) -> None:
    pivot = frame.pivot(index="scenario", columns="method", values=value)
    pivot.plot(kind="bar", ax=ax)
    ax.set_xlabel("scenario")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="method", fontsize="small")


def _save_figure(fig: Any, path: Path, *, dpi: int = 150) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    _ensure_matplotlib_config_dir()
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def _ensure_matplotlib_config_dir() -> None:
    if "MPLCONFIGDIR" in os.environ:
        return
    config_dir = Path("/tmp") / "adp_matplotlib"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(config_dir)
