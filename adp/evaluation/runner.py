from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..core import ADP, ADPConfig, ADPData
from ..common.resource_monitor import ResourceMonitor
from .baselines import fit_sklearn_pls, fit_statsmodels_dimred
from .metrics import direction_metrics
from .scenarios import BenchmarkMethod, BenchmarkScenario, default_scenarios


def run_benchmark_suite(
    scenarios: Iterable[BenchmarkScenario] | None = None,  # Сценарии или None.
    *,
    methods: Iterable[BenchmarkMethod] = ("adp_new", "statsmodels_sir", "statsmodels_save", "statsmodels_phd", "sklearn_pls"),
    random_state: int = 0,  # Общее начальное число.
    show_progress: bool = False,  # Показывать прогресс ADP.
) -> pd.DataFrame:
    """Запускает набор benchmark-сценариев.

    Вход:
        scenarios: список сценариев или None для default_scenarios().
        methods: имена методов.
        random_state: общий seed.
        show_progress: флаг tqdm для ADP.
    Выход:
        DataFrame со строкой на каждый метод и trial.
    """

    scenario_list = list(scenarios) if scenarios is not None else default_scenarios()
    method_list = list(methods)
    rows: list[dict[str, Any]] = []

    # Последовательность начальных чисел раздает независимые значения на повтор,
    # чтобы методы сравнивались на одних данных, но не делили random_state обучения.
    seed_seq = np.random.SeedSequence(random_state)
    scenario_seeds = seed_seq.spawn(sum(scenario.trials for scenario in scenario_list))
    seed_index = 0

    for scenario in scenario_list:
        for trial in range(scenario.trials):
            trial_seed = int(scenario_seeds[seed_index].generate_state(1)[0])
            seed_index += 1
            for method_index, method in enumerate(method_list):
                method_seed = trial_seed + 10_000 + method_index
                rows.append(
                    execute_benchmark_method(
                        method,
                        scenario,
                        trial_seed,
                        trial,
                        method_seed,
                        show_progress,
                    )
                )

    return pd.DataFrame(rows)


def make_data(
    scenario: BenchmarkScenario,  # Сценарий замеров.
    seed: int,  # Начальное число генерации данных.
) -> ADPData:
    """Генерирует данные для одного benchmark-сценария.

    Вход:
        scenario: параметры задачи.
        seed: seed генератора.
    Выход:
        ADPData с X, y и истинным beta.
    """

    # Данные генерирует тот же интерфейс ADP, чтобы замеры использовали ровно ту
    # одноиндексную модель, на которую настроен рабочий new-вариант.
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


def execute_benchmark_method(
    method: BenchmarkMethod,  # Имя метода для замера.
    scenario: BenchmarkScenario,  # Сценарий.
    data_seed: int,  # Начальное число генерации данных.
    trial: int,  # Номер повтора.
    seed: int,  # Начальное число метода.
    show_progress: bool,  # Показывать прогресс.
) -> dict[str, Any]:
    """Запускает один метод на одном наборе данных.

    Вход:
        method: имя метода.
        scenario: параметры сценария.
        data_seed: seed воспроизводимой генерации данных.
        trial: номер повтора.
        seed: seed метода.
        show_progress: флаг tqdm.
    Выход:
        Словарь метрик, времени и памяти.
    """

    failed = False
    error = ""
    objective = np.nan
    beta_hat = np.full(scenario.d, np.nan)
    beta_true = np.full(scenario.d, np.nan)
    algorithm_usage = _empty_resource_usage("algorithm")
    model = None
    algorithm_monitor = None
    full_monitor = ResourceMonitor()
    with full_monitor:
        try:
            data = make_data(scenario, data_seed)
            beta_true = data.beta
            if method == "adp_new":
                # new соответствует manifold_new.tex: случайные проекции phi.
                model = ADP.create(
                    "new",
                    scenario.adp_config(
                        random_state=seed,
                        show_progress=show_progress,
                    ),
                )
                result = model.fit(data.X, data.y, centers=data.centers)
                beta_hat = result.beta
                objective = result.objective
                algorithm_usage = dict(result.resource_usage)
            else:
                algorithm_monitor = ResourceMonitor()
                with algorithm_monitor:
                    if method == "statsmodels_sir":
                        beta_hat = fit_statsmodels_dimred(data.X, data.y, "sir")
                    elif method == "statsmodels_save":
                        beta_hat = fit_statsmodels_dimred(data.X, data.y, "save")
                    elif method == "statsmodels_phd":
                        beta_hat = fit_statsmodels_dimred(data.X, data.y, "phd")
                    elif method == "sklearn_pls":
                        beta_hat = fit_sklearn_pls(data.X, data.y)
                    else:
                        raise ValueError(f"Неизвестный метод benchmark: {method}")
                algorithm_usage = algorithm_monitor.usage.to_dict("algorithm")
        except Exception as exc:
            failed = True
            error = f"{type(exc).__name__}: {exc}"
            if model is not None and model.last_resource_usage_:
                algorithm_usage = dict(model.last_resource_usage_)
            elif algorithm_monitor is not None:
                algorithm_usage = algorithm_monitor.usage.to_dict("algorithm")
        metrics = direction_metrics(beta_hat, beta_true)

    full_usage = full_monitor.usage.to_dict("full_run")
    fit_time = float(algorithm_usage["algorithm_time_sec"])
    return {
        "scenario": scenario.name,
        "trial": trial,
        "method": method,
        "n": scenario.n,
        "d": scenario.d,
        "n_directions": scenario.n_directions,
        "n_centers": scenario.n_centers or min(scenario.n, max(20, scenario.n // 4)),
        "outer_steps": scenario.outer_steps,
        "inner_steps": scenario.inner_steps,
        "link": scenario.link,
        "noise": scenario.noise,
        "corr": scenario.corr,
        "cosine": metrics["cosine"],
        "cosine_abs": metrics["cosine_abs"],
        "angle_deg": metrics["angle_deg"],
        "signed_l2": metrics["signed_l2"],
        "fit_time_sec": fit_time,
        "peak_memory_kib": float(full_usage["full_run_rss_peak_delta_mib"]) * 1024.0,
        **algorithm_usage,
        **full_usage,
        "objective": objective,
        "failed": failed,
        "error": error,
    }


def run_method(
    method: BenchmarkMethod,
    scenario: BenchmarkScenario,
    data_seed: int,
    trial: int,
    seed: int,
    show_progress: bool,
) -> dict[str, Any]:
    """Backward-compatible alias for the legacy benchmark helper."""

    return execute_benchmark_method(
        method,
        scenario,
        data_seed,
        trial,
        seed,
        show_progress,
    )


def _empty_resource_usage(prefix: str) -> dict[str, float | int | str]:
    return {
        f"{prefix}_time_sec": np.nan,
        f"{prefix}_rss_start_mib": np.nan,
        f"{prefix}_rss_min_mib": np.nan,
        f"{prefix}_rss_mean_mib": np.nan,
        f"{prefix}_rss_max_mib": np.nan,
        f"{prefix}_rss_peak_delta_mib": np.nan,
        f"{prefix}_memory_samples": 0,
        f"{prefix}_memory_source": "unavailable",
    }
