from __future__ import annotations

import time
import tracemalloc
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..core import ADP, ADPConfig, ADPData
from .baselines import fit_sklearn_pls, fit_statsmodels_dimred
from .metrics import direction_metrics
from .scenarios import BenchmarkMethod, BenchmarkScenario, default_scenarios


def run_benchmark_suite(
    scenarios: Iterable[BenchmarkScenario] | None = None,  # Сценарии или None.
    *,
    methods: Iterable[BenchmarkMethod] = ("adp_new", "adp_old", "statsmodels_sir", "statsmodels_save", "statsmodels_phd", "sklearn_pls"),
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
            data = make_data(scenario, trial_seed)
            for method_index, method in enumerate(method_list):
                method_seed = trial_seed + 10_000 + method_index
                rows.append(run_method(method, scenario, data, trial, method_seed, show_progress))

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
    # одноиндексную модель, на которую настроены old/new варианты.
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


def run_method(
    method: BenchmarkMethod,  # Имя метода для замера.
    scenario: BenchmarkScenario,  # Сценарий.
    data: ADPData,  # Данные сценария.
    trial: int,  # Номер повтора.
    seed: int,  # Начальное число метода.
    show_progress: bool,  # Показывать прогресс.
) -> dict[str, Any]:
    """Запускает один метод на одном наборе данных.

    Вход:
        method: имя метода.
        scenario: параметры сценария.
        data: сгенерированные данные.
        trial: номер повтора.
        seed: seed метода.
        show_progress: флаг tqdm.
    Выход:
        Словарь метрик, времени и памяти.
    """

    started = time.perf_counter()
    failed = False
    error = ""
    objective = np.nan
    beta_hat: np.ndarray

    # Окно измерения памяти намеренно охватывает только один вызов метода; так
    # peak_memory_kib сопоставим между ADP и базовыми методами.
    started_tracing = not tracemalloc.is_tracing()
    if started_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    try:
        if method == "adp_new":
            # new соответствует manifold_new.tex: случайные проекции phi.
            model = ADP.create("new", scenario.adp_config(random_state=seed, show_progress=show_progress))
            result = model.fit(data.X, data.y, centers=data.centers)
            beta_hat = result.beta
            objective = result.objective
        elif method == "adp_old":
            # old соответствует manifold_old.tex: полные локальные моменты без phi.
            model = ADP.create("old", scenario.adp_config(random_state=seed, show_progress=show_progress))
            result = model.fit(data.X, data.y, centers=data.centers)
            beta_hat = result.beta
            objective = result.objective
        elif method == "statsmodels_sir":
            beta_hat = fit_statsmodels_dimred(data.X, data.y, "sir")
        elif method == "statsmodels_save":
            beta_hat = fit_statsmodels_dimred(data.X, data.y, "save")
        elif method == "statsmodels_phd":
            beta_hat = fit_statsmodels_dimred(data.X, data.y, "phd")
        elif method == "sklearn_pls":
            beta_hat = fit_sklearn_pls(data.X, data.y)
        else:
            raise ValueError(f"Неизвестный метод benchmark: {method}")
    except Exception as exc:
        failed = True
        error = f"{type(exc).__name__}: {exc}"
        beta_hat = np.full_like(data.beta, np.nan)
    finally:
        _, peak_memory = tracemalloc.get_traced_memory()
        if started_tracing:
            tracemalloc.stop()

    fit_time = time.perf_counter() - started
    metrics = direction_metrics(beta_hat, data.beta)
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
        "peak_memory_kib": peak_memory / 1024.0,
        "objective": objective,
        "failed": failed,
        "error": error,
    }
