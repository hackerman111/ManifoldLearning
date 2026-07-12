from __future__ import annotations

from dataclasses import replace

from .types import SingleIndexScenario


_TITLES = {
    "C01": "Прямое вычисление локальных статистик I и U",
    "C02": "Инвариантность к константному сдвигу отклика",
    "C03": "Точное восстановление трёхмерной линейной модели",
    "C04": "Ортогональная эквивариантность",
    "C05": "Эквивариантность к масштабу признаков",
    "C06": "Инвариантность к сдвигу и масштабу отклика",
    "C07": "Монотонность локальной массы и binary search",
    "C08": "Изолированный ALS на noiseless statistics",
    "C09": "Matrix-free CG совпадает с dense reference",
    "C10": "Objective ALS не возрастает",
    "C11": "Stopping устойчив к масштабу objective",
    "C12": "Воспроизводимость dtype, chunking и threads",
    "S01": "Качество по семейству функций связи",
    "S02": "Рост n при фиксированном алгоритме",
    "S03": "Рост n в практическом режиме",
    "S04": "Деградация при Gaussian noise",
    "S05": "Граница по размерности и n/d",
    "S06": "Цена nuisance dimensions",
    "T01": "Чувствительность к числу направлений P",
    "T02": "Взаимодействие P и d",
    "T03": "Чувствительность к числу центров J",
    "T04": "Рабочий диапазон локальной массы",
    "T05": "Чувствительность к lambda",
    "T06": "Скорость уменьшения bandwidth",
    "T07": "Точность текущего ALS и CG",
    "T08": "Расписание обновления направлений",
    "T09": "Mean против quantile local mass",
    "T10": "Perturbation центров",
    "R01": "Gaussian noise robustness",
    "R02": "Тяжёлые хвосты шума",
    "R03": "Выбросы в отклике",
    "R04": "Выбросы в признаках",
    "R05": "Гетероскедастичность",
    "R06": "Коррелированные признаки",
    "R07": "Почти вырожденная covariance",
    "R08": "Разные масштабы координат",
    "R09": "Нелинейная зависимость признаков",
    "R10": "Sparse и dense истинное направление",
    "R11": "Осциллирующая функция связи",
    "R12": "Нарушение гладкости",
    "R13": "Неоднородная плотность",
    "R14": "Принудительно разреженные окрестности",
    "R15": "Misspecification single-index модели",
    "M01": "Масштабирование времени по n",
    "M02": "Масштабирование времени по d",
    "M03": "Масштабирование времени по J",
    "M04": "Масштабирование времени по P",
    "M05": "Стоимость outer и inner iterations",
    "M06": "Модель process RSS",
    "M07": "Process-level parallel scaling",
    "M08": "Chunking memory-quality trade-off",
    "I01": "Область притяжения по initial cosine",
    "I02": "Link-specific basin",
    "I03": "Практические инициализации",
    "I04": "Sampling и algorithmic variance",
    "B01": "Сравнение с EDR baseline",
    "A01": "Step0 против полного внешнего цикла",
    "A02": "Ablation без анизотропии",
    "A03": "Ablation без уменьшения bandwidth",
    "A04": "Full directional basis",
    "A05": "Fixed directions",
    "A06": "Ablation без regularization",
    "A07": "Mean mass против q05 mass",
    "A08": "Центры без perturbation",
    "A09": "Negative controls",
}

_SMOKE_IDS = ("C01", "S01", "M01", "B01")
_MINIMAL_IDS = (
    *(f"C{index:02d}" for index in range(1, 13)),
    "S01",
    "S02",
    "S04",
    "T01",
    "T04",
    "T07",
    "I01",
    "B01",
    "A01",
    "A02",
    "A05",
    "M01",
)
_FULL_IDS = tuple(_TITLES)
_PUBLICATION_IDS = tuple(
    scenario_id for scenario_id in _FULL_IDS if not scenario_id.startswith("C")
)

PROFILE_IDS = {
    "smoke": _SMOKE_IDS,
    "minimal": _MINIMAL_IDS,
    "full": _FULL_IDS,
    "publication": _PUBLICATION_IDS,
}


def scenario_registry() -> tuple[SingleIndexScenario, ...]:
    return tuple(_make_scenario(scenario_id) for scenario_id in _FULL_IDS)


def scenarios_for_profile(profile: str) -> tuple[SingleIndexScenario, ...]:
    if profile not in PROFILE_IDS:
        raise ValueError(f"unknown single-index profile: {profile}")
    by_id = {scenario.scenario_id: scenario for scenario in scenario_registry()}
    selected = tuple(by_id[scenario_id] for scenario_id in PROFILE_IDS[profile])
    if profile == "smoke":
        return tuple(_smoke_variant(scenario) for scenario in selected)
    return selected


def _make_scenario(scenario_id: str) -> SingleIndexScenario:
    family = scenario_id[0]
    if family == "C":
        executor = "correctness"
    elif family == "M":
        executor = "scaling"
    else:
        executor = "recovery"

    data = {
        "n": 1000,
        "d": 20,
        "link": "tanh",
        "noise": 0.1,
        "corr": 0.0,
        "sigma_x": 1.0,
    }
    algorithm = {
        "n_centers": 200,
        "n_directions": 32,
        "min_neighbors": 64.0,
        "statistics_workers": 1,
    }
    solver = {"outer_steps": 8, "inner_steps": 20}
    methods = ("full_adp",)
    repeats = 50

    if family == "C":
        data.update({"n": 100, "d": 5, "noise": 0.0})
        algorithm.update({"n_centers": 10, "n_directions": 5})
        solver.update({"outer_steps": 1, "inner_steps": 8})
        repeats = 1
    elif family == "M":
        repeats = 20
    elif family == "I":
        repeats = 100
    elif family == "R":
        repeats = 100

    if scenario_id == "R06":
        data["corr"] = 0.7
    elif scenario_id == "R07":
        data["corr"] = 0.98
    elif scenario_id == "B01":
        methods = (
            "random_direction",
            "ols",
            "statsmodels_sir",
            "statsmodels_save",
            "statsmodels_phd",
            "sklearn_pls",
            "full_adp",
        )
    elif family == "A":
        methods = ("full_adp", _ablation_method(scenario_id))

    return SingleIndexScenario(
        scenario_id=scenario_id,
        family=family,
        executor=executor,
        hypothesis=_TITLES[scenario_id],
        data=data,
        algorithm=algorithm,
        solver=solver,
        repeats=repeats,
        methods=methods,
        record_solver_trace=scenario_id in {"C08", "C09", "C10", "C11", "T07"},
    )


def _ablation_method(scenario_id: str) -> str:
    return {
        "A01": "step0_only",
        "A02": "no_anisotropy",
        "A03": "fixed_h",
        "A04": "full_directional_basis",
        "A05": "fixed_directions",
        "A06": "no_regularization",
        "A07": "mean_mass",
        "A08": "no_center_perturbation",
        "A09": "negative_control",
    }[scenario_id]


def _smoke_variant(scenario: SingleIndexScenario) -> SingleIndexScenario:
    data = dict(scenario.data)
    if "n" in data:
        data.update({"n": 80, "d": 4})
    algorithm = dict(scenario.algorithm)
    algorithm.update({"n_centers": 12, "n_directions": 4, "min_neighbors": 5.0})
    solver = dict(scenario.solver)
    solver.update({"outer_steps": 1, "inner_steps": 3})
    return replace(
        scenario,
        data=data,
        algorithm=algorithm,
        solver=solver,
        repeats=1,
    )
