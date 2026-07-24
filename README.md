# Average Derivative Procedure

Основная точка входа сохраняет прежний вид:

```python
from adp import ADP, ADPConfig

model = ADP.create("new", ADPConfig(show_progress=False))
result = model.fit(X, y)
```

## Параллельное вычисление статистик NumPy

2 workers включаются явно через `statistics_workers=2`; по умолчанию
остаётся безопасный serial-режим с одним worker:

```python
config = ADPConfig(statistics_workers=2, show_progress=False)
model = ADP.create("new", config)
```

## Замена этапов алгоритма

Основные этапы ADP создаются через изолированный `StageRegistry`. Встроенные
имена можно посмотреть через `registry.available(category)` и выбрать при
создании модели:

```python
from adp import ADP, ADPConfig, StageRegistry

registry = StageRegistry.with_defaults()
model = ADP.create(
    "new",
    ADPConfig(show_progress=False),
    stages={
        "bandwidth_selector": "adaptive_mass",
        "statistics_builder": "random_projection",
        "beta_solver": "cg",
    },
    registry=registry,
)
```

Новый исследовательский solver можно передать напрямую:

```python
model = ADP.create(
    "new",
    config,
    stage_factories={
        "beta_solver": lambda context: ExperimentalBetaSolver(context.config),
    },
)
```

Доступные категории:

- `beta_initializer`;
- `center_selector`;
- `bandwidth_selector`;
- `direction_sampler`;
- `statistics_builder`;
- `local_solver`;
- `beta_solver`;
- `stop_rule`.

После `fit()` результат содержит выбранные реализации, накопленное время и
число вызовов каждого этапа:

```python
result.stage_names
result.stage_timings
result.stage_calls
```

### Варианты начального направления и \(h_0\)

Фиксированное направление штрафа хранится отдельно как `result.beta_ref`, а
нормированный результат первого outer-шага — как `result.beta_hat0`.
Встроенные варианты `beta_initializer`:

- `e1` — контрольное \(e_1\);
- `pca` — первый главный компонент центрированной выборки \(X\);
- `ridge_0`, `ridge_1e-4`, `ridge_1e-2` — нормированное направление
  \((X^\mathsf{T}X+\eta I)^{-1}X^\mathsf{T}(Y-\bar Y)\), где суффикс задаёт
  множитель перед \(\operatorname{tr}(X^\mathsf{T}X)/d\).

Например:

```python
model = ADP.create(
    "new",
    ADPConfig(show_progress=False),
    stages={"beta_initializer": "ridge_1e-4"},
)
```

Варианты `bandwidth_selector`:

- `local_mass_mean` — прежнее среднее условие;
- `local_mass_q0`, `local_mass_q05`, `local_mass_q10`, `local_mass_q25` —
  условие по соответствующему нижнему квантилю локальной массы;
- `knn_q90_k1`, `knn_q90_k2`, `knn_q90_k4` — \(Q_{0.9}\) расстояния до
  \(K\)-го наблюдения для \(K=n_{\min},2n_{\min},4n_{\min}\).

Все варианты меняют только начальный \(h_0\); последующий anisotropy-шаг
остаётся общим. При `record_telemetry=True` первый элемент
`result.outer_telemetry` содержит `h`, `local_mass_min`, `local_mass_q05`,
`local_mass_q10` и `local_mass_q25`.

## Время и потребление памяти

Каждый вызов `fit()` автоматически измеряет wall-clock время алгоритма и RSS
текущего процесса. RSS включает память массивов NumPy и записывается в MiB:

```python
result = model.fit(X, y)
result.resource_usage
```

Словарь содержит:

- `algorithm_time_sec`;
- `algorithm_rss_start_mib`;
- `algorithm_rss_min_mib`;
- `algorithm_rss_mean_mib`;
- `algorithm_rss_max_mib`;
- `algorithm_rss_peak_delta_mib`;
- число измерений и источник RSS.

Минимум, среднее и максимум являются абсолютным RSS процесса внутри окна
`fit()`. `algorithm_rss_peak_delta_mib` показывает прирост максимума относительно
начала вызова. Те же значения доступны через `model.summary()["resource_usage"]`.
Если `fit()` завершился ошибкой, последнее измерение остаётся в
`model.last_resource_usage_`.

Экспериментальные runners дополнительно записывают поля `full_run_*`. Это окно
начинается перед генерацией данных и созданием модели, включает `fit()` и расчёт
метрик. В confirmatory-сериях оно заканчивается после записи iteration rows в
worker CSV, поэтому учитывает сохранение основного результата. Время самой
записи отдельно находится в `result_persist_time_sec`.

## CSV-логи серий экспериментов

Confirmatory-эксперименты 4, 5 и 6 больше не создают JSON-манифесты. Для серии с
префиксом `<prefix>` сохраняются:

- `<prefix>_runs.csv` — один job на строку, статус, ошибки, время и память;
- `<prefix>_iterations.csv` — показатели каждой outer-итерации;
- `<prefix>_initial_parameters.csv` — seeds и развёрнутые настройки каждого job;
- `<prefix>_summary.csv` и `<prefix>_final_success.csv` — агрегаты;
- `<prefix>_series.csv` — параметры и итог всей серии;
- `<prefix>_artifacts.csv` — пути к таблицам и графикам.

Таблицы связываются по `run_id` и содержат `schema_version`. При параллельном
запуске workers пишут отдельные временные CSV-шарды, которые объединяются без
загрузки всех строк в память.

Stress runner аналогично сохраняет
`adp_single_index_stress_series.csv` и
`adp_single_index_stress_artifacts.csv` вместо JSON manifest. Benchmark
низкоуровневых NumPy-статистик принимает только CSV-путь:

```bash
python experiments/benchmark_numpy_statistics.py \
  --case primary \
  --repetitions 7 \
  --output outputs/numpy_statistics.csv
```

## Воспроизводимый single-index benchmark

Полный план из 24 000 независимых запусков стартует отдельной подкомандой:

```bash
python run_benchmarks.py single-index \
  --profile full \
  --jobs auto \
  --output benchmark_outputs/single_index
```

Можно выбрать эксперименты, диапазон seed и диагностические seed:

```bash
python run_benchmarks.py single-index \
  --profile full \
  --experiments 2,3,4 \
  --seeds 0:9 \
  --diagnostic-seeds 0,1,2 \
  --jobs 4 \
  --output benchmark_outputs/single_index
```

Для быстрой проверки используется профиль `smoke`; `--max-runs` оставляет
детерминированный префикс развернутого списка запусков:

```bash
python run_benchmarks.py single-index \
  --profile smoke \
  --jobs 2 \
  --max-runs 2 \
  --output /tmp/adp_new_benchmark_smoke
```

Dry-run проверяет конфигурацию, печатает число запусков и ничего не записывает:

```bash
python run_benchmarks.py single-index --profile full --dry-run
```

Прерванную серию можно продолжить. При resume нужно повторить исходные
`--profile`, `--experiments`, `--seeds`, `--diagnostic-seeds` и
`--center-fraction`; число `--jobs` можно изменить. Уже зафиксированные
`run_id` пропускаются:

```bash
python run_benchmarks.py single-index \
  --profile full \
  --jobs 4 \
  --resume benchmark_outputs/single_index/<series_id>
```

Запуски со статусом `numerical_failure` повторяются только с явным флагом и
атомарно заменяют прежний shard:

```bash
python run_benchmarks.py single-index \
  --profile full \
  --resume benchmark_outputs/single_index/<series_id> \
  --retry-failed
```

Графики можно полностью перестроить из сохранённых CSV, не выполняя `fit()`:

```bash
python run_benchmarks.py single-index \
  --profile smoke \
  --resume benchmark_outputs/single_index/<series_id> \
  --reports-only
```

`--jobs` задаёт число независимых worker-процессов. В каждом worker переменные
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` и
`NUMEXPR_NUM_THREADS` ограничиваются единицей; сам `model.fit(...)` выполняется
внутри `threadpoolctl` с лимитом `1`. Для benchmark-конфигурации
`ADPConfig.statistics_workers` всегда равен `1`, поэтому вложенного
параллелизма нет.

Каждая серия находится в отдельном подкаталоге `<series_id>` и содержит семь
публичных таблиц:

- `run_summary.csv` — одна итоговая строка на запуск;
- `outer_iterations.csv` — внешние итерации и разложение времени;
- `inner_iterations.csv` — внутренние итерации и поля решателя;
- `local_diagnostics.csv` — полные локальные диагностики выбранных seed и ошибок;
- `solver_iterations.csv` — трассы невязки линейного решателя;
- `series.csv` — конфигурация, окружение и состояние серии;
- `artifacts.csv` — относительные пути, размеры, статусы и ошибки CSV/PNG.

PNG строятся только из этих CSV и сохраняются в `plots/experiment_<selector>/`
и `plots/summary/`. JSON-файлы новый benchmark не создаёт.

Для сравнения нескольких реализаций достаточно создать Python-файл с
именованными фабриками:

```python
from adp import ADP, ADPConfig


def baseline():
    return ADP.create("new", ADPConfig(show_progress=False))


def candidate():
    return ADP.create(
        "new",
        ADPConfig(show_progress=False),
        stages={"local_solver": "zero_intercept"},
    )


MODELS = {
    "baseline": baseline,
    "candidate": candidate,
}
```

Первая модель считается baseline, остальные сравниваются с ней. Порядок
словаря сохраняется. Готовый пример с тремя реализациями находится в
`examples/model_comparison_models.py`:

```bash
python run_benchmarks.py compare \
  --models examples/model_comparison_models.py \
  --profile smoke \
  --seeds 0:4 \
  --jobs 1 \
  --output benchmark_outputs/model_comparison
```

Каждый `fit` выполняется в новом процессе. Все модели одной группы получают
одинаковые `X`, `y`, центры, начальное `beta` и направления, запускаются
последовательно на одном CPU, а их порядок циклически меняется между seed.
`--jobs` задаёт число параллельных групп. Для измерения latency без конкуренции
используйте `--jobs 1`; большие значения измеряют throughput под параллельной
нагрузкой.

Для эксперимента с внутренними `beta_initializer` общий внешний `beta0`
необходимо отключить. Два готовых набора моделей запускаются так:

```bash
python run_benchmarks.py compare \
  --models examples/initial_direction_models.py \
  --use-model-initializers \
  --experiments 2:6 \
  --profile smoke \
  --seeds 0:4 \
  --jobs 1 \
  --output benchmark_outputs/initial_direction

python run_benchmarks.py compare \
  --models examples/initial_bandwidth_models.py \
  --use-model-initializers \
  --experiments 2:6 \
  --profile smoke \
  --seeds 0:4 \
  --jobs 1 \
  --output benchmark_outputs/initial_bandwidth
```

Для отдельного сравнения контрольного направления с малыми значениями
`ridge_eta` используйте готовый набор из четырёх режимов:

```bash
python run_benchmarks.py compare \
  --models examples/ridge_eta_comparison_models.py \
  --use-model-initializers \
  --experiments 2:6 \
  --profile smoke \
  --seeds 0:25 \
  --jobs 9 \
  --output benchmark_outputs/ridge_eta_comparison
```

Первым baseline идёт обычный контрольный режим `e1_control`, затем
`ridge_eta_1e-4`, `ridge_eta_1e-5` и `ridge_eta_1e-6`.

Для отдельной ручной сетки эксперимента 2 доступны `--d` и `--n-over-d`.
Обе опции обязательны вместе и принимают списки через запятую. Например,
сравнение `e1_control`, `ridge_eta_1e-6` и `ridge_eta_1e-7` запускается так:

```bash
python run_benchmarks.py compare \
  --models examples/small_ridge_comparison_models.py \
  --use-model-initializers \
  --experiments 2 \
  --d 5,10 \
  --n-over-d 5,10 \
  --seeds 0:25 \
  --jobs 9 \
  --output benchmark_outputs/small_ridge_comparison
```

Ручная сетка использует генератор и параметры эксперимента 2. Поэтому
результаты получают те же таблицы и отдельные графики времени, памяти и
абсолютного косинуса для каждого `d`.

Финальное двухмодельное сравнение `e1_control` против `ridge_eta_1e-6`
использует отдельный конфиг:

```bash
python run_benchmarks.py compare \
  --models examples/final_ridge_comparison_models.py \
  --use-model-initializers \
  --experiments 2 \
  --d 5,10 \
  --n-over-d 5,10 \
  --seeds 0:25 \
  --jobs 9 \
  --output benchmark_outputs/final_ridge_comparison
```

В этом режиме `X`, `y`, центры и случайные направления остаются одинаковыми,
но каждая модель сама строит `beta_ref`. `runs.csv` дополнительно сохраняет
`beta_initializer`, `bandwidth_selector`, `beta_ref_encoded`,
`beta_hat0_encoded`, начальный `h` и квантильные диагностики локальной массы.
В готовых файлах `min_neighbors=4`, чтобы условие было достижимо даже на
минимальном smoke-наборе с `n=8`; для основной серии значение меняется в
функции `_config()` обоих файлов.

`--experiments` принимает список (`2,4,6`) или включительный диапазон
 (`2:6`). Для каждого выбранного эксперимента используются его штатные
 `smoke_parameter_grid()` или `full_parameter_grid()`:

- 2 — размерность и отношение `n/d`;
- 3 — уровень шума `sigma_eps`;
- 4 — корреляция признаков `rho_corr`;
- 5 — масштаб признаков `sigma_x`;
- 6 — функция связи `link`.

`runs.csv`, `model_summary.csv`, `comparisons.csv` и
`comparison_summary.csv` содержат номер эксперимента и полный набор параметров.
При выборе нескольких экспериментов heatmap сохраняются раздельно в
`plots/experiment_2/`, ..., `plots/experiment_6/`.

Результат содержит:

- `runs.csv` — все запуски, метрики времени/памяти/качества и ошибки;
- `model_summary.csv` — агрегаты отдельно для каждой модели;
- `comparisons.csv` — каждый candidate против baseline на том же seed;
- `comparison_summary.csv` — агрегированные speedup, memory ratio и
  эквивалентность;
- `manifest.json` — команда, модели и параметры серии;
- `plots/` — для каждого `d` отдельные графики runtime, peak RSS и
  медианного `|cos|` по `n/d`, а также heatmap каждого candidate. При
  нескольких экспериментах эти графики лежат в `plots/experiment_<id>/`.

Падение одной реализации записывается в `runs.csv`, остальные запуски
продолжаются; после сохранения артефактов CLI возвращает ненулевой код. Разные
численные ответы по умолчанию допустимы. Флаг `--require-equivalent` делает
расхождение с baseline ошибкой команды.

Программный API сравнения двух ADP-совместимых моделей остаётся доступен:

```python
from experiments.compare_model_efficiency import (
    compare_models,
    write_comparison_artifacts,
)

runs = compare_models(
    first_model,
    second_model,
    model_names=("first", "second"),
    seeds=(0, 1, 2),
    jobs=4,
)
write_comparison_artifacts(
    runs,
    "benchmark_outputs/model_comparison",
    model_names=("first", "second"),
)
```

Модели должны поддерживать вызов
`fit(X, y, centers=..., beta0=..., directions=...)` и сериализацию через
`cloudpickle`. Каждый `fit` выполняется в новом процессе на одинаковых данных,
центрах, начальном направлении и случайных направлениях. Время запуска процесса
и сериализации не входит в `fit_time_sec`; RSS измеряется только во время
`fit`. `jobs` задаёт число параллельных пар: внутри каждой пары модели
запускаются строго последовательно в порядке AB или BA, но каждый `fit` остаётся
в отдельном свежем процессе. На системах с process affinity за активной парой
закрепляется одно разрешённое CPU-ядро; назначение и фактическая affinity
сохраняются в `runs.csv`. `tqdm` показывает число завершённых запусков. Для
измерения latency без внутренней конкуренции benchmark-процессов используйте
`jobs=1`; большие значения измеряют пропускную способность при параллельной
нагрузке. Таблицы сохраняют как запрошенное, так и фактическое `n/d` после
округления `n`.

`paired.csv` содержит sign-invariant ошибку направления, ошибку проектора,
разность objective и итоговый флаг `numerically_equivalent`. Встроенный CLI
является self-check двух независимых экземпляров корректной реализации
`random_projection` и завершится с ненулевым кодом при расхождении результатов:

```bash
python experiments/compare_model_efficiency.py \
  --profile full \
  --seeds 0:99 \
  --jobs 4 \
  --output benchmark_outputs/model_comparison
```

Запускаемый пример честного сравнения matrix-free CG и плотного direct solver
на одинаковых данных, начальном `beta`, центрах и направлениях:

```bash
python examples/compare_adp_solvers.py --n 120 --d 8
```
