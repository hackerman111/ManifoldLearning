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

Сравнение времени и памяти двух ADP-совместимых моделей на сетке эксперимента 2
вынесено в отдельный модуль:

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
