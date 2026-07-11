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

Новая серия запускается отдельной подкомандой. По умолчанию используются один
процесс и один worker статистик, поэтому вложенный параллелизм не включается:

```bash
python run_benchmarks.py single-index \
  --profile smoke \
  --jobs 1 \
  --statistics-workers 1 \
  --data-dir adp_D1_data \
  --output benchmark_outputs/single_index
```

Эксперименты D01–D04 читают `dataset_manifest.csv` и подготовленные файлы из
`adp_D1_data/prepared`. Перед запуском проверяются target, размерность и SHA-256.
Данные не копируются в каталог серии и не заменяются сетевой версией.

Два process workers или два NumPy workers включаются только явно. Обычно
следует увеличивать один уровень параллелизма за раз:

```bash
python run_benchmarks.py single-index --profile minimal --jobs 2
python run_benchmarks.py single-index --profile minimal --statistics-workers 2
```

Прерванную серию можно продолжить. Успешные `run_id` пропускаются, а failed jobs
повторяются только с явным `--retry-failed` и заменяют прежний commit-marker:

```bash
python run_benchmarks.py single-index \
  --resume benchmark_outputs/single_index/<series_id>

python run_benchmarks.py single-index \
  --resume benchmark_outputs/single_index/<series_id> \
  --retry-failed
```

Каталог серии содержит нормализованные
`single_index_series.csv`, `single_index_runs.csv`,
`single_index_iterations.csv`, `single_index_initial_parameters.csv`,
`single_index_summary.csv`, `single_index_failures.csv` и
`single_index_artifacts.csv`. Численные отчёты и графики G01–G21 заново строятся
только из этих CSV; JSON для новой серии не создаётся.

Запускаемый пример честного сравнения matrix-free CG и плотного direct solver
на одинаковых данных, начальном `beta`, центрах и направлениях:

```bash
python examples/compare_adp_solvers.py --n 120 --d 8
```
