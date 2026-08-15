# Unified ADP CLI для single-index, multi-index и сравнительных экспериментов

## Цель

Расширить `ADP/cli.py`, сохранив ручной single-index режим и добавив:

- multi-index режим;
- выбор встроенного solver-а;
- несколько повторов одного запуска;
- загрузку Python-файла с объектом `ADP_Experiment`;
- запуск одной или двух конфигураций на одинаковых данных и случайности;
- возобновление прерванной серии;
- архив исходных данных, результатов и main-compatible CSV;
- построение применимых графиков из manifest ветки `main`;
- общий и внутренний progress через `tqdm`.

Система не назначает автоматического победителя. Она сохраняет исходные
парные наблюдения, разности метрик и агрегированные сравнения.

## Публичные типы

Общие experiment-типы размещаются в корне пакета `ADP`, а не внутри
`single_index` или `multi_index`.

```python
@dataclass(frozen=True, slots=True)
class ADP_ExperimentPoint:
    name: str
    n: int
    d: int
    noise: float = 0.05
    metadata: Mapping[str, str | int | float | bool | None] = field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True)
class ADP_ExperimentVariant:
    config: ADP_Config
    solver: Literal["auto", "lsmr", "varpro"] = "auto"
    solver_settings: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ADP_Experiment:
    name: str
    mode: Literal["single", "multi"]
    points: tuple[ADP_ExperimentPoint, ...]
    variants: Mapping[str, ADP_ExperimentVariant]
    runs: int = 1
    seed: int = 7
    index_dim: int = 1
    data_factory: Callable[
        [ADP_ExperimentPoint, np.random.Generator], ADP_Data
    ] | None = None
```

Публичные классы экспортируются из `ADP.__init__`. `ADP_Data.true_index`
принимает `np.ndarray | None`: отсутствие ground truth не является ошибкой,
но quality-графики для таких данных пропускаются.

Experiment-файл является обычным доверенным Python-модулем и обязан
экспортировать объект:

```python
from ADP import (
    ADP_Config,
    ADP_Experiment,
    ADP_ExperimentPoint,
    ADP_ExperimentVariant,
)

experiment = ADP_Experiment(
    name="solver_compare",
    mode="single",
    runs=20,
    seed=42,
    points=(
        ADP_ExperimentPoint(
            name="noise_005",
            n=500,
            d=10,
            noise=0.05,
            metadata={"sigma_eps": 0.05},
        ),
        ADP_ExperimentPoint(
            name="noise_020",
            n=500,
            d=10,
            noise=0.20,
            metadata={"sigma_eps": 0.20},
        ),
    ),
    variants={
        "lsmr": ADP_ExperimentVariant(
            config=ADP_Config(),
            solver="lsmr",
            solver_settings={"max_steps": 5},
        ),
        "varpro": ADP_ExperimentVariant(
            config=ADP_Config(),
            solver="varpro",
        ),
    },
)
```

В репозиторий добавляется минимальный готовый пример с двумя точками и двумя
solver-ами.

## Валидация experiment-файла

Loader проверяет до создания каталога серии:

- экспорт `experiment` и его тип;
- непустое безопасное имя experiment-а;
- уникальные непустые имена точек и вариантов;
- одну или две конфигурации;
- `runs >= 1`, `seed >= 0`, `n > d + 1`, конечный неотрицательный `noise`;
- JSON-совместимые скалярные значения metadata;
- `index_dim == 1` для single и `1 <= index_dim < d` для каждой multi-точки;
- допустимые имена solver-ов;
- запрет `varpro` в multi-index режиме;
- отсутствие неизвестных solver settings через фактическую сигнатуру solver-а.

Параметры модели не исправляются автоматически. Например, `N_J > n`
завершает соответствующий job технической ошибкой, которая архивируется; все
следующие jobs продолжают выполняться.

## Генерация данных и paired randomness

Для каждого сочетания `point x seed` данные создаются ровно один раз. Все
точки используют одинаковую последовательность seed:

```text
experiment.seed, experiment.seed + 1, ..., experiment.seed + runs - 1
```

Оба варианта получают одни и те же сохранённые `X`, `Y`, `true_index` и один
и тот же effective `ADP_Config.seed`. Runner создаёт effective config через
`dataclasses.replace(variant.config, seed=seed)` и не изменяет остальные поля.

Если `data_factory` не задан, используется встроенная генерация:

- `X` — независимые стандартные Gaussian-признаки;
- single-index: нормированный Gaussian `beta_true`,
  `Y = sin(X @ beta_true) + noise * rng.normal(size=n)`;
- multi-index: QR-ортонормированный Gaussian `basis_true`,
  `Y = sum(sin(X @ basis_true), axis=1) + noise * rng.normal(size=n)`.

Пользовательская функция имеет единый контракт
`data_factory(point, rng) -> ADP_Data`; отдельные factories на точках не
поддерживаются. Factory может использовать `point.metadata` как параметры
распределения. Возвращённые формы, конечность массивов и согласованность `n,d`
проверяются до fit. `true_index=None` разрешён.

## Solver-ы

Новая registry/factory-иерархия не вводится. Runner напрямую сопоставляет
имена с существующими функциями `LSMR.solve` и `VarPro.solve`, после чего
создаёт существующий `ADP_solver(method, **solver_settings)`.

- `auto` использует текущий default соответствующей модели; на момент
  утверждения spec это LSMR и для single-index, и для multi-index;
- `auto` без дополнительных settings передаёт модели `solver=None`, а при
  наличии settings накладывает их поверх settings текущего model solver;
- `lsmr` доступен в обоих режимах;
- `varpro` доступен только в single-index;
- explicit solver settings принадлежат варианту и одинаковы для всех точек и
  seed этого варианта.

## План jobs и порядок выполнения

Job идентифицируется тройкой `point_name, seed, variant_name`. Для одной или
двух конфигураций runner создаёт соответственно `points * runs` или
`points * runs * 2` jobs.

Выполнение строго последовательное. При двух вариантах порядок чередуется по
индексу seed:

```text
run 0: A -> B
run 1: B -> A
run 2: A -> B
```

Это уменьшает систематический эффект прогрева. Параллельный `jobs>1` не
поддерживается, потому что конкуренция за CPU/RAM искажает time/RSS-сравнение.
На время fit native BLAS/OpenMP pools ограничиваются одним потоком.

## Progress

Runner использует два уровня `tqdm`:

- верхний bar — завершённые jobs из общего числа;
- вложенный bar — внешние итерации текущего fit с текущими `h` и `rho` либо
  `alpha` в postfix.

Обе модели получают необязательный progress callback в `fit`; без callback
обычный библиотечный вызов остаётся тихим и сохраняет обратную совместимость.
В non-TTY runner печатает одну короткую строку после каждого job.

## Telemetry и статусы

Для каждого job архивируются:

- wall time только model fit;
- существующий `model.profile_`: суммарное время стадий и `tracemalloc`;
- sampled RSS текущего процесса через `psutil`: start/min/mean/max,
  peak delta, число samples и source;
- число внешних итераций, stop reason и доступная solver diagnostics;
- cosine для single-index при наличии ground truth;
- нормированная projector error для single- и multi-index при наличии ground
  truth.

RSS sampler делает обязательные start/end samples и периодические samples во
время fit. Data generation, сериализация и построение графиков не входят во
время алгоритма.

Статус отделён от качества:

- `success` — fit завершился через `h_min` или `local_mass_limit`;
- `nonconverged` — исчерпан явный iteration limit;
- `numerical_failure` — data/model/solver exception или невалидный результат.

Низкое качество само по себе не меняет технический статус. Исключение одного
варианта сохраняется вместе с traceback, второй вариант и следующие jobs
продолжаются. Наличие `nonconverged` или `numerical_failure` даёт итоговый exit
code `1`.

## Хранилище и resume

Новая серия создаётся по пути:

```text
<output-dir>/<experiment.name>/<timestamp>/
```

Внутри используются:

```text
data/<point>/seed_<seed>.npz
commits/<run_id>.json
models/<run_id>.npz
run_summary.csv
outer_iterations.csv
inner_iterations.csv
local_diagnostics.csv
solver_iterations.csv
paired_comparison.csv
comparison_summary.csv
series.csv
artifacts.csv
plots/points/<point>/<plot>.png
plots/summary/<plot>.png
```

`X`, `Y` и `true_index` атомарно записываются в compressed NPZ до запуска
первого варианта. Если ground truth отсутствует, NPZ содержит явный marker,
а не object array.

После каждого job его полный результат сначала записывается во временный
commit-файл и публикуется через `os.replace`. Public CSV пересобираются из
commit-файлов, поэтому оборванная запись не считается завершённым job.
`--resume PATH` загружает уже сохранённые data NPZ, сверяет спецификацию
завершённых jobs с текущим experiment-объектом и пропускает совпавшие jobs.
Изменение config или solver settings под тем же именем варианта считается
ошибкой resume.

По умолчанию `models/<run_id>.npz` содержит финальные `beta`/`basis`,
`eigenvalues` и `coefficients`. Флаг `--no-save-models` отключает только эти
snapshots; input NPZ и CSV остаются обязательными.

По явному решению пользователя не сохраняются копия experiment-файла, его
hash, git metadata и версии окружения.

`Ctrl-C` закрывает progress bars, пересобирает partial CSV/графики, записывает
`series.status=partial` и возвращает exit code `130`. `--reports-only PATH`
пересобирает summary/artifacts/plots только из архива, не импортируя
experiment-файл и не выполняя fit.

## CSV-схема

Имена основных таблиц совместимы с evaluation-системой `main`.

`run_summary.csv` содержит:

- series/run identity, point, variant, mode, seed;
- metadata точки;
- requested и effective поля `ADP_Config`;
- solver name/settings;
- размеры данных и `index_dim`;
- quality, stop/status/error;
- fit/stage timings, RSS и `tracemalloc` metrics;
- количество строк детальной telemetry и ссылки на NPZ artifacts.

`outer_iterations.csv` использует один mode-aware trace:

- общие поля: `outer_k`, `h_k`, mean local mass, solver diagnostics;
- single: `rho_k`, `beta_k`, cosine, projector error;
- multi: `alpha_k`, `basis_k`, eigenvalues, projector error.

`inner_iterations.csv`, `local_diagnostics.csv` и `solver_iterations.csv`
имеют стабильные заголовки из `main`, но строки создаются только при наличии
реальной telemetry. Aggregate solver counters не выдаются за per-iteration
историю.

`paired_comparison.csv` соединяет варианты строго по `point,seed`. Варианты
`A` и `B` определяются порядком insertion в mapping `variants`. Таблица хранит
исходные значения обеих сторон и signed `B - A` deltas отдельно для
`cosine_abs`, `projector_error`, fit/stage time, RSS и числа итераций. Если
одна сторона завершилась ошибкой, статусы сохраняются, а недоступные deltas
остаются пустыми.

`comparison_summary.csv` группирует paired rows по точке и содержит count,
median, q05 и q95. Поле winner отсутствует.

## Графики из `main`

Report layer переносит компактный `PlotSpec` manifest и визуальные функции из:

- `main:adp/evaluation/single_index/reports.py`;
- `main:adp/evaluation/single_index/plots.py`;
- `main:adp/common/plotting.py`.

Сохраняются принятые в `main` правила: median и интервал 5-95%, box
25-75% с whiskers 5-95%, Wilson intervals для долей, heatmaps, stacked
runtime/status, ADP palette, русские подписи и log/log2/symlog scales.
`variant` добавляется в grouping каждого сравнительного графика.

Всегда при наличии соответствующих колонок рассматриваются диагностические
графики:

- `quality_vs_outer_iteration.png`;
- `projector_error_vs_outer_iteration.png`;
- `bandwidth_vs_outer_iteration.png`;
- `rho_vs_outer_iteration.png` или multi-аналог
  `alpha_vs_outer_iteration.png`;
- `beta_step_vs_outer_iteration.png` с mode-aware подписью шага basis;
- `objective_vs_outer_iteration.png`;
- `objective_vs_inner_iteration.png`;
- `beta_step_vs_inner_iteration.png`;
- `solver_residual_vs_iteration.png`;
- `local_mass_by_outer_iteration.png`;
- `effective_neighbors_by_outer_iteration.png`;
- `local_condition_by_outer_iteration.png`;
- `mass_vs_condition.png`;
- `local_slopes_by_outer_iteration.png`.

При наличии соответствующей metadata автоматически рассматриваются main
families:

- `d,n_over_d`: quality/projector heatmap, technical success heatmap, runtime,
  RSS memory и outer-iterations heatmap;
- `sigma_eps`: quality/projector error, technical success, runtime, outer
  iterations и final objective;
- `rho_corr`: quality/projector error, technical success, local condition,
  solver iterations и runtime;
- `sigma_x`: quality/projector error, initial/final bandwidth, local mass,
  runtime и bandwidth ratio;
- `link`: quality/projector error, technical success, outer iterations,
  objective и local slopes;
- `x_distribution`, `noise_distribution`: quality/projector error, failure
  rate и runtime;
- `heteroscedastic`: quality/projector error;
- `effective_outlier_fraction`: quality/projector error и failure rate;
- `delta`: quality/projector error и objective;
- общие: `runtime_breakdown.png`, `runtime_share_breakdown.png` и
  `status_breakdown.png`.

Точное соответствие metadata и исходных имён manifest:

| Metadata | Файлы из `main` |
|---|---|
| `d,n_over_d` | `quality_heatmap_d_nd_ratio.png`, `success_rate_heatmap.png`, `runtime_vs_dimension.png`, `memory_vs_dimension.png`, `iterations_heatmap_d_nd_ratio.png` |
| `sigma_eps` | `quality_vs_sigma_eps.png`, `success_rate_vs_sigma_eps.png`, `runtime_vs_sigma_eps.png`, `outer_iterations_vs_sigma_eps.png`, `final_objective_vs_sigma_eps.png` |
| `rho_corr` | `quality_vs_correlation.png`, `success_rate_vs_correlation.png`, `local_condition_vs_correlation.png`, `solver_iterations_vs_correlation.png`, `runtime_vs_correlation.png`, `singular_fraction_vs_correlation.png` |
| `sigma_x` | `quality_vs_sigma_x.png`, `h0_vs_sigma_x.png`, `final_bandwidth_vs_sigma_x.png`, `local_mass_vs_sigma_x.png`, `runtime_vs_sigma_x.png`, `bandwidth_ratio_vs_sigma_x.png` |
| `link` | `quality_by_link_function.png`, `success_rate_by_link_function.png`, `outer_iterations_by_link_function.png`, `objective_by_link_function.png`, `local_slopes_by_link_function.png` |
| `x_distribution` | `quality_by_x_distribution.png`, `failure_rate_by_distribution.png`, `runtime_by_distribution.png` |
| `noise_distribution` | `quality_by_noise_distribution.png`, `failure_rate_by_distribution.png`, `runtime_by_distribution.png` |
| `heteroscedastic` | `quality_by_heteroscedasticity.png` |
| `effective_outlier_fraction` | `quality_vs_outlier_fraction.png`, `failure_rate_vs_outliers.png` |
| `delta` | `quality_vs_model_misspecification.png`, `objective_vs_model_misspecification.png` |

В multi-index режиме каждый filename с token `quality` получает аналог с
token `projector_error`: например, `quality_vs_sigma_eps.png` становится
`projector_error_vs_sigma_eps.png`, а
`quality_heatmap_d_nd_ratio.png` становится
`projector_error_heatmap_d_nd_ratio.png`. Значение
`1 - projector_error` не создаётся. Single-only filename
`rho_vs_outer_iteration.png` заменяется на
`alpha_vs_outer_iteration.png`; неприменимые scalar-slope графики
пропускаются.

График `correctness_rate.png` из `main` не создаётся, потому что он превращает
quality threshold в success/failure, а согласованный контракт разделяет
технический статус и качество.

График создаётся только когда присутствуют все обязательные поля и хотя бы
одно конечное наблюдение. Для каждого created/skipped/error artifact в
`artifacts.csv` сохраняются filename, source tables, status и причина.

## CLI

Ручной режим сохраняет текущие параметры `ADP_Config` и добавляет:

```text
--mode single|multi
--index-dim M
--solver auto|lsmr|varpro
--solver-tol FLOAT
--solver-max-steps N
--runs N
--output-dir PATH
--no-save-models
--no-progress
```

Он создаёт внутри один `ADP_ExperimentPoint` и один variant, затем использует
тот же runner/storage/reporting path. Сравнение двух вариантов доступно через
experiment-файл.

Experiment-режим:

```text
--experiment-file PATH
--output-dir PATH
--resume PATH
--reports-only PATH
--dry-run
--no-save-models
--no-progress
```

При `--experiment-file` вычислительные параметры берутся только из файла;
обычные single-run флаги не переопределяют experiment. `--dry-run` импортирует
и валидирует объект, печатает точки, варианты и число jobs, но не создаёт
каталог. `--reports-only` взаимоисключён с запуском и не требует
experiment-файла.

Примеры:

```bash
python ADP/cli.py --mode multi --index-dim 2 --solver lsmr --runs 5
python ADP/cli.py --experiment-file ADP/examples/experiment_compare.py
python ADP/cli.py --experiment-file ADP/examples/experiment_compare.py \
  --resume ADP/experiment_outputs/solver_compare/20260810T120000
python ADP/cli.py --reports-only \
  ADP/experiment_outputs/solver_compare/20260810T120000
```

## Проверка

Один сфокусированный test-модуль проверяет:

- loader и ошибки схемы;
- single/multi manual CLI;
- одинаковые input arrays и model seed у двух вариантов;
- одинаковую seed-последовательность на всех точках;
- чередование порядка вариантов;
- raw NPZ, optional model NPZ и main-compatible CSV;
- paired rows/deltas без winner;
- технический статус отдельно от quality;
- продолжение после одного failed job;
- resume без повторного fit;
- `--reports-only` и partial report;
- применимый single plot, multi projector/alpha plot и skipped artifacts;
- `--dry-run` без созданных файлов.

Финальная проверка включает compilation, текущие multi-index/statistics tests,
новый experiment test и реальные малые CLI smoke для single, multi и paired
experiment-файла.

## Не входит в изменение

- параллельное выполнение jobs;
- TOML/JSON experiment-файлы;
- пользовательские solver-ы через `module:function`;
- автоматическое исправление невалидных configs;
- автоматический выбор победителя или единый composite quality score;
- восстановление полного evaluation framework и каталога сценариев из `main`;
- архивирование исходника experiment-файла, git state и environment versions.
