# Каталог single-index экспериментов для uppercase ADP

## Цель

Добавить в `ADP/cli.py` выбор экспериментов из каталога `main` и перенести
сценарии 1–8.3, генерацию синтетических данных, CSV-отчётность и графики в
отдельный пакет uppercase `ADP/single_index/experiments`.

Текущий ручной сценарий с параметрами `--n`, `--d` и `--noise` сохраняется как
эксперимент `custom`.

## Структура пакета

```text
ADP/single_index/experiments/
├── __init__.py
├── base.py
├── types.py
├── registry.py
├── data.py
├── reports.py
├── plots.py
├── scenario_custom.py
├── scenario_1.py
├── scenario_2.py
├── scenario_3.py
├── scenario_4.py
├── scenario_5.py
├── scenario_6.py
├── scenario_7_1.py
├── scenario_7_2.py
├── scenario_8_1.py
├── scenario_8_2.py
└── scenario_8_3.py
```

`base.py` содержит общий `BaseScenario`: разворачивание parameter grid по seed,
запуск модели, преобразование `ADP_single_index_result` в строки CSV и
продолжение серии после ошибки отдельного запуска.

Каждый `scenario_*.py` экспортирует класс с именем `Scenario`. Класс содержит
selector, русское название, smoke/full parameter grid и список графиков этого
эксперимента. `registry.py` импортирует классы под уникальными alias и является
единственным источником выбора по selector.

`types.py` содержит immutable `ExperimentParameters`, `SeedBundle` и описание
одного задания. `data.py` генерирует готовые `X`, `Y` и `beta_true`; модель
по-прежнему получает только `fit(X, Y)` и не отвечает за генерацию данных.

`reports.py` содержит `PlotSpec`, plot manifest и CSV-driven агрегацию из
`main:adp/evaluation/single_index/reports.py`. `plots.py` содержит общие
Matplotlib-функции и ADP-оформление из
`main:adp/evaluation/single_index/plots.py` и `main:adp/common/plotting.py`.

Существующий `ADP/single_index/scenario.py` перестаёт содержать собственные
CSV/plot helpers и остаётся compatibility import класса из
`experiments/scenario_custom.py`.

## CLI

Добавляются аргументы:

- `--experiment custom|1|2|3|4|5|6|7.1|7.2|8.1|8.2|8.3|all`;
- допускается список через запятую, например `--experiment 1,4,8.2`;
- `--profile smoke|full`, по умолчанию `smoke`;
- `--runs N` остаётся числом seed для каждой точки parameter grid;
- `--output-dir PATH` остаётся корнем каталога серии;
- `--list-experiments` печатает selector и русское название без запуска.

`--experiment custom` используется по умолчанию и сохраняет текущее поведение.
Параметры `--n`, `--d`, `--noise` относятся только к `custom`. Аргументы
`ADP_Config` применяются ко всем выбранным экспериментам.

Примеры:

```bash
python ADP/cli.py --experiment 3 --profile smoke --runs 10
python ADP/cli.py --experiment 1,4,8.2 --profile full --runs 100
python ADP/cli.py --experiment all --profile smoke
python ADP/cli.py --experiment custom --n 240 --d 3 --noise 0.05
```

## Эксперименты

Smoke-профиль содержит одну маленькую representative point на selector, как в
`main`. Full-профиль переносит буквальные независимые grids:

- `1`: корректность по `d ∈ {5,25}`, `n/d ∈ {5,10}` и link
  `linear|quadratic`, без шума;
- `2`: масштабирование по `d ∈ {5,25,50,100}` и
  `n/d ∈ {1,1.15,2,5,10}`;
- `3`: устойчивость к `sigma_eps ∈ {0,0.316,0.5,0.707,1,1.414,2}`;
- `4`: корреляция AR(1) `rho_corr ∈ {0,0.25,0.5,0.75,0.9,0.95}`;
- `5`: масштаб признаков `sigma_x ∈ {0.25,0.5,1,2,4}`;
- `6`: функции связи `linear|quadratic|square|sin|tanh|oscillating`;
- `7.1`: признаки `gaussian|uniform|student_t5`;
- `7.2`: шум `gaussian|student_t5|student_t3`;
- `8.1`: гетероскедастичность `False|True`;
- `8.2`: доля/масштаб выбросов `(0,1)`, `(0.01,5)`, `(0.01,10)`,
  `(0.05,5)`, `(0.05,10)`;
- `8.3`: misspecification `delta ∈ {0,0.1,0.25,0.5}`.

Если отдельно не указано обратное, эксперименты 3–6 используют
`d ∈ {25,100}`, `n/d ∈ {2,5,10}`, а 7–8 используют `d ∈ {25,100}` и
`n/d ∈ {2,5}`.

## Данные и воспроизводимость

Переносятся генераторы из `main`:

- независимые seed для beta, признаков, шума, инициализации, выбросов,
  misspecification и внутренних случайных направлений;
- Gaussian AR(1), uniform и Student-t признаки;
- Gaussian и Student-t шум;
- стандартизированные функции связи;
- гетероскедастичный шум, замена части наблюдений выбросами и дополнительное
  направление `gamma`, ортогональное `beta`;
- масштабная нормализация link для эксперимента 5.

Идентичная точка grid и seed создаёт идентичные `X`, `Y`, `beta_true` независимо
от порядка выбора экспериментов.

## Совместимость размеров с uppercase моделью

Full-grid эксперимента 2 содержит `n/d=1`, тогда как локальная инициализация
uppercase модели требует `n>d+1`. Для таких точек runner выбирает
`index_init=random`, ограничивает effective `N_loc`, `N_lin`, `N_J` размером
выборки и записывает одновременно requested/effective параметры. Для обычных
точек сохраняется выбранный пользователем `index_init`.

Автоматическая адаптация не скрывает значения: обе версии параметров попадают
в `runs.csv`. Численная ошибка одной точки записывается как `status=error`, но
не останавливает остальные задания.

## CSV

Серия создаёт публичные таблицы:

- `runs.csv`: selector, grid parameters, seed bundle, requested/effective ADP
  config, итоговые cosine, stop reason, время, память, status и error;
- `outer_iterations.csv`: `outer_k`, `h_k`, `rho_k`, beta, `cosine_abs`,
  `beta_delta`, LSMR-метрики и средняя локальная масса;
- `inner_iterations.csv`, `local_diagnostics.csv`, `solver_iterations.csv`:
  стабильные заголовки для совместимости manifest; доступные значения
  записываются, недоступные для текущей модели остаются пустыми;
- `series.csv`: выбранные selectors/profile, число requested/completed/error
  jobs и временные метки;
- `artifacts.csv`: пути ко всем CSV и PNG.

Строки задания сохраняются после каждого завершённого fit. Графики строятся
исключительно из сохранённых CSV.

## Графики

Из `main` переносятся plot manifest, русские подписи, масштабы осей, facet/group
правила, median/quantile bands, Wilson intervals, boxplot, scatter, heatmap и
stacked runtime. Файлы раскладываются по `plots/experiment_<selector>` и
`plots/summary`.

Для метрик, которых текущий `ADP_single_index_result` не предоставляет
(`objective`, локальная condition, подробные local slopes и некоторые
per-iteration timings), manifest сохраняется, а график явно показывает
`нет наблюдений`. Значения не восстанавливаются эвристически и не подменяются
другими метриками. Доступные графики строятся полностью.

## Проверка

1. Registry test подтверждает selectors, порядок, уникальность и точное число
   smoke/full grid points относительно `main`.
2. Data tests подтверждают детерминизм, AR(1), распределения, выбросы,
   гетероскедастичность и ортогональность gamma.
3. Report tests строят по synthetic CSV все виды графиков, включая no-data.
4. CLI test проверяет `--list-experiments`, одиночный selector, список и `all`.
5. Реальный `--experiment all --profile smoke --runs 1` проверяет CSV и
   автосоздание графиков; технический status анализируется отдельно от качества.
6. После focused tests запускаются полный доступный `pytest -q`, compilation и
   инспекция артефактов smoke-серии.
