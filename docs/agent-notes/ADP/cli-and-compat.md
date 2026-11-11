# CLI, experiments и legacy/compatibility paths

## Два CLI-сценария

### Синтетический runner

`python -m ADP.cli` проходит через `ADP/cli/__main__.py` к `cli.main.main`; прямой `python -m ADP.cli.main` также поддержан. CLI генерирует reproducible X, истинный basis и отклик с выбранной link/noise либо принимает внедренные данные из внутренних функций. Параметр mode переключает single, multi, manifold; single/multi используют общий `fit_index`, manifold — `ADP_Manifold`. Выход печатает качество, итоговые настройки, stop reason и профиль времени/памяти.

CLI валидирует комбинации до запуска: single требует `index_dim=1`, для multi `index_dim<d`; hybrid для single запрещен; `gpu-solver` разрешен только с LSMR; GPU для manifold запрещен. Manifold принимает solver cg/hybrid; при выборе lsmr эта ветка фактически создает модель с cg. Источники: **SRC-CLI-PARSER**, **SRC-CLI-CHECKS**, **SRC-CLI-MANIFOLD**.

### Experiment wrapper

`python -m ADP.cli.experiment` перечисляет каталог и передает управление верхнеуровневому package `benchmarks/experiments/` (models/registry/data/runner/single/multi/manifold). Сам ADP-модуль задает CLI аргументы, собирает `Build`, выбирает profile/runs/seed/output-dir и получает код завершения с учетом численных failures. Не переносите логику каталогов или генерации экспериментов в этот wrapper: определения находятся вне `ADP/`. Источник: **SRC-CLI-EXP-WRAPPER**.

`experiment_utils.py` централизует проверки точек/конфигураций и метрик. `experiment_plots.py` читает `runs.csv`/manifest, сохраняет summary, trace summary, failure/phase/boundary CSV и Markdown; когда plots включены — формирует PNG для качества, runtime, memory, failures, trajectories и paired deltas. Recovery single/multi определяется качеством общего подпространства; convergence считается отдельно только для диагностики. Manifold дополнительно проходит порог максимальной ошибки локального projector по всем центрам. Numerical failures остаются отдельным исходом. Источники: **SRC-CLI-EXP-VALIDATION**, **SRC-CLI-REPORT**, **SRC-CLI-RECOVERY**.

`python -m experiments.diagnostic` — отдельный парный benchmark поверх `Experiment.run`: базовая точка и локальные варианты параметров на общих seed, сценарии base/noise/correlation/scarce, отдельные seed для отбора и проверки. Выходы: `run.json`, `diagnostics.{csv,json,md}` и обычные `series/*/{series.json,runs.csv,...}`; `--analyze` пересобирает отчёт по сохранённым series без повторного fit. Выбор параметра является предварительным и относится только к проверенному сценарию. При ошибке fit общий `benchmarks/experiments/runner.py` записывает его wall-clock время; traced peak остаётся пустым, если fit не вернул профиль. Источник: `benchmarks/experiments/diagnostic.py`, `benchmarks/experiments/runner.py:run_experiment`.

`python -m experiments.r_sdr` — отдельный парный benchmark ADP с обычным MAVE,
пакетным MEANMAVE, классическим SIR, dMAVE и refined MAVE на R. Требует минимум
10 повторов, сохраняет исходники/протокол, данные и результаты, поддерживает
возобновление и отчёты. Реализации, ограничения и маршруты:
[r-sdr-comparison.md](r-sdr-comparison.md).

## Автономный экспорт (2026-10-23)

Исторически `export/README.md` описывал переносимый source snapshot multi-index/current Grassmann,
основного CLI, каталога multi, paired SVD/Grassmann benchmark и R SDR CLI.
`export/pyproject.toml` устанавливается отдельно; worker.R включён в wheel.
В исходном экспорте численные ADP-модули побайтно совпадали с live; только экспортированный
cli/main.py добавляет `--solver grassman` для multi с default20.
На момент экспорта оригинальный CLI выбора Grassmann не имел; при объединении
2026-10-26 добавлен явный выбор grassman без изменения default LSMR. SOURCE_MANIFEST.json фиксирует
снимок; автоматической синхронизации нет. VERIFY.md и benchmarks/results/verification/ хранят
автономные проверки (139 pass/2 inherited manifold xfail, 2/2 paired SVD/core,
20/20 actual R MEANMAVE/core fits, installed wheel smoke). Дополнительный
multiv2 исключён из export: original import падает на mi-spokoini-m2-n;
main multi/Spokoiny catalog сохранён. Root full pytest имеет 3 collection errors
из-за этого existing multiv2; original manifold CLI также имеет m=2/default
local_quadratic mismatch. Эти исходные проблемы не исправлялись при упаковке.

## Объединение с Vlad_EDR_M (2026-10-26)

Текущий CLI дополнительно принимает `--solver grassman` для single/multi;
бюджет этой явной опции — 20 без переопределения. Существующие LSMR/CG/HYBRID
и default LSMR с бюджетами 3/5 сохранены. Grassmann для manifold отвергается;
правила GPU остаются прежними. Для single CLI адаптирует вектор к матрице
ранга 1 и возвращает вектор и локальные скаляры из HPAOResult. Численный grassman.py из EDR_code не заменялся.
`ADP/core/solver_common.py` экспортирует тот же HPAOResult из LSMR для
перенесённых экспериментов, сохраняя isinstance-контракт движка.

`export/` в текущей рабочей копии отсутствует; исторические сведения раздела
выше не подтверждают наличие или актуальность переносимого комплекта сейчас.
Документы исходного M-экспорта сохранены отдельно в
[протоколе объединения](../../integration/vlad_edr_m_20261026/README.md).
Темы и пути новых исследований: [benchmarks/experiments/IMPORTED_FROM_M.md](../../../benchmarks/experiments/IMPORTED_FROM_M.md).

## Legacy и aliases

- `ADP/__init__.py` экспортирует публичные классы и helper-функции, а старые имена модулей подключает через `sys.modules` aliases (source: **SRC-API-ALIASES**).
- `ADP/engine/__init__.py` делает то же для common-функций/модулей и лениво разрешает исторические single/multi engine paths (source: **SRC-ENGINE-ALIASES**).
- `ADP/cli.py` — совместимый thin shim; актуальный модульный запуск — пакет `ADP.cli` (source: **SRC-CLI-COMPAT**, **SRC-CLI-MAIN-MODULE**).
- `ADP/core/ADP_Data.py` и класс `ADP_Solver` — старый stateful API, не основной facade моделей (source: **SRC-LEGACY-DATA**, **SRC-LEGACY-ORCH**).
- `ADP/engine/common/ADP_Statistic_engine.py` поддерживает плотные и sparse neighborhood inputs и имеет собственную CPU/CUDA реализацию. Текущий public single/multi путь по умолчанию подключает `engine.common.statistic.calculate_statistics`; при GPU использует `GPUStatistics` (source: **SRC-LEGACY-STAT-FLOW**, **SRC-LEGACY-GPU-STAT**, **SRC-IFIT-LOOP**, **SRC-STAT-GPU**).
- `ADP/engine/common/box_kernel.py` строит compact-support neighborhoods и sparse block objects; current `weights.py` не создает эти блоки и текущий `index_fit` не выбирает их по `smart_weights` (source: **SRC-BOX-REP**, **SRC-BOX-ENGINE**, **SRC-WEIGHT**, **SRC-IFIT-LOOP**).

Важно: в `ADP_Data.__post_init__` поле `n` присваивается `X.shape[1]`, тогда как остальные текущие алгоритмы трактуют X как `(n,d)` и число строк — n. Это live-наблюдение, не исправление; legacy consumer перед использованием требует проверки/теста. Кроме того, single-метрика `ADP_Data.Calculate_metric` знакочувствительна, в то время как current single result использует абсолютный cosine (source: **SRC-LEGACY-DATA**, **SRC-SI-RESULT**).

## Локаторы

| ID | Фрагмент | Команда sed |
|---|---|---|
| SRC-CLI-PARSER | аргументы synthetic runner | `rtk proxy sed -n '58,186p' ADP/cli/main.py` |
| SRC-CLI-CHECKS | сбор config и mode/solver/input validation | `rtk proxy sed -n '189,253p' ADP/cli/main.py` |
| SRC-CLI-DATA | synthetic sample and reproducible data generation | `rtk proxy sed -n '256,273p' ADP/cli/main.py` |
| SRC-CLI-INDEX | single/multi run, solver dispatch and shared fit | `rtk proxy sed -n '276,388p' ADP/cli/main.py` |
| SRC-CLI-MANIFOLD | manifold run and diagnostics | `rtk proxy sed -n '391,521p' ADP/cli/main.py` |
| SRC-CLI-OUTPUT | quality/report/profile and main | `rtk proxy sed -n '531,633p' ADP/cli/main.py` |
| SRC-CLI-EXP-WRAPPER | experiment parser, runner handoff and exit status | `rtk proxy sed -n '1,111p' ADP/cli/experiment.py` |
| SRC-CLI-EXP-VALIDATION | validation helpers and point/config invariants | `rtk proxy sed -n '1,348p' ADP/cli/experiment_utils.py` |
| SRC-CLI-REPORT | runs/manifest aggregation and report outputs | `rtk proxy sed -n '153,268p' ADP/cli/experiment_plots.py` |
| SRC-CLI-RECOVERY | convergence diagnostics, geometric recovery, failure classification | `rtk proxy sed -n '461,689p' ADP/cli/experiment_plots.py` |
| SRC-CLI-MAIN-MODULE | executable package entry point | `rtk proxy sed -n '1,5p' ADP/cli/__main__.py` |
| SRC-CLI-EXPORTS | lazy exports | `rtk proxy sed -n '1,22p' ADP/cli/__init__.py` |
| SRC-API-ALIASES | public API and old module aliases | `rtk proxy sed -n '1,68p' ADP/__init__.py` |
| SRC-ENGINE-ALIASES | engine facade and lazy old module aliases | `rtk proxy sed -n '1,82p' ADP/engine/__init__.py` |
| SRC-LEGACY-DATA | legacy state fields and metric behavior | `rtk proxy sed -n '9,55p' ADP/core/ADP_Data.py` |
| SRC-LEGACY-ORCH | legacy ADP_Solver.fit outer orchestration | `rtk proxy sed -n '99,179p' ADP/core/ADP_Solver.py` |
| SRC-LEGACY-STAT-FLOW | sparse/dense CPU statistics and sparse cache | `rtk proxy sed -n '1,140p' ADP/engine/common/ADP_Statistic_engine.py` |
| SRC-LEGACY-GPU-STAT | compatibility statistics CUDA path | `rtk proxy sed -n '141,239p' ADP/engine/common/ADP_Statistic_engine.py` |
| SRC-BOX-REP | box/plateau kernels, sparse block representation and cache | `rtk proxy sed -n '18,283p' ADP/engine/common/box_kernel.py` |
| SRC-BOX-ENGINE | KD-tree neighborhood builder and support logic | `rtk proxy sed -n '284,895p' ADP/engine/common/box_kernel.py` |
| SRC-BOX-SEARCH | sparse bandwidth/scale and initializer utilities | `rtk proxy sed -n '895,1100p' ADP/engine/common/box_kernel.py` |
| SRC-GPU-BACKEND | CuPy availability, backend dispatch and conversion | `rtk proxy sed -n '9,36p' ADP/gpu.py` |

Команда experiment CLI ссылается на top-level `benchmarks/experiments/`, не раскрытый в этом наборе заметок. Внешние source files, которые потребуется читать дальше: `benchmarks/experiments/models.py`, `benchmarks/experiments/registry.py`, `benchmarks/experiments/data.py`, `benchmarks/experiments/runner.py`, `benchmarks/experiments/single.py`, `benchmarks/experiments/multi.py`, `benchmarks/experiments/manifold.py`, `benchmarks/experiments/diagnostic.py`.

Полная карта ADP source paths находится в [README.md](README.md#каталог-исходников). Исторические алиасы — интерфейс совместимости, не доказательство, что файл и модуль имеют одинаковую реализацию.
