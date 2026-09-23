# Карта кодовой базы ADP

Эти заметки описывают текущее содержимое пакета `ADP/`, чтобы агент мог сначала прочитать относящийся к задаче фрагмент, а не повторно сканировать весь репозиторий. Источник истины для поведения — live-код; заметки не утверждают эквивалентность формулам `tex/` или старым веткам. Ветка и файлы могут быть изменены пользователем, поэтому перед математическими правками перепроверьте соответствующие локаторы.

## Как читать карту источников

Каждый ID вида **SRC-IFIT** — устойчивый код фрагмента. Его команда ниже выводит исходник без преобразования. Для дословного повторного чтения используйте диапазон из таблицы; в тематических заметках даны более узкие диапазоны по функциям.

    rtk proxy sed -n '174,540p' ADP/engine/common/index_fit.py

Команда рассчитана на запуск из корня репозитория. RTK обязателен по локальным инструкциям. Если нужна другая часть файла, используйте тот же путь и собственный диапазон sed.

## Входные точки и маршрут вычислений

Текущий публичный сценарий single/multi (source: **SRC-SI**, **SRC-MI**, **SRC-IFIT**):

    ADP_single_index.fit / ADP_multi_index.fit
      -> engine.common.index_fit.fit_index
      -> инициализация -> центры/расстояния -> поиск h -> направления
      -> веса -> локальные статистики I/U -> solver HPAO
      -> новый индекс -> trace/выбор best или last

Single и multi отличаются геометрией локализации и формой индекса, но разделяют управляющий цикл. В single индекс внутри — вектор `(d,)`; в multi внутри solver — ортонормированные строки `(m,d)`, а публичный `basis_` имеет форму `(d,m)`. Главный путь по умолчанию использует HPAO-LSMR; `ADP/cli/main.py` отдельно позволяет выбрать CG или HYBRID.

Manifold — другая модель, с отдельной оркестрацией: сначала полные локальные градиенты, затем граф между центрами и изменяющиеся по центрам проекторы `(J,m,d)`. После синхронизирующих шагов уменьшается bandwidth функции и переоцениваются локальные подпространства (source: **SRC-MAN**, **SRC-MAN-FIT**, **SRC-MAN-GRAPH**, **SRC-MAN-OPT**).

## Каталог исходников

Полный диапазон даёт агенту возможность получить модуль целиком. В описаниях указаны роли, а не гарантия, что каждый путь подключён к каждому публичному API.

| ID | Файл и роль | Команда извлечения |
|---|---|---|
| SRC-PKG | `ADP/__init__.py` — публичные экспорты и исторические import aliases | `rtk proxy sed -n '1,68p' ADP/__init__.py` |
| SRC-GPU | `ADP/gpu.py` — ленивое подключение CuPy, выбор backend и возврат в NumPy | `rtk proxy sed -n '1,36p' ADP/gpu.py` |
| SRC-CLI-COMPAT | `ADP/cli.py` — совместимый CLI shim | `rtk proxy sed -n '1,8p' ADP/cli.py` |
| SRC-CFG | `ADP/core/ADP_Config.py` — kernel и конфигурация single/multi | `rtk proxy sed -n '1,112p' ADP/core/ADP_Config.py` |
| SRC-DATA | `ADP/core/ADP_Data.py` — прежний контейнер данных/состояния | `rtk proxy sed -n '1,55p' ADP/core/ADP_Data.py` |
| SRC-SOLVER-API | `ADP/core/ADP_Solver.py` — адаптер актуального solver и старый класс solver | `rtk proxy sed -n '1,179p' ADP/core/ADP_Solver.py` |
| SRC-STAT-RECORD | `ADP/core/ADP_Statistic.py` — запись локальных статистик | `rtk proxy sed -n '1,37p' ADP/core/ADP_Statistic.py` |
| SRC-CORE-INIT | `ADP/core/__init__.py` — пакетный init | `rtk proxy sed -n '1,1p' ADP/core/__init__.py` |
| SRC-SI | `ADP/core/single/ADP_single_index.py` — публичный single fit/transform state | `rtk proxy sed -n '1,93p' ADP/core/single/ADP_single_index.py` |
| SRC-SI-RESULT | `ADP/core/single/ADP_single_index_result.py` — результат и sign-invariant cosine | `rtk proxy sed -n '1,43p' ADP/core/single/ADP_single_index_result.py` |
| SRC-SI-UTIL | `ADP/core/single/ADP_single_index_utils.py` — валидация single API | `rtk proxy sed -n '1,60p' ADP/core/single/ADP_single_index_utils.py` |
| SRC-SI-INIT | `ADP/core/single/__init__.py` — экспорты single | `rtk proxy sed -n '1,4p' ADP/core/single/__init__.py` |
| SRC-MI | `ADP/core/multi/ADP_multi_index.py` — публичный multi fit/transform state | `rtk proxy sed -n '1,104p' ADP/core/multi/ADP_multi_index.py` |
| SRC-MI-RESULT | `ADP/core/multi/ADP_multi_index_result.py` — результат и расстояние подпространств | `rtk proxy sed -n '1,51p' ADP/core/multi/ADP_multi_index_result.py` |
| SRC-MI-UTIL | `ADP/core/multi/ADP_multi_index_utils.py` — валидация multi API/базиса | `rtk proxy sed -n '1,95p' ADP/core/multi/ADP_multi_index_utils.py` |
| SRC-MI-INIT | `ADP/core/multi/__init__.py` — экспорты multi | `rtk proxy sed -n '1,4p' ADP/core/multi/__init__.py` |
| SRC-MAN | `ADP/core/manifold/ADP_Manifold.py` — фасад manifold fit/predict и совместимые hooks | `rtk proxy sed -n '1,424p' ADP/core/manifold/ADP_Manifold.py` |
| SRC-MAN-RESULT | `ADP/core/manifold/ADP_Manifold_result.py` — result record manifold | `rtk proxy sed -n '1,27p' ADP/core/manifold/ADP_Manifold_result.py` |
| SRC-MAN-UTIL | `ADP/core/manifold/ADP_Manifold_utils.py` — конфигурация, формы и проверки инвариантов | `rtk proxy sed -n '1,269p' ADP/core/manifold/ADP_Manifold_utils.py` |
| SRC-MAN-INIT | `ADP/core/manifold/__init__.py` — экспорты manifold | `rtk proxy sed -n '1,4p' ADP/core/manifold/__init__.py` |
| SRC-ENGINE-INIT | `ADP/engine/__init__.py` — фасад и aliases engine | `rtk proxy sed -n '1,82p' ADP/engine/__init__.py` |
| SRC-LEGACY-STAT | `ADP/engine/common/ADP_Statistic_engine.py` — dense/sparse CPU/GPU совместимая статистика | `rtk proxy sed -n '1,330p' ADP/engine/common/ADP_Statistic_engine.py` |
| SRC-COMMON-INIT | `ADP/engine/common/__init__.py` — экспорты common | `rtk proxy sed -n '1,8p' ADP/engine/common/__init__.py` |
| SRC-BOX | `ADP/engine/common/box_kernel.py` — отдельный движок компактных sparse neighborhoods | `rtk proxy sed -n '1,1100p' ADP/engine/common/box_kernel.py` |
| SRC-CALC | `ADP/engine/common/calculus.py` — расстояния, h/rho/alpha, random directions | `rtk proxy sed -n '1,434p' ADP/engine/common/calculus.py` |
| SRC-GPU-STAT | `ADP/engine/common/gpu_statistics.py` — GPU-расчёт статистик общего fit_index | `rtk proxy sed -n '1,167p' ADP/engine/common/gpu_statistics.py` |
| SRC-IFIT | `ADP/engine/common/index_fit.py` — общий контроллер single/multi | `rtk proxy sed -n '1,540p' ADP/engine/common/index_fit.py` |
| SRC-INIT | `ADP/engine/common/initialize.py` — локальный gradient-PCA, pilot, random init | `rtk proxy sed -n '1,424p' ADP/engine/common/initialize.py` |
| SRC-LOGGER | `ADP/engine/common/logger.py` — старый tracker и IndexProfiler | `rtk proxy sed -n '1,194p' ADP/engine/common/logger.py` |
| SRC-STAT | `ADP/engine/common/statistic.py` — актуальные normalized/legacy CPU-моменты | `rtk proxy sed -n '1,229p' ADP/engine/common/statistic.py` |
| SRC-UTIL | `ADP/engine/common/utils.py` — проверка входов, форм и weight blocks | `rtk proxy sed -n '1,297p' ADP/engine/common/utils.py` |
| SRC-WEIGHT | `ADP/engine/common/weights.py` — single/multi kernel weights блоками | `rtk proxy sed -n '1,184p' ADP/engine/common/weights.py` |
| SRC-SI-ENGINE | `ADP/engine/single_index/ADP_single_index_engine.py` — ориентация линии | `rtk proxy sed -n '1,22p' ADP/engine/single_index/ADP_single_index_engine.py` |
| SRC-SI-ENGINE-INIT | `ADP/engine/single_index/__init__.py` — экспорты single engine | `rtk proxy sed -n '1,5p' ADP/engine/single_index/__init__.py` |
| SRC-MI-ENGINE | `ADP/engine/multi_index/ADP_multi_index_engine.py` — QR, знаки и projector-distance | `rtk proxy sed -n '1,69p' ADP/engine/multi_index/ADP_multi_index_engine.py` |
| SRC-MI-ENGINE-INIT | `ADP/engine/multi_index/__init__.py` — экспорты multi engine | `rtk proxy sed -n '1,5p' ADP/engine/multi_index/__init__.py` |
| SRC-MAN-ENGINE-INIT | `ADP/engine/manifol_engine/__init__.py` — фасад manifold hooks | `rtk proxy sed -n '1,65p' ADP/engine/manifol_engine/__init__.py` |
| SRC-MAN-FIT | `ADP/engine/manifol_engine/fit.py` — manifold fit, transform, predict | `rtk proxy sed -n '1,234p' ADP/engine/manifol_engine/fit.py` |
| SRC-MAN-GRAPH | `ADP/engine/manifol_engine/graphs.py` — CSR graph и первичная projectors SVD | `rtk proxy sed -n '1,67p' ADP/engine/manifol_engine/graphs.py` |
| SRC-MAN-OPT | `ADP/engine/manifol_engine/optimisation.py` — local slopes, B-system, projector recovery | `rtk proxy sed -n '1,301p' ADP/engine/manifol_engine/optimisation.py` |
| SRC-MAN-UTIL-ENGINE | `ADP/engine/manifol_engine/utils.py` — scale boundary, chart query и trace | `rtk proxy sed -n '1,127p' ADP/engine/manifol_engine/utils.py` |
| SRC-MAN-WEIGHT | `ADP/engine/manifol_engine/weights.py` — manifold weights, bandwidth и I/U | `rtk proxy sed -n '1,224p' ADP/engine/manifol_engine/weights.py` |
| SRC-CG | `ADP/solver/CG.py` — отдельный current proximal-CG solver | `rtk proxy sed -n '1,276p' ADP/solver/CG.py` |
| SRC-HYBRID | `ADP/solver/HYBRID.py` — HPAO HYBRID и manifold линейная подзадача | `rtk proxy sed -n '1,703p' ADP/solver/HYBRID.py` |
| SRC-LEGACY-LSMR | `ADP/solver/legacy_lsmr.py` — прежний solver-контракт | `rtk proxy sed -n '1,350p' ADP/solver/legacy_lsmr.py` |
| SRC-LSMR | `ADP/solver/LSMR.py` — текущий HPAO-LSMR, local refit/correction/certificates | `rtk proxy sed -n '1,652p' ADP/solver/LSMR.py` |
| SRC-MULTIOP | `ADP/solver/_multi_operator.py` — joint multi forward/adjoint | `rtk proxy sed -n '1,29p' ADP/solver/_multi_operator.py` |
| SRC-CLI-INIT | `ADP/cli/__init__.py` — ленивые CLI exports | `rtk proxy sed -n '1,22p' ADP/cli/__init__.py` |
| SRC-CLI-MAIN-ENTRY | `ADP/cli/__main__.py` — `python -m ADP.cli` entry | `rtk proxy sed -n '1,5p' ADP/cli/__main__.py` |
| SRC-CLI-MAIN | `ADP/cli/main.py` — synthetic single/multi/manifold runner | `rtk proxy sed -n '1,607p' ADP/cli/main.py` |
| SRC-CLI-EXP | `ADP/cli/experiment.py` — передача управления пакету `experiments/` | `rtk proxy sed -n '1,111p' ADP/cli/experiment.py` |
| SRC-CLI-PLOTS | `ADP/cli/experiment_plots.py` — сборка отчетов, сводок и графиков | `rtk proxy sed -n '1,1246p' ADP/cli/experiment_plots.py` |
| SRC-CLI-EXP-UTIL | `ADP/cli/experiment_utils.py` — проверки и общая логика experiment runner | `rtk proxy sed -n '1,348p' ADP/cli/experiment_utils.py` |

## Что читать по задаче

- Сквозной путь общей single/multi-модели: [index-pipeline.md](index-pipeline.md).
- Индекс размерности 1: [single-index.md](single-index.md).
- Несколько направлений/EDR-базис: [multi-index.md](multi-index.md).
- Меняющееся по центрам локальное подпространство: [manifold.md](manifold.md).
- Объективы, внутренние операторы и сертификаты: [solvers.md](solvers.md).
- CLI и границы совместимости: [cli-and-compat.md](cli-and-compat.md).

## Важные различия путей

1. `ADP_single_index.fit` и `ADP_multi_index.fit` идут через `engine.common.index_fit.fit_index`. Это текущий общий outer-loop.
2. `ADP/core/ADP_Solver.py::ADP_Solver.fit` — legacy orchestration поверх `ADP_Data`; не считайте его текущей реализацией моделей.
3. `ADP/engine/common/ADP_Statistic_engine.py` умеет принимать `SparseNeighborhoodBlock` и отдельно содержит CUDA-путь. `fit_index` по умолчанию вызывает другой модуль, `engine/common/statistic.py`; GPU для текущего fit проходит через `GPUStatistics`.
4. `engine/common/box_kernel.py` экспортирует sparse compact-support engine, но текущий `fit_index` не выбирает его по `ADP_Config.smart_weights`. По live-ссылкам `smart_weights` валидируется и сохраняется в metadata, но в `fit_index`/`weights.py` не читается. Не предполагайте sparse execution без отдельного явного вызова.
5. `ADP/solver/legacy_lsmr.py` — не то же самое, что актуальный `ADP/solver/LSMR.py`, несмотря на схожее название.

Утверждение о неиспользовании `smart_weights` основано на поиске по `ADP/`; перед изменением wiring повторите поиск, так как дерево меняется (source: **SRC-CFG**, **SRC-IFIT**, **SRC-WEIGHT**, **SRC-BOX**).
