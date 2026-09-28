# Эксперименты ADP

Самостоятельные запуски из корня репозитория:

```bash
uv sync --group bench
uv run python -m experiments.single
uv run python -m experiments.multi
uv run python -m experiments.multiv2
uv run python -m experiments.manifold
```

Также поддерживается `uv run python experiments/single.py` (аналогично для
`multi.py`, `multiv2.py` и `manifold.py`). Каждый запуск создаёт отдельную папку
`benchmark_outputs/experiments/<timestamp>-<mode>/`.

`experiments.multiv2` содержит 36 серий каталога `multi` под отдельными
селекторами `multiv2-*`; серии `mi-3` и `mi-5` исключены из v2. По умолчанию
запускаются оставшиеся 22 основных серии; другие доступные сценарии показывают
`--list` и `--experiment all`.
Обзорный профиль v2 сохраняет полную сетку. Семь коротких бинарных серий
(включая две повторные проверки) раскрыты по четырём уровням шума
`sigma_eps = 0, 0.2, 0.6, 1.0`;
сравнение функций связи раскрыто по четырём `link_scale` для каждого типа связи.
Это даёт минимум семь точек на серию и позволяет сравнить качество и частоту
восстановления у границы шума/частоты. Выходные файлы v2 отделены от `multi`.
Стандартный обзор v2 планирует 815 fits при пяти повторах; при `--runs 30` —
4890 fits. Весь оставшийся каталог — 9524 fits при одном повторе, включая
breaking-сетку на 8960 точек.

## Диагностика параметров и слабых режимов

Отдельное исследование информативных центров multi-index остановлено на C2:
пилот C1 прошёл gate неоднородности, но зафиксированное правило с покрытием
`>=0.99` не нашло `J<500` для 11/20 selection задач. Full fits и held-out
не выполнялись, public estimator не менялся. Протокол, raw paths и точная
причина остановки: `docs/experiments/multi_center_selection_2026-09-25/`
(`c1_protocol.md`, `c1_result.md`, `c2_rule.md`, `c2_result.md`).
Для воспроизведения из корня репозитория с одним BLAS-потоком:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m experiments.multi_center_diagnostic
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m experiments.multi_center_select
```

Модули отказываются перезаписывать существующий output-dir; для повтора
укажите новый `--output-dir`.

Для двух зафиксированных точек Multi v2 есть отдельный парный протокол:

```bash
python -m experiments.multiv2_quality --stage inner --dry-run
python -m experiments.multiv2_quality --stage inner
python -m experiments.multiv2_quality --stage outer --inner-steps 80 --dry-run
python -m experiments.multiv2_quality --stage validation --candidate-file candidate.json --dry-run
```

`inner` сравнивает лимиты HPAO 5/50/80 при трёх outer шагах; `outer`
сравнивает 3/6/9 outer шагов с указанным лимитом HPAO. `factor` принимает
`--inner-steps`, `--outer-steps` и один `--factor FIELD=VALUE` для
`N_loc`, `N_phi`, `N_J`, `lambda_penalty` или `index_init`. Все эти режимы
используют только selection seed 1000–1009. `validation` отдельно сравнивает
исходный baseline 5/3 с замороженным кандидатом на seed 2000–2019 сразу
на обеих точках. Файл кандидата содержит ровно `d10` и `n1000`, например
`{"d10":{"solver_max_steps":80},"n1000":{"solver_max_steps":80}}`.
Validation не выбирает победителя. `--dry-run` показывает точный бюджет
без создания результата. `run.json`, `summary.json`, `series.json` и
`runs.csv` хранят конфигурации, seed, outcomes и solver traces; анализ
отклоняет неполные или непарные наборы строк.

Отдельный парный бенчмарк сравнивает базовую конфигурацию с локальными
однофакторными вариантами для single, multi и manifold:

```bash
uv run python -m experiments.diagnostic --dry-run
uv run python -m experiments.diagnostic --profile smoke
uv run python -m experiments.diagnostic --mode multi --scenario noise --runs 12
uv run python -m experiments.diagnostic --analyze benchmark_outputs/diagnostic/<run>
```

`--mode` принимает `single`, `multi`, `manifold` или `all` (по умолчанию);
`--scenario` — `base`, `noise`, `correlation`, `scarce` или `all`.
`--threads`, `--seed` и `--output-dir` задают среду запуска. Полный профиль
по умолчанию делает 12 seed на вариант: первые 6 используются для выбора,
последние 6 — только для проверки. Полный набор всех семейств и сценариев —
1440 fits; `--dry-run` показывает точное число. Smoke делает 48 fits, но
не выдаёт рекомендацию по двум seed.

Внутри сценария все варианты получают одинаковые данные, истинное
подпространство и random streams; меняется один параметр. Проверяются
`N_loc`, `N_phi`, `N_J`, шаги solver и `lambda_penalty` для single/multi;
`N_loc`, `N_phi`, `N_manifold`, `lambda_manifold`, `sync_steps` для manifold.
Сценарии меняют шум, корреляцию или отношение `n/d`. Полный декартов поиск
и взаимодействия параметров этот набор не оценивает.

В `diagnostics.md` видны recovery, Wilson 95% интервал, численные ошибки,
несходимость, качество, время, память и парные изменения от baseline.
`diagnostics.csv` содержит также качество и прирост от инициализации,
нижний квантиль качества, числа измеренных fits и фактические конфигурации.
`diagnostics.json` сохраняет правило выбора и seed split; `run.json` —
профиль, конфигурацию запуска и хеш исходников. Исходные `runs.csv` и
`series.json` остаются под `series/`.

Отбор использует только первую половину seed: максимум восстановлений,
минимум численных ошибок, затем качество; сходимость solver показывается
для диагностики и не входит в recovery или допуск кандидата. Разница до 0.01 по нормированной
метрике считается практическим равенством. При равенстве сохраняется
baseline, если парное ускорение меньше 10%. Новый вариант получает
предварительную рекомендацию только после проверки на других seed:
не менее пяти повторов, recovery ≥ 0.8, без численных ошибок и с
улучшением восстановления либо существенным выигрышем качества/времени.
`baseline_retained` означает, что базовая настройка остаётся предпочтительной;
`no_reliable_candidate` означает, что ни один вариант не прошёл критерий.
Эти выводы относятся только к записанному сценарию и проверенной сетке.
Интервалы при шести проверочных seed широки, поэтому результаты стоит
подтверждать на других данных. Численные ошибки остаются в знаменателе.

## Что исследуется

- `single`: корректность, объём выборки и размерность, шум и корреляция,
  классы функций, чётность и частота, локализация производной, инициализация,
  распределения, тяжёлые хвосты, гетероскедастичность, выбросы и нарушение
  модели; массы, направления, центры, регуляризация и параметры итераций.
- `multi`: объём выборки, внешняя и внутренняя размерности, шум и корреляция,
  аддитивные и мультипликативные функции, частота; инициализация, локализация,
  тензор, распределение и обновление направлений, регуляризация и итерации.
- `manifold`: радиальная модель с меняющимся локальным направлением,
  объём выборки, размерность, шум, масштаб, корреляция; локальные массы,
  центры, направления, соседи графа, регуляризация и синхронизация.

Manifold-набор исследует одну существующую радиальную модель
`0.5 * (x_1² + x_2²)`. Его результаты нельзя обобщать на произвольные многообразия.
Гипотезы в каталогах — проверяемые предположения, не установленные свойства метода.

## Объём работы

По умолчанию используется `--profile overview`: крайние и средний уровни каждого
числового фактора, все категориальные уровни и 5 повторений на точку. Для всех
серий `manifold` и `multiv2` overview запускает полную сетку.
Параметры и seed-протокол исходных точек сохраняются; сетка находит переход
между уровнями, но точный порог после этого требует более узкого дополнительного
прогона. Отчёты сравнивают recovery, quality, convergence и ошибки на каждом
уровне; с `--runs 30` частота срыва оценивается отдельно для каждого уровня.

Все 13 manifold-серий теперь содержат семь уровней. На конфигурационных осях
сетка доходит до валидных краёв: `n=61` при `N_lin=60`, `d=58` при `N_lin=60`,
`N_lin=6`/`239` при `d=4,n=240`, `N_loc=10`/`239`, `N_J=12`/`240`, `N_phi=1`,
`N_manifold=2`/`23`; остальные оси расширены до режимов высокого шума,
корреляции, масштаба, penalty и числа синхронизаций. Основная серия `manifold`
совместно повышает `d`, `n` и шум, а `manifold-d` отдельно меняет только `d`.
В `manifold-n` фиксируется `N_J=36`, чтобы изменение n не вызвало автоматического
роста числа центров. Для `manifold-nloc` минимум `N_loc=10` сохраняет `N_J=24`,
а `manifold-centers` начинается с эффективного минимума `N_J=12`.
При пяти повторах текущий объём по умолчанию: single — 25 серий / 720 fits,
multi — 24 серии / 930 fits, manifold — 13 серий / 455 fits. Команда с
`--runs 30 --solver hybrid` планирует 2730 fits; она показывает частоту отказов
на семи уровнях каждой серии, а точный порог можно уточнить между соседними уровнями.

Перед длительным запуском можно посмотреть точное число вычислений:

```bash
uv run python -m experiments.single --dry-run
uv run python -m experiments.multi --list
uv run python -m experiments.manifold --runs 30 --solver hybrid --dry-run
uv run python -m experiments.multiv2 --experiment all --runs 1 --dry-run
```

Профили:

| Профиль | Точки | Повторы по умолчанию | Назначение |
|---|---|---|---|
| `smoke` | Одна уменьшенная точка каждой серии | 1 | Проверка запуска и отчётов |
| `overview` | Крайние/средние числовые уровни и категории; manifold и multiv2 — полные сетки | 5 | Обзор факторов и границ отказа |
| `full` | Полная исходная сетка выбранных серий | По каталогу | Подробное исследование |

`--runs N` переопределяет число повторений; `--seed N` задаёт начало диапазона.
Потоки BLAS ограничиваются в CLI через `--threads` (по умолчанию 1).
Серии выполняются последовательно.

```bash
# Проверка всех трёх наборов на маленьких задачах
uv run python -m experiments.single --profile smoke --no-plots
uv run python -m experiments.multi --profile smoke --no-plots
uv run python -m experiments.manifold --profile smoke --no-plots

# Только выбранные факторы и больше повторений
uv run python -m experiments.single --experiment si-n,si-d,si-noise --runs 10
uv run python -m experiments.multi --experiment mi-tensor,mi-nphi --profile full
uv run python -m experiments.manifold --experiment manifold-neighbors,manifold-lambda
```

Флаг `--list` перечисляет все доступные серии семейства. Дорогие `*-breaking`,
`scale-*`, уточняющие и фокусные сетки доступны через `--experiment`, но не входят
в наборы по умолчанию. Одна полная `*-breaking` сетка с её исходными 100 повторами
требует 896 000 fits. `--profile full` сам по себе её не добавляет.

`--solver {lsmr,cg,hybrid}`, `--solver-max-steps` и `--cg-maxiter` передаются
общему исполнителю. Для single доступны только `lsmr` и `cg`;
`hybrid` поддерживает multi и manifold. Несовместимый выбор отклоняется до
создания результатов. Как и раньше, значение `solver_max_steps`, явно заданное
в точке каталога, имеет приоритет над настройкой build. Запрошенная и эффективная
конфигурации записываются раздельно.

## Результаты

Начинайте с `overview.md`: таблица завершённых серий, ссылки на подробности,
причины остановок и тексты ошибок. Отчёт обновляется после каждой серии.

| Файл | Содержимое |
|---|---|
| `suite.json` | Профиль, seed, потоки, объём работы, завершённые серии и статус |
| `overview.csv` | Число сходимостей, прохождений порога, восстановлений и ошибок; квантили качества, времени и памяти |
| `<series>/series.json` | Точные точки запуска, версии, git-состояние, BLAS, seed-протокол, параметры и критерии |
| `<series>/runs.csv` | Каждый fit, включая неудачи: качество, selected error, конфигурация, trace и solver diagnostics |
| `<series>/summary.csv`, `summary.md` | Сравнение уровней факторов |
| `<series>/phase_summary.csv` | Качество, сходимость и восстановление раздельно, интервалы Уилсона |
| `<series>/trace_summary.csv` | Диагностика по итерациям |
| `<series>/failures.csv` | Неудачные исходы |
| `<series>/plots/` | Графики качества, стоимости, итераций и восстановления |

Графики включены; `--no-plots` оставляет таблицы и текстовые отчёты.
При Ctrl+C завершённые серии и строки текущей серии остаются на диске;
`suite.json` получает статус `interrupted`. Автоматического продолжения нет.
Численные ошибки не останавливают последующие fits; код возврата 1 означает,
что хотя бы один fit завершился численной ошибкой. Несходимость учитывается
отдельно и сама по себе не меняет код возврата. Ctrl+C возвращает 130.

## Как читать качество

Восстановление определяется геометрией подпространства, а сходимость solver
остаётся отдельной диагностикой. Для новых семейных запусков пороги заданы
явно: single `abs(cos) >= 0.9`, multi `trace_score >= 0.95`, manifold —
full-center RMS локальной ошибки projector `<= 0.2` и максимальная ошибка
по всем центрам `<= 0.2`. Уже заданные пороги сохраняются. Настройка влияет
только на классификацию результата, не на оцениватель.

### Дайджест для автоматического анализа

Существующий каталог можно разобрать без повторного запуска fits:

```bash
uv run python -m experiments.analyze_suite benchmark_outputs/experiments/<run-id>
uv run python -m experiments.analyze_suite benchmark_outputs/experiments/<run-id> --format md
```

По умолчанию JSON печатается в stdout. Он содержит planned/observed coverage,
статус каждой серии, число fits, convergence/quality/recovery с Wilson 95% CI,
квантили качества, prediction RMSE, selected error, HPAO solver loss,
manifold objective/penalty, времени, traced memory и outer iterations,
лучшие/худшие условия по recovery в фазовых таблицах, boundary-группы,
stop/failure modes и provenance. Частичные серии не попадают в завершенные
итоги. Quality-pass имеет знаменатель только из fits с конечной метрикой;
convergence и recovery делятся на все попытки. Recovery single/multi
определяется только метрикой общего подпространства. Для manifold дополнительно
требуется, чтобы максимальная ошибка локального projector по всем центрам
была не выше 0.2; solver convergence остаётся диагностикой. Общие медианы по каталогу являются описательными
для смеси настроек; фазовые условия также не являются рекомендацией параметров.
Loss-поля имеют разные определения. Для single/multi `selected_error` — сумма
квадратов ошибки ответа в центрах на выбранном внешнем шаге; в общем случае
это не held-out ошибка. В manifold это сумма квадратов между `Y` и
`predict(X)` на той же сгенерированной выборке. `solver_loss` — отдельный
внутренний HPAO objective, а manifold trace отдельно хранит `objective` и
`manifold_penalty`. Сырые SSE зависят от масштаба `Y` и размера оцениваемого
набора, поэтому предиктивную пользу сравнивайте по RMSE/MAE на заранее
отложенных наблюдениях и относительно baseline.
Сохранённый здесь manifold `prediction_rmse` вычислен на той же сгенерированной
выборке, а не на held-out данных; в рассмотренных single/multi каталогах эта
метрика отсутствует.
Вызовы старого CLI сохраняют исходный протокол, включая отсутствие порога.
В семейных отчётах фазовые вероятности разделены по остальным факторам;
категориальные уровни (функция, инициализация, распределение) не смешиваются.

Для multi сохраняются также `projector_distance = sum(sin(theta_j)²)` и
максимальный главный угол. Основной `trace_score = mean(cos(theta_j)²)`
взят из текущей реализации; метрики разных семейств не усредняются вместе.

Квантили общего отчёта описывают смесь точек одной серии; для сравнения
параметров используйте таблицы по уровням. Они не являются доверительными
интервалами среднего. Пять повторений дают первичный обзор, не доказательство
устойчивого превосходства. Smoke не проверяет научные гипотезы.

Время — wall-clock fit, без генерации данных и отчётов. Память измеряется
`tracemalloc` внутри fit и не равна полному RSS процесса. Для общей длительности
запуска есть `elapsed_sec` в `suite.json`.

## Устройство кода

Определения сеток находятся в `single.py`, `multi.py`, `multiv2.py`, `manifold.py`;
общие конструкции сеток — в `_grids.py`, контракты — в `models.py`,
генерация данных — в `data.py`, выполнение одной серии — в `runner.py`,
объединение каталогов и старые псевдонимы — в `registry.py`.
`suite.py` управляет семейным запуском, `profiles.py` выбирает обзорные точки,
`suite_report.py` составляет общий отчёт. Графики используют существующий
`ADP/cli/experiment_plots.py`.

`ADP/cli/experiment.py` остаётся совместимым CLI и экспортирует прежние
`Build`, `Experiment`, `ExperimentPoint`, `CATALOG`, `custom_experiment`,
`run_experiment`. Например:

```bash
uv run python -m ADP.cli.experiment --experiment manifold-basic --profile smoke
```

## Proof-first скетч H1

Изолированный `experiments.proof_first_h1` сравнивает ортогональные блоки
направлений с исходными изотропными направлениями multi и оригинального
manifold estimator. Доказательство, reference, замороженный протокол,
66 selection fit и отрицательный результат находятся в
`docs/experiments/proof_first_shared_2026-09-26/` (`r2_protocol.md`,
`r3_result.md`, `selection/runs.jsonl`). Ни один случай не прошёл gate
качества/стоимости; held-out seed не запускались, public defaults не менялись.

### Manifold при m>1

Отдельный эксперимент `manifold_generalization` сравнивает m=1,2,3 при d=8,
n=600, кривизне 0/0.35/0.8, шуме 0/0.1 и двух фиксированных режимах
поддержки. Все 180 fits (5 seed) сохраняются, включая numerical failures.
Используется явный `estimator="manifold"`; default `local_quadratic` m>1
не поддерживает. Аналитическая геометрия, ограничения интерпретации и
результаты — в `docs/experiments/manifold_generalization_2026-09-28/report.md`.

```bash
UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m experiments.manifold_generalization --self-check
UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m experiments.manifold_generalization --profile smoke --out /tmp/manifold-generalization-smoke
UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m experiments.manifold_generalization --profile main --out /tmp/manifold-generalization-main
```

Каталог вывода должен быть новым. Manifest фиксируется до fits, JSONL и
summary обновляются после каждого fit. По исчерпании бюджета 600 s серия
помечается incomplete (бюджет проверяется между fits). Recovery требует
RMS и максимальную principal sine <=0.2 на всех центрах и independent
queries; center-only и query-only успех также показаны отдельно. Solver
residuals/stop reason не входят в recovery.

Отдельное парное exploratory сравнение использует существующую опцию
`--scale-boundary stop`; default эксперимента остаётся `raise`.
Их результаты сохраняются в разных каталогах и не смешиваются.
Исправленный отдельный протокол `experiments.manifold_generalization_repair`
сравнивает исходный относительный спектр локализации с явным `"unit"` на
парных seed. Он проверяет качество центров и ошибку оценки выбранного chart
отдельно от неизбежной ошибки nearest-chart на query. Для искривлённых
`m>=2` генераторное поле `row(Dz)` не идентифицируется по одному scalar Y,
поэтому эти профили остаются описательной диагностикой. Формулы, критерий,
команды и результаты — в `docs/experiments/manifold_generalization_2026-09-28/repair.md`.

Разбор провала на кривизне и честный критерий для скалярного отклика —
[`docs/experiments/manifold_curvature_2026-09-28/report.md`](../docs/experiments/manifold_curvature_2026-09-28/report.md).
Для `m>=2,c>0` проверяется включение наблюдаемого градиента в локальное
пространство; это необходимое условие, а не восстановление всего `row(Dz)`.
На проверенном `n=600,d=8` профиле даже broad/unit не прошёл all-center
порог ни в одной из 60 попыток, поэтому положительной рекомендации для
искривлённых m=1/2/3 нет. На плоской широкой поддержке broad/unit
восстановил 30/30 попыток отдельной validation серии.

### Единая сетка manifold

`experiments.manifold_grid` реализует таблицу из `Manifold exp.md`: Gaussian
признаки, вращённую квадратичную геометрию, стандартизованный безшумный
сигнал с добавочным Gaussian noise и текущий estimator `manifold`. Серии
выбираются через `--series`; одинаковые конфигурации между сериями выполняются
один раз и остаются помечены всеми именами серий. `development` использует
30 seed на конфигурацию, `full` — 250; `--runs` переопределяет число повторов.

```bash
uv run --no-sync python -m experiments.manifold_grid --profile development --dry-run
uv run --no-sync python -m experiments.manifold_grid --profile full --dry-run
uv run --no-sync python -m experiments.manifold_grid --profile full
uv run --no-sync python -m experiments.manifold_grid --profile full --scale-boundary raise --dry-run
```

Выводы записываются в новый каталог `benchmark_outputs/experiments/`:
`manifest.json` фиксирует генератор, конфигурацию и окружение, `runs.jsonl`
содержит каждую оценку и ошибку, `summary.json` обновляется после каждого fit.
По умолчанию сетка использует `scale_boundary=stop`: fit завершается на
последнем допустимом масштабе с `stop_reason=function_mass_boundary` или
`manifold_mass_boundary`. Ошибки ранга и solver по-прежнему записываются как
ошибки fit. Исходный строгий протокол из `Manifold exp.md` доступен через
`--scale-boundary raise`; его результаты нельзя объединять с режимом `stop`.

Для `m=1` и плоского `c=0` recovery требует RMS и максимум principal sine
не выше 0.2 на всех центрах и при оценке chart тех же ближайших центров
для 512 независимых запросов. Для искривлённых `m>=2` поле генераторного
`row(Dz)` не определяется однозначно одним scalar Y, поэтому `recovered=null`.
Сырые ошибки на query и oracle nearest-chart сохраняются отдельно как
описательные метрики. В `summary.json` есть число оцениваемых fits, границы
массы и численные ошибки; схема новых результатов — версия 2.

В таблице 26 строк, но строго одинаковых конфигураций 22, поэтому полная
сетка после дедупликации составляет 5,500 fits, а не указанные в документе
5,750. Также `N_lin=200` не проходит live-валидацию для `n<=200` или `d=200`;
поэтому текущий runner применяет ближайшую допустимую целевую массу
`max(d+2, min(200, n-1))` для таких крайних точек. Остальные параметры
совпадают со спецификацией, кроме явного выбора `scale_boundary=stop` для
основного запуска сетки; production defaults не меняются. Старые результаты
строгого протокола не перезаписываются.
Результаты ограниченной проверки всех 22 точек и оценка качества — в
[`docs/experiments/manifold_grid_2026-09-28/report.md`](../docs/experiments/manifold_grid_2026-09-28/report.md).

### Сетка Spokoini для multi-index

`mi-spokoini-*` запускает только текущий multi-index ADP на генеративных
моделях из `Spokoiny.md`; ADE, SIR II и PHD в этот эксперимент не входят.
Для воспроизведения полного числа повторов:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m experiments.multi \
  --experiment spokoini --profile full --threads 1 --dry-run
```

Убрать `--dry-run`, чтобы выполнить запланированные 5450 fits. Полная сетка
содержит m=1 (7 точек), m=2 (13 точек в четырёх sweeps) и m=3 (3 точки):
обычно 250 повторов на точку, 100 при m=2 и d>10. `--profile smoke` проверяет
по одному сокращённому fit на selector; `overview` сохраняет все точки с пятью
повторами. Каждый fit использует независимый seed; все численные ошибки
остаются в `runs.csv`.

Данные генерируются точно по описанию: независимые координаты
`X=2*Beta(1,tau)-1`, фиксированные ортонормированные направления, исходные
link-функции и ненормированный отклик плюс гауссов шум с заданным sigma.
Существующее значение `tau` для других экспериментов не меняется. Текущий
движок начинает multi anisotropy factor с 1 и получает `a` из опубликованного
`a_h`; он не предоставляет независимые `rho_min`, `a_rho`, `h_1` и `h_max`,
поэтому это grid текущей реализации на заданном дизайне, а не побитовая
репликация параметризации кода 2001 года. В полной сетке явно фиксируются
`N_loc=10`, `N_lin=2d`, `N_J=n`, `N_phi=max(m+1,min(10,d))`,
`index_init=local`, выбор последнего шага и лимит solver в 5 шагов; другие
параметры остаются текущими ADP defaults. Полный эффективный конфиг записывается
в каждой строке `runs.csv`.

`checkpoint_summary.csv` в каждой серии содержит среднее и IQR геометрической
ошибки `m * (1 - trace_score)` на шагах 1, 2, 4, 8 и на последнем доступном
шаге. В нём отдельно указаны число fit-ов с trace и численные ошибки; строки
`runs.csv`/`trace_summary.csv` остаются первичными данными.
