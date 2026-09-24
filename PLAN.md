---
task_id: improve-multiindex-quality-2026-09-24
status: active
created_utc: 2026-09-24
updated_utc: 2026-09-24
change_class: EXPERIMENT_DESIGN
---

# PLAN — улучшить качество Multi-index

## Цель

Найти воспроизводимую конфигурацию или подтвержденную причину низкого качества Multi-index на трудных точках Multi v2. Разделить влияние внутреннего solver budget, числа внешних шагов и настроек оценивателя. Рекомендовать изменение только после проверки на новых seed; если надежного кандидата нет, оставить текущий default и зафиксировать отрицательный результат.

## Вне задачи

- Менять критерий `trace_score >= 0.95`, правило сходимости или состав ошибок ради роста recovery.
- Менять математическую цель, kernel, tensor, локализацию или default без отдельного варианта и парного свидетельства.
- Перезапускать весь каталог из 22 серий, manifold/single, GPU или оптимизацию времени до подтверждения причин качества.
- Использовать незавершенный `multiv2` каталог как завершенный benchmark.

## Известные свидетельства

- `benchmark_outputs/experiments/20260924T012524585775-multiv2/suite.json`: запланировано 22 серии, отмечено 7 завершенных; статус остался `running`. Завершенные серии содержат 1590 fits: 0 формально сошлись, 197 прошли порог качества, ошибок вычисления нет; все остановлены по `outer_steps`.
- В записанной конфигурации Multi v2 стоят `outer_steps=3`, `solver=lsmr`, `solver_max_steps=5`, `index_init=local`, `select_step=best`. Для `multiv2-mi-frequency-multiplicative` есть лишь 24 строки первой точки; ее нет в агрегированной сводке.
- На размерностной сетке качество проходит порог у 30/30 запусков при `d=10`, 28/30 при `d=20`, 19/30 при `d=30`; число проходов быстро снижается при дальнейшем росте d. В шкале признаков проходы сосредоточены около `sigma_X=0.75–1.0`. Полные срезы: `multiv2-mi-d/phase_summary.csv`, `multiv2-mi-scale/phase_summary.csv`.
- Отдельный diagnostic на базовой точке `d=6, n=240` дал предварительный сигнал в пользу `solver_max_steps=80` (5/6 validation recovery против 4/6 у baseline), но он не подтверждает перенос результата на v2 с `d=100` и большим числом центров. См. `agent-notes/DECISIONS.md`, решение от 2026-09-23.
- Основная метрика — schema-9 `trace_score`, выше лучше; recovery требует одновременно сходимости и прохождения порога. Прохождение качества при `nonconverged` не считать восстановлением.

## Проверяемые гипотезы

- **H1 — ограничение внутреннего solver.** Лимит в 5 шагов может не давать HPAO завершить подзадачу на каждом внешнем шаге и ухудшать последующее качество. Сначала проверить статусы и сертификаты LSMR в trace; затем сравнить лимиты 5, 50 и 80 при неизменных `outer_steps=3`.
- **H2 — ограничение внешнего цикла.** Если внутренние подзадачи сертифицируются, 3 внешних шага могут быть недостаточны для остановки fit. Отдельно сравнить 3, 6 и 9 внешних шагов при фиксированном достаточном solver budget.
- **H3 — недостаточное качество после сходимости.** Если формальная сходимость достигнута, отдельно проверить по одному фактору `N_loc`, `N_phi`, `N_J` и `lambda_penalty` на трудной точке. Не запускать полный декартов поиск.

H1–H3 — гипотезы, а не установленные причины. Если inner solver не сертифицируется при лимите 80, остановить подбор внешних параметров и исследовать trace/solver прежде, чем менять оцениватель.

## Инварианты и класс изменений

- Сохранять objective, joint multi operator, корректную пару forward/adjoint, ортонормированность basis и сравнение подпространств через projector/principal angles.
- Оставить `trace_score`, порог 0.95 и классификацию `recovered = convergence_pass AND quality_pass` неизменными. Любые numerical failures и nonconverged fits входят в знаменатель.
- Каждая пара baseline/candidate использует одинаковые данные и seed bundle; validation seed не участвуют в отборе. Фиксировать requested/effective config, stop reason, выбранный outer iteration и inner solver status/residual на каждом шаге.
- Текущая работа — `EXPERIMENT_DESIGN`. Изменение `solver_max_steps` или `outer_steps` сначала остается явной `APPROXIMATE` конфигурацией. `N_loc`, `N_phi`, `N_J`, `lambda_penalty`, initialization, tensor и другие меняющие процедуру настройки исследуются только как отдельные `ESTIMATOR`-варианты. Default менять лишь после held-out проверки.
- Не вводить плотные `d×d`/`(J,n,d)` массивы или normal equations как часть последующей реализации.

## Точный маршрут

- Сначала: `agent-notes/ADP/multi-index.md`, `agent-notes/ADP/index-pipeline.md`, `agent-notes/ADP/solvers.md`, `agent-notes/contracts/numerics.md`, `agent-notes/contracts/research.md`, `agent-notes/WORKFLOW.md`.
- Экспериментальный runner: `experiments/diagnostic.py`, `experiments/multiv2.py`, `experiments/multi.py`, `experiments/models.py`, `experiments/data.py`, `experiments/runner.py`, `experiments/README.md`.
- Численный путь только для H1/H2: `ADP/engine/common/index_fit.py` (`SRC-MI-DRIVER`), `ADP/engine/common/initialize.py` (`SRC-MI-INIT`), `ADP/solver/LSMR.py` (`SRC-HPAO-ENTRY`, `SRC-HPAO-GAUGE`). Для H3 читать только нужные локаторы `SRC-MI-WEIGHTS`, `SRC-MI-ALPHA`, `SRC-MI-DIRECTIONS`.
- Артефакты: `benchmark_outputs/experiments/20260924T012524585775-multiv2/{suite.json,overview.csv,overview.md}`, выбранные `series.json`, `runs.csv`, `phase_summary.csv`, `trace_summary.csv`; текущее paired diagnostic описано в `experiments/README.md` и `agent-notes/DECISIONS.md`.
- Тесты и конфигурацию открыть перед правками: соответствующие multi/solver случаи из `tests/` и experiment tests; не сканировать остальные подсистемы.

## Ограниченные шаги

| ID | Статус | Работа | Свидетельство завершения |
|---|---|---|---|
| T1 | pending | Сверить строки `runs.csv` и traces с конфигурацией; проверить, означает ли `outer_steps` только внешний лимит либо вместе с ним регулярно не сходится inner HPAO. Зафиксировать две sentinel точки: качественную (`multiv2-mi-d`, `d=10`) и трудную (`multiv2-mi-n`, `n=1000, d=100`). | Небольшой baseline-отчет с точными точками/effective config, распределением stop/solver status и ограничениями исходного незавершенного каталога. Fits не запускать. |
| T2 | pending | Минимально расширить paired diagnostic: принимать выбранные точки Multi v2, иметь selection-only screening и отдельный validation запуск замороженного кандидата. Существующие base/noise/correlation/scarce сценарии и defaults не менять. | Тесты подтверждают одинаковые seed bundles у вариантов, раздельные selection/validation seed, запись effective config и отбраковку неполного набора; `--dry-run` показывает точный бюджет. Validation режим не выбирает победителя. |
| T3 | pending | На selection seed `1000–1009` сравнить `solver_max_steps=5/50/80`, сохраняя `outer_steps=3`, на двух sentinel точках. Между этапами использовать только эти selection результаты. | Парная таблица trace_score/recovery, inner convergence и сертификатов по каждому внешнему шагу, ошибок, времени и памяти. Если 80 не дает сертифицированного inner solve, остановиться на диагнозе solver. |
| T4 | pending | Только после inner convergence сравнить `outer_steps=3/6/9` при выбранном внутреннем лимите на тех же selection seed, без изменения других параметров. | Отдельная парная оценка внешней сходимости и качества; найден минимальный бюджет, дающий устойчивое улучшение, либо зафиксировано отсутствие эффекта. |
| T5 | pending | Если качество остается ниже порога при корректной сходимости, провести ограниченный однофакторный selection по `N_loc`, `N_phi`, `N_J`, `lambda_penalty` или initialization; добавлять только кандидаты, обоснованные traces. | Отбор только по selection seed, без полного перебора и без просмотра validation; сохранены варианты, включая численные ошибки и неудачи. Любая смена оценивателя помечена `ESTIMATOR`. |
| T6 | pending | Один раз проверить baseline и замороженного лучшего кандидата на обеих sentinel точках с новыми validation seed `2000–2019`; после проверки при необходимости внести минимальное изменение только в явную конфигурацию/экспериментальный вариант. | Не менее 20 held-out seed на вариант, фиксированное парное сравнение без повторного выбора победителя и воспроизводимая конфигурация. Обновлены затронутые notes и итоговый `STATE.md`; изменение default отдельно обосновано или отклонено. |

Seed root selection и validation должны быть новыми относительно просмотренного v2 набора; предлагаемое разбиение — selection `1000–1009`, validation `2000–2019`. Общая validation запускается только после фиксации параметров.

## Критерии решения и остановки

- Recovery на каждой sentinel точке: не менее 16/20 held-out fits, ноль numerical failures и не хуже baseline. Для общего default кандидат не должен ухудшать качественную точку `d=10` ради улучшения только трудной точки.
- Если recovery равен, считать медианный paired прирост `trace_score >= 0.01` практически значимым; разницу меньше 0.01 считать ничьей и сохранять baseline, если нет подтвержденного выигрыша времени не менее 10% без потери качества.
- Если ни один вариант не проходит held-out критерий, оставить default без изменений и завершить отрицательным научным результатом. Не снижать порог, не смешивать качество с технической сходимостью и не продолжать tuning на validation seed.
- Любой конкретный дефект формулы/оператора переводит работу в отдельный этап: dense/reference сравнение, focused regression tests и новая классификация `EXACT`/`NUMERICAL`/`APPROXIMATE`/`ESTIMATOR` до кода.
