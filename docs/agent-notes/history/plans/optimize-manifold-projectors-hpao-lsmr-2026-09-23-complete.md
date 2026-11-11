---
task_id: optimize-manifold-projectors-hpao-lsmr-2026-09-23
status: done
created_utc: 2026-09-23T14:40:59Z
updated_utc: 2026-09-23T15:30:22Z
change_class: pending_per_candidate
---

# PLAN — локальные проекторы manifold и HPAO/LSMR

## Цель и порядок

Ускорить два подтвержденных CPU bottleneck: синхронное обновление локальных проекторов `ADP_Manifold` и current HPAO/LSMR для single/multi-index. **Сначала математически вывести и проверить более эффективный вариант для каждого пути; только после этого оптимизировать соответствующий production-код.** Сравнивать полный `fit`, время фазы, пик памяти и численный результат на парных входах.

## Вне задачи

- Менять estimator, kernel, bandwidth, случайные направления, граф manifold, критерии остановки, допуски solver или публичный API ради ускорения.
- Подменять current HPAO на `legacy_lsmr.py`, переносить результаты CPU на GPU без отдельной проверки или переписывать остальные фазы `fit`.
- Делать приблизительный solver либо новую статистическую процедуру режимом по умолчанию. Такие идеи допускаются только как отдельно названный вариант с собственным протоколом.

## Известные свидетельства

- `docs/adp_fit_bottlenecks.md`: при измеренных сходящихся CPU-конфигурациях manifold projector update занял 101.54/180.42 ms (`J=40`) и 199.44/343.62 ms (`J=80`); HPAO/LSMR — 51.29/70.07 ms (`J=48`) и 63.24/97.83 ms (`J=96`). Это локальные результаты для одного набора конфигураций, не гарантия ускорения в других режимах.
- Внутри manifold `_one_step` время распределено между `_local_slopes`, `_build_B_system`, `_solve_B`, `_recover_projector` и `_objective`; ускорение одного CG solve не покрывает всю фазу. Текущий CG применяет matrix-free normal operator и отдельно проверяет исходный residual.
- HPAO решает ridge correction через matrix-free forward/adjoint LSMR, затем проверяет normal residual certificate, trust radius, gauge, objective и stationarity. Последние профили выполнялись с `max_steps=50,tol=1e-6`; default `max_steps=5` там не сходился.
- Предыдущие завершенные планы: `agent-notes/history/plans/bottlenecks-multi-manifold-2026-09-23.md` и `agent-notes/history/plans/optional-agent-json-2026-09-23.md`. Исходные benchmark-артефакты не перезаписывать.

## Математические инварианты и классификация

- Для manifold зафиксировать **реально реализованную**, а не автоматически рукописную, локальную задачу при данных slopes: `Σ_j γ_j ||I_j − U_j Bᵀs_j||² + λ tr[B(I−E)Bᵀ]`, где `γ_j=mass_j·ω_j`, `E=Σ_j(ω_j/Σω)P_jᵀP_j`. Проверить по live-коду нормировки, ориентацию CSR и восстановление локального projector из малого rank-`m` фактора. Различия с `manifold-ade.tex` описать явно, не исправлять их внутри оптимизации.
- Для HPAO зафиксировать при данном refit коэффициентов задачу correction `min_δ ||Aδ−r||²+λ_prox||δ||²` с mass-scaled `A,r`; сохранить сопряженность `A/A*`, норму проксимального штрафа, residual certificate и правила принятия outer-шагов. Не смешивать HPAO-LSMR с manifold B-solver.
- Каждую идею до реализации классифицировать по `agent-notes/contracts/research.md`: `EXACT`, `NUMERICAL`, `APPROXIMATE` или `ESTIMATOR`. В основной путь допускаются только `EXACT`/обоснованные `NUMERICAL` изменения, сохраняющие цель. Для нового предобусловливания доказать корректность преобразованного ridge-штрафа и обратного перехода; не формировать плотные `(m·d)²`, `(J,n,d)` или `(J,P,n,d)` массивы.
- Сохранять `float64`, локальные массы и порядок центров, детерминированность seed, конечность результатов, orthonormal projectors, rank/failure checks и публичные поля результатов. Ориентация базиса сравнивается через projectors/principal angles, а не покомпонентно.

## Точный маршрут чтения

- Правила: `agent-notes/contracts/{numerics,research,engineering}.md`, `agent-notes/WORKFLOW.md`; профиль: `docs/adp_fit_bottlenecks.md`, `benchmarks/fit_bottlenecks.py`.
- Manifold: `agent-notes/ADP/manifold.md`, `agent-notes/ADP/solvers.md`, `agent-notes/tex/manifold-adp.md`; источники `SRC-MAN-FIT-ORCH`, `SRC-MAN-STEP`, `SRC-MAN-B-SYSTEM`, `SRC-MAN-RECOVER`, `SRC-MAN-HYBRID`, `A-OBJECTIVE`, `A-ONESTEP`; релевантные проверки `tests/test_manifold.py`, `tests/test_hybrid.py`.
- HPAO: `agent-notes/ADP/{solvers,index-pipeline,multi-index}.md`; источники `SRC-HPAO-ENTRY`, `SRC-HPAO-LOCAL`, `SRC-HPAO-OP`, `SRC-HPAO-GAUGE`, `SRC-MULTI-OP`, `SRC-HYBRID-INDEX`; релевантные проверки `tests/test_index_models.py`, `tests/test_hybrid_optimization.py`, `tests/test_hybrid_recycling.py`, `tests/test_gpu_index.py` для общего operator-контракта.
- Открывать только нужные диапазоны и вызовы выбранного варианта; перед правкой общей функции найти ее конкретных callers. Проверить live-код и тесты при начале каждого шага: заметки служат маршрутом, а не доказательством текущего поведения.

## Проверяемые математические гипотезы

| Путь | Вопрос для вывода, а не готовое решение | Условие принятия |
| --- | --- | --- |
| Manifold | Можно ли точно сократить повторные contractions/сборку B-системы и вычисление objective, используя общие величины соседей или факторизацию penalty, без `d×d` матриц и роста памяти по `E·P·d`? Может ли иная эквивалентная формулировка уменьшить число/цену матричных действий? | Формула для каждого forward/adjoint/штрафа и результата совпадает с исходной локальной задачей; cost model показывает уменьшение работы на измеренном режиме. |
| HPAO/LSMR | Можно ли уменьшить стоимость одного `A/A*` действия или число Krylov-итераций точным преобразованием/предобусловливанием, сохранив ridge-геометрию и исходный residual certificate? | Выведены преобразованная задача и обратное отображение correction; явный малый reference и оценка обусловленности подтверждают решение той же задачи без скрытой смены tolerance. |

## Ограниченные шаги и свидетельства

| ID | Статус | Работа | Свидетельство завершения |
| --- | --- | --- | --- |
| M1 | done | Воспроизвести baseline на текущем checkout; выписать точные функции, формы, цели, один hot target/iteration и reference для обеих задач. | `docs/manifold_hpao_optimization_math.md` и `docs/experiments/manifold_hpao_opt_2026-09-23/baseline_*.json`, `reference_*.npz`; 10 процессов на серию, direct reference совпал. |
| M2 | done | Математически исследовать manifold: проверить минимум два точных кандидата, для отклоненных указать причину; сравнить цену построения, действия и recovery. | `docs/manifold_hpao_optimization_math.md`: M-A/M-B EXACT, прямой reference и microtiming; плотный solve отклонен по памяти. Production-код не изменен. |
| M3 | done | Математически исследовать HPAO/LSMR: проверить точные factorization/операторные или предобусловленные варианты против текущей correction-задачи и certificate. | В math-документе H-A EXACT и H-B NUMERICAL; direct reference, adjoint, certificate, итерации/время/память. H-B отклонен по времени. Production-код не изменен. |
| G | done | Принять или отвергнуть каждого кандидата отдельно. | Решение записано в `agent-notes/DECISIONS.md`: M-A/M-B/H-A EXACT приняты для условной реализации, H-B и dense отклонены; formulas/reference/microtiming в math-документе. |
| C1 | done | Минимально реализовать принятый manifold-вариант, сохранив старый reference путь для сравнения. | `ADP/engine/manifol_engine/optimisation.py`, `tests/test_manifold.py`, reference в `benchmarks/optimization_candidates.py`; 12+14 тестов, full fit 178.747→167.827 и 342.202→317.379 ms, проекторы ≤`3.22e-15`, RSS без роста. |
| C2 | done | Минимально реализовать принятый HPAO-вариант для CPU, сохранив контракт общего operator и проверив затронутые backend/callers. | `LSMR.py`/`tests/test_lsmr.py`, 49 CPU tests passed (28 GPU skipped: no device), direct ridge/adjoint/certificate; single 46.316→42.676 ms, multi 69.815→67.116 и 98.671→94.096 ms, RSS без роста. |
| V | done | Проверить оба изменения вместе на base/J2 и еще одной форме по `d` или `P`; обновить затронутые маршруты `agent-notes` и отчет. | 180 fit runs на шести формах и обратный порядок multi; снижение median 6.48–8.91% manifold, 2.83–5.37% multi в первой серии, без заметного роста RSS/смены stop. На малом multi повторный эффект 1.14% в пределах шума. `docs/manifold_hpao_optimization_math.md`, 305 CPU tests passed, Ruff/Pyright/diff checks. |

## Протокол сравнения и границы

- Сначала малый auditable reference: dense локальная manifold-задача и direct ridge correction при малом размере; сравнивать исходный и кандидатный операторы на одинаковых векторах до сравнения `fit`. Для iterative методов проверять adjoint identity, исходный objective/gradient, status и residual.
- Бенчмарк: фиксированные данные, направления, seeds, kernel, bandwidth, lambda, solver tolerances, dtype, один поток BLAS, прогрев, не менее 10 повторов в отдельных процессах. Отдельно учитывать profiler overhead; измерять полный `fit`, фазу, число итераций/matvec, peak RSS и изменение качества. Не смешивать несошедшиеся runs с успешными.
- Начальные формы и команды брать из `docs/adp_fit_bottlenecks.md`; артефакты новой серии хранить отдельно. Показать выигрыш и возможный проигрыш по каждому режиму; если speedup только в одном размере, оставить условное применение или отказаться от замены.
- Остановить кандидат и вернуться к M2/M3, если исчезает эквивалентность, ломается certificate/degenerate case, растет память сверх установленного бюджета, меняется estimator или нет воспроизводимого выигрыша полного `fit`. Не ослаблять допуски ради прохождения тестов. Если точный вариант для одного пути не найден, сохранить старый код этого пути и документировать отрицательный результат перед новой гипотезой.
