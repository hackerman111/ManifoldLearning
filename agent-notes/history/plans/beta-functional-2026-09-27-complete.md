---
task_id: beta-functional-complexity-2026-09-27
status: done
created_utc: 2026-09-27
updated_utc: 2026-09-27
change_class: EXACT / research
---

# PLAN — функционал β, математическая сложность и HYBRID

## Цель и границы

Формализовать live single/multi ADP functional и β-подзадачу AO,
категоризировать способы сокращения вычислений, доказать или опровергнуть
применимость конкретных структурных сокращений. Если нового применимого
солвера не получается, выполнить измеренный аудит CPU index HYBRID и его
общего HPAO пути. Manifold отметить как отдельную цель; не переносить на
него выводы без доказательства. Итог — русский математический отчёт,
воспроизводимые reference/профиль/benchmark и честные границы вывода.
Не менять estimator, defaults, API, tolerances/rank/certificates; не запускать
новый широкий full-fit/held-out поиск. Старый завершённый план сохранён в
`agent-notes/history/plans/proof-first-shared-2026-09-26-complete.md`.

## Известное, гипотезы и инварианты

- Прежние reduced L-BFGS/GN и уточнение HYBRID не прошли совместный gate;
  причины проверять по локальным отчётам, не повторять те же кандидаты.
- Исходная цель F(B,C)=1/2 sum_j mass_j ||I_j-U_j B^T c_j||²,
  BB^T=I_m. Proximal lambda относится к расслабленному correction.
- H1: точное сокращение по достаточным статистикам/row-space или общему
  metric может снизить размер; нужны точные предпосылки и стоимость setup.
- H2: явные cross-terms исключают независимые solves по строкам B;
  усреднение полных градиентов без их локальной metric меняет estimator.
- H3 fallback: в HYBRID/HPAO скрыта повторная contraction planning,
  лишняя projected temporary, factorization или certificate/operator work.
  Принимать только после профиля и reference сравнения.
- Сохранить float64, mass, minimum-norm local refits, QR gauge,
  исходные adjoint/residual/stationarity guards, bounded memory.
  Запрещены dense d×d, (md)², полный (JP,md), (J,n,d).

## Точный read set

`AGENTS.md`, `agent-notes/{STATE,WORKFLOW}.md`,
`agent-notes/contracts/{numerics,research,engineering}.md`,
`agent-notes/ADP/{solvers,multi-index,index-pipeline}.md`,
`agent-notes/tex/multi-index-adp.md`;
`ADP/solver/{LSMR,HYBRID,_multi_operator}.py` (SRC-HPAO-LOCAL/OP/GAUGE,
SRC-HYBRID-INDEX/HPAO), `ADP/engine/common/statistic.py`;
`tex/multiindex.tex:1010–1236`; профильные существующие tests;
`docs/experiments/multi_solver_{search,retry}_2026-09-24/` только
proof/completion/profile; frozen diagnostics по их точным маршрутам.

## Ограниченные work units

| Шаг | Работа | Acceptance evidence |
|---|---|---|
| B0 | Проверить формулы/live shapes/TeX и написать категории методов. | Формула с forward/adjoint/Hessian, distinction original/prox/reduced. |
| B1 | Не более трёх структурных сокращений: exact row compression, common metric, ADP-gradient replacement. | Полные выводы, контрпримеры/условия, time/memory и verdict применимости. Численные независимые reference проверки. |
| B2 | Если нет применимого нового solver: профиль HYBRID/HPAO на frozen и synthetic shapes, проверить NumPy альтернативы. | Raw profile, медианы, env/shapes, reference error, peak memory. Не более двух kernel кандидатов; точные варианты в изолированном benchmark. |
| B3 | Завершить отчёт и routing/checkpoint. | Целевые pytest/Ruff/Pyright/diff checks по реально затронутому коду; PLAN done, STATE и DECISIONS. |

## Verification и stop conditions

Сначала deterministic dense/reference и adjoint; timing отдельно, фиксированный
CPU/BLAS thread configuration, warmup, paired repetition, wall + allocation/RSS.
Никакого вывода о full-fit recovery из kernel timings. Новый solver допускается
лишь при доказанных live-предпосылках и полной цене setup/certification.
При отсутствии таких предпосылок закрыть B1 как bounded negative и выполнить
B2. Не ослаблять gates и не изменять estimator ради сокращения сложности.
Engineering-кандидаты оставить изолированными, если выигрыш нестабилен или
не проверена полная trajectory; отчёт об аудите достаточен для запроса.

## Итоговый checkpoint — done

- B0: live functional, β ridge correction, adjoint/Hessian/certificate и
  TeX normalization сверены. Полный вывод и восемь категорий методов:
  `docs/experiments/beta_functional_2026-09-27/report.md`.
- B1: augmented QR без rank truncation доказан и reference-tested;
  d10/P40 сокращается до 11 строк, d100/P40 не сокращается.
  Common-metric spectral solve доказан условно, proportional metrics
  отвергнуты frozen inputs. Gradient averaging/PCA не equivalent в общем.
- B2: six hash-verified snapshots, 9 paired repeats и два synthetic kernel
  regimes; raw `audit_final.json` содержит profiles/env/source hashes,
  wall/allocation, reference errors и correction certificates. Для d100
  основные A/A* уже matmul; инженерные substitutions не дали убедительного
  общего trajectory gain. Стресс с большими mass выявил certificate
  sensitivity к reassociation. Все варианты остались isolated.
- B3: новые 8 tests passed; совместная regression 70 passed / 1 failed
  (известный manifold default local_quadratic при m=2). Target Ruff,
  Pyright, format и diff checks прошли. Итоги проверки:
  `docs/experiments/beta_functional_2026-09-27/verification.md`.
  Исправлены только affected solver/TeX notes, PLAN/STATE/DECISIONS.

Production `ADP/`, estimator/default/API/tolerances не менялись. Новые
full-fit selection/held-out/GPU прогоны не запускались. Это завершённая
формализация и bounded audit, не обещание скорости/recovery полного fit.
