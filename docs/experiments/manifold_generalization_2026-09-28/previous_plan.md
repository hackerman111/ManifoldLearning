task_id: single-multi-recovery-2026-09-27
status: active
change_class: ESTIMATOR (initialization), APPROXIMATE (solver comparison)
---

# Улучшение восстановления single и multi index

## Цель и границы

Проверить улучшение полного восстановления через начальное подпространство
и solver, разобрав идеи Method.md. Новые варианты сначала изолировать в
experiments/. Не менять defaults по одному запуску. Manifold вне scope.
Предыдущий PLAN сохранён в
docs/experiments/sparse_solver_suitability_2026-09-27/completed_plan.md.
Исходные dirty edits сохраняются.

## Известное и гипотезы

- local уже OPG: SVD локальных ridge-gradients; local-cv подбирает ridge
  по weighted LOO. pilot — tanh MLP, только multi.
- Прежние HYBRID/reduced-L-BFGS/GN не прошли gates. HPAO cap 80 помогает
  d10, но не устраняет плохое качество d100. Sparse audit: плотные матрицы.
- H1: SIR+SAVE start лучше выделяет симметричные Gaussian зависимости.
- H2: существующие local-cv/pilot улучшают start.
- H3: CG при одинаковом cap меняет endpoint; linear certificate недостаточен.

## Инварианты и read set

Float64, unit single / row-orthonormal multi, прежние rank guards, mass,
moments, kernel max(1-(distance2/h²)²,0), independent seed streams,
F=1/2 sum mass ||I-U B.T c||². Без (J,n,d), dense normal matrix и inverse.
Новый initializer не получает true basis; whitening/back-transform явные.

Method.md; contracts numerics/research/engineering; WORKFLOW;
agent-notes/ADP/{index-pipeline,solvers,single-index,multi-index}.md;
ADP/engine/common/{initialize,index_fit}.py; ADP_Config; solver LSMR/CG;
experiments/{data,models,runner,diagnostic,multiv2,multi}.py;
точечные initializer/model/solver tests. Primary literature — в отчёте.

## Этапы и acceptance evidence

- [x] R0: Method triage, baseline, конкретный manifest до selection.
- [x] R1: population scope SIR+SAVE, whitening/back-transform, operator,
  rank/eigengap/memory limits; отдельный аудит.
- [x] R2: isolated prototype, small dense reference, invariance/rank checks,
  memory stress. Failed formula/rank gate блокирует full fits.
- [ ] R3: frozen paired fits: до 6 профилей, 4 selection seeds, до 5 вариантов
  (local/LSMR, local-cv/LSMR, pilot/LSMR multi, SIR+SAVE/LSMR, local/CG).
  Приоритет n=1000,d=10/100 и symmetric links. Все failures в знаменателе.
  Initial/final quality, convergence, recovery, wall time и process RSS
  отдельно. Selection wall budget <=40 min.
- [ ] R4: только прошедшие профильные кандидаты, 10 untouched seeds;
  никаких изменений candidate/parameters/gates после selection.
- [ ] R5: opt-in только после validation; отчёт, runnable commands,
  STATE/DECISIONS/routes и статус done.

Recovery is geometric and independent of solver convergence: single uses
`abs(beta_true @ beta) >= 0.95`, multi uses `trace_score >= 0.95`. Manifold
requires both full-center RMS local-projector error and the maximum local
projector error across all centers to be <=0.2. Solver convergence and
stationarity remain experiment diagnostics and do not gate recovery or
candidate admission.
Per-profile gate: все пары без quality-регрессии >1e-10, не меньше
recovery, нет новых numerical failures, median quality gain >=0.01 либо
recovery gain >=1/4. Median time <=2x, RSS <=1.5x. Validation повторяет
per-pair nonregression и требует положительного gain.
Это scoped opt-in, не универсальное превосходство.

Stop: failed proof/reference — без full fits; failed selection — без
validation варианта; budget — явно incomplete. Gates не ослаблять.
Отрицательный исход закрыть с диагностикой, без смены defaults.

## Текущий checkpoint

R1: proof.md + независимый read-only аудит; Gaussian/coverage/domain
ограничения явные. R2: 4 numerical reference tests; n=10000,d=1000,m=2
stress 3.588 s, process RSS 392.24 MiB, eigen residual 1.36e-11.
Ruff/Pyright пройдены. Manifest фиксируется перед full fits:
single d10/d100 quadratic, single d100 square; multi d10/d100 additive,
multi d100 multiplicative; n=1000,tau=0.4,sigma_eps=0.2,cap=80,outer=3.
Selection 73000–73003, validation 74000–74009. 108 selection fits.
CG native convergence имеет иной step-count certificate; отдельно сохраняется
общий критерий stationarity max(grad,local_grad,orthogonality)<1e-6.
