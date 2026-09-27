---
task_id: multi-index-solver-retry-2026-09-24
status: done
created_utc: 2026-09-24
updated_utc: 2026-09-24
change_class: EXPERIMENT_DESIGN
---

# PLAN — повторная оптимизация multi-index solver

## Цель и границы

Повторить ограниченный поиск после анализа прежних неудач: ускорить достижение
прежнего внутреннего certificate, сохранив статистический objective и качество
полного ADP fit. Вне задачи estimator/kernel/локализация/инициализация, GPU,
manifold-модель и смена публичного default без evidence.
Предыдущий завершённый план и его математический контракт/proof gate сохранены
в `agent-notes/history/plans/multi-index-solver-search-2026-09-24-complete.md`.

## Инварианты и точный маршрут

`F(B,C)=1/2 Σ mass_j ||I_j-U_j Bᵀc_j||²`, `BBᵀ=I_m`, прежний minimum-norm
local refit/cutoff, float64, HPAOResult/orientation и нормировки общего
certificate. `lambda_prox` относится только к correction, не входит в F.
Два последовательных реальных принятых шага; не снижать tol/не скрывать
rank/failure. Production без d×d, (md)², (JP,md); диагностический dense
reference ограничен d<=100. Новый конечный solver — явный APPROXIMATE вариант;
ускорение алгебры при сохранении цели — EXACT/NUMERICAL.

Маршрут: `AGENTS.md -> PLAN.md -> agent-notes/STATE.md ->
agent-notes/ADP/solvers.md`; numerics/research/engineering contracts и
`agent-notes/WORKFLOW.md` при продолжении. Live locators:
`ADP/solver/LSMR.py:40–340,429–700`, `HYBRID.py:29–434`,
`_multi_operator.py`. Эксперименты:
`experiments/{multi_solver_retry_diagnose,multi_solver_frozen,
reduced_gauss_newton}.py`; тест `tests/test_reduced_gauss_newton.py`.

## Выполненные bounded этапы

|Этап|Статус и evidence|
|---|---|
|R1|done — две frozen HPAO диагностики, четыре точных linear probes, три bounded curvature probes; `docs/experiments/multi_solver_retry_2026-09-24/r1.md`.|
|R2|done — две заранее сформулированные гипотезы; вывод/отдельный audit для H6 и H7 до прототипов; `hypotheses.md`, `proof_certified_inner.md`, `proof_reduced_gn.md`.|
|R3|done — H6 24 paired frozen trials, H7 12 frozen trials, без validation; `h6.md`, `h7.md`, raw под `benchmark_outputs/diagnostic/multi_solver_retry_*_20260924/`. Оба кандидата провалили frozen gate.|
|R4|skipped — full-fit selection и held-out не начаты по stop condition; seed3000–3019/2000–2019 не использованы.|
|R5|done — отрицательный итог, raw/проверки/отклонённый patch сохранены; `completion.md`, STATE/DECISIONS и affected solver note обновлены.|

## Результат и stop condition

На d100 H6 получил 3/6 certificates как baseline при median paired wall
ratio1.264; H7 — 4/6, median spent wall ratio3.883, peakRSS1.748, хуже
objective на сертифицированной задаче и один явный line-search failure.
Порог требовал не меньше baseline certificates, допустимый objective,
median solver ratio<=0.70, d10 slowdown<=10%, peakRSS<=1.2; H7 провалил
несколько условий, H6 скорость/сходимость. Failed/time-limited trials
остались в знаменателе, их wall не признан time-to-certificate.

Оба выделенных кандидата исчерпаны. Новый поиск потребует отличающей
математической гипотезы, заранее заданного бюджета и независимого gate.
Низкую loss не трактовать как восстановление. Публичный solver/default и
estimator сохранены; H6 production patch отменён и архивирован, H7 остался
изолированным `experiments/` прототипом. Полный проверяемый отчёт —
`docs/experiments/multi_solver_retry_2026-09-24/completion.md`.
