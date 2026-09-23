# Current agent state

## Task

- ID: optimize-manifold-projectors-hpao-lsmr-2026-09-23
- Status: active; plan written, mathematical work not started
- Goal: после математического вывода точных более эффективных вариантов ускорить manifold projector update и current HPAO/LSMR
- Plan: `PLAN.md`; next step M1

## Established facts

- `docs/adp_fit_bottlenecks.md` измерил manifold projector update как 56–58% полного CPU fit и HPAO/LSMR как 65–73% сходящегося multi-index fit на указанных там формах. Это исходные, а не универсальные оценки.
- Manifold `_one_step` включает slopes, B-system, CG, recovery и objective; HPAO/LSMR включает matrix-free ridge correction и строгий normal residual certificate. Точные read routes и математические инварианты записаны в `PLAN.md`.
- Production-код для этой оптимизации еще не менялся. Сначала завершить M1–M3 и математический gate G; затем отдельно менять код каждого принятого варианта.
- Предыдущий план инструкций сохранен в `agent-notes/history/plans/optional-agent-json-2026-09-23.md`; более ранний план профилирования — в `agent-notes/history/plans/bottlenecks-multi-manifold-2026-09-23.md`. Существующие незакоммиченные изменения и экспериментальные данные сохранены.

## Last verified evidence

- План составлен по live-фрагментам `SRC-MAN-STEP`, `SRC-MAN-B-SYSTEM`, `SRC-MAN-RECOVER`, `SRC-HPAO-ENTRY`, `SRC-HPAO-OP` и отчету `docs/adp_fit_bottlenecks.md`.
- Математический вывод новых вариантов, новые benchmark-запуски и кодовая верификация еще не выполнялись.

## Open questions / blockers

- Неизвестно, какой точный вариант даст устойчивый выигрыш полного `fit` при ограниченной памяти. Это предмет M2/M3, а не предположение плана.

## Next action

M1: проверить текущий checkout, зафиксировать точные локальные задачи и воспроизвести парные baseline/reference для manifold и HPAO без правок production-кода.
