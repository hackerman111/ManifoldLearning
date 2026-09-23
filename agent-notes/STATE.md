# Current agent state

## Task

- ID: bottlenecks-multi-manifold-2026-09-23
- Status: active (planning checkpoint; measurements pending)
- Goal: найти и формализовать bottleneck времени и памяти CPU-путей multi-index и manifold ADP
- Plan: `PLAN.md`

## Established facts

- Multi-index идет через `engine.common.index_fit.fit_index`; его `IndexProfiler` уже разделяет основные фазы.
- Manifold идет через `engine.manifol_engine.fit`; его trace дает solver diagnostics, но не полный профиль времени по фазам.
- Дерево содержит незавершенные правки; стартовый HEAD `31b7ae7`. Измерения должны фиксировать dirty state.
- Подтвержденных измерениями этой задачи bottleneck пока нет.

## Current hypothesis / approach

Сначала зафиксировать контрольную и представительскую CPU-конфигурации для каждой модели и формат воспроизводимых результатов; затем измерить фазы, проверить масштабирование и оформить выводы.

## Active read set

- `agent-notes/ADP/index-pipeline.md`
- `agent-notes/ADP/multi-index.md`
- `agent-notes/ADP/manifold.md`
- `agent-notes/ADP/solvers.md`
- Source IDs и benchmark/test paths указаны в `PLAN.md`.

## Last verified evidence

План составлен по live-маршрутам `SRC-IFIT-LOOP` и `SRC-MAN-FIT-ORCH`; замеры еще не запускались.

## Open questions / blockers

Нет.

## Next action

Выполнить P1 из `PLAN.md`: уточнить параметры запусков по существующим benchmark/experiment entry points и сохранить baseline evidence.
