# ADP agent notes

Исследование идей e-means с сохранением анизотропного ADP:
[e-means-adp.md](e-means-adp.md) (изолированный optimizer, доказательства и gates).

Последняя интеграционная проверка: [adp-moment-integration.md](adp-moment-integration.md)
(остаточный старт, предложения при прежнем функционале, локальный WLS2).

Текущая завершённая проверка слабых моментов: [weak-moment.md](weak-moment.md)
(WLS-метрика, residual-pHd, общий quadratic profile, Fourier-пределы).

This directory is the durable routing layer for agent work. Do not scan the repository from scratch when these notes already identify the relevant path.

Operational files:

- [`STATE.md`](STATE.md) — current compact handoff state; rewrite in place.
- [`WORKFLOW.md`](WORKFLOW.md) — `PLAN.md`, Markdown checkpoints, handoff, map maintenance, and optional machine-readable history.
- [`DECISIONS.md`](DECISIONS.md) — append-only durable decisions.
- [`history/state.json`](history/state.json) — optional machine-readable state snapshot; may be older than `STATE.md`.
- [`history/events.jsonl`](history/events.jsonl) — optional append-only development events.
- [`contracts/`](contracts/) — on-demand numerical, research, and engineering rules moved out of root `AGENTS.md` to save context.

For any task, read this file only far enough to choose a thematic route, then open that route and its cited source locators. Do not preload all notes.

Сжатая документация для агентской работы с алгоритмами ADP собрана здесь в двух независимых разделах:

- [TeX-спецификации](tex/README.md) — пересказ математических фрагментов трех исходных файлов в `tex/`, с кодами локаторов и командами `sed`.
- [Кодовая база ADP](ADP/README.md) — описание live-архитектуры, pipeline, single-index, multi-index, manifold, solver-ов, CLI и legacy-контуров. Каталог охватывает Python-модули пакета `ADP/`; каждый source ID раскрывается командой `rtk proxy sed -n`.

Это два разных источника: TeX описывает рукописные математические спецификации, а заметки `ADP/` — фактическое поведение текущего кода. Если формулы расходятся, не объединяйте их молча: сверяйте нужный локатор и фиксируйте выбранную интерпретацию.

Все команды локаторов запускаются из корня репозитория. Например:

    rtk proxy sed -n '174,280p' ADP/engine/common/index_fit.py
    rtk proxy sed -n '2066,2174p' docs/tex/manifold.tex

После объединения с Vlad_EDR_M (2026-10-26): [общий README](../../README.md),
[каталог экспериментов](../../benchmarks/experiments/README.md), [TeX](../tex/README.md),
[сохранённые серии](../../benchmarks/results/verification/README.md),
[целостность и проверки переноса](../integration/vlad_edr_m_20261026/README.md).


После разделения исследовательского архива и JSS-пакета (2026-10-29): [карта переноса и SHA-256 аудит](../integration/vlad_edr_m_20261029/README.md). `Vlad_EDR_M/Code` теперь содержит ADP-код и рукопись; эксперименты, гипотезы, тесты и результаты ведутся в `EDR_code`.
