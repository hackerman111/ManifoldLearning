---
task_id: optional-agent-json-2026-09-23
status: done
created_utc: 2026-09-23T14:31:19Z
updated_utc: 2026-09-23T14:32:54Z
change_class: N/A
---

# PLAN — необязательная JSON-история агентов

## Цель

Убрать из инструкций обязательное обновление `agent-notes/history/state.json` и `events.jsonl`. Сохранить короткий и достаточный Markdown handoff через `PLAN.md` и `agent-notes/STATE.md`.

## Вне задачи

- Менять данные прошлых экспериментов или удалять существующую JSON/JSONL-историю.
- Менять вычислительный код и контракты ADP.

## Известное состояние

- До этой правки обязательные JSON/JSONL-записи предписывались в `AGENTS.md`, `agent-notes/WORKFLOW.md`, `agent-notes/history/README.md`; верхний индекс `agent-notes/README.md` называл их операционными файлами.
- Предыдущий завершенный план сохранен без изменений в `agent-notes/history/plans/bottlenecks-multi-manifold-2026-09-23.md`; его отчет и результаты остаются по адресам, указанным там.
- `PLAN.md`, `STATE.md`, `DECISIONS.md` и файлы истории уже имели незакоммиченные изменения до этой задачи.

## Рабочие шаги и проверка

| Шаг | Статус | Работа | Проверка |
| --- | --- | --- | --- |
| P1 | done | Обновить правила в `AGENTS.md`, `WORKFLOW.md` и индексах `agent-notes`. | Обязательные записи JSON/JSONL удалены из checkpoint, handoff, map maintenance и completion; формат оставлен необязательным. |
| P2 | done | Обновить `STATE.md` и завершить план без записи нового JSON/JSONL. | `git diff --check` прошел; прошлый план сохранен в Markdown-архиве; в `STATE.md` есть результат и следующий шаг. |

## Условие остановки

Если правила начинают требовать дублировать одни и те же факты в нескольких файлах, сократить их до одного актуального Markdown handoff с ссылками на доказательства.
