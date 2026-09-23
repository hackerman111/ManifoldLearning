# Persistent research workflow

This file defines how an agent carries a PhD-level research/software task across sessions without carrying chat history.

## State model

Use four layers with distinct jobs:

| File | Role | Update style |
|---|---|---|
| `PLAN.md` | current task and bounded work units | edit in place |
| `agent-notes/STATE.md` | concise current truth needed to resume | rewrite in place |
| `agent-notes/DECISIONS.md` | durable decisions and why | append only |
| `agent-notes/history/events.jsonl` | machine-readable development history | append only |

`agent-notes/history/state.json` is the machine-readable mirror of the current handoff state. It should agree with `STATE.md` and `PLAN.md`.

The architecture map under `agent-notes/ADP/` and `agent-notes/tex/` is a fifth layer: a routing cache for the repository. Update it only when the mapped facts change.

## Starting or resuming work

For a non-trivial task:

1. Read `PLAN.md` and `agent-notes/STATE.md` before exploring code.
2. Use `agent-notes/README.md` to select the smallest relevant architecture/theory note.
3. Follow its source IDs/locators to verify only the source needed for the current step.
4. Read relevant tests and configuration before editing.
5. If the map cannot identify the needed sources without broad scanning, use the scout protocol from root `AGENTS.md`.

Do not treat stale chat statements as project state when durable files disagree.

## Planning

Create or refresh `PLAN.md` before implementation when work is non-trivial. Keep one active task in the file.

A useful plan answers only what execution needs:

- what is being changed and what is explicitly out of scope;
- what is already established versus still hypothetical;
- which scientific/mathematical invariants must survive;
- which exact files/source IDs are expected to matter;
- what small work units will be executed;
- what evidence makes each work unit complete;
- when to stop rather than continue optimizing or patching.

For research questions, prefer an experiment that distinguishes competing hypotheses before building a large implementation around one hypothesis.

## Bounded execution

Execute one plan work unit at a time. After editing, run the focused verification for that unit before starting the next one.

If verification fails twice for the same conceptual reason, stop local patching. Record the failed assumption in `STATE.md` and `events.jsonl`, revise the plan or hypothesis, then proceed.

Large mechanical edits may span many files, but conceptual changes should remain narrow enough to review and verify independently.

## Checkpoint protocol

At a mandatory checkpoint:

1. Update statuses/evidence in `PLAN.md`.
2. Rewrite `agent-notes/STATE.md` with only the facts a fresh agent needs now.
3. Update `agent-notes/history/state.json`.
4. Append exactly one compact JSON object to `agent-notes/history/events.jsonl` for the checkpoint.
5. If a durable decision was made, append it to `agent-notes/DECISIONS.md`.
6. If code changes made an architecture/theory note stale, update only the affected note and include its path in the event.

Do not use `STATE.md` as an append-only log. Old details belong in events, decisions, experiment artifacts, commits, or archived plans.

## What belongs in `STATE.md`

Keep it roughly <= 120 lines. Include:

- active task/status;
- established facts that materially constrain the next step;
- current hypothesis or selected approach;
- files/source IDs currently relevant;
- last verified evidence;
- unresolved blockers/questions;
- exact next action.

Exclude raw tool output, long diffs, chat transcript, broad repo summaries already present in `agent-notes`, and completed history that no longer affects the next action.

## Decision log

Append to `DECISIONS.md` when choosing between materially different mathematical, architectural, compatibility, performance, or experimental approaches.

A decision entry records the decision, evidence/reason, alternatives rejected, affected files/contracts, and whether it is active or superseded. Do not log formatting choices or routine implementation details.

## Machine-readable history

Each line of `events.jsonl` is one object conforming to `history/event.schema.json`.

Use event types:

- `plan` — plan created/materially revised;
- `checkpoint` — implementation slice completed;
- `experiment` — experiment/benchmark result changed or confirmed direction;
- `decision` — durable decision recorded;
- `failure` — failed approach invalidated an assumption;
- `map_update` — `agent-notes` route/description refreshed;
- `verification` — significant verification completed;
- `handoff` — state prepared for a fresh session;
- `completion` — task finished.

Prefer paths to evidence over embedding evidence. Store benchmark/result files under the project's normal experiment layout and reference them from the event.

Do not rewrite prior JSONL events. Corrections are new events referring to the earlier event ID in `supersedes`.

## Architecture-map maintenance

`agent-notes` saves tokens only while it is trustworthy.

When editing a mapped source:

- check whether the relevant note's behavior statement or locator changed;
- refresh only affected ranges/IDs;
- preserve the distinction between live-code notes and TeX notes;
- never change a note to hide a code/theory disagreement;
- if a source path is removed or responsibility moves, update the thematic route and top-level README when necessary;
- append a `map_update` event listing both the note and verified source paths.

Do not perform a repository-wide map rebuild merely because one line number moved.

## Handoff / new chat

Before a context reset or new chat, create a checkpoint and then append a `handoff` event.

A fresh agent must be able to resume with this minimal read sequence:

`AGENTS.md -> PLAN.md -> agent-notes/STATE.md -> relevant thematic note -> cited source/tests`

If that sequence is insufficient, improve the durable files before handing off rather than pasting more chat history.

## Completion

A task is complete only when implementation/research evidence is verified and the durable state has been updated. Mark the plan and `state.json` as `done`, leave `STATE.md` with the final result and any follow-up, and append a `completion` event.
