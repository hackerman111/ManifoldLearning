# AGENTS.md

## Purpose

This repository is research software for Average Derivative Procedure (ADP), sufficient dimension reduction, single-index and multi-index models, manifold variants, and related numerical experiments.

Priorities, in order: mathematical correctness, numerical stability, bounded memory, measured wall-clock performance, maintainability, public API compatibility. Do not trade an earlier item for a later one without an explicit reason and evidence.

## Durable context

Chat history is disposable. Repository files are the durable memory.

Do not begin a non-trivial task by rescanning the repository. Start from the maintained routing notes and read source code only where the task requires verification or editing.

Read in this order:

1. `AGENTS.md`.
2. `PLAN.md` when a task is active.
3. `agent-notes/STATE.md`.
4. `agent-notes/README.md`, then only the thematic note(s) relevant to the task.
5. The exact source ranges, tests, configuration, or TeX locators referenced by those notes.

`agent-notes/` is a routing cache, not source of truth. Live code is authoritative for current behavior. TeX/manuscript sources are authoritative only for the formulas they actually state. If code, notes, and TeX disagree, do not silently reconcile them.

## Context discipline

Prefer targeted reads over discovery scans.

- Use the source IDs and `sed` locators already recorded under `agent-notes/ADP/` and `agent-notes/tex/`.
- Read a whole large module only when narrower ranges cannot answer the question.
- Search callers/references only for the symbol or contract being changed.
- Do not reread a paper, long log, or source region when an existing maintained note already answers the question; verify the original only when the claim matters to the current change.
- Do not load every contract below. Load only the contract relevant to the task.

Task routing:

- numerical algorithms, ADP formulas, solvers, performance, memory: `agent-notes/contracts/numerics.md`;
- experiments, scientific claims, benchmarks, reproducibility, numerical verification: `agent-notes/contracts/research.md`;
- Python/API/dependencies/backends/parallelism/tooling: `agent-notes/contracts/engineering.md`;
- persistent-state protocol and handoff rules: `agent-notes/WORKFLOW.md`.

## Maintaining `agent-notes`

Treat the notes as maintained indices. If a code change invalidates a documented route, behavior, source locator, architecture statement, compatibility boundary, or mathematical interpretation, update the affected note in the same task.

Do not regenerate or reread the entire map after each change. Refresh only the affected note and source ranges. Update `agent-notes/README.md` only when top-level routing changes.

When a note conflicts with code in a way that could affect the current task, verify the minimal relevant source, fix the note, then continue. Record a `map_update` event in `agent-notes/history/events.jsonl`.

## Work lifecycle

A non-trivial task means any multi-file change, mathematical or behavioral change, performance work, experiment, architecture decision, or work likely to span a context reset. Such a task must have an active `PLAN.md` before implementation.

Use a bounded cycle:

`orient -> plan -> execute one slice -> verify -> checkpoint -> next slice`

Do not mix unrelated refactors into the slice. A plan step is complete only when its stated evidence exists.

At every meaningful checkpoint update the durable state before continuing. A checkpoint is mandatory after:

- an accepted or materially revised plan;
- a non-trivial implementation slice;
- an experiment or benchmark whose result affects the next decision;
- a key mathematical, architectural, or compatibility decision;
- a failure that invalidates the current hypothesis or approach;
- immediately before context reset/handoff and at task completion.

The checkpoint protocol is defined in `agent-notes/WORKFLOW.md`. In short, keep `PLAN.md` current, rewrite `agent-notes/STATE.md` to the latest concise truth, update `agent-notes/history/state.json`, append one event to `events.jsonl`, and append to `DECISIONS.md` only for durable decisions.

## `PLAN.md`

`PLAN.md` is the current plan-of-record, not a diary. Keep it short enough to reread in a new session.

For research or numerical work it must state: goal, non-goals, known evidence, hypotheses when applicable, mathematical invariants, change class when applicable, exact read set, bounded work units, verification, and stop conditions.

Do not keep completed implementation chatter in the plan. Preserve durable outcomes in `STATE.md`, `DECISIONS.md`, experiment artifacts, and the machine-readable history.

## Reconnaissance and GPT-6 Luna

When answering the current question requires broad reconnaissance rather than local verification, delegate the corpus scan to a read-only **GPT-6 Luna** scout when the host exposes that model. Typical triggers are an unfamiliar subsystem spanning many files, a large experiment/log corpus, repository-wide dependency tracing, or broad literature/source discovery.

The scout must not implement or edit. Give it one precise question, explicit corpus boundaries, and this output contract:

- return only sources relevant to the question;
- for repository material, return `path:line-range` or an existing source ID plus one-line relevance;
- for external material, return the URL/citation plus one-line relevance;
- report contradictions and missing evidence explicitly;
- do not dump long excerpts or generic summaries;
- keep the report compact enough that the lead agent can use it without importing the scout's working context.

The lead agent should trust the scout for routing, not for final scientific correctness. Re-open only the few cited sources needed to edit code, establish a mathematical claim, or resolve a contradiction. If GPT-6 Luna is unavailable, use the cheapest capable read-only scout with the same contract.

Do not spawn a scout for a local change whose relevant files are already identified by `agent-notes`.

## Scientific change control

Before changing mathematical behavior, classify the change as `EXACT`, `NUMERICAL`, `APPROXIMATE`, or `ESTIMATOR` according to `agent-notes/contracts/research.md`.

`APPROXIMATE` and `ESTIMATOR` changes must be explicit variants/options/experimental branches unless the user specifically requests a behavior change. Performance work must not silently change the estimator.

For mathematical or performance work, preserve or create a small auditable reference path and compare against it before relying on large benchmarks.

## Verification and completion

Do not claim completion from inspection alone. Run the smallest relevant checks first, then broader checks when the change can affect shared behavior.

Before finishing a non-trivial task:

- verify the implementation against the plan's acceptance evidence;
- update affected `agent-notes` routes;
- update `PLAN.md`, `STATE.md`, and machine-readable state/history;
- append any durable decision to `DECISIONS.md`;
- leave the repository in a state from which a fresh agent can resume by reading only `AGENTS.md -> PLAN.md -> STATE.md -> relevant agent-notes`.

If the task is complete, mark `PLAN.md` and `history/state.json` as `done`; do not erase the evidence or append-only history.
