# Single-index benchmark worker lifecycle fix

## Goal

Remove the benchmark-side process-lifecycle regression that repeatedly tears
down Python interpreters during a series and provide a genuinely process-free
safe mode for `--jobs 1`. The numerical algorithm, generated datasets,
persisted schema, reports, and public CLI remain unchanged.

## Evidence and failure mechanism

Commit `c37fd41` changed the runner in two relevant ways: it removed the serial
path and added `max_tasks_per_child=1` to `ProcessPoolExecutor`. With that
setting every completed fit retires its worker and starts a fresh spawn worker.
The failed 1,080-job series therefore requested up to 1,080 interpreter
lifecycles instead of reusing six workers. Four recorded workers subsequently
failed during interpreter shutdown before the kernel reported a corrupted page
table.

Earlier benchmark revisions completed series by reusing a fixed worker pool,
and `jobs=1` executed directly in the parent process. The fix restores those
two known behaviors without claiming to repair the separate kernel, driver, or
hardware fault exposed by the workload.

## Chosen approach

Make two targeted changes in `adp.evaluation.single_index.runner`:

1. Omit `max_tasks_per_child` when constructing `ProcessPoolExecutor`, leaving
   the default worker lifetime equal to the pool lifetime.
2. Route `process_jobs == 1` through a restored `_run_serial` function. Each
   outcome is committed before progress advances, matching the parallel
   commit-before-progress invariant.

The parallel path keeps its existing worker initializer, one-thread runtime
limits, per-worker persistence, run-id validation, cancellation, and error
propagation. The change does not select `fork`, add retries, or silently resume
after a crashed worker.

## Alternatives not selected

Moving pandas and report imports out of spawned workers would require changing
the package's eager public imports and the dataset module, which also imports
pandas. That is a broader dependency-boundary refactor and is not necessary to
remove the proven per-fit teardown regression.

Replacing the scheduler with a bounded submission window and additional
circuit breakers could further harden large series, but it changes scheduling
semantics and is outside this focused repair. It can be designed separately if
worker failures remain after the lifecycle regression is removed.

## Failure handling

An abrupt parallel worker termination continues to surface through
`BrokenProcessPool` or the original future exception. The `finally` block still
shuts down the pool with pending futures cancelled. Serial execution propagates
unexpected exceptions immediately and leaves already committed jobs available
for the existing resume workflow.

## Test strategy

Implementation follows RED-GREEN-REFACTOR without running a numerical
benchmark:

1. Replace the old recycling assertion with a regression test that records
   pool constructor arguments and proves `max_tasks_per_child` is absent.
2. Add a runner-level test proving `jobs=1` selects `_run_serial` and never
   constructs a process pool. Use fake execution and storage boundaries so no
   ADP fit or spawned interpreter is started.
3. Verify the focused runner tests, the complete single-index unit-test slice,
   and `git diff --check`.

No `smoke`, `minimal`, or `full` benchmark profile is executed as part of this
repair. A real workload remains explicitly deferred because the host has
already experienced kernel page-table corruption.

## Success criteria

- Parallel workers are reused for the lifetime of one pool.
- `--jobs 1` performs no process-pool construction.
- Commit-before-progress and resume behavior remain intact.
- Focused and single-index unit tests pass without starting the real benchmark.
- Existing unrelated working-tree changes are preserved.
