# Research, experiments, and verification contract

Load this file for changes that affect mathematical meaning, experiments, benchmarks, scientific claims, tests, diagnostics, or reproducibility. It contains the project's durable scientific checks.

## Source of truth

The implementation, manuscript, experiment specifications, and older branches may disagree.

Never silently reconcile them.

Before changing mathematical code, identify the exact implemented formula and classify the change:

- `EXACT`: algebraic or implementation change that preserves the current estimator and objective;
- `NUMERICAL`: mathematically equivalent reformulation intended to improve conditioning or floating-point behavior;
- `APPROXIMATE`: same target objective, but solved approximately or with a changed tolerance/sketch;
- `ESTIMATOR`: changes weights, bandwidth rules, centers, kernel, random-direction law, regularization, objective, or statistical procedure.

`APPROXIMATE` and `ESTIMATOR` changes require an explicit option, separate variant, or clearly isolated experimental branch. Do not hide them inside a performance refactor.

When code and theory disagree, preserve the current behavior unless the task explicitly asks to change it. Add a focused test that records the chosen interpretation.

## Working method

For nontrivial numerical changes:

1. Read the current implementation, its tests, and the formula it implements.
2. Write down array shapes and the dominant time/memory complexity.
3. Identify the actual hot path. Do not optimize from intuition alone when profiling is cheap.
4. Keep or create a small dense/reference implementation.
5. Implement the smallest useful change.
6. Compare the optimized implementation with the reference on small deterministic problems.
7. Measure wall-clock time and peak memory on a representative larger problem.
8. Record any changed numerical tolerance, iteration count, dtype, seed scheme, or algorithmic assumption.

A performance patch without correctness checks is incomplete.
A numerical patch without a failure-mode test is incomplete.
A research claim without a reproducible experiment is a hypothesis.

## Research integrity

Every benchmark that supports a research conclusion must record enough information to reproduce the run.

At minimum record:

- code version/commit and dirty state when available;
- exact kernel formula;
- whether the kernel argument is distance or squared distance;
- bandwidth rule;
- anisotropy rule and search direction;
- center-selection rule;
- random-direction distribution and refresh rule;
- solver and tolerances;
- ridge parameters;
- dtype;
- chunk size;
- thread/process configuration;
- Python, NumPy, SciPy and BLAS environment;
- seed scheme.

Do not silently update old experiment outputs after changing any of these.

Failed runs remain in result tables with an error field. Do not drop failures before aggregating metrics.

Do not present a one-seed improvement as a stable result.

When writing research notes, distinguish:

- literature result;
- mathematical consequence of the implemented formulas;
- experimental hypothesis;
- engineering proposal.

Do not turn a measured correlation into a theorem.

## Benchmarking

Performance means both time and memory.

Use `time.perf_counter()` for wall-clock timing.

Use an appropriate memory tool such as `tracemalloc` for Python allocations, and remember that it does not capture every native allocation. State the measurement method.

For hot-path changes report at least:

- problem shape `(n, d, J, P[, m])`;
- dtype;
- wall-clock;
- peak memory;
- solver iteration count when relevant;
- output error versus reference;
- quality metric when the change can affect estimation.

Warm-up effects, BLAS threading, and GPU transfer time must be accounted for when applicable.

Do not optimize plotting, CSV output, or CLI formatting before the numerical core unless profiling shows they matter.

### Performance claims

Do not write “faster”, “memory efficient”, or “more stable” without one of:

- a complexity argument tied to the actual allocation/operation removed;
- a benchmark;
- a numerical error comparison.

Prefer concrete statements such as:

- removes a persistent `(J, P, d)` array;
- replaces `O(J n d)` working memory with `O(B n + B d)`;
- reduces LSMR iterations from 84 to 27 on the recorded benchmark;
- keeps relative error below `1e-10` in `float64`.

## Testing

New numerical code needs more than ordinary unit tests.

### Reference tests

For small dimensions, compare optimized code against the simplest trustworthy dense/direct implementation.

Reference code may be slow. It should be easy to audit.

### Algebraic equivalence

Test identities used for optimization, for example:

- GEMM pairwise distances versus explicit differences;
- dense versus chunked statistics;
- dense weights versus sparse/neighbor representation;
- explicit `U @ v` versus factorized operator action;
- forward/adjoint identity;
- direct ridge solution versus matrix-free solve on a small problem.

### Numerical stress

Include tests with:

- large constant offsets in `X`;
- correlated or nearly collinear features;
- highly unequal local masses;
- nearly degenerate local slopes;
- small and large ridge values.

### Invariances

Test invariances that belong to the model:

- `beta` versus `-beta`;
- chunk size should not materially change a `float64` result;
- center batching must preserve row ordering;
- repeated runs with the same seed are reproducible.

For multi-index code test:

- orthonormality;
- invariance to a change of basis inside the same subspace when the objective has that invariance;
- principal-angle/projector agreement.

### Solver tests

For each iterative solver test:

- finite output;
- stop/status handling;
- residual;
- iteration limit behavior;
- warm start;
- preconditioner on/off consistency;
- failure behavior.

### Tolerances

Use `rtol` and `atol` that are justified by scale and dtype.

Do not use a loose tolerance merely to make a failing derivation pass.

Do not require bitwise equality for reductions whose order can legitimately differ.

## Performance regression tests

Keep microbenchmarks separate from correctness tests.

A benchmark should test a known hot path with a stable synthetic input.

Do not make normal `pytest` depend on noisy wall-clock thresholds unless the threshold is extremely coarse and only detects catastrophic regression.

For important optimizations, retain a benchmark script or reproducible benchmark fixture so the claim can be rechecked later.

## Diagnostics

Diagnostics must be cheap relative to the algorithm unless explicitly enabled.

Do not store full weight matrices, direction tensors, or per-iteration large arrays merely for plotting.

Prefer scalar summaries:

- objective;
- `h`;
- `rho`;
- local-mass quantiles;
- effective sample-size quantiles;
- solver iterations;
- residuals;
- `beta` change;
- orthogonality/subspace error.

If array tracing is enabled, make the memory cost explicit.

## Completion checklist

Before declaring a numerical task complete, verify:

- the implemented formula is identified;
- the change classification is stated internally (`EXACT`, `NUMERICAL`, `APPROXIMATE`, or `ESTIMATOR`);
- shapes are correct;
- no forbidden large allocation was introduced;
- non-finite and degenerate cases are handled;
- iterative operators pass an adjoint test when applicable;
- iterative solver status/residual is checked;
- a small reference test passes;
- deterministic seed behavior is preserved;
- hot-path changes have before/after timing and memory measurements;
- estimator-changing changes are isolated and not silently made the default;
- comments explain non-obvious mathematical/performance decisions;
- unrelated code was not reformatted or renamed.
