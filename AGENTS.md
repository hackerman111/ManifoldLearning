# AGENTS.md

## Scope

This repository is research software for Average Derivative Procedure (ADP), sufficient dimension reduction, single-index and multi-index models, and related numerical experiments.

Write code for research use first:

1. mathematical correctness;
2. numerical stability;
3. bounded memory;
4. measured wall-clock performance;
5. readability and maintainability;
6. public API compatibility.

Do not trade an earlier item for a later one without an explicit reason and a benchmark or mathematical argument.

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

## Python style

Use modern, explicit Python.

- Put `from __future__ import annotations` in new Python modules.
- Type public functions, methods, dataclasses, and non-obvious internal interfaces.
- Prefer `dataclass(frozen=True, slots=True)` for immutable configuration/data records.
- Prefer `dataclass(slots=True)` for mutable result/state records when mutation is intentional.
- Keep configuration separate from runtime state.
- Use `pathlib.Path` for filesystem paths.
- Use `collections.abc` types for runtime-facing protocols such as `Callable`, `Iterable`, and `Iterator`.
- Use `np.random.default_rng`; never introduce new code based on the global NumPy RNG.
- Use explicit exceptions with actionable messages.
- Validate user-facing inputs at API boundaries. Do not repeat expensive validation inside hot loops.
- Do not catch broad exceptions in numerical core code unless the failure is re-raised with context.
- Do not silently replace an invalid solver result with a previous iterate unless that fallback is part of the documented algorithm.

Keep functions narrow. Split orchestration from numerical kernels.

Prefer names that match the mathematics already used by the project: `beta`, `rho`, `h`, `mass`, `directions`, `slopes`, `centers`, `weights`, `I`, `U`. Do not introduce synonyms for the same object only for stylistic variety.

Preserve existing public names even if an older API does not follow current naming conventions. Do not perform API renames as a drive-by cleanup.

## Comments and docstrings

The project uses concise Russian comments and docstrings. Follow that style in research-facing modules.

Comments should explain one of:

- the mathematical identity being used;
- the reason for a numerically safer formulation;
- a shape or memory constraint that is not obvious from the code;
- why an apparently simpler implementation is intentionally avoided;
- the provenance of an empirical threshold.

Do not narrate obvious Python syntax.

For important numerical functions, document:

- input meaning;
- output meaning;
- non-obvious shapes;
- mathematical invariant;
- failure conditions.

Do not put long textbook explanations into hot-path functions.

## Array conventions

Use row-major observation layout unless the existing module explicitly defines another convention:

- `X`: `(n, d)`;
- `y`: `(n,)`;
- `centers`: `(J, d)`;
- `directions`: `(J, P, d)`;
- local projected statistic `I`: `(J, P)`;
- explicit local operator `U`: `(J, P, d)`;
- weights: `(J, n)` only when a dense representation is intentionally justified.

Treat shapes as contracts. Check them at module boundaries.

When a shape is non-obvious, annotate it next to the operation:

```python
projected = np.einsum("jpd,d->jp", U, beta, optimize=True)  # (J, P)
```

Avoid transposes whose only purpose is to repair an unclear upstream layout.

## Allocation rules

Memory behavior is part of correctness for this project.

### Forbidden by default

Do not materialize:

- a differences tensor of shape `(J, n, d)`;
- a projected differences tensor of shape `(J, P, n, d)`;
- a dense `d x d` localization tensor when its action has a low-rank formula;
- a dense `d x d` normal matrix only to call a generic dense solver;
- a full dense weight matrix if compact support is already sparse enough that a neighbor representation is cheaper;
- repeated copies of large arrays for diagnostics.

For the stress regime `n = 10_000`, `d = 1_000`, any proposed persistent allocation must be mentally costed in bytes before implementation.

### Pairwise distances

Use the Gram identity rather than broadcasting a `(J, n, d)` difference tensor:

```python
distance2 = (
    np.square(centers).sum(axis=1)[:, None]
    + np.square(X).sum(axis=1)[None, :]
    - 2.0 * centers @ X.T
)
np.maximum(distance2, 0.0, out=distance2)
```

The clipping is required because roundoff can make squared distances slightly negative.

If `J * n` itself is too large, tile over centers or observations instead of falling back to `(J, n, d)` broadcasting.

### Chunking

Chunk over the dimension that bounds working memory, usually centers.

A chunk is an implementation detail and should not change the mathematical result beyond normal floating-point reduction differences.

Choose chunk sizes from memory behavior and benchmarks. Do not encode a new unexplained magic number.

### Sparse neighborhoods

Compact-support weights should lead to sparse computation when sparsity is material.

Preferred representations:

- CPU with irregular row lengths: CSR or an equivalent row-oriented structure;
- bounded/fixed neighborhood size: neighbor lists;
- GPU-oriented batched kernels: neighbor lists are usually preferable to generic sparse assembly.

Do not convert sparse support back to dense merely because a dense NumPy expression is shorter.

## Vectorization policy

Vectorization is a means, not a goal.

Prefer BLAS-backed `@`, `np.einsum(..., optimize=True)`, reductions, and batched matrix multiplication when they reduce Python overhead without creating oversized intermediates.

A small Python loop over:

- outer ADP steps;
- random directions `P`;
- center chunks;
- a small target dimension `m`;

is acceptable when it avoids an allocation proportional to `J * n * d`, `E * P * d`, or another large product.

A Python loop over all observations, features, or all `(j, i)` pairs in a hot path is normally not acceptable.

Never use `np.vectorize` as a performance optimization.

## Reuse and caching

Reuse expensive quantities only when the cache has a clear invalidation rule.

Good candidates include:

- `||X_i - x_j||^2` when centers and data are unchanged;
- `X @ beta` within a step;
- low-dimensional projections used by both screening and weights;
- factorized local statistics reused by multiple solver iterations.

Cache keys must reflect every quantity that changes the result.

Do not cache large arrays merely because recomputation feels wasteful. Compare recomputation cost with memory pressure.

Do not keep both a dense representation and an equivalent sparse/factorized representation unless a benchmark proves the duplication useful.

## Numerical linear algebra

### Never invert explicitly

Do not use `np.linalg.inv(A) @ b`.

Use, depending on the problem:

- `np.linalg.solve`;
- `np.linalg.lstsq`;
- QR;
- SVD;
- LSMR/LSQR;
- a justified Krylov method.

For rank-deficient or nearly rank-deficient local systems, prefer rank-revealing QR/SVD or `lstsq`.

### Least squares and ridge

For large ridge least-squares problems, prefer the augmented operator formulation:

\[
\min_x \left\|
\begin{bmatrix}
A \\
\sqrt{\lambda}I
\end{bmatrix}
x -
\begin{bmatrix}
b \\
\sqrt{\lambda}x_0
\end{bmatrix}
\right\|_2^2.
\]

Use matrix-free LSMR/LSQR when `A` is large and only forward/adjoint actions are needed.

Do not form `A.T @ A` only to use CG or Cholesky. Normal equations square the condition number of the unregularized least-squares operator.

CG is acceptable when:

- the actual problem is naturally SPD;
- the operator is not an avoidable normal equation;
- or a benchmark shows a justified advantage and the conditioning consequences are documented.

### LinearOperator

Every nontrivial `LinearOperator` must have:

- a forward action;
- the mathematically correct adjoint;
- a small test of the adjoint identity

\[
\langle Ax, y\rangle \approx \langle x, A^\top y\rangle;
\]

- finite-result checks;
- solver diagnostics exposed to the caller or result object.

Do not trust a Krylov solve only because it returned an array.

Record or test solver status, iteration count, and a residual certificate when practical.

### Factorized local operators

If a local ADP operator admits a factorization such as

\[
U_j = Q_j R_j,
\]

prefer storing/applying the factors when this reduces memory.

Do not materialize `U_j` only to multiply it by a vector if an equivalent `Q_j @ (R_j @ v)` action is available.

If the exact finite-sketch objective is preserved, treat the factorization as `EXACT`.

Replacing the finite sketch by a full local row-space objective is an `ESTIMATOR` change and must be isolated accordingly.

## ADP-specific performance rules

### Localization tensors

Do not build a dense localization matrix when the quadratic form can be evaluated directly.

Single-index quantities should use formulas based on:

- `||X_i - x_j||^2`;
- `beta.T @ (X_i - x_j)`.

Multi-index quantities should use low-dimensional projected terms such as:

- `P @ (X_i - x_j)`;
- the orthogonal residual norm;
- the small `m x m` matrix acting in the EDR space.

Do not create a `d x d` tensor merely to evaluate a scalar quadratic form.

### Structural matrices

For multi-index code, do not form a `d x d` matrix such as

\[
\mathcal J = B^\top M B
\]

if `rank(M) <= m << d`.

Work through the small factor and use an SVD/eigendecomposition in dimension `m`.

### Screening

Exact zero-weight screening is preferred when it provably removes work without changing any nonzero weight.

Keep exact screening separate from approximate pruning.

If a pruning rule can produce false negatives, it is an `APPROXIMATE` or `ESTIMATOR` change and must be opt-in.

### Support-aware computation

When compact support makes the average neighborhood size `k << n`, target complexity in terms of the number of surviving edges

\[
E = \operatorname{nnz}(W)
\]

rather than `J * n`.

Do not quote sparse speedups using only asymptotic notation. Measure the actual crossover because irregular memory access can dominate.

## Numerical stability

Research code must prefer stable formulas over shorter formulas.

### Dtypes

Use `float64` as the default reference dtype for numerical research unless the task explicitly targets another dtype.

`float32` is allowed only when:

- it is an explicit configuration;
- correctness is compared with `float64`;
- the quality/runtime/memory tradeoff is measured.

Never silently downcast user data.

### Centering and local statistics

For weighted local statistics:

- compute normalized weights when appropriate;
- use the weighted local mean;
- center before reductions when this reduces cancellation;
- explicitly correct the weighted zero first moment when the algorithm relies on it.

Prefer evaluating a projection as

```python
phi @ (x - local_mean)
```

or an algebraically stabilized batched equivalent over subtracting two large nearly equal projected values when data have a large offset.

If using a global centering `Xc = X - x_bar`, do it once and reuse it.

### Degenerate neighborhoods

Check local mass and effective sample size.

A row with a tiny mass must not quietly produce `NaN`, `inf`, or a meaningless local coefficient.

Treat the effective sample size

\[
n_{\mathrm{eff},j}
=

\frac{(\sum_i w_{ij})^2}{\sum_i w_{ij}^2}
\]

as distinct from the number of nonzero weights.

### Denominators

A floating-point tiny constant is a division guard, not a statistical regularizer.

Use a named local ridge parameter when regularization is part of the method.

Do not hide an ill-conditioned local fit by replacing its denominator with `np.finfo(...).tiny` and proceeding as if the fit were reliable.

### Normalization and sign

Single-index directions represent a line, not an oriented vector.

After solving:

- verify finite values;
- verify nonzero norm;
- normalize deliberately;
- align sign to a prior/reference when continuity is needed.

Quality metrics must use sign-invariant quantities such as `abs(beta_true @ beta_hat)` when appropriate.

For multi-index code, compare subspaces with principal angles/projectors, not elementwise matrix distance between arbitrary bases.

## Randomness and reproducibility

Use local generators:

```python
rng = np.random.default_rng(seed)
```

For experiment suites, use `np.random.SeedSequence` to derive independent streams.

Do not reuse one opaque seed for all of:

- data generation;
- center selection;
- random directions;
- initialization;
- bootstrap;
- competing methods.

When methods are compared, use the same generated dataset but independent method-specific randomness unless common random numbers are intentionally part of the protocol.

Changing the order of unrelated computations should not accidentally change all subsequent random draws. Split random streams when ordering stability matters.

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

## Parallelism

Do not combine process-level parallelism with uncontrolled BLAS threading.

Experiment workers should normally pin BLAS thread counts to avoid oversubscription.

Parallelize at one dominant level.

Do not add multiprocessing inside a BLAS-heavy inner kernel without measuring it.

Do not introduce hidden global thread settings in importable library modules unless the project explicitly owns process configuration. Process-level settings belong in experiment/CLI entry points.

## Backend boundaries

Keep numerical solver code independent from plotting and pandas.

Core numerical modules should operate on arrays and small structured result objects.

Convert to pandas only in evaluation/reporting code.

Import matplotlib lazily in plotting/reporting paths.

If multiple array backends are supported, keep data on one backend throughout a numerical step. Avoid CPU/GPU ping-pong.

Backend abstractions must not force materialization of arrays that the NumPy backend can stream or factorize.

## API and architecture

Keep responsibilities separated:

- data preparation;
- bandwidth/anisotropy selection;
- weight/support construction;
- local statistic construction;
- solver;
- diagnostics;
- benchmarks/reports.

A variant class should contain formulas specific to that estimator, not generic CLI/reporting logic.

A solver should consume a clearly defined statistics/operator interface and return a structured result with diagnostics.

Avoid dictionaries whose required keys are known and stable. Prefer dataclasses/typed records for long-lived internal interfaces.

Do not expose implementation caches as public API.

Do not add dependency-heavy abstractions for one short kernel.

## Error handling

Reject invalid inputs early:

- wrong shape;
- complex or non-numeric data;
- non-finite values;
- impossible local-mass targets;
- zero direction;
- zero/invalid normalized index;
- non-finite solver output.

Distinguish:

- `TypeError`: wrong kind of object/dtype;
- `ValueError`: invalid value/shape/configuration;
- `RuntimeError`: numerical procedure could not produce a valid result;
- `NotImplementedError`: deliberately unsupported algorithmic case.

Do not convert numerical failure into a plausible-looking estimate.

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

## Dependencies

Prefer NumPy/SciPy primitives before adding a new dependency.

A new dependency is justified only if it provides a substantial capability that would otherwise require nontrivial, fragile code.

Do not add a framework only to replace a short `LinearOperator`, a small QR/SVD, or a few NumPy reductions.

Do not add JIT/GPU code before identifying a hot path and keeping a NumPy reference implementation.

## Anti-patterns

Do not introduce:

```python
diff = X[None, :, :] - centers[:, None, :]  # (J, n, d)
```

for large production paths.

Do not introduce:

```python
beta = np.linalg.inv(A) @ b
```

Do not introduce:

```python
normal = A.T @ A
rhs = A.T @ b
```

for a large least-squares problem only because CG expects an SPD operator.

Do not introduce `np.vectorize` for speed.

Do not repeatedly call `X @ beta` in the same step when it can be reused.

Do not call `astype(...)`, `np.asarray(...)`, or `.copy()` repeatedly inside a hot loop when conversion can happen once at the boundary.

Do not build a huge intermediate merely to reduce it immediately afterward.

Do not move pandas or matplotlib objects into the solver.

Do not hide empirical constants. Name them and state why they exist.

Do not “clean up” a mathematical formula while optimizing unrelated code.

## Preferred optimization order

When a function is too slow or memory-heavy, try in this order:

1. remove mathematically unnecessary work;
2. remove unnecessary large intermediates;
3. reuse already computed projections/reductions;
4. exploit compact support or low rank;
5. use matrix-free forward/adjoint actions;
6. batch/chunk to fit cache and memory;
7. use BLAS-backed vectorized operations;
8. add a preconditioner;
9. change algorithm/estimator only as a separate research option;
10. consider JIT/GPU only after the CPU reference path is correct and profiled.

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

## Project configuration

`pyproject.toml` is the source of truth for Python tooling, dependencies,
supported Python versions, linting, formatting, typing, and test configuration.

Before modifying Python code:

1. Read `pyproject.toml`.
2. Respect all configuration under:
   - `[project]`
   - `[dependency-groups]`
   - `[tool.ruff]`
   - `[tool.pyright]`
   - `[tool.pytest.ini_options]`
   - `[tool.coverage.*]`
3. Do not introduce configuration that conflicts with `pyproject.toml`.
4. Do not duplicate tool configuration in separate files unless explicitly requested.
5. Do not manually emulate lint/type/test rules. Run the configured tools.

Use the project environment and commands:

```bash
uv sync

uv run ruff format .
uv run ruff check .
uv run pyright
uv run pytest
```
