# Numerical and ADP contract

Load this file only for mathematical, numerical, solver, performance, memory, or estimator changes. These rules were preserved from the previous repository-level `AGENTS.md` and remain binding unless the user explicitly changes them.

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
