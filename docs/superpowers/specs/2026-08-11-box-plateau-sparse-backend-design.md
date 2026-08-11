# Box and Plateau Sparse Kernel Backend

Date: 2026-08-11

## Goal

Add exact `box-only` and `plateau-only` kernel modes to the uppercase `ADP`
single- and multi-index algorithms. Both modes use one sparse neighborhood
engine and feed the existing statistics builder without materializing dense
`J x n` distance or weight matrices.

The new implementation logic lives in `ADP/engine/box_kernel.py`. Existing
files receive only the wiring needed to select the backend and consume its
sparse blocks.

## Non-goals

- No automatic `box -> plateau` transition.
- No heuristic reuse that can change the estimator.
- No replacement of existing Epanechnikov or user-provided callable kernels.
- No center collapsing when centers use different random directions.

## Public API

`ADP/engine/box_kernel.py` exports:

- `box_kernel(q)`, with value one for `q < 1` and zero otherwise;
- `plateau_kernel(q, tau=0.5)`, with value one for `q <= tau`, the specified
  quintic taper for `tau < q < 1`, and zero for `q >= 1`;
- a helper that creates and identifies a plateau callable with a validated
  `tau`;
- the sparse neighborhood block and state used by the models and statistics;
- the single-index, multi-index, isotropic-bandwidth, local-initialization,
  scale-search, cache, and sparse-block conversion functions required by the
  new backend.

The package exports the two kernels for direct Python use. The CLI accepts:

```text
--kernel box
--kernel plateau --kernel-tau 0.5
```

`0 < tau < 1` is required. `--kernel-tau` is rejected for other kernels so it
cannot be silently ignored. Existing `epanechnikov` and `module:function`
values retain their current behavior.

Sparse routing is determined from the selected callable. `ADP_Config.kernel`
remains callable, preserving experiment serialization and existing Python
experiment files. A plateau configuration uses a stable `functools.partial`
of `plateau_kernel`, which the existing experiment serializer already
supports.

`smart_weights` is rejected for box and plateau. These kernels always use
their exact sparse screening path, while the current smart scale selector is
specific to Epanechnikov mass.

## Kernel Representation

Neighborhoods use a CSR-like block:

- `indptr`: row offsets;
- `indices`: observation indices for all support edges;
- for plateau only, positions and values for edges in `tau < q < 1`;
- row mass and diagnostics derived without a dense weight matrix.

Box stores no persistent floating-point value per edge. Plateau treats all
non-boundary support values as implicit one and stores floating-point values
only for boundary edges.

The statistics path may create a temporary padded local array of size
`B x max_neighbors` for a center batch. It must not create `B x n` or `J x n`
weight matrices.

## Exact Neighborhood Search

One neighborhood engine is created per model fit. It owns the fixed input
data, centers, an Euclidean `scipy.spatial.cKDTree`, and exact reuse state.
SciPy is already a project dependency.

For each outer iteration:

1. Build the current projected coordinates.
2. Use projected-radius and Euclidean-radius queries to form candidate lists.
3. Intersect the safe candidate lists where both bounds are available.
4. Compute the exact anisotropic `q` only for candidates.
5. Keep exactly the pairs with `q < 1` and encode them in CSR order.

For single-index localization, the projected lower bound uses the current
one-dimensional projection and the Euclidean bound uses `h / rho` when
`rho > 0`. For multi-index localization, a projected tree uses coordinates
scaled by the square roots of the localization eigenvalues, and the Euclidean
bound uses `h / alpha` when `alpha > 0`. A zero scale disables only the
corresponding Euclidean bound; exact projected screening remains valid.

Strict final comparison against `q < 1` handles tree-query boundary inclusion
and floating-point roundoff consistently.

## Bandwidth, Initialization, and Scale Search

New kernels do not build the existing dense `pairwise_distance2(X, centers)`
matrix.

- Initial isotropic bandwidth is found by bracketed bisection over sparse
  radius queries and the actual kernel mass.
- Local initialization removes zero-weight rows and solves the same weighted
  local regressions using only each center's sparse neighborhood.
- Random and pilot initialization remain unchanged.
- `rho` and `alpha` use bracketed bisection over the actual sparse mass and
  return the largest feasible scale represented by the feasible lower bound.

Thus box mass is the exact neighbor count, while plateau mass is the sum of
implicit interior ones and explicit boundary weights. The target remains the
current mean local mass `N_loc` (or `N_lin` for local initialization), so the
algorithmic contract does not change.

## Statistics Integration

`calculate_statistics` and `calculate_statistics_gpu` accept either their
existing dense/iterated weight blocks or the new sparse blocks.

For sparse blocks, a shared adapter supplies local indices, normalized local
weights, row mass, and cached local data to the existing local-statistics
calculation. The mathematical formulas for `I`, `U`, `mean`, `n_eff`, and
`eta` remain common. CPU consumes the local arrays directly; GPU transfers
only the padded local arrays, not dense `B x n` weights.

Existing callable kernels continue through the current dense/block iterator
without behavior changes.

## Exact Reuse

Changing `P`, `Lambda`, `alpha`, or `h` can move any pair across a kernel
boundary. Because the source document gives no conservative motion bound,
the engine always recomputes exact candidate distances after such a change.

After recomputation:

- box cache keys are sorted support indices and can reuse local means,
  centered local `X`, local `Y`, and other support-only preparation;
- plateau cache keys include support and explicit boundary weights, so reuse
  occurs only when the weighted neighborhood is exactly unchanged;
- centers with identical box support share prepared local geometry, but retain
  their own directions and statistic rows;
- previous support and active/boundary values remain available for exact
  comparison and diagnostics.

No cached value is used merely because an edge was previously inside the
support.

## Diagnostics

Each outer trace row records:

- `support_edges`;
- `boundary_edges`;
- `support_reuse_hits`.

Effective parameters record the kernel mode and effective plateau `tau`.
Requested configuration remains represented by the callable already stored in
`ADP_Config`.

## Validation and Errors

The backend rejects:

- non-finite kernel inputs or geometry;
- `tau` outside the open interval `(0, 1)`;
- malformed CSR arrays, invalid indices, or negative boundary weights;
- any center row with non-positive or non-finite mass;
- `smart_weights` combined with box or plateau.

At `q = tau`, plateau returns exactly one. At `q = 1`, both kernels return
exactly zero.

## Verification

Focused tests cover:

1. Box and plateau values, support boundaries, and plateau first/second
   derivative continuity at `tau` and `1`.
2. Sparse single- and multi-index blocks against explicit dense reference
   formulas.
3. CPU statistics from sparse blocks against dense weights.
4. The GPU sparse adapter when CUDA is available, with normal explicit skip
   behavior otherwise.
5. Exact reuse hits for unchanged neighborhoods and invalidation after a
   support or boundary-weight change.
6. A guard proving new-kernel fits do not call the dense pairwise-distance or
   dense weight path.
7. CLI parsing, `tau` validation, configuration serialization, and single- and
   multi-index smoke fits.
8. The complete existing test suite after focused checks pass.

No performance claim is made from correctness tests alone. Runtime and memory
improvements require a matched benchmark with identical prepared input,
centers, seeds, and effective parameters.
