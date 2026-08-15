# Exact Box Order-Statistics Fast Path

Date: 2026-08-12

## Goal

Replace repeated sparse bisection for the box kernel with exact global order
statistics while preserving the current mean-mass estimator. The immediate
target is single-index `h0` and `rho`; isotropic box bandwidth selection is
shared by local initialization and the multi-index model.

Row-wise coverage is not part of this change because it changes the estimator.
Plateau behavior and automatic `box -> plateau` switching remain unchanged.

## Dense Geometry Cache and Fallback

`NeighborhoodEngine` lazily caches the fixed squared Euclidean distance matrix
`D` in `float64` only for the box fast path. The existing pairwise-distance
formula is reused. A cache is allowed when the resulting `D` occupies at most
64 MiB; larger problems retain the current cKDTree and sparse-bisection path.

The dense object is geometry, not a weight matrix. Box output remains CSR-like
neighbor indices with implicit unit weights. Temporary selection arrays may be
dense only while the cache-enabled path is active and are released after the
selection call.

## Exact Bandwidth Selection

For `J` centers, target mean mass `N_loc`, and
`L = ceil(J * N_loc)`, select the `L`-th smallest entry of `D` with
`np.partition(..., L - 1)`. The strict box boundary is represented by

```python
h = np.nextafter(np.sqrt(threshold), np.inf)
```

and the configured lower bound is retained when it already has sufficient
mass. Ties may produce more than `L` edges, matching the strict mathematical
condition. Plateau and over-budget box problems retain the existing bisection.

## Exact Single-Index Scale Selection

For fixed `h` and unit `beta`, compute

```text
s2[j, i] = ((centers[j] - X[i]) @ beta) ** 2
tau[j, i] = (h**2 - s2[j, i]) / D[j, i]
```

Pairs with `D == 0` and `s2 < h**2` receive `tau = +inf`, so self-edges and
duplicate observations count as permanent support. Other non-feasible pairs
receive non-positive breakpoints.

If mass at `rho = 1` is sufficient, return one. If mass at `rho = 0` is
insufficient, return `None`. Otherwise select the `L`-th largest breakpoint
with one partition and return

```python
rho = np.nextafter(np.sqrt(breakpoint), 0.0)
```

The returned value is checked against the original strict comparison
`rho**2 * D + s2 < h**2`. If floating-point evaluation still places a selected
edge on the boundary, move `rho` by representable steps toward zero until the
required mass is present. Feasibility at zero was already checked, so this
terminates. This selection is the largest feasible scale; the opposite wording
in `tex/manifold_v2.tex` is corrected.

## Sparse Support

When the dense box cache is available, isotropic and single-index support is
built by scanning `D` in center blocks and encoding only matching observation
indices. No floating-point edge weights are stored. Plateau and box fallback
continue through the existing exact cKDTree screening.

The cache does not alter support keys, statistics reuse, trace diagnostics, or
the CPU/GPU statistics interface.

## Model Routing

The single-index update calls the exact box scale selector when available and
otherwise calls the existing sparse scale bisection. Multi-index `alpha`
selection remains on the existing path; extending the same breakpoint method
to its low-rank quadratic form is a separate optimization.

## Verification

Tests must cover:

1. Exact bandwidth equivalence to dense box counts, including ties and the
   strict boundary.
2. Exact `rho` equivalence to the former bisection within its tolerance.
3. `D == 0` permanent edges and infeasible `rho = 0` behavior.
4. Sparse support equality against the direct dense quadratic form.
5. Cache-limit fallback without allocating dense geometry.
6. A model-level guard proving box single-index routing does not call sparse
   scale bisection when the cache is active.
7. Focused and full test suites plus a matched one-step timing comparison on
   the reported `n=J=1000, d=100` configuration.

Timing results are reported separately from correctness. No end-to-end quality
or universal performance claim is inferred from a microbenchmark.
