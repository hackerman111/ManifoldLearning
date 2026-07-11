# ADP NumPy Statistics Optimization Design

## Goal

Reduce the wall-clock time of the active lowercase `adp/` NumPy statistics
stage while preserving the current random-projection ADP estimator, public
stage contracts, result shapes, numeric dtypes, and CuPy behavior.

The primary acceptance target is the warmed NumPy CPU path. On the audit case
`n=1000`, `d=100`, `J=150`, and `P=16`, the compact-kernel center loop accounts
for almost all of `_compute_statistics` time. The optimized implementation must
reach at least a 1.5x median speedup for this statistics stage without degrading
direction quality or objective values. The serial optimized path must not
increase measured peak memory; an explicitly enabled parallel path has a 5%
peak-memory overhead limit.

## Current Bottleneck

The current path is:

1. `RandomProjectionADP._compute_statistics_default` obtains cached dense
   `J x n` squared distances and, after Step 0, cached projected distances.
2. It builds a new kernel argument `q` for each center chunk.
3. `NumpyBackend.random_projection_sums` sends compact kernels to
   `_compact_random_projection_sums`.
4. The compact implementation loops over centers in Python, selects active
   observations, and allocates `centered`, `projected`, and
   `weighted_projected` temporaries for every center.
5. Chunk results are copied into the full statistics accumulator.

The compact path avoids a dense `C x n x P` tensor, but it still pays for dense
`J x n` distance data and repeated per-center allocations. The existing
neighbor index only supplies a bandwidth-search hint and does not reduce the
statistics workload.

## Chosen Approach

Use a dependency-free fused NumPy implementation first, followed by an
explicit, bounded center-parallel path only if the serial implementation does
not meet the 1.5x speed target.

This approach was selected over two alternatives:

- A radius/kNN statistics pipeline could avoid dense `J x n` data in genuinely
  sparse neighborhoods, but the observed active fractions are commonly
  60-87%. In high dimension or for small anisotropy, the candidate set can
  approach the full sample and make index construction/querying a regression.
- A mandatory compiled dependency such as Numba would add installation,
  warm-up, and cache behavior to a repository that currently has no dependency
  manifest. It is not needed before exhausting NumPy and standard-library
  improvements.

## Architecture and Scope

The stage-factory architecture remains unchanged. The built-in
`statistics_builder` continues to call the model's default statistics method,
which delegates heavy arithmetic to the selected backend.

The implementation is limited to these responsibilities:

- `adp/variants/random_projection.py` orchestrates chunks and requests the
  backend kernel argument without backend-specific branches.
- `adp/backends/numpy_backend.py` owns fused NumPy construction of `q` and the
  compact local sums.
- `adp/backends/cupy_backend.py` mirrors the shared kernel-argument method, but
  its device accumulation and mathematical implementation do not change.
- `adp/common/types.py` gains a statistics-worker setting only if the serial
  benchmark gate fails.
- `tests/test_performance_optimizations.py` holds numerical and dispatch
  regression coverage.

No existing uncommitted stage-factory work is reverted or duplicated. The
optimization must fit the current `statistics_builder` boundary.

## NumPy Data Flow

### Kernel argument

Introduce a backend method that builds the kernel argument for one chunk:

```text
isotropic:   q = norm2 / h^2
anisotropic: q = (rho^2 * norm2 + projection2) / h^2
```

The NumPy implementation uses a single output buffer with `np.multiply` and
`np.add` `out=` operations. This avoids the extra full-size intermediate in the
anisotropic expression. The CuPy implementation retains its existing device
expression behind the same method.

### Fused compact sums

For each center with active indices `q < 1`:

1. compute compact-kernel weights, their mass `N`, and the weighted mean
   `xbar`;
2. form `centered = X_active - xbar`;
3. form `projected = centered @ directions.T`;
4. multiply `projected` by weights in place;
5. compute `imav = y_active @ projected`;
6. compute `U = projected.T @ centered`;
7. write `N` and output arrays directly.

This removes the separate `weighted_projected` allocation and the separate
`weights * y` temporary.

The statistic

```text
S = sum_i w_i * <X_i - xbar, phi>
```

is identically zero because `xbar` is the weighted mean. The implementation
therefore preserves the existing `S` array, dtype, and shape but fills it with
zeros rather than performing another reduction. Existing custom stage APIs
remain structurally compatible; the only difference is removal of floating
round-off around mathematical zero.

Centers with no active observations keep zero `imav`, `S`, `U`, and `N`, as in
the current implementation.

## Conditional Parallel Path

The serial fused kernel is implemented and measured first. If its median
speedup on the acceptance case is below 1.5x, add
`ADPConfig.statistics_workers: int = 1`.

Values must be positive integers. A value greater than one partitions center
ranges across a bounded `ThreadPoolExecutor`. Each worker writes to disjoint
output slices, so reduction order within each center and deterministic results
remain unchanged. Inputs stay serial when the conservative work proxy
`J * n * P * d` is below `1_000_000`, even when more workers are requested,
because thread scheduling would dominate their work.

The default remains one worker to avoid nested oversubscription in the existing
process-parallel experiment runners. Parallel benchmark commands set an
explicit worker count and cap BLAS/OpenMP worker counts to one. No automatic
CPU-count-based spawning is introduced.

## Compatibility and Error Handling

- `ADP.create`, `fit`, stage selection, and `LocalStatistics` keep their public
  signatures and shapes.
- `epanechnikov`, `quartic`, and `gaussian` remain supported. Gaussian keeps
  the existing dense path; the fused compact path is used only for compact
  kernels.
- `float64` and `float32` remain backend-controlled end to end.
- CuPy input caching, on-device accumulation, final transfer count, and memory
  release behavior remain unchanged.
- Invalid worker counts fail during `ADPConfig` validation if the conditional
  parallel phase is needed.
- Worker exceptions propagate to the statistics stage and are wrapped by the
  existing stage-execution boundary. Executors are always closed before the
  exception leaves the backend.
- Numerical mismatch is never handled by silently falling back to the old
  implementation; it fails regression tests.

## Verification

### Correctness tests

Add tests before production edits for:

1. fused compact `imav`, `S`, `U`, and `N` against the existing reference for
   Epanechnikov and quartic kernels;
2. isotropic and anisotropic kernel-argument construction;
3. empty and mixed empty/nonempty neighborhoods;
4. `float64` and `float32` output dtypes and tolerances;
5. unchanged Gaussian dense behavior;
6. unchanged fake-CuPy device-transfer and accumulator contracts;
7. serial and multi-worker equivalence plus invalid worker validation, only if
   the conditional parallel phase is activated.

Use `rtol=1e-11, atol=1e-12` for float64 and
`rtol=2e-5, atol=2e-6` for float32.

Correctness tests do not assert wall-clock durations. Performance acceptance is
evaluated by the repeatable benchmark protocol below, avoiding flaky pytest
failures from scheduler and BLAS noise.

### Performance protocol

Measure before and after in the same checkout and process environment:

- reuse identical `X`, `y`, centers, directions, beta, and bandwidth;
- warm each implementation once;
- record at least seven repetitions and compare medians;
- set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`;
- use the primary case `n=1000`, `d=100`, `J=150`, `P=16` with
  Epanechnikov weights;
- include one denser and one sparser active-neighborhood case;
- run one existing `large` stress case afterward.

Acceptance requires:

- at least 1.5x median warmed statistics-stage speedup, using the explicit
  worker setting if the conditional parallel phase is activated;
- NumPy results within the stated dtype-specific tolerances;
- no degradation beyond numerical tolerance in cosine, objective, and final
  beta for the stress case;
- no increase beyond measurement noise in peak memory for the serial optimized
  path and no more than a 5% increase for the explicit parallel path;
- less than a 5% median runtime regression on the smoke/small case.

Final verification includes the focused performance tests, the complete pytest
suite, Python compilation of changed modules, `git diff --check`, and the
recorded before/after benchmark values.

## Non-Goals

This change does not:

- redesign bandwidth selection or introduce per-center bandwidths;
- change center generation, OPG/PHD initialization, or regularization;
- turn the neighbor index into the primary statistics data structure;
- change the mathematical ADP objective or beta solver;
- optimize the CuPy kernel beyond preserving its shared contract;
- add Numba, Cython, or another compiled runtime dependency.
