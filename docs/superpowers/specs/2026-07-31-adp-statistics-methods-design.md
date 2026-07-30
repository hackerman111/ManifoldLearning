# ADP Statistics Methods Design

## Goal

Add three interchangeable implementations of the Average Derivative Procedure
statistics from `manifold_v2.tex` and `adp_matrix_statistics.tex` to a
hypothesis-specific subclass of `ADP/ADP_single_index.py`. Compare only the
statistics kernels on identical precomputed data, weights, and directions.

## Scope

The implementation creates:

- `hypo/statistic/ADP_single_index_hypo_stat.py` with the modified class;
- `hypo/statistic/tester.py` with deterministic benchmark experiments;
- `hypo/tester.py` as the common CLI for hypothesis testers.

`ADP/__init__.py` is repaired only enough to stop importing the deleted
`ADP_model_base.py` and to expose the current single-index class.

The work does not implement the unfinished ADP fitting loop, bandwidth
selection, weight construction, plots, GPU kernels, or approximate sparsity.

## Public API

`ADP_single_index_hypo_stat` subclasses `ADP_single_index` and exposes:

```python
Calculate_statistic(
    weights,
    directions,
    *,
    method="direct",
    X=None,
    Y=None,
    batch_size=32,
)
```

The explicit methods are also public:

- `Calculate_statistic_direct`;
- `Calculate_statistic_matrix`;
- `Calculate_statistic_sparse`.

`X` has shape `(n, d)`, `Y` has shape `(n,)`, dense `weights` has shape
`(J, n)`, and `directions` has shape `(J, P, d)`. When `X` and `Y` are
omitted, the subclass uses the generated data from the parent class and
transposes its feature-first `X`.

Each method returns the same dictionary:

```python
{
    "I": ...,       # (J, P)
    "U": ...,       # (J, P, d)
    "mass": ...,    # (J,)
    "mean": ...,    # (J, d)
    "n_eff": ...,   # (J,)
    "eta": ...,     # (J, P)
}
```

Shared validation rejects incompatible shapes, non-finite values, negative
weights, and centers with zero local mass.

## Statistics Implementations

### Direct NumPy

The reference implementation follows the direct sums:

```text
M_j   = sum_i(w_ji X_i) / sum_i(w_ji)
Q_jri = phi_jr^T (X_i - M_j)
I_jr  = sum_i(Y_i Q_jri w_ji)
U_jr  = sum_i((X_i - M_j) Q_jri w_ji)
```

It uses broadcasting, `einsum`, and matrix multiplication without Python
loops. It materializes the `(J, n, d)` difference tensor and is therefore the
simple correctness reference, not the memory-efficient production method.

### Dense Matrix

The matrix implementation uses the stable factorization from
`adp_matrix_statistics.tex`. It globally centers `X`, processes centers in
batches, computes local means with `A @ X`, explicitly removes the weighted
mean from projected differences, and evaluates stable `H @ Y` and `H @ X`
forms. It never materializes a `(J, n, d)` difference tensor.

The only Python loop is over center batches. All work inside a batch is NumPy
array arithmetic.

### Sparse CSR

The sparse implementation uses SciPy CSR with exactly the nonzero pattern of
the supplied weights. It does not discard positive weights and therefore
computes the same statistics as the two dense methods.

Masses and means use sparse-dense multiplication. Each direction reuses the
CSR `indices` and `indptr`, replaces only its values, and computes the stable
sparse matrix-vector and sparse matrix-dense products. The only Python loop is
over directions because SciPy CSR matrices are two-dimensional.

The benchmark constructs dense and CSR representations before measurement, so
format conversion is excluded while both methods still receive mathematically
identical weights.

## Benchmark

`hypo/statistic/tester.py` provides:

- `smoke`: `(n, d, J, P) = (256, 12, 16, 4)` at 10% nonzero weights;
- `standard-sparse`: `(1000, 32, 64, 8)` at 5%;
- `standard-medium`: `(1500, 64, 96, 8)` at 15%;
- `standard-dense`: `(1000, 32, 64, 8)` at 75%.

Data, centers, weights, and unit directions are deterministic for a supplied
seed. Weights come from a compact Epanechnikov-style kernel with a radius
chosen from the requested distance quantile. Every center is an observed
point, which guarantees positive local mass.

For every method and case the tester records:

- median and minimum wall time from warmed runs;
- peak allocations measured by `tracemalloc`, excluding prepared inputs;
- relative errors of `I` and `U` against direct `float64`;
- relative errors after large common translations of `X` and `Y`;
- finiteness, maximum `eta`, density, and effective-neighbor diagnostics;
- time speedup and memory ratio relative to the direct method.

Rows are written with the standard-library `csv` module. A smoke run fails
instead of writing misleading success if matrix or sparse results disagree
with the unshifted direct `float64` reference beyond the configured tolerance.

## Common CLI

`hypo/tester.py` discovers files matching `hypo/*/tester.py`. It lists the
available hypothesis names and forwards remaining arguments to the selected
tester in a subprocess using the current Python interpreter.

Examples:

```bash
python hypo/tester.py --list
python hypo/tester.py statistic --profile smoke --repetitions 3 \
  --output /tmp/adp-statistics-smoke.csv
```

This convention lets a future hypothesis become runnable by adding one
`hypo/<name>/tester.py`; the common CLI needs no registry edits.

## Verification

Verification consists of:

1. compiling all changed Python files;
2. checking direct, matrix, and sparse agreement on deterministic small data;
3. checking the common CLI discovery and delegated help;
4. running the smoke benchmark and inspecting its CSV;
5. running the standard benchmark for the reported time, memory, and numerical
   stability comparison;
6. running `git diff --check`.
