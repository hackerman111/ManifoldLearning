# Batched matrix ADP statistics

## Scope

Implement only the dense batched matrix formulas from
`adp_matrix_statistics.tex` in `ADP/ADP_statistic.py`. Direct, CSR, and GPU
neighbor-list implementations are outside this change.

## API

```python
calculate_statistics(X, Y, weights, directions, batch_size=32)
```

Inputs follow the notation from the document:

- `X`: observations with shape `(n, d)`;
- `Y`: responses with shape `(n,)`;
- `weights`: nonnegative weights with shape `(J, n)` and positive row masses;
- `directions`: directions with shape `(J, P, d)`;
- `batch_size`: positive integer number of centers processed together.

The function returns a dictionary containing:

- `I` with shape `(J, P)`;
- `U` with shape `(J, P, d)`;
- `mass` with shape `(J,)`;
- `mean` with shape `(J, d)`;
- `n_eff` with shape `(J,)`;
- `eta` with shape `(J, P)`.

## Computation

Inputs are converted to finite real NumPy arrays and their shapes, weights, row
masses, and batch size are validated once. Weights are normalized once.

`X` is globally centered. For each center batch, projections are computed with
stacked matrix multiplication. Their weighted residual is recorded in `eta` and
then subtracted. The corrected projections form `H`; `I` and `U` are computed
with matrix multiplication using the numerically stable formulas from the
document. Result arrays are allocated once and filled by slices.

The Python `for` loop only selects center batches. All expensive work inside a
batch uses NumPy matrix multiplication and broadcasting.

## Internal structure

Keep one public calculation function and two private helpers:

- `_prepare_inputs` validates and converts inputs;
- `_normalized_residual` calculates `eta`.

Extracting `I` and `U` into separate functions is intentionally avoided because
they share the same projections and `H`.

## Verification

One focused test compares `I` and `U` with literal direct sums in `float64` and
checks invariance under constant shifts of `X` and `Y`.
