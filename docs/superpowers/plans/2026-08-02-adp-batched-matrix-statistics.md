# Batched Matrix ADP Statistics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the numerically stable dense batched ADP statistics from `adp_matrix_statistics.tex`.

**Architecture:** One public NumPy function validates inputs, prepares shared normalized quantities, and processes center slices. Two private helpers keep validation and the `eta` diagnostic separate without splitting the shared `I`/`U` computation.

**Tech Stack:** Python, NumPy, pytest for one focused numerical check.

---

### Task 1: Batched matrix statistics

**Files:**
- Create: `test_ADP_statistic.py`
- Modify: `ADP/ADP_statistic.py`

- [ ] **Step 1: Write the failing numerical test**

```python
import numpy as np

from ADP_statistic import calculate_statistics


def test_matrix_statistics_match_direct_sums_and_are_shift_invariant():
    rng = np.random.default_rng(42)
    n, d, J, P = 11, 4, 5, 3
    X = rng.normal(size=(n, d))
    Y = rng.normal(size=n)
    weights = rng.uniform(0.1, 1.0, size=(J, n))
    directions = rng.normal(size=(J, P, d))
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)

    result = calculate_statistics(X, Y, weights, directions, batch_size=2)

    mass = weights.sum(axis=1)
    mean = weights @ X / mass[:, None]
    centered = X[None, :, :] - mean[:, None, :]
    projections = np.einsum("jpd,jnd->jpn", directions, centered)
    weighted = weights[:, None, :] * projections

    np.testing.assert_allclose(result["I"], weighted @ Y, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        result["U"], weighted @ centered, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(result["mass"], mass)
    np.testing.assert_allclose(result["mean"], mean)
    normalized = weights / mass[:, None]
    np.testing.assert_allclose(result["n_eff"], 1 / np.square(normalized).sum(1))
    assert np.all(np.isfinite(result["eta"]))
    assert np.all((0 <= result["eta"]) & (result["eta"] < 1e-12))

    shifted = calculate_statistics(
        X + 1e6, Y - 1e6, weights, directions, batch_size=3
    )
    np.testing.assert_allclose(shifted["I"], result["I"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(shifted["U"], result["U"], rtol=1e-9, atol=1e-9)
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
PYTHONPATH=ADP rtk python -m pytest test_ADP_statistic.py -q
```

Expected: collection fails because `calculate_statistics` is not defined.

- [ ] **Step 3: Implement the minimal batched matrix function**

```python
import numpy as np


def calculate_statistics(X, Y, weights, directions, batch_size=32):
    """Calculate stable dense ADP statistics in batches of local centers."""
    if (
        not isinstance(batch_size, (int, np.integer))
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")

    X, Y, W, Phi, mass = _prepare_inputs(X, Y, weights, directions)
    J, P = Phi.shape[:2]
    A = W / mass[:, None]
    x_bar = X.mean(axis=0)
    Xc = X - x_bar
    Mc = A @ Xc
    mean = Mc + x_bar
    y_bar = A @ Y
    I = np.empty((J, P))
    U = np.empty((J, P, X.shape[1]))
    eta = np.empty((J, P))

    for start in range(0, J, batch_size):
        batch = slice(start, min(start + batch_size, J))
        Ab, Phib, Mcb = A[batch], Phi[batch], Mc[batch]
        Q = Phib @ Xc.T - Phib @ Mcb[..., None]
        residual = (Q @ Ab[..., None]).squeeze(-1)
        eta[batch] = _normalized_residual(Ab, Q, residual)
        H = (Q - residual[..., None]) * Ab[:, None, :]
        s = H.sum(axis=2)
        I[batch] = mass[batch, None] * (
            H @ Y - s * y_bar[batch, None]
        )
        U[batch] = mass[batch, None, None] * (
            H @ Xc - s[..., None] * Mcb[:, None, :]
        )

    return {
        "I": I,
        "U": U,
        "mass": mass,
        "mean": mean,
        "n_eff": 1.0 / np.square(A).sum(axis=1),
        "eta": eta,
    }


def _normalized_residual(A, Q, residual):
    denominator = (np.abs(Q) @ A[..., None]).squeeze(-1)
    return np.divide(
        np.abs(residual),
        denominator,
        out=np.zeros_like(residual),
        where=denominator != 0,
    )


def _prepare_inputs(X, Y, weights, directions):
    arrays = []
    for value, name in (
        (X, "X"),
        (Y, "Y"),
        (weights, "weights"),
        (directions, "directions"),
    ):
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
            raise TypeError(f"{name} must have a real numeric dtype")
        array = array.astype(float, copy=False)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        arrays.append(array)

    X, Y, W, Phi = arrays
    if X.ndim != 2 or 0 in X.shape:
        raise ValueError("X must have non-empty shape (n, d)")
    if Y.shape != (X.shape[0],):
        raise ValueError("Y must have shape (n,)")
    if W.ndim != 2 or W.shape[1] != X.shape[0] or W.shape[0] == 0:
        raise ValueError("weights must have non-empty shape (J, n)")
    if Phi.ndim != 3 or Phi.shape != (W.shape[0], Phi.shape[1], X.shape[1]):
        raise ValueError("directions must have shape (J, P, d)")
    if Phi.shape[1] == 0:
        raise ValueError("directions must contain at least one direction")
    if np.any(W < 0):
        raise ValueError("weights must be nonnegative")

    mass = W.sum(axis=1)
    if not np.all(np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError("every weight row must have positive finite mass")
    return X, Y, W, Phi, mass
```

- [ ] **Step 4: Run the focused test**

Run:

```bash
PYTHONPATH=ADP rtk python -m pytest test_ADP_statistic.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Compile the module**

Run:

```bash
rtk python -m py_compile ADP/ADP_statistic.py
```

Expected: command exits successfully without output.

- [ ] **Step 6: Commit the implementation**

```bash
rtk git add ADP/ADP_statistic.py test_ADP_statistic.py
rtk git commit -m "feat: add batched matrix ADP statistics"
```
