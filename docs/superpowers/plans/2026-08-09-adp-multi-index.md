# ADP Multi-Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a fitted multi-index Average Derivative Procedure with local-PCA and random initialization, matrix-free low-rank localization, and a multi-index-capable LSMR solver.

**Architecture:** Keep `ADP_multi_index` independent from the single-index class while reusing configuration, statistics, tracking, and solver adapter code. Add only multi-index numerical helpers to the existing calculus module; represent localization by `(basis, eigenvalues, alpha, h)` and never allocate a `d x d` tensor. Extend the existing LSMR entry point by dispatching on a one- or two-dimensional initial index so current single-index callers remain unchanged.

**Tech Stack:** Python 3, NumPy, SciPy `LinearOperator`/`lsmr`, pytest.

---

## File map

- Modify `ADP/engine/calculus.py`: local-gradient PCA and matrix-free multi-index localization helpers.
- Modify `ADP/single_index/solvers/LSMR.py`: retain the vector solver and add matrix-index alternating LSMR.
- Modify `ADP/multi_index/ADP_multi_index.py`: replace the current user-owned stub with the complete result and model classes while preserving its intended fields.
- Modify `ADP/__init__.py`: export the multi-index model and low-rank weight iterator.
- Create `test_ADP_multi_index.py`: one focused test module for initialization, tensor equivalence, solver compatibility, and model smoke coverage.

### Task 1: Matrix-free multi-index calculus

**Files:**
- Modify: `ADP/engine/calculus.py:69-213`
- Create: `test_ADP_multi_index.py`

- [ ] **Step 1: Write failing initialization and tensor-equivalence tests**

```python
import numpy as np

from ADP.engine.calculus import (
    calculate_multi_weight,
    initialize_basis_local,
    initialize_basis_random,
)


def test_multi_initializers_are_orthonormal():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 5))
    true_basis, _ = np.linalg.qr(rng.normal(size=(5, 2)))
    Y = np.sin(X @ true_basis[:, 0]) + np.square(X @ true_basis[:, 1])
    centers = X[:16]
    distance2 = np.maximum(
        np.square(centers).sum(1)[:, None]
        + np.square(X).sum(1)[None, :]
        - 2 * centers @ X.T,
        0,
    )

    local = initialize_basis_local(
        X, Y, centers, distance2, 20, lambda q: np.maximum(1 - q, 0), 1e-8, 2
    )
    random = initialize_basis_random(rng, 5, 2)

    assert local.shape == random.shape == (5, 2)
    np.testing.assert_allclose(local.T @ local, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(random.T @ random, np.eye(2), atol=1e-12)


def test_multi_weight_matches_explicit_low_rank_tensor():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(9, 5))
    centers = rng.normal(size=(4, 5))
    basis, _ = np.linalg.qr(rng.normal(size=(5, 2)))
    eigenvalues = np.array([1.7, 0.4])
    alpha, h = 0.35, 1.2
    kernel = lambda q: np.exp(-q)

    actual = np.vstack(
        [
            block
            for _, block in calculate_multi_weight(
                X, centers, basis, eigenvalues, h, alpha, kernel, block_size=2
            )
        ]
    )
    projector = basis @ basis.T
    tensor2 = (
        alpha**2 * (np.eye(5) - projector)
        + basis @ np.diag(eigenvalues) @ basis.T
    ) / h**2
    differences = centers[:, None, :] - X[None, :, :]
    argument = np.einsum("jnd,de,jne->jn", differences, tensor2, differences)
    np.testing.assert_allclose(actual, kernel(argument), rtol=1e-12, atol=1e-12)
```

- [ ] **Step 2: Run the tests and verify imports fail**

Run: `rtk pytest -q test_ADP_multi_index.py -k 'initializers or weight'`

Expected: collection fails because the three multi-index calculus functions do not exist.

- [ ] **Step 3: Extract local-gradient PCA and add random initialization**

Implement
`initialize_basis_local(X, Y, centers, distance2, N_lin, kernel, local_ridge, index_dim)`
by reusing the current local-linear gradient calculation and returning
`right_vectors[:index_dim].T`. Keep `initialize_beta_local` as the compatibility
wrapper:

```python
def initialize_beta_local(X, Y, centers, distance2, N_lin, kernel, local_ridge):
    return initialize_basis_local(
        X, Y, centers, distance2, N_lin, kernel, local_ridge, 1
    )[:, 0]


def initialize_basis_random(rng, n_features, index_dim):
    basis, _ = np.linalg.qr(
        rng.standard_normal((n_features, index_dim)), mode="reduced"
    )
    signs = np.sign(basis[np.argmax(np.abs(basis), axis=0), np.arange(index_dim)])
    basis *= np.where(signs == 0, 1, signs)
    return basis
```

Reject `index_dim < 1`, `index_dim > d`, and a local gradient matrix whose `index_dim`-th singular value is numerically zero.

- [ ] **Step 4: Add low-rank quadratic-form, alpha search, weights, and directions**

Add private validation for finite `(d,m)` orthonormal bases and finite nonnegative `(m,)` eigenvalues. Compute the two quadratic terms one basis column at a time so no `(B,n,m)` or `(d,d)` array is retained:

```python
def _multi_components(X, centers, basis, eigenvalues, distance2):
    orthogonal2 = distance2.copy()
    principal2 = np.zeros_like(distance2)
    for vector, eigenvalue in zip(basis.T, eigenvalues):
        difference = (centers @ vector)[:, None] - (X @ vector)[None, :]
        component2 = np.square(difference)
        orthogonal2 -= component2
        principal2 += eigenvalue * component2
    np.maximum(orthogonal2, 0.0, out=orthogonal2)
    return orthogonal2, principal2
```

Expose:

```python
calculate_alpha_k(
    X, centers, basis, eigenvalues, h_k, N_loc, kernel, *, distance2=None
) -> float | None

calculate_multi_weight(
    X, centers, basis, eigenvalues, h, alpha, kernel,
    block_size=128, *, distance2=None
) -> Iterator[tuple[int, np.ndarray]]

generate_multi_proj(
    rng, n_centers, n_directions, basis, eigenvalues, alpha
) -> np.ndarray
```

`calculate_alpha_k` uses `(alpha**2 * orthogonal2 + principal2) / h_k**2` and the same largest-feasible binary search as the current `calculate_rho_k`. `generate_multi_proj` samples

```python
z = rng.standard_normal((n_centers, n_directions, d))
coordinates = z @ basis
orthogonal = z - coordinates @ basis.T
principal = (
    rng.standard_normal((n_centers, n_directions, m))
    * np.sqrt(eigenvalues)
) @ basis.T
values = alpha * orthogonal + principal
return values / np.linalg.norm(values, axis=2, keepdims=True)
```

and raises if any direction has zero norm.

- [ ] **Step 5: Run focused calculus tests**

Run: `rtk pytest -q test_ADP_multi_index.py -k 'initializers or weight'`

Expected: `2 passed`.

- [ ] **Step 6: Commit the calculus slice**

```bash
rtk git add ADP/engine/calculus.py test_ADP_multi_index.py
rtk git commit -m "feat: add matrix-free multi-index calculus"
```

### Task 2: Multi-index LSMR

**Files:**
- Modify: `ADP/single_index/solvers/LSMR.py:10-120`
- Modify: `test_ADP_multi_index.py`

- [ ] **Step 1: Add a failing solver compatibility test**

Construct exact synthetic statistics without fitting the full model:

```python
from ADP.ADP_Statistic import ADP_Statistics
from ADP.single_index.solvers.LSMR import solve


def _statistics(I, U):
    J = len(I)
    return ADP_Statistics(
        I=I,
        U=U,
        mass=np.ones(J),
        mean=np.zeros((J, U.shape[2])),
        n_eff=np.ones(J),
        eta=np.zeros_like(I),
    )


def test_lsmr_accepts_vector_and_matrix_indices():
    rng = np.random.default_rng(5)
    J, directions, d, m = 12, 5, 4, 2
    U = rng.normal(size=(J, directions, d))

    beta = rng.normal(size=d)
    beta /= np.linalg.norm(beta)
    slopes = rng.normal(size=J)
    single = solve(
        _statistics(slopes[:, None] * (U @ beta), U),
        beta,
        lambda_penalty=1e-3,
        local_ridge=1e-8,
        max_steps=2,
    )

    basis, _ = np.linalg.qr(rng.normal(size=(d, m)))
    coefficients = rng.normal(size=(J, m))
    I = np.einsum("jpd,dm,jm->jp", U, basis, coefficients)
    multi = solve(
        _statistics(I, U),
        basis,
        lambda_penalty=1e-3,
        local_ridge=1e-8,
        max_steps=2,
    )

    assert single.index.shape == (d,)
    assert multi.index.shape == (d, m)
    assert multi.coefficients.shape == (J, m)
    np.testing.assert_allclose(multi.index.T @ multi.index, np.eye(m), atol=1e-10)
    assert np.asarray(multi.diagnostics["eigenvalues"]).shape == (m,)
```

- [ ] **Step 2: Run the solver test and verify the matrix case fails**

Run: `rtk pytest -q test_ADP_multi_index.py::test_lsmr_accepts_vector_and_matrix_indices`

Expected: FAIL in the current scalar `_slopes`/`_solve_beta` path.

- [ ] **Step 3: Dispatch the public solver by index rank**

Keep the existing vector path intact and move it to `_solve_single`. In `solve`, validate common settings and dispatch:

```python
initial_index = np.asarray(beta, dtype=float)
if initial_index.ndim == 1:
    return _solve_single(
        statistics,
        initial_index,
        lambda_penalty=lambda_penalty,
        local_ridge=local_ridge,
        max_steps=max_steps,
        tol=tol,
    )
if initial_index.ndim == 2:
    return _solve_multi(
        statistics,
        initial_index,
        lambda_penalty=lambda_penalty,
        local_ridge=local_ridge,
        max_steps=max_steps,
        tol=tol,
    )
raise ValueError("initial index must have shape (d,) or (d, m)")
```

- [ ] **Step 4: Implement alternating matrix LSMR**

For fixed basis, solve all local coefficient systems:

```python
projected = U @ basis
gram = np.einsum("jpm,jpn->jmn", projected, projected, optimize=True)
gram += ridge * np.eye(basis.shape[1])
rhs = np.einsum("jpm,jp->jm", projected, I, optimize=True)
coefficients = np.linalg.solve(gram, rhs)
```

For fixed coefficients, use a `(I.size + d*m, d*m)` `LinearOperator` whose data prediction is

```python
np.einsum("jpd,dm,jm->jp", U, matrix, coefficients, optimize=True)
```

and whose transpose is

```python
np.einsum("jpd,jp,jm->dm", U, residual, coefficients, optimize=True)
```

plus the existing square-root penalty block. Canonicalize without a dense structural matrix:

```python
coefficient_gram = coefficients.T @ coefficients
values, vectors = np.linalg.eigh(coefficient_gram)
factor = raw_basis @ (vectors * np.sqrt(np.maximum(values, 0)))
basis, singular_values, _ = np.linalg.svd(factor, full_matrices=False)
eigenvalues = singular_values**2
```

Orient columns deterministically, recompute coefficients in the canonical basis, use projector Frobenius distance for convergence, and place a copy of `eigenvalues` in diagnostics.

- [ ] **Step 5: Run solver and existing statistics tests**

Run: `rtk pytest -q test_ADP_multi_index.py::test_lsmr_accepts_vector_and_matrix_indices test_ADP_statistic.py`

Expected: `3 passed`.

- [ ] **Step 6: Commit the solver slice**

```bash
rtk git add ADP/single_index/solvers/LSMR.py test_ADP_multi_index.py
rtk git commit -m "feat: extend LSMR to multi-index bases"
```

### Task 3: Full multi-index model and result

**Files:**
- Modify: `ADP/multi_index/ADP_multi_index.py:1-45`
- Modify: `test_ADP_multi_index.py`

- [ ] **Step 1: Add failing result and model tests**

```python
from ADP import ADP_Config, ADP_SolverResult, ADP_solver
from ADP.multi_index.ADP_multi_index import ADP_multi_index, ADP_multi_index_result


def test_multi_result_uses_projector_distance():
    basis = np.eye(4)[:, :2]
    rotated = basis @ np.array([[0.0, -1.0], [1.0, 0.0]])
    result = ADP_multi_index_result(beta_init=basis, beta_final=rotated, beta_true=basis)
    result.Calculate_dist()
    assert result.dist_init == 0.0
    assert result.dist_final < 1e-15


def test_multi_model_random_initialization_and_fit_contract():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(48, 4))
    Y = np.sin(X[:, 0]) + X[:, 1] ** 2

    def unchanged(statistics, initial_index, **params):
        return ADP_SolverResult(
            index=initial_index,
            coefficients=np.ones((len(statistics.I), 2)),
            diagnostics={"eigenvalues": np.ones(2)},
        )

    config = ADP_Config(
        seed=6,
        N_loc=12,
        N_lin=12,
        N_J=4,
        N_phi=3,
        h_min=1e6,
        index_init="random",
    )
    model = ADP_multi_index(2, config=config, solver=ADP_solver(unchanged)).fit(X, Y)

    assert model.basis_.shape == (4, 2)
    np.testing.assert_allclose(model.basis_.T @ model.basis_, np.eye(2), atol=1e-12)
    assert model.transform(X).shape == (48, 2)
    assert model.result_.stop_reason == "h_min"
```

- [ ] **Step 2: Run the model tests and verify the stub fails**

Run: `rtk pytest -q test_ADP_multi_index.py -k 'result or model'`

Expected: collection or construction fails because `ADP_multi_index` is not implemented and the current result class is incomplete.

- [ ] **Step 3: Replace the stub with the fitted model**

Mirror the single-index orchestration without inheritance:

```python
class ADP_multi_index:
    def __init__(self, index_dim, config=None, solver=None):
        if isinstance(index_dim, bool) or not isinstance(index_dim, (int, np.integer)):
            raise TypeError("index_dim must be an integer")
        if index_dim < 1:
            raise ValueError("index_dim must be positive")
        self.index_dim = int(index_dim)
        self.config = config or ADP_Config()
        self.solver = solver or ADP_solver(solve_lsmr, tol=5e-8)
```

Implement `fit/_fit` with the single-index stage tracker and the approved flow:

- validate `index_dim < d`;
- select centers and pairwise distances exactly as single-index does;
- choose `initialize_basis_local` or `initialize_basis_random` from `config.index_init`;
- start with `eigenvalues=np.ones(m)` and `alpha=1`, which makes the tensor isotropic;
- call `generate_multi_proj`, `calculate_multi_weight`, unified `calculate_statistics`, and the solver adapter;
- require solver index `(d,m)`, orthonormalize it with reduced QR, require diagnostics eigenvalues `(m,)`, finite and nonnegative;
- update `h`, `alpha`, basis, and eigenvalues until `h_min` or `local_mass_limit`;
- assign fitted state only after successful completion.

Implement:

```python
def transform(self, X):
    self._check_fitted()
    return utils._prepare_transform(X, self.n_features_in_) @ self.basis_
```

and fix `ADP_multi_index_result.Calculate_dist` to compare `(d,m)` projector matrices with denominator `sqrt(2*m)`.

- [ ] **Step 4: Run model tests**

Run: `rtk pytest -q test_ADP_multi_index.py -k 'result or model'`

Expected: `2 passed`.

- [ ] **Step 5: Commit the model slice**

```bash
rtk git add ADP/multi_index/ADP_multi_index.py test_ADP_multi_index.py
rtk git commit -m "feat: implement multi-index ADP model"
```

### Task 4: Public API and end-to-end verification

**Files:**
- Modify: `ADP/__init__.py:1-18`
- Modify: `test_ADP_multi_index.py`

- [ ] **Step 1: Add public import assertions**

```python
def test_multi_index_public_api():
    from ADP import ADP_multi_index, _weight_blocks_multi

    assert ADP_multi_index.__name__ == "ADP_multi_index"
    assert callable(_weight_blocks_multi)
```

- [ ] **Step 2: Export the model and low-rank iterator**

Add:

```python
from ADP.engine.calculus import calculate_multi_weight as _weight_blocks_multi
from .multi_index.ADP_multi_index import ADP_multi_index
```

and include both names in `__all__`.

- [ ] **Step 3: Run the focused suite**

Run: `rtk pytest -q test_ADP_multi_index.py test_ADP_statistic.py`

Expected: all tests pass.

- [ ] **Step 4: Run compilation and whitespace checks**

Run: `rtk python -m compileall -q ADP test_ADP_multi_index.py`

Expected: exit status `0` with no output.

Run: `rtk git diff --check`

Expected: exit status `0` with no output.

- [ ] **Step 5: Run a real LSMR model smoke**

Run a short `python -c` program that generates `n=120, d=5, m=2`, sets
`Y = sin(X @ B[:,0]) + (X @ B[:,1])**2`, fits `ADP_multi_index` with local
initialization, and asserts:

```python
assert model.basis_.shape == (5, 2)
assert np.all(np.isfinite(model.basis_))
assert np.all(model.eigenvalues_ >= 0)
assert model.transform(X).shape == (120, 2)
```

Expected: exit status `0` and printed finite initial/final projector distances.

- [ ] **Step 6: Commit public wiring and final checks**

```bash
rtk git add ADP/__init__.py test_ADP_multi_index.py
rtk git commit -m "test: verify multi-index ADP pipeline"
```
