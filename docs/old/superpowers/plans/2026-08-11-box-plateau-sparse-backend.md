# Box and Plateau Sparse Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add exact box-only and plateau-only sparse neighborhood backends to uppercase ADP single- and multi-index fits without dense `J x n` distance or weight matrices.

**Architecture:** `ADP/engine/box_kernel.py` owns kernel evaluation, CSR neighborhoods, cKDTree screening, sparse bandwidth/scale search, local initialization, and support caches. Existing models select this path only for the two new kernels; existing statistics functions accept sparse blocks while preserving the current dense callable-kernel path.

**Tech Stack:** Python 3.14, NumPy, SciPy `cKDTree`, optional CuPy, pytest, argparse.

---

## File Map

- Create `ADP/engine/box_kernel.py`: all new kernel and sparse-backend logic.
- Create `test/test_ADP_box_kernel.py`: focused correctness and integration tests.
- Modify `ADP/ADP_Config.py`: recognize and validate sparse kernels.
- Modify `ADP/cli.py`: add `box`, `plateau`, and `--kernel-tau` parsing.
- Modify `ADP/ADP_Statistic.py`: consume sparse blocks on CPU and GPU.
- Modify `ADP/single_index/ADP_single_index.py`: route sparse single-index fits.
- Modify `ADP/multi_index/ADP_multi_index.py`: route sparse multi-index fits.
- Modify `ADP/__init__.py`: export public kernel helpers.
- Modify `ADP/CLI.md`: document the two independent modes.

The checkout already contains unrelated modifications in several target files.
Before every commit, stage only task-owned hunks and verify them with
`rtk git diff --cached`; never stage a whole pre-modified file blindly.

### Task 1: Kernel functions and stable kernel identification

**Files:**
- Create: `ADP/engine/box_kernel.py`
- Create: `test/test_ADP_box_kernel.py`

- [ ] **Step 1: Write failing kernel boundary tests**

```python
from functools import partial

import numpy as np
import pytest

from ADP.engine.box_kernel import (
    box_kernel,
    make_plateau_kernel,
    plateau_kernel,
    sparse_kernel_parameters,
)


def test_box_and_plateau_kernel_boundaries():
    q = np.array([0.0, 0.5, 0.75, 1.0, 2.0])
    np.testing.assert_array_equal(box_kernel(q), [1.0, 1.0, 1.0, 0.0, 0.0])
    expected = np.array([1.0, 1.0, 0.5, 0.0, 0.0])
    np.testing.assert_allclose(plateau_kernel(q, tau=0.5), expected, atol=1e-15)


def test_plateau_tau_and_kernel_identity_are_validated():
    kernel = make_plateau_kernel(0.4)
    assert isinstance(kernel, partial)
    assert sparse_kernel_parameters(box_kernel) == ("box", None)
    assert sparse_kernel_parameters(kernel) == ("plateau", 0.4)
    with pytest.raises(ValueError, match="tau"):
        make_plateau_kernel(1.0)
    with pytest.raises(ValueError, match="finite"):
        plateau_kernel(np.array([np.nan]))
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run:

```bash
rtk proxy python -m pytest -q test/test_ADP_box_kernel.py
```

Expected: collection fails with `ModuleNotFoundError: ADP.engine.box_kernel`.

- [ ] **Step 3: Implement the kernel primitives**

Create `ADP/engine/box_kernel.py` with these public signatures and formulas:

```python
from functools import partial

import numpy as np


def _validated_tau(tau: float) -> float:
    tau = float(tau)
    if not np.isfinite(tau) or not 0.0 < tau < 1.0:
        raise ValueError("tau must be finite and lie in (0, 1)")
    return tau


def _kernel_input(value) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(value)):
        raise ValueError("kernel input must be finite")
    return value


def box_kernel(value) -> np.ndarray:
    return np.asarray(_kernel_input(value) < 1.0, dtype=float)


def plateau_kernel(value, *, tau: float = 0.5) -> np.ndarray:
    value = _kernel_input(value)
    tau = _validated_tau(tau)
    result = np.zeros_like(value)
    result[value <= tau] = 1.0
    boundary = (value > tau) & (value < 1.0)
    s = (value[boundary] - tau) / (1.0 - tau)
    result[boundary] = 1.0 - 10.0 * s**3 + 15.0 * s**4 - 6.0 * s**5
    return result


def make_plateau_kernel(tau: float = 0.5):
    return partial(plateau_kernel, tau=_validated_tau(tau))


def sparse_kernel_parameters(kernel) -> tuple[str, float | None] | None:
    if kernel is box_kernel:
        return "box", None
    if kernel is plateau_kernel:
        return "plateau", 0.5
    if isinstance(kernel, partial) and kernel.func is plateau_kernel:
        if kernel.args or set(kernel.keywords or {}) - {"tau"}:
            raise ValueError("plateau kernel accepts only keyword tau")
        return "plateau", _validated_tau((kernel.keywords or {}).get("tau", 0.5))
    return None
```

- [ ] **Step 4: Run the focused tests**

Run `rtk proxy python -m pytest -q test/test_ADP_box_kernel.py`.
Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit only the new module and test**

```bash
rtk git add ADP/engine/box_kernel.py test/test_ADP_box_kernel.py
rtk git diff --cached --check
rtk git commit -m "feat: add box and plateau kernels"
```

### Task 2: CSR neighborhood blocks and exact cKDTree screening

**Files:**
- Modify: `ADP/engine/box_kernel.py`
- Modify: `test/test_ADP_box_kernel.py`

- [ ] **Step 1: Add failing dense-reference tests for single and multi support**

Use deterministic `X`, centers, a unit `beta`, and an orthonormal two-column
basis. For every returned block, reconstruct a dense matrix only in the test
and compare it with:

```python
q_single = (rho**2 * distance2 + projection2) / h**2
q_multi = (alpha**2 * orthogonal2 + principal2) / h**2
expected_box = box_kernel(q_single)
expected_plateau = plateau_kernel(q_multi, tau=0.5)
```

Also assert:

```python
assert block.boundary_weights.size == 0  # box
assert block.edge_count == np.count_nonzero(expected_box)
assert block.boundary_count == np.count_nonzero(
    (q_multi > 0.5) & (q_multi < 1.0)
)
```

- [ ] **Step 2: Run both tests and verify imports or attributes fail**

Run:

```bash
rtk proxy python -m pytest -q \
  test/test_ADP_box_kernel.py -k 'single_neighborhood or multi_neighborhood'
```

Expected: failure because `SparseNeighborhoodBlock` and `NeighborhoodEngine`
do not exist.

- [ ] **Step 3: Add the validated CSR block**

Implement a frozen slotted dataclass containing `start`, `indptr`, `indices`,
`boundary_positions`, `boundary_weights`, `mode`, and `tau`. Its validation
must enforce monotone offsets, in-range positions, finite weights in `(0, 1)`,
and positive row mass. Provide `edge_count`, `boundary_count`, `mass`,
`row(row_index)`, and a support/weight cache key.

```python
@dataclass(frozen=True, slots=True)
class SparseNeighborhoodBlock:
    start: int
    indptr: np.ndarray
    indices: np.ndarray
    boundary_positions: np.ndarray
    boundary_weights: np.ndarray
    mode: str
    tau: float | None

    @property
    def mass(self):
        counts = np.diff(self.indptr).astype(float)
        if self.boundary_count:
            rows = np.searchsorted(self.indptr[1:], self.boundary_positions, side="right")
            np.add.at(counts, rows, self.boundary_weights - 1.0)
        return counts
```

- [ ] **Step 4: Implement `NeighborhoodEngine` candidate generation**

Construct one Euclidean `cKDTree(X)`. Implement block iterators:

```python
engine.single_blocks(beta, h, rho, kernel, record=True)
engine.multi_blocks(basis, eigenvalues, h, alpha, kernel, record=True)
engine.isotropic_blocks(h, kernel, record=False)
```

Single candidates use the exact lower bounds
`abs((x-center) @ beta) < h / sqrt(1 + rho**2)` and, for `rho > 0`,
`||x-center|| < h/rho`. Multi candidates use a projected `cKDTree` on
`(X @ basis) * sqrt(eigenvalues)` with radius `h` and, for `alpha > 0`, the
Euclidean radius `h/alpha`. Intersect sorted candidate arrays, compute exact
`q`, then apply strict `q < 1`.

Encode box with indices only. Encode plateau boundary positions and values
computed by `plateau_kernel`; do not retain dense candidates or distances.

- [ ] **Step 5: Run the neighborhood tests**

Run `rtk proxy python -m pytest -q test/test_ADP_box_kernel.py`.
Expected: kernel and dense-reference neighborhood tests pass.

- [ ] **Step 6: Commit the engine changes**

```bash
rtk git add ADP/engine/box_kernel.py test/test_ADP_box_kernel.py
rtk git diff --cached --check
rtk git commit -m "feat: build exact sparse neighborhoods"
```

### Task 3: Sparse bandwidth, scale search, and local initialization

**Files:**
- Modify: `ADP/engine/box_kernel.py`
- Modify: `test/test_ADP_box_kernel.py`

- [ ] **Step 1: Add failing search and initialization tests**

Test that `search_sparse_bandwidth` returns a bandwidth whose mean mass is at
least the target and that a one-ULP downward move is infeasible for a box case
with distinct distances. Compare `search_sparse_scale` with a dense bisection
reference for both kernels. Verify `initialize_basis_local_sparse` returns an
orthonormal `(d, m)` basis and matches the dense weighted local-gradient
reference up to projector distance.

- [ ] **Step 2: Run the focused failing tests**

Run:

```bash
rtk proxy python -m pytest -q test/test_ADP_box_kernel.py \
  -k 'bandwidth or scale or local_initialization'
```

Expected: missing-function failures.

- [ ] **Step 3: Implement generic sparse mass and searches**

Add a helper summing `block.mass` without concatenating blocks. Implement the
same brackets as the current calculus path:

```python
def search_sparse_scale(build_blocks, target_mass):
    if _mean_mass(build_blocks(1.0)) >= target_mass:
        return 1.0
    if _mean_mass(build_blocks(0.0)) < target_mass:
        return None
    low, high = 0.0, 1.0
    while high - low > np.sqrt(np.finfo(float).eps):
        middle = (low + high) / 2.0
        if _mean_mass(build_blocks(middle)) >= target_mass:
            low = middle
        else:
            high = middle
    return float(low)
```

`search_sparse_bandwidth` brackets upward from `lower` and bisects 60 times,
calling `engine.isotropic_blocks(h, kernel, record=False)`. Use the combined
bounding-box diagonal as a finite initial upper bound and retain the existing
100-doubling failure guard.

- [ ] **Step 4: Implement sparse local initialization**

`initialize_basis_local_sparse(X, Y, engine, N_lin, kernel, local_ridge,
index_dim)` must:

1. obtain `h_lin` from sparse bandwidth search;
2. materialize only each support row's one-dimensional local weights;
3. solve the existing intercept-plus-gradient ridge least squares after
   removing zero-weight observations;
4. extract and orient the leading right singular vectors;
5. retain the current rank checks and error messages.

- [ ] **Step 5: Run all backend tests and commit**

```bash
rtk proxy python -m pytest -q test/test_ADP_box_kernel.py
rtk git add ADP/engine/box_kernel.py test/test_ADP_box_kernel.py
rtk git diff --cached --check
rtk git commit -m "feat: add sparse ADP searches and initialization"
```

Expected: all backend tests pass before the commit.

### Task 4: Sparse statistics adapter and exact reuse cache

**Files:**
- Modify: `ADP/engine/box_kernel.py`
- Modify: `ADP/ADP_Statistic.py`
- Modify: `test/test_ADP_box_kernel.py`

- [ ] **Step 1: Add failing statistics-equivalence and cache tests**

Create identical directions and compare every field of `ADP_Statistics` from
the sparse block iterator against dense reference weights for box and plateau.
Call the box path twice and assert the second call reports support cache hits.
Change one support index and assert the changed row is not reused. Change one
plateau boundary weight while preserving support and assert invalidation.

- [ ] **Step 2: Run the tests and verify sparse blocks are rejected**

Run:

```bash
rtk proxy python -m pytest -q test/test_ADP_box_kernel.py \
  -k 'statistics or reuse'
```

Expected: `calculate_statistics` rejects the sparse payload as a weight array.

- [ ] **Step 3: Implement `SparseStatisticsCache` and grouped local statistics**

Cache entries by exact block row key. A box key contains sorted indices. A
plateau key additionally contains boundary positions and exact weight bytes.
Each entry stores local indices, normalized weights, mass, local `X`, local
`Y`, mean, centered `X`, and local response mean.

Group rows sharing a key and evaluate their distinct `Phi` values together.
Return `(I, U, mean, n_eff, eta)` for the block using the same centered formulas
as `_local_block`. Accept `xp=np` or CuPy so only local padded/grouped arrays are
transferred to the device.

- [ ] **Step 4: Wire sparse payloads into both statistics functions**

Extend both signatures compatibly:

```python
def calculate_statistics(
    X, Y, weights, directions, batch_size=32, *, sparse_cache=None
):
    ...

def calculate_statistics_gpu(
    X, Y, weights, directions, batch_size=32, *, sparse_cache=None
):
    ...
```

Inside their existing block loops, dispatch `SparseNeighborhoodBlock` to the
new grouped helper and retain the current dense/local dispatch for ndarray
blocks. Do not import `ADP_Statistic` from `box_kernel.py`; the dependency must
remain one-way to avoid a cycle.

- [ ] **Step 5: Verify CPU equivalence and optional GPU behavior**

```bash
rtk proxy python -m pytest -q test/test_ADP_box_kernel.py
rtk proxy python -m pytest -q test/test_ADP_gpu.py -k sparse
```

Expected: CPU tests pass; CUDA tests pass when a device exists or use the
suite's explicit no-device skip behavior.

- [ ] **Step 6: Stage only task-owned hunks and commit**

Use `rtk git add ADP/engine/box_kernel.py test/test_ADP_box_kernel.py`, then
interactively stage only the sparse-statistics hunks from the already modified
`ADP/ADP_Statistic.py`. Verify with `rtk git diff --cached`, then commit:

```bash
rtk git commit -m "feat: consume sparse ADP neighborhoods"
```

### Task 5: Config, CLI, and public exports

**Files:**
- Modify: `ADP/ADP_Config.py`
- Modify: `ADP/cli.py`
- Modify: `ADP/__init__.py`
- Modify: `ADP/CLI.md`
- Modify: `test/test_ADP_box_kernel.py`

- [ ] **Step 1: Add failing CLI and validation tests**

Assert `parse_kernel("box") is box_kernel`, `parse_kernel("plateau")` resolves
to plateau with `tau=0.5`, and parser arguments produce `tau=0.3` for
`--kernel plateau --kernel-tau 0.3`. Assert invalid tau, tau with Epanechnikov,
and `smart_weights=True` with either sparse kernel raise clear errors. Verify
the experiment job spec serializes plateau as the existing partial structure.

- [ ] **Step 2: Run the focused tests and observe parser failures**

Run `rtk proxy python -m pytest -q test/test_ADP_box_kernel.py -k 'cli or config'`.
Expected: new CLI names and option are rejected.

- [ ] **Step 3: Add minimal config and CLI wiring**

Import `sparse_kernel_parameters` in `ADP_Config.__post_init__` and validate the
resolved mode. Keep `kernel` callable. Reject `smart_weights` when parameters
are non-`None`.

Extend `parse_kernel` for `box` and `plateau`. Add
`--kernel-tau` with default `None`; after parsing, replace plateau with
`make_plateau_kernel(args.kernel_tau or 0.5)` and reject a supplied tau for any
other kernel. Export `box_kernel`, `plateau_kernel`, and
`make_plateau_kernel` from `ADP/__init__.py`.

- [ ] **Step 4: Document the two CLI modes**

Add runnable examples for box and plateau, state `tau=0.5`, explain that they
always use exact sparse screening, and state that no automatic hybrid exists.

- [ ] **Step 5: Run focused CLI/config tests**

Run:

```bash
rtk proxy python -m pytest -q test/test_ADP_box_kernel.py ADP/test_experiment.py \
  -k 'kernel or cli or job_spec'
```

Expected: focused tests pass and existing custom callable parsing remains
covered.

- [ ] **Step 6: Stage only new hunks and commit**

Inspect each pre-existing diff, interactively stage only Task 5 hunks, verify
`rtk git diff --cached`, then commit with:

```bash
rtk git commit -m "feat: expose sparse kernel modes"
```

### Task 6: Single- and multi-index model routing

**Files:**
- Modify: `ADP/single_index/ADP_single_index.py`
- Modify: `ADP/multi_index/ADP_multi_index.py`
- Modify: `test/test_ADP_box_kernel.py`

- [ ] **Step 1: Add failing single/multi smoke and dense-path guard tests**

For each mode and new kernel, fit a small deterministic dataset with bounded
`outer_steps`. Monkeypatch each model module's `pairwise_distance2`,
`calculate_weight`, or `calculate_multi_weight` to raise if called. Assert a
fitted normalized index/basis, finite statistics, and trace fields
`support_edges`, `boundary_edges`, and `support_reuse_hits`.

- [ ] **Step 2: Run smoke tests and verify the dense guard fires**

Run:

```bash
rtk proxy python -m pytest -q test/test_ADP_box_kernel.py \
  -k 'single_fit or multi_fit or dense_path'
```

Expected: failure from the monkeypatched dense path.

- [ ] **Step 3: Route single-index fits**

Resolve sparse kernel parameters once during initialization. For sparse mode:

- construct `NeighborhoodEngine(X, centers, batch_size)`;
- use sparse local initialization when requested;
- use sparse initial bandwidth;
- generate `engine.single_blocks(...)` and pass a persistent
  `SparseStatisticsCache` to the selected CPU/GPU statistics function;
- update `rho` through sparse scale search;
- add engine/cache counts to each trace row.

Leave the existing dense distance, weight, and smart paths textually intact in
the non-sparse branch.

- [ ] **Step 4: Route multi-index fits**

Apply the same branch using `engine.multi_blocks(...)` and sparse alpha search.
Random and pilot initialization remain unchanged; local mode uses sparse local
initialization. Preserve eigenvalue validation and matrix-free LSMR behavior.

Set effective parameters in both models:

```python
"kernel_mode": sparse_parameters[0] if sparse_parameters else "callable",
"kernel_tau": sparse_parameters[1] if sparse_parameters else None,
```

- [ ] **Step 5: Run backend and existing model tests**

```bash
rtk proxy python -m pytest -q \
  test/test_ADP_box_kernel.py \
  test/test_ADP_multi_index.py \
  test/test_ADP_smart.py
```

Expected: all selected tests pass; Epanechnikov smart tests prove the old path
is unchanged.

- [ ] **Step 6: Stage only Task 6 hunks and commit**

Interactively stage model hunks because both files were dirty before this
task. Verify the cached diff contains no pre-existing GPU or experiment edits,
then commit:

```bash
rtk git commit -m "feat: run ADP with sparse kernels"
```

### Task 7: Final validation and artifact inspection

**Files:**
- Modify only if a verification failure identifies an in-scope defect.

- [ ] **Step 1: Compile the package**

Run `rtk python -m compileall -q ADP`.
Expected: exit code zero and no output.

- [ ] **Step 2: Run the complete test suite transparently**

Run `rtk proxy python -m pytest -q`.
Expected: all collected tests pass; CUDA-only tests may explicitly skip when no
CUDA device is present.

- [ ] **Step 3: Run real CLI smoke commands**

```bash
rtk python ADP/cli.py --terminal-only --mode single --kernel box \
  --n 80 --d 3 --N_J 16 --N_loc 5 --outer-steps 3 --no-progress

rtk python ADP/cli.py --terminal-only --mode multi --index-dim 2 \
  --kernel plateau --kernel-tau 0.5 --n 100 --d 4 --N_J 20 --N_loc 5 \
  --outer-steps 3 --no-progress
```

Expected: each command finishes normally and reports its requested/effective
kernel mode.

- [ ] **Step 4: Inspect memory-shape diagnostics**

Run a small direct probe that iterates one box and one plateau block. Assert no
attribute has shape `(J, n)`, box has zero boundary floats, plateau stores
exactly `boundary_edges` floats, and reported masses match dense references.

- [ ] **Step 5: Audit requirements and the final diff**

Re-read
`docs/superpowers/specs/2026-08-11-box-plateau-sparse-backend-design.md` and
check every requirement against code or a passing test. Run:

```bash
rtk git diff --check
rtk git status --short
```

Report unrelated pre-existing changes separately. Do not claim runtime or
memory improvement without a matched benchmark.
