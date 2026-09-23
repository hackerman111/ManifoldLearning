# Hypo VarPro Riemannian L-BFGS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a directly comparable single-index ADP variant whose local
coefficients are eliminated analytically and whose unit direction is optimized
by Riemannian L-BFGS.

**Architecture:** Reuse the current `hypo.single_index.ADP_single_index` class
for input preparation, initialization, structural adaptation, statistics, and
telemetry. Subclass it under `hypo/varpro/` and replace only `_alternating`;
copy the current tester scenario and change only its import and
solver-specific checks.

**Tech Stack:** Python, NumPy, existing ADP statistics and root tester
discovery.

---

### Task 1: Implement the VarPro solver subclass

**Files:**

- Create: `hypo/varpro/ADP_single_index.py`

- [ ] **Step 1: Add constructor and timing compatibility**

Import the baseline class under a private alias. Add `lbfgs_memory`, default
`inner_steps` to 50 through the inherited constructor, and rename the
parent-loop timing slot after `fit`:

```python
class ADP_single_index(_ADP_single_index):
    def __init__(self, *, lbfgs_memory=10, **kwargs):
        if isinstance(lbfgs_memory, bool) or not isinstance(
            lbfgs_memory, (int, np.integer)
        ) or lbfgs_memory < 1:
            raise ValueError("lbfgs_memory must be a positive integer")
        kwargs.setdefault("inner_steps", 50)
        super().__init__(**kwargs)
        self.lbfgs_memory = lbfgs_memory

    def fit(self, X, Y):
        super().fit(X, Y)
        timings = self.timings_
        self.timings_ = {
            "initialization": timings["initialization"],
            "rho": timings["rho"],
            "directions": timings["directions"],
            "weights": timings["weights"],
            "statistics": timings["statistics"],
            "rlbfgs": timings["lsmr"],
            "total": timings["total"],
        }
        return self
```

- [ ] **Step 2: Implement the profiled objective and Riemannian gradient**

Use the stable residual form of the profiled objective and the envelope
gradient:

```python
def _profiled_value_gradient(self, I, U, beta):
    projected = U @ beta
    numerator = np.sum(I * projected, axis=1)
    denominator = np.sum(projected * projected, axis=1) + self.local_ridge
    slopes = numerator / denominator
    residual = I - slopes[:, None] * projected
    value = 0.5 * (
        np.sum(residual * residual)
        + self.local_ridge * np.dot(slopes, slopes)
    )
    euclidean = -np.einsum(
        "j,jpd,jp->d", slopes, U, residual, optimize=True
    )
    gradient = euclidean - beta * np.dot(beta, euclidean)
    if not (
        np.isfinite(value)
        and np.all(np.isfinite(slopes))
        and np.all(np.isfinite(gradient))
    ):
        raise RuntimeError("VarPro objective returned nonfinite values")
    return float(value), gradient, slopes
```

- [ ] **Step 3: Implement two-loop recursion**

Operate only on accepted positive-curvature tangent pairs:

```python
@staticmethod
def _lbfgs_direction(gradient, history):
    vector = gradient.copy()
    coefficients = []
    for step, change in reversed(history):
        inverse_curvature = 1.0 / np.dot(step, change)
        coefficient = inverse_curvature * np.dot(step, vector)
        coefficients.append(coefficient)
        vector -= coefficient * change
    if history:
        step, change = history[-1]
        vector *= np.dot(step, change) / np.dot(change, change)
    for (step, change), coefficient in zip(
        history, reversed(coefficients), strict=True
    ):
        inverse_curvature = 1.0 / np.dot(step, change)
        vector += step * (
            coefficient - inverse_curvature * np.dot(change, vector)
        )
    return -vector
```

- [ ] **Step 4: Implement retraction, transport, Armijo search, and stopping**

In `_riemannian_lbfgs`, project the direction, fall back to the negative
gradient for a non-descent direction, normalize `beta + alpha * direction`,
backtrack at most 30 times, transport all history by tangent projection, and
retain a new pair only when
`dot(step, change) > 1e-10 * norm(step) * norm(change)`. Stop on gradient norm,
relative objective change, or `self.inner_steps`.

Return `(beta, slopes, diagnostics)` with the exact trace keys listed in the
design.

- [ ] **Step 5: Replace the inherited solver boundary**

Time the entire VarPro solve in the inherited temporary `lsmr` slot:

```python
def _alternating(self, statistics, beta):
    started = perf_counter()
    result = self._riemannian_lbfgs(statistics["I"], statistics["U"], beta)
    self.timings_["lsmr"] += perf_counter() - started
    return result
```

- [ ] **Step 6: Compile the solver**

Run:

```bash
rtk python -m py_compile hypo/varpro/ADP_single_index.py
```

Expected: exit code 0.

### Task 2: Add the paired VarPro tester

**Files:**

- Create: `hypo/varpro/tester.py`

- [ ] **Step 1: Copy the benchmark surface**

Keep `--n`, `--d`, `--noise`, `--seed`, `--threshold`, and `--beta-init`;
generate the same normalized `beta_true`, Gaussian `X`, sinusoidal response,
and Gaussian noise as the current single-index tester.

- [ ] **Step 2: Add the gradient self-check**

Build deterministic small `I`, `U`, unit beta, and unit tangent direction.
Compare `dot(gradient, direction)` with a central finite difference along the
normalized sphere retraction using `epsilon=1e-6`, `rtol=1e-5`, and
`atol=1e-7`.

- [ ] **Step 3: Update implementation-specific checks**

Require finite unit beta, nonempty trace, finite objective and gradient norm,
known solver statuses, at least one solver iteration, the timing keys

```python
(
    "initialization",
    "rho",
    "directions",
    "weights",
    "statistics",
    "rlbfgs",
    "total",
)
```

and the existing weight-density conditions. Print the same recovery and
timing summaries plus the last solver status.

- [ ] **Step 4: Compile and discover the tester**

Run:

```bash
rtk python -m py_compile hypo/varpro/tester.py
rtk python hypo/tester.py --list
```

Expected: both commands exit 0 and `varpro` appears in the list.

### Task 3: Verify mathematics, recovery, and compatibility

**Files:**

- Verify: `hypo/varpro/ADP_single_index.py`
- Verify: `hypo/varpro/tester.py`
- Verify unchanged: `hypo/single_index/ADP_single_index.py`
- Verify unchanged: `hypo/single_index/tester.py`

- [ ] **Step 1: Run the new paired benchmark**

Run:

```bash
rtk python hypo/tester.py varpro --seed 42
```

Expected for the current high-dimensional scenario: the gradient and
technical diagnostics pass, while the command exits `1` because the measured
quality is `cosine_final=0.766502 < 0.90`. Record this quality failure
separately from `solver_status=max_iterations`; do not change the dataset,
threshold, or TeX objective to force a pass.

- [ ] **Step 2: Re-run the LSMR baseline on the identical seed**

Run:

```bash
rtk python hypo/tester.py single_index --seed 42
```

Expected: reproduce the baseline quality failure
`cosine_final=0.864072 < 0.90`; record it separately from the VarPro result.

- [ ] **Step 3: Run the shared-statistics smoke benchmark**

Run:

```bash
rtk python hypo/tester.py statistic --profile smoke --repetitions 1 \
    --output /tmp/adp_varpro_stat_smoke.csv
```

Expected: exit code 0.

- [ ] **Step 4: Check the complete diff**

Run:

```bash
rtk git diff --check
rtk git status --short
```

Expected: no whitespace errors; only the pre-existing working-tree changes and
the four requested design/plan/VarPro files are present.

No commit is part of this plan because the working tree already contains
uncommitted user changes that the VarPro subclass intentionally consumes.
