# Hybrid Krylov Dynamic Lambda Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separately runnable `hypo/Krylov` ADP variant whose beta update uses matrix-free Golub--Kahan projection and projected GCV instead of fixed-lambda LSMR.

**Architecture:** Subclass `hypo.single_index.ADP_single_index` so the current initialization, weights, statistics, slopes, and outer loop remain the comparison baseline. Override only the beta update, alternating-solver telemetry, and timing label; keep a dedicated tester beside the variant.

**Tech Stack:** Python 3, NumPy, SciPy `minimize_scalar`, existing ADP statistics and `hypo/tester.py` discovery.

---

### Task 1: Add a failing projected-ridge check

**Files:**
- Create: `hypo/Krylov/tester.py`

- [ ] **Step 1: Create the deterministic check**

Create `hypo/Krylov/tester.py` with:

```python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hypo.Krylov.ADP_single_index import ADP_single_index


def projected_ridge_check():
    diagonal = np.linspace(1.0, 2.0, 6)
    subdiagonal = np.linspace(0.1, 0.6, 6)
    B = np.zeros((7, 6))
    B[np.arange(6), np.arange(6)] = diagonal
    B[np.arange(1, 7), np.arange(6)] = subdiagonal
    rho = 2.5

    model = ADP_single_index()
    selected_lambda, gcv, coordinates, _ = model._select_lambda(B, rho, None)
    rhs = np.zeros(7)
    rhs[0] = rho
    direct = np.linalg.solve(
        B.T @ B + selected_lambda * np.eye(6),
        B.T @ rhs,
    )
    return bool(
        np.isfinite(selected_lambda)
        and selected_lambda > 0
        and np.isfinite(gcv)
        and gcv >= 0
        and np.allclose(coordinates, direct, rtol=1e-9, atol=1e-11)
    )


if __name__ == "__main__":
    raise SystemExit(not projected_ridge_check())
```

- [ ] **Step 2: Run the check and confirm the missing implementation**

Run:

```bash
rtk python hypo/Krylov/tester.py
```

Expected: nonzero exit with `ModuleNotFoundError` for
`hypo.Krylov.ADP_single_index`.

### Task 2: Implement Hybrid Golub--Kahan and projected GCV

**Files:**
- Create: `hypo/Krylov/ADP_single_index.py`
- Test: `hypo/Krylov/tester.py`

- [ ] **Step 1: Add the isolated solver subclass**

Create `hypo/Krylov/ADP_single_index.py` with one subclass and these constants:

```python
from __future__ import annotations

import math
from time import perf_counter

import numpy as np
from scipy.optimize import minimize_scalar

from hypo.single_index.ADP_single_index import ADP_single_index as _ADP_single_index


class ADP_single_index(_ADP_single_index):
    _CHECK_EVERY = 5
    _KRYLOV_LIMIT = 100
    _GCV_GRID_SIZE = 21
    _LAMBDA_LOG_TOL = 0.1
    _DIRECTION_TOL = 1e-3
    _GCV_TOL = 1e-2
```

Override `fit()` so lambda continuation is reset for each fit and the inherited
temporary timing slot is renamed:

```python
    def fit(self, X, Y):
        self.lambda_ = None
        super().fit(X, Y)
        timings = self.timings_
        self.timings_ = {
            "initialization": timings["initialization"],
            "rho": timings["rho"],
            "directions": timings["directions"],
            "weights": timings["weights"],
            "statistics": timings["statistics"],
            "slopes": timings["slopes"],
            "krylov": timings["lsmr"],
            "total": timings["total"],
        }
        return self
```

- [ ] **Step 2: Implement one-SVD projected GCV selection**

Add `_select_lambda(B, rho, lambda_prior)`. It must:

1. compute `np.linalg.svd(B, full_matrices=False)` exactly once;
2. form `rhs = rho * e1`;
3. evaluate
   `y = V @ ((s / (s**2 + lambda)) * (U.T @ rhs))`;
4. evaluate
   `||B @ y - rhs||**2 / (B.shape[0] - sum(s**2/(s**2+lambda)))**2`;
5. search a 21-point geometric grid over the spectral and continuation union;
6. expand one boundary by one decade once;
7. refine an interior minimum with bounded `minimize_scalar` in log-lambda;
8. return `(selected_lambda, gcv, y, boundary)`.

Use this exact search core:

```python
    def _select_lambda(self, B, rho, lambda_prior):
        left, singular_values, right = np.linalg.svd(B, full_matrices=False)
        if (
            singular_values.size == 0
            or not np.all(np.isfinite(singular_values))
            or singular_values[0] <= 0
        ):
            raise RuntimeError("Golub--Kahan projection is singular")

        rhs = np.zeros(B.shape[0])
        rhs[0] = rho
        projected_rhs = left.T @ rhs
        squared = singular_values * singular_values

        def evaluate(lambda_value):
            coordinates = right.T @ (
                singular_values / (squared + lambda_value) * projected_rhs
            )
            residual = B @ coordinates - rhs
            denominator = B.shape[0] - np.sum(
                squared / (squared + lambda_value)
            )
            if denominator <= np.finfo(float).eps:
                return math.inf, coordinates
            gcv = np.dot(residual, residual) / denominator**2
            return float(gcv), coordinates

        scale = max(float(squared[0]), np.finfo(float).tiny)
        lower, upper = 1e-8 * scale, 1e2 * scale
        if (
            lambda_prior is not None
            and np.isfinite(lambda_prior)
            and lambda_prior > 0
        ):
            lower = min(lower, lambda_prior / 100)
            upper = max(upper, lambda_prior * 100)

        for expansion in range(2):
            grid = np.geomspace(lower, upper, self._GCV_GRID_SIZE)
            scores = np.array([evaluate(value)[0] for value in grid])
            if not np.any(np.isfinite(scores)):
                raise RuntimeError("projected GCV returned no finite value")
            best = int(np.argmin(scores))
            boundary = best in {0, grid.size - 1}
            if expansion == 0 and boundary:
                if best == 0:
                    lower /= 10
                else:
                    upper *= 10
                continue
            break

        selected_lambda = float(grid[best])
        selected_gcv, coordinates = evaluate(selected_lambda)
        if not boundary:
            result = minimize_scalar(
                lambda value: evaluate(math.exp(value))[0],
                bounds=(math.log(grid[best - 1]), math.log(grid[best + 1])),
                method="bounded",
                options={"xatol": 1e-4},
            )
            if result.success and np.isfinite(result.fun) and result.fun < selected_gcv:
                selected_lambda = float(math.exp(result.x))
                selected_gcv, coordinates = evaluate(selected_lambda)

        if not (
            np.isfinite(selected_lambda)
            and selected_lambda > 0
            and np.isfinite(selected_gcv)
            and np.all(np.isfinite(coordinates))
        ):
            raise RuntimeError("projected GCV returned an invalid solution")
        return selected_lambda, selected_gcv, coordinates, boundary
```

- [ ] **Step 3: Implement the matrix-free Golub--Kahan process**

Add `_hybrid_krylov(I, U, slopes, beta_prior)` with matrix-free functions:

```python
        def matvec(vector):
            return (slopes[:, None] * (U @ vector)).ravel()

        def rmatvec(vector):
            data = vector.reshape(I.shape)
            return np.einsum(
                "j,jpd,jp->d", slopes, U, data, optimize=True
            )
```

Use `r0 = I.ravel() - matvec(beta_prior)`, standard Golub--Kahan recurrences,
and store only the right basis plus bidiagonal coefficients. Use the exact
recurrence order below so the subdiagonal coefficient is included in the
current projected matrix before testing for breakdown:

```python
        rhs = I.ravel()
        residual = rhs - matvec(beta_prior)
        rho = np.linalg.norm(residual)
        scale = max(1.0, np.linalg.norm(rhs))
        epsilon = np.finfo(float).eps
        if not np.isfinite(rho):
            raise RuntimeError("Golub--Kahan residual is nonfinite")
        if rho <= epsilon * scale:
            return beta_prior.copy(), {
                "solver_status": "zero_residual",
                "krylov_iterations": 0,
                "selected_lambda": self.lambda_,
                "projected_gcv": 0.0,
                "lambda_at_boundary": False,
            }

        u = residual / rho
        raw_v = rmatvec(u)
        alpha = np.linalg.norm(raw_v)
        if not np.isfinite(alpha):
            raise RuntimeError("Golub--Kahan alpha is nonfinite")
        if alpha <= epsilon:
            return beta_prior.copy(), {
                "solver_status": "breakdown",
                "krylov_iterations": 0,
                "selected_lambda": self.lambda_,
                "projected_gcv": 0.0,
                "lambda_at_boundary": False,
            }
        v = raw_v / alpha

        basis = []
        diagonal = []
        subdiagonal = []
        candidate = previous = None
        stable_checks = 0
        status = "max_iterations"
        krylov_limit = min(U.shape[2], self._KRYLOV_LIMIT)

        for step in range(1, krylov_limit + 1):
            basis.append(v.copy())
            diagonal.append(alpha)
            raw_next_u = matvec(v) - alpha * u
            beta_coefficient = np.linalg.norm(raw_next_u)
            if not np.isfinite(beta_coefficient):
                raise RuntimeError("Golub--Kahan beta is nonfinite")
            subdiagonal.append(beta_coefficient)

            beta_breakdown = beta_coefficient <= epsilon * max(1.0, alpha)
            alpha_breakdown = False
            if not beta_breakdown:
                next_u = raw_next_u / beta_coefficient
                raw_next_v = rmatvec(next_u) - beta_coefficient * v
                next_alpha = np.linalg.norm(raw_next_v)
                if not np.isfinite(next_alpha):
                    raise RuntimeError("Golub--Kahan alpha is nonfinite")
                alpha_breakdown = next_alpha <= epsilon * max(
                    1.0, beta_coefficient
                )
            breakdown = beta_breakdown or alpha_breakdown
            must_check = (
                step % self._CHECK_EVERY == 0
                or step == krylov_limit
                or breakdown
            )
```

At each `must_check`, construct:

```python
        B = np.zeros((step + 1, step))
        indices = np.arange(step)
        B[indices, indices] = diagonal
        B[indices + 1, indices] = subdiagonal
        basis_matrix = np.column_stack(basis)
```

Select lambda using the preceding checkpoint lambda when available, falling
back to `self.lambda_`, then normalize and sign-align the projected update:

```python
        lambda_prior = (
            self.lambda_
            if previous is None
            else previous["selected_lambda"]
        )
        selected_lambda, selected_gcv, coordinates, boundary = (
            self._select_lambda(B, rho, lambda_prior)
        )
        beta = beta_prior + basis_matrix @ coordinates
        beta_norm = np.linalg.norm(beta)
        if (
            not np.all(np.isfinite(beta))
            or not np.isfinite(beta_norm)
            or beta_norm == 0
        ):
            raise RuntimeError("Hybrid Krylov returned an invalid beta")
        beta /= beta_norm
        if np.dot(beta, beta_prior) < 0:
            beta = -beta
```

Retain:

```python
        candidate = {
            "beta": beta,
            "solver_status": status,
            "krylov_iterations": step,
            "selected_lambda": selected_lambda,
            "projected_gcv": selected_gcv,
            "lambda_at_boundary": bool(boundary),
        }
```

Check stabilization against the preceding checkpoint with:

```python
        lambda_stable = abs(
            math.log10(selected_lambda)
            - math.log10(previous["selected_lambda"])
        ) < self._LAMBDA_LOG_TOL
        cosine = min(1.0, abs(np.dot(beta, previous["beta"])))
        direction_stable = math.sqrt(max(0.0, 2 * (1 - cosine))) < (
            self._DIRECTION_TOL
        )
        gcv_stable = abs(selected_gcv - previous["projected_gcv"]) / max(
            1.0, abs(selected_gcv)
        ) < self._GCV_TOL
```

Require all three conditions, an interior minimum, and two consecutive stable
checks. Use statuses `stabilized`, `max_iterations`, `breakdown`, and
`zero_residual`. Set `self.lambda_` only after a valid projected candidate.
After a nonterminal iteration assign:

```python
        u = next_u
        v = raw_next_v / next_alpha
        alpha = next_alpha
```

On breakdown return the current projected candidate with status `breakdown`.
On the second stable check return it with status `stabilized`. At the
projection limit return the final candidate with status `max_iterations`.
Before returning, set `self.lambda_ = candidate["selected_lambda"]` and remove
the internal `beta` key from the telemetry dictionary. Raise `RuntimeError` if
the loop ends without a projected candidate.

- [ ] **Step 4: Replace only the alternating beta update**

Override `_alternating()` while reusing `_slopes()`:

```python
    def _alternating(self, statistics, beta):
        I, U = statistics["I"], statistics["U"]
        delta = math.inf
        solver_record = None
        for inner in range(self.inner_steps):
            prior = beta
            started = perf_counter()
            slopes = self._slopes(I, U, prior)
            self.timings_["slopes"] += perf_counter() - started
            started = perf_counter()
            beta, solver_record = self._hybrid_krylov(I, U, slopes, prior)
            self.timings_["lsmr"] += perf_counter() - started
            delta = min(
                np.linalg.norm(beta - prior),
                np.linalg.norm(beta + prior),
            )
            if delta < self.tol:
                break

        started = perf_counter()
        slopes = self._slopes(I, U, beta)
        self.timings_["slopes"] += perf_counter() - started
        record = dict(solver_record)
        record["inner_iterations"] = inner + 1
        record["beta_delta"] = float(delta)
        return beta, slopes, record
```

- [ ] **Step 5: Run the projected-ridge check**

Run:

```bash
rtk python hypo/Krylov/tester.py
```

Expected: exit code `0` and no output.

- [ ] **Step 6: Commit the solver core**

Run:

```bash
rtk git add hypo/Krylov/ADP_single_index.py hypo/Krylov/tester.py
rtk git commit -m "feat: add hybrid Krylov ADP solver"
```

Expected: a commit containing only the two `hypo/Krylov` files.

### Task 3: Restore the baseline benchmark surface

**Files:**
- Modify: `hypo/Krylov/tester.py`
- Reference: `hypo/single_index/tester.py`

- [ ] **Step 1: Extend the tester without duplicating its parser**

Import the baseline CLI helpers:

```python
from hypo.single_index.tester import absolute_cosine, parse_args
```

Keep `projected_ridge_check()` and add `main(argv=None)` using the same data:

```python
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    beta_true = rng.normal(size=args.d)
    beta_true /= np.linalg.norm(beta_true)
    X = rng.normal(size=(args.n, args.d))
    Y = np.sin(X @ beta_true) + args.noise * rng.normal(size=args.n)
    model = ADP_single_index(
        seed=args.seed + 1, beta_init=args.beta_init
    ).fit(X, Y)
```

Validate timing names:

```python
    timing_names = (
        "initialization",
        "rho",
        "directions",
        "weights",
        "statistics",
        "slopes",
        "krylov",
        "total",
    )
```

For every trace row, accept only the four documented statuses, require a
nonnegative integer `krylov_iterations`, a boolean `lambda_at_boundary`, and
finite nonnegative projected GCV. Require a finite positive selected lambda
whenever `krylov_iterations > 0`.

Compute two explicit booleans:

```python
    technical_valid = (
        projected_ridge_check()
        and np.all(np.isfinite(model.beta_))
        and np.isclose(np.linalg.norm(model.beta_), 1.0, atol=1e-10)
        and bool(model.trace_)
        and trace_valid
        and any(step["krylov_iterations"] > 0 for step in model.trace_)
        and tuple(model.timings_) == timing_names
        and all(
            np.isfinite(model.timings_[name]) and model.timings_[name] >= 0
            for name in timing_names
        )
        and model.timings_["total"] > 0
        and np.isfinite(model.weight_density_)
        and 0 <= model.weight_density_ <= 1
        and np.isfinite(model.mean_zero_weight_fraction_)
        and 0 <= model.mean_zero_weight_fraction_ <= 1
        and np.isclose(
            model.mean_zero_weight_fraction_,
            np.mean([step["zero_weight_fraction"] for step in model.trace_]),
        )
    )
    quality_valid = final_cosine >= args.threshold
```

Print `technical_status`, `quality_status`, final solver status, selected
lambda, and the same timing/weight information as the baseline. Return nonzero
when either technical validity or quality fails. Replace the temporary
`projected_ridge_check()` entrypoint with:

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify automatic discovery**

Run:

```bash
rtk python hypo/tester.py --list
```

Expected output contains a separate `Krylov` line.

- [ ] **Step 3: Run a low-dimensional deterministic benchmark**

Run:

```bash
rtk python hypo/tester.py Krylov --n 1200 --d 6 --noise 0.05 --seed 42
```

Expected: `technical_status=PASS`, finite positive selected lambda, normalized
beta, and `quality_status=PASS`.

- [ ] **Step 4: Run the matching high-dimensional benchmark**

Run:

```bash
rtk python hypo/tester.py Krylov --n 1000 --d 100 --noise 0.5 --seed 42
```

Expected: `technical_status=PASS`. Record the measured quality status and cosine
without treating a threshold miss as a Krylov numerical failure.

- [ ] **Step 5: Commit the benchmark tester**

Run:

```bash
rtk git add hypo/Krylov/tester.py
rtk git commit -m "test: add hybrid Krylov benchmark"
```

Expected: a commit containing only `hypo/Krylov/tester.py`.

### Task 4: Final focused verification

**Files:**
- Verify: `hypo/Krylov/ADP_single_index.py`
- Verify: `hypo/Krylov/tester.py`

- [ ] **Step 1: Compile both Python files**

Run:

```bash
rtk python -m py_compile \
  hypo/Krylov/ADP_single_index.py \
  hypo/Krylov/tester.py
```

Expected: exit code `0` and no output.

- [ ] **Step 2: Re-run the deterministic projected solve**

Run:

```bash
rtk python -c "from hypo.Krylov.tester import projected_ridge_check; assert projected_ridge_check()"
```

Expected: exit code `0` and no output.

- [ ] **Step 3: Check patch hygiene**

Run:

```bash
rtk git diff --check
rtk git status --short
```

Expected: no whitespace errors; pre-existing unrelated dirty files remain
untouched.
