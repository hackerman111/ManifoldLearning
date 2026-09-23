# Hypo Single-Index ADP LSMR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a complete adaptive single-index ADP implementation and a deterministic beta-recovery tester discoverable through `hypo/tester.py`.

**Architecture:** A standalone class under `hypo/single_index` owns the ADP loop and matrix-free LSMR operator while reusing `ADP.ADP_statistic.calculate_statistics`. A neighboring tester is the single acceptance check; the root tester discovers it without modification.

**Tech Stack:** Python, NumPy, SciPy `LinearOperator` and `lsmr`.

---

### Task 1: Add the failing recovery tester

**Files:**
- Create: `hypo/single_index/tester.py`

- [ ] **Step 1: Create the tester**

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hypo.single_index.ADP_single_index import ADP_single_index


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Проверка восстановления beta методом ADP.")
    parser.add_argument("--n", type=int, default=1200)
    parser.add_argument("--d", type=int, default=6)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args(argv)
    if args.n <= args.d + 1:
        parser.error("--n must exceed --d + 1")
    if args.d < 1:
        parser.error("--d must be positive")
    if not np.isfinite(args.noise) or args.noise < 0:
        parser.error("--noise must be finite and nonnegative")
    if not np.isfinite(args.threshold) or not 0 < args.threshold <= 1:
        parser.error("--threshold must be in (0, 1]")
    return args


def absolute_cosine(left, right):
    return float(abs(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))))


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    beta_true = rng.normal(size=args.d)
    beta_true /= np.linalg.norm(beta_true)
    X = rng.normal(size=(args.n, args.d))
    Y = np.square(X @ beta_true) + args.noise * rng.normal(size=args.n)

    model = ADP_single_index(seed=args.seed + 1).fit(X, Y)
    initial_cosine = absolute_cosine(model.beta_init_, beta_true)
    final_cosine = absolute_cosine(model.beta_, beta_true)
    valid = (
        np.all(np.isfinite(model.beta_))
        and np.isclose(np.linalg.norm(model.beta_), 1.0, atol=1e-10)
        and bool(model.trace_)
        and any(step["lsmr_iterations"] > 0 for step in model.trace_)
        and final_cosine >= args.threshold
    )
    print(
        f"seed={args.seed} n={args.n} d={args.d} "
        f"cosine_init={initial_cosine:.6f} cosine_final={final_cosine:.6f} "
        f"threshold={args.threshold:.2f} outer_steps={len(model.trace_)} "
        f"status={'PASS' if valid else 'FAIL'}"
    )
    return int(not valid)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify the root tester discovers the route but execution fails**

Run:

```bash
rtk python hypo/tester.py --list
rtk python hypo/tester.py single_index
```

Expected: `single_index` appears in the list; execution fails because
`hypo.single_index.ADP_single_index` does not exist.

### Task 2: Implement adaptive ADP and matrix-free LSMR

**Files:**
- Create: `hypo/single_index/ADP_single_index.py`
- Modify: `ADP/__init__.py:3`

- [ ] **Step 1: Repair the existing package import**

```python
from .single_index.ADP_single_index import ADP_single_index
```

- [ ] **Step 2: Implement `ADP_single_index`**

```python
from __future__ import annotations

import math

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr

from ADP.ADP_statistic import calculate_statistics


class ADP_single_index:
    def __init__(
        self,
        *,
        n_centers=64,
        n_directions=8,
        N_loc=10,
        N_lin=None,
        lambda_penalty=1.0,
        local_ridge=1e-8,
        outer_steps=4,
        inner_steps=5,
        bandwidth_decay=math.sqrt(2.0),
        h_min=None,
        tol=1e-6,
        batch_size=32,
        seed=42,
    ):
        for name, value in (
            ("n_centers", n_centers),
            ("n_directions", n_directions),
            ("outer_steps", outer_steps),
            ("inner_steps", inner_steps),
            ("batch_size", batch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("N_loc", N_loc),
            ("lambda_penalty", lambda_penalty),
            ("local_ridge", local_ridge),
            ("tol", tol),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if N_lin is not None and (not np.isfinite(N_lin) or N_lin <= 0):
            raise ValueError("N_lin must be finite and positive")
        if not np.isfinite(bandwidth_decay) or bandwidth_decay <= 1:
            raise ValueError("bandwidth_decay must exceed one")
        if h_min is not None and (not np.isfinite(h_min) or h_min <= 0):
            raise ValueError("h_min must be finite and positive")

        self.n_centers = n_centers
        self.n_directions = n_directions
        self.N_loc = float(N_loc)
        self.N_lin = None if N_lin is None else float(N_lin)
        self.lambda_penalty = float(lambda_penalty)
        self.local_ridge = float(local_ridge)
        self.outer_steps = outer_steps
        self.inner_steps = inner_steps
        self.bandwidth_decay = float(bandwidth_decay)
        self.h_min = None if h_min is None else float(h_min)
        self.tol = float(tol)
        self.batch_size = batch_size
        self.seed = seed
        self.beta_ = None
        self.beta_init_ = None
        self.slopes_ = None
        self.h0_ = None
        self.trace_ = []

    def fit(self, X, Y):
        X, Y = self._prepare_inputs(X, Y)
        n, d = X.shape
        if self.N_loc > n:
            raise ValueError("N_loc cannot exceed n")
        N_lin = self.N_lin
        if N_lin is None:
            N_lin = min(n, max(2 * d + 2, n // max(1, int(self.N_loc))))
        if N_lin > n:
            raise ValueError("N_lin cannot exceed n")

        rng = np.random.default_rng(self.seed)
        centers = X[rng.choice(n, size=min(self.n_centers, n), replace=False)]
        distance2 = self._pairwise_distance2(X, centers)
        h_lin = self._search_bandwidth(distance2, N_lin)
        beta = self._initial_beta(X, Y, centers, distance2, h_lin)
        h = self._search_bandwidth(distance2, self.N_loc)
        feature_scale = float(np.mean(np.std(X, axis=0)))
        h_min = self.h_min or max(10 * feature_scale / n, np.finfo(float).eps)

        self.beta_init_ = beta.copy()
        self.h0_ = h
        self.trace_ = []
        slopes = None
        for outer in range(self.outer_steps):
            rho = None
            if outer:
                next_h = h / self.bandwidth_decay
                if next_h < h_min:
                    break
                h = next_h
                rho = self._search_rho(X, centers, distance2, h, beta)

            directions = self._directions(
                rng, centers.shape[0], d, beta, rho
            )
            weights = self._weights(X, centers, distance2, h, beta, rho)
            statistics = calculate_statistics(
                X, Y, weights, directions, batch_size=self.batch_size
            )
            beta, slopes, record = self._alternating(statistics, beta)
            record.update(
                outer=outer,
                h=float(h),
                rho=None if rho is None else float(rho),
                mean_mass=float(np.mean(statistics["mass"])),
            )
            self.trace_.append(record)

        if slopes is None:
            raise RuntimeError("ADP did not complete an outer step")
        self.beta_ = beta
        self.slopes_ = slopes
        return self

    @staticmethod
    def _prepare_inputs(X, Y):
        arrays = []
        for value, name in ((X, "X"), (Y, "Y")):
            array = np.asarray(value)
            if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
                raise TypeError(f"{name} must have a real numeric dtype")
            array = array.astype(float, copy=False)
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values")
            arrays.append(array)
        X, Y = arrays
        if X.ndim != 2 or 0 in X.shape:
            raise ValueError("X must have non-empty shape (n, d)")
        if Y.shape != (X.shape[0],):
            raise ValueError("Y must have shape (n,)")
        if X.shape[0] <= X.shape[1] + 1:
            raise ValueError("n must exceed d + 1")
        return X, Y

    @staticmethod
    def _kernel(argument):
        return np.maximum(1 - np.square(argument), 0)

    @staticmethod
    def _pairwise_distance2(X, centers):
        distance2 = (
            np.square(centers).sum(axis=1)[:, None]
            + np.square(X).sum(axis=1)[None, :]
            - 2 * centers @ X.T
        )
        np.maximum(distance2, 0, out=distance2)
        return distance2

    def _mass(self, argument):
        return float(np.mean(np.sum(self._kernel(argument), axis=1)))

    def _search_bandwidth(self, distance2, target):
        low = np.finfo(float).eps
        high = max(float(np.sqrt(np.max(distance2, initial=0))), 1.0)
        for _ in range(80):
            if self._mass(distance2 / high**2) >= target:
                break
            high *= 2
        else:
            raise RuntimeError("could not bracket a feasible bandwidth")
        for _ in range(60):
            middle = (low + high) / 2
            if self._mass(distance2 / middle**2) >= target:
                high = middle
            else:
                low = middle
        return float(high)

    def _initial_beta(self, X, Y, centers, distance2, h_lin):
        weights = self._kernel(distance2 / h_lin**2)
        d = X.shape[1]
        ridge_rows = np.zeros((d, d + 1))
        ridge_rows[:, 1:] = np.sqrt(self.local_ridge) * np.eye(d)
        gradients = np.empty((centers.shape[0], d))
        for j, center in enumerate(centers):
            design = np.column_stack((np.ones(X.shape[0]), X - center))
            root_weight = np.sqrt(weights[j])
            augmented_design = np.vstack((design * root_weight[:, None], ridge_rows))
            augmented_y = np.concatenate((Y * root_weight, np.zeros(d)))
            gradients[j] = np.linalg.lstsq(
                augmented_design, augmented_y, rcond=None
            )[0][1:]
        _, singular_values, right_vectors = np.linalg.svd(
            gradients, full_matrices=False
        )
        if singular_values[0] <= np.finfo(float).eps:
            raise RuntimeError("local gradients do not identify beta")
        beta = right_vectors[0]
        beta /= np.linalg.norm(beta)
        if beta[np.argmax(np.abs(beta))] < 0:
            beta = -beta
        return beta

    def _search_rho(self, X, centers, distance2, h, beta):
        projected = centers @ beta
        projection2 = np.square(projected[:, None] - (X @ beta)[None, :])

        def mass(rho):
            return self._mass((rho**2 * distance2 + projection2) / h**2)

        if mass(1.0) >= self.N_loc:
            return 1.0
        if mass(0.0) < self.N_loc:
            raise RuntimeError("N_loc cannot be preserved at rho=0")
        low, high = 0.0, 1.0
        for _ in range(50):
            middle = (low + high) / 2
            if mass(middle) >= self.N_loc:
                low = middle
            else:
                high = middle
        return float(low)

    def _directions(self, rng, centers, d, beta, rho):
        values = rng.normal(size=(centers, self.n_directions, d))
        if rho is not None:
            values = rho * values + rng.normal(
                size=(centers, self.n_directions, 1)
            ) * beta
        norms = np.linalg.norm(values, axis=2, keepdims=True)
        if np.any(norms == 0):
            raise RuntimeError("generated a zero direction")
        return values / norms

    def _weights(self, X, centers, distance2, h, beta, rho):
        if rho is None:
            argument = distance2 / h**2
        else:
            projection2 = np.square(
                (centers @ beta)[:, None] - (X @ beta)[None, :]
            )
            argument = (rho**2 * distance2 + projection2) / h**2
        return self._kernel(argument)

    def _slopes(self, I, U, beta):
        projected = U @ beta
        return np.sum(I * projected, axis=1) / (
            np.sum(projected * projected, axis=1) + self.local_ridge
        )

    def _solve_beta(self, I, U, slopes, beta_prior):
        rows = I.size
        d = U.shape[2]
        sqrt_lambda = math.sqrt(self.lambda_penalty)

        def matvec(vector):
            data = (slopes[:, None] * (U @ vector)).ravel()
            return np.concatenate((data, sqrt_lambda * vector))

        def rmatvec(vector):
            data = vector[:rows].reshape(I.shape)
            return (
                np.einsum("j,jpd,jp->d", slopes, U, data, optimize=True)
                + sqrt_lambda * vector[rows:]
            )

        operator = LinearOperator(
            (rows + d, d), matvec=matvec, rmatvec=rmatvec, dtype=float
        )
        rhs = np.concatenate((I.ravel(), sqrt_lambda * beta_prior))
        result = lsmr(
            operator,
            rhs,
            atol=min(self.tol, 1e-8),
            btol=min(self.tol, 1e-8),
            maxiter=max(50, 5 * d),
        )
        beta = result[0]
        norm = np.linalg.norm(beta)
        if not np.all(np.isfinite(beta)) or not np.isfinite(norm) or norm == 0:
            raise RuntimeError("LSMR returned an invalid beta")
        beta /= norm
        if np.dot(beta, beta_prior) < 0:
            beta = -beta
        return beta, int(result[1]), int(result[2])

    def _alternating(self, statistics, beta):
        I, U = statistics["I"], statistics["U"]
        stop = iterations = 0
        delta = math.inf
        for inner in range(self.inner_steps):
            prior = beta
            slopes = self._slopes(I, U, prior)
            beta, stop, iterations = self._solve_beta(I, U, slopes, prior)
            delta = min(
                np.linalg.norm(beta - prior), np.linalg.norm(beta + prior)
            )
            if delta < self.tol:
                break
        slopes = self._slopes(I, U, beta)
        return beta, slopes, {
            "inner_iterations": inner + 1,
            "lsmr_stop": stop,
            "lsmr_iterations": iterations,
            "beta_delta": float(delta),
        }
```

- [ ] **Step 3: Compile and run the recovery route**

Run:

```bash
rtk python -m py_compile hypo/single_index/ADP_single_index.py hypo/single_index/tester.py
rtk python hypo/tester.py single_index
```

Expected: tester prints `status=PASS`, final absolute cosine is at least
`0.90`, and the command exits zero.

### Task 3: Regression-check the existing statistics route

**Files:**
- Verify: `hypo/statistic/ADP_single_index_hypo_stat.py`
- Verify: `hypo/statistic/tester.py`

- [ ] **Step 1: Run the existing smoke profile**

Run:

```bash
rtk python hypo/tester.py statistic --profile smoke --repetitions 1 --output /tmp/adp-statistic-smoke.csv
```

Expected: direct, matrix, and sparse rows complete without an import error and
the command exits zero.

- [ ] **Step 2: Verify route discovery and workspace scope**

Run:

```bash
rtk python hypo/tester.py --list
rtk git diff --check
rtk git status --short
```

Expected: both `single_index` and `statistic` are listed; no whitespace errors;
pre-existing user modifications remain present and uncommitted.

- [ ] **Step 3: Commit only implementation-owned files**

```bash
rtk git add ADP/__init__.py hypo/single_index/ADP_single_index.py hypo/single_index/tester.py docs/superpowers/plans/2026-08-02-hypo-single-index-adp-lsmr.md
rtk git commit -m "feat: add hypo single-index ADP LSMR"
```
