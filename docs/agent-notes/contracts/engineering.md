# Engineering contract

Load this file for Python implementation, API, dependency, backend, parallelism, error-handling, or tooling changes. It is intentionally separate from the root instructions so routine research sessions do not pay for unrelated implementation rules.

## Python style

Use modern, explicit Python.

- Put `from __future__ import annotations` in new Python modules.
- Type public functions, methods, dataclasses, and non-obvious internal interfaces.
- Prefer `dataclass(frozen=True, slots=True)` for immutable configuration/data records.
- Prefer `dataclass(slots=True)` for mutable result/state records when mutation is intentional.
- Keep configuration separate from runtime state.
- Use `pathlib.Path` for filesystem paths.
- Use `collections.abc` types for runtime-facing protocols such as `Callable`, `Iterable`, and `Iterator`.
- Use `np.random.default_rng`; never introduce new code based on the global NumPy RNG.
- Use explicit exceptions with actionable messages.
- Validate user-facing inputs at API boundaries. Do not repeat expensive validation inside hot loops.
- Do not catch broad exceptions in numerical core code unless the failure is re-raised with context.
- Do not silently replace an invalid solver result with a previous iterate unless that fallback is part of the documented algorithm.

Keep functions narrow. Split orchestration from numerical kernels.

Prefer names that match the mathematics already used by the project: `beta`, `rho`, `h`, `mass`, `directions`, `slopes`, `centers`, `weights`, `I`, `U`. Do not introduce synonyms for the same object only for stylistic variety.

Preserve existing public names even if an older API does not follow current naming conventions. Do not perform API renames as a drive-by cleanup.

## Comments and docstrings

The project uses concise Russian comments and docstrings. Follow that style in research-facing modules.

Comments should explain one of:

- the mathematical identity being used;
- the reason for a numerically safer formulation;
- a shape or memory constraint that is not obvious from the code;
- why an apparently simpler implementation is intentionally avoided;
- the provenance of an empirical threshold.

Do not narrate obvious Python syntax.

For important numerical functions, document:

- input meaning;
- output meaning;
- non-obvious shapes;
- mathematical invariant;
- failure conditions.

Do not put long textbook explanations into hot-path functions.

## Parallelism

Do not combine process-level parallelism with uncontrolled BLAS threading.

Experiment workers should normally pin BLAS thread counts to avoid oversubscription.

Parallelize at one dominant level.

Do not add multiprocessing inside a BLAS-heavy inner kernel without measuring it.

Do not introduce hidden global thread settings in importable library modules unless the project explicitly owns process configuration. Process-level settings belong in experiment/CLI entry points.

## Backend boundaries

Keep numerical solver code independent from plotting and pandas.

Core numerical modules should operate on arrays and small structured result objects.

Convert to pandas only in evaluation/reporting code.

Import matplotlib lazily in plotting/reporting paths.

If multiple array backends are supported, keep data on one backend throughout a numerical step. Avoid CPU/GPU ping-pong.

Backend abstractions must not force materialization of arrays that the NumPy backend can stream or factorize.

## API and architecture

Keep responsibilities separated:

- data preparation;
- bandwidth/anisotropy selection;
- weight/support construction;
- local statistic construction;
- solver;
- diagnostics;
- benchmarks/reports.

A variant class should contain formulas specific to that estimator, not generic CLI/reporting logic.

A solver should consume a clearly defined statistics/operator interface and return a structured result with diagnostics.

Avoid dictionaries whose required keys are known and stable. Prefer dataclasses/typed records for long-lived internal interfaces.

Do not expose implementation caches as public API.

Do not add dependency-heavy abstractions for one short kernel.

## Error handling

Reject invalid inputs early:

- wrong shape;
- complex or non-numeric data;
- non-finite values;
- impossible local-mass targets;
- zero direction;
- zero/invalid normalized index;
- non-finite solver output.

Distinguish:

- `TypeError`: wrong kind of object/dtype;
- `ValueError`: invalid value/shape/configuration;
- `RuntimeError`: numerical procedure could not produce a valid result;
- `NotImplementedError`: deliberately unsupported algorithmic case.

Do not convert numerical failure into a plausible-looking estimate.

## Dependencies

Prefer NumPy/SciPy primitives before adding a new dependency.

A new dependency is justified only if it provides a substantial capability that would otherwise require nontrivial, fragile code.

Do not add a framework only to replace a short `LinearOperator`, a small QR/SVD, or a few NumPy reductions.

Do not add JIT/GPU code before identifying a hot path and keeping a NumPy reference implementation.

## Anti-patterns

Do not introduce:

```python
diff = X[None, :, :] - centers[:, None, :]  # (J, n, d)
```

for large production paths.

Do not introduce:

```python
beta = np.linalg.inv(A) @ b
```

Do not introduce:

```python
normal = A.T @ A
rhs = A.T @ b
```

for a large least-squares problem only because CG expects an SPD operator.

Do not introduce `np.vectorize` for speed.

Do not repeatedly call `X @ beta` in the same step when it can be reused.

Do not call `astype(...)`, `np.asarray(...)`, or `.copy()` repeatedly inside a hot loop when conversion can happen once at the boundary.

Do not build a huge intermediate merely to reduce it immediately afterward.

Do not move pandas or matplotlib objects into the solver.

Do not hide empirical constants. Name them and state why they exist.

Do not “clean up” a mathematical formula while optimizing unrelated code.

## Project configuration

`pyproject.toml` is the source of truth for Python tooling, dependencies,
supported Python versions, linting, formatting, typing, and test configuration.

Before modifying Python code:

1. Read `pyproject.toml`.
2. Respect all configuration under:
   - `[project]`
   - `[dependency-groups]`
   - `[tool.ruff]`
   - `[tool.pyright]`
   - `[tool.pytest.ini_options]`
   - `[tool.coverage.*]`
3. Do not introduce configuration that conflicts with `pyproject.toml`.
4. Do not duplicate tool configuration in separate files unless explicitly requested.
5. Do not manually emulate lint/type/test rules. Run the configured tools.

Use the project environment and commands:

```bash
uv sync

uv run ruff format .
uv run ruff check .
uv run pyright
uv run pytest
```
