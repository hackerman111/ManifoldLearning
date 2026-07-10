# ADP Stage Factories Design

## Goal

Refactor the lowercase `adp/` implementation so hypotheses about solver and
stage efficiency can be tested by replacing one algorithmic component at a
time. Preserve the existing public API and the numerical behavior of the
default random-projection ADP implementation.

## Public API

The existing construction path remains valid:

```python
model = ADP.create("new", config)
```

Named implementations can be selected through `stages`:

```python
model = ADP.create(
    "new",
    config,
    stages={
        "bandwidth_selector": "adaptive_mass",
        "statistics_builder": "random_projection",
        "beta_solver": "cg",
        "stop_rule": "convergence",
    },
)
```

Research components can be supplied without global registration:

```python
model = ADP.create(
    "new",
    config,
    stage_factories={
        "beta_solver": lambda context: ExperimentalBetaSolver(),
    },
)
```

Reusable named components can be registered in an isolated registry:

```python
registry = StageRegistry.with_defaults()
registry.register("beta_solver", "direct", DirectBetaSolverFactory())

model = ADP.create(
    "new",
    config,
    stages={"beta_solver": "direct"},
    registry=registry,
)
```

Resolution precedence is:

1. an entry in `stage_factories`;
2. a named implementation from `stages` and the supplied registry;
3. the built-in default for the `new` variant.

The built-in registry is copied for each model and is not globally mutable.

## Architecture

`ADP` remains the public model factory. A dedicated `ADPAlgorithm` owns the
outer and inner training loops and invokes replaceable stages in this order:

1. initialize beta;
2. select centers;
3. select initial bandwidth;
4. create directions;
5. prepare the current outer localization step;
6. compute local statistics;
7. solve local coefficients;
8. solve the global beta update;
9. evaluate the stop rule;
10. assemble and store the result.

The inner alternating loop remains in `ADPAlgorithm`; changing a beta solver
does not require inheriting the model, the training loop, or the statistics
implementation.

The replaceable stage categories are:

- `beta_initializer`;
- `center_selector`;
- `bandwidth_selector`;
- `direction_sampler`;
- `statistics_builder`;
- `local_solver`;
- `beta_solver`;
- `stop_rule`.

The existing `RandomProjectionADP` remains the default model for variant
`"new"`. Its protected methods become compatibility adapters that delegate to
the selected stages. There must be one implementation path rather than a new
factory path alongside the old mixin path.

## Component Contracts

`StageContext` contains immutable model-level dependencies:

- `ADPConfig`;
- the selected numerical backend;
- the model random-number generator.

`ADPState` contains mutable state for one `fit` call:

- validated `X` and `y`;
- centers;
- current beta and prior beta;
- bandwidth and anisotropy;
- directions;
- local statistics;
- local coefficients;
- training history and progress records.

Each public stage interface is a small typing `Protocol`. Methods receive only
the arguments needed by that stage. In particular, the beta solver contract is
equivalent to:

```python
class BetaSolver(Protocol):
    def solve(
        self,
        statistics: LocalStatistics,
        intercepts: np.ndarray,
        slopes: np.ndarray,
        prior: np.ndarray,
        lambda_penalty: float,
        x0: np.ndarray,
    ) -> np.ndarray: ...
```

Factories receive `StageContext` and return a component implementing the
corresponding protocol. Stage instances are created when the model is created,
not on every iteration.

The stop rule receives a phase (`"inner"` or `"outer"`) together with the
current state and most recent training step. The built-in `"convergence"`
implementation therefore preserves both existing termination points: inner
objective/beta convergence and the optional outer `anisotropy_min` condition.

## Registry and Validation

`StageRegistry` maps `(category, implementation_name)` to a factory. It exposes
registration, lookup, copying, and enumeration of available names. Duplicate
registration in the same registry is rejected unless replacement is explicit.

Model construction validates:

- all category names are supported;
- all named implementations exist;
- factories are callable;
- returned components expose the required stage method.

An unknown implementation error includes the category, requested name, and
available registered names.

## Execution Errors

Exceptions raised by custom components retain their original exception as the
cause and are wrapped in `StageExecutionError`. The error records the stage
category, implementation name, outer iteration, and inner iteration when
available.

Outputs are validated at stage boundaries. A beta solver result must have the
expected dimension, contain only finite values, and have nonzero norm. Invalid
custom output raises `StageExecutionError` before it can corrupt subsequent
iterations.

The built-in conjugate-gradient implementation preserves its current fallback
to the prior beta when SciPy reports failure or returns an invalid vector.

## Experiment Diagnostics

`ADPResult` gains these fields:

```python
stage_names: dict[str, str]
stage_timings: dict[str, float]
stage_calls: dict[str, int]
```

`stage_names` stores stable registry names, not inferred class names.
`stage_timings` stores cumulative wall-clock time for each category.
`stage_calls` stores the number of invocations. Timing wraps only the stage
call and boundary validation; progress formatting and result serialization are
not charged to a solver.

The existing `timings` field and its aggregate `statistics` and `solve` keys
remain available. `summary()` includes the resolved stage names, timings, and
call counts so experiment manifests can persist them directly.

`model.algorithm` exposes the resolved components for inspection without
requiring private attribute discovery.

## Fair Solver Comparison Example

Documentation includes a short comparison that generates one dataset and then
reuses the same `X`, `y`, `beta0`, centers, and directions for each solver. The
example reports direction quality, objective value, beta-solver time, and beta
solver call count. It does not claim statistically meaningful benchmark
results from a single run.

## Compatibility

The following existing calls retain their meaning:

```python
ADP.create("new", config)
ADP.create("new", random_state=1)
model.fit(X, y, centers=centers, beta0=beta0, directions=directions)
```

The default stage selection uses the current OLS/random beta initialization,
center sampling, quantile/mean bandwidth logic, random directions,
random-projection statistics, local coefficient update, matrix-free CG beta
update, and existing convergence conditions.

Protected methods used by current tests and research scripts remain as thin
delegating adapters during this refactor.

## Testing

Tests are added before production changes and cover:

1. default stage resolution;
2. named implementation lookup;
3. isolated custom registration;
4. direct factory precedence over a named selection;
5. actual stage invocation order;
6. replacement of only the beta solver;
7. stage names, timings, and call counts;
8. unknown category and implementation diagnostics;
9. invalid factory results and invalid beta solver output;
10. default numerical smoke behavior;
11. the existing focused and full regression suites.

Recording test components verify orchestration through real `fit` calls.
Mocks are limited to unavoidable timing/error boundaries. The final
verification includes the focused factory tests, the complete pytest suite,
Python compilation checks for changed modules, and `git diff --check`.

## Scope

This change does not add a new mathematical ADP variant, replace the numerical
backend abstraction, redesign experiment runners, introduce global plugin
discovery, or provide statistically conclusive solver benchmarks. It creates
the component boundaries, diagnostics, and example needed for those experiments.
