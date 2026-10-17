# Structured EDR solver experiment

Implementation is opt-in. Production `grassman.solve` is unchanged.
See [PROOF.md](PROOF.md) for the mathematical contract and [REPORT.md](REPORT.md)
for Luna's measured results and their limits.

Existing dense statistics / public model callback:

```python
from functools import partial

from ADP.core.ADP_Solver import ADP_solver
from ADP.solver.grassman_research import solve

solver = ADP_solver(partial(solve, method="schur_newton"), precondition=True,
                    recycle_steps=2, max_steps=30)
# Pass solver to the existing ADP_multi_index constructor.
result = solve(B, U, I, mass=mass, method="schur_gn")
```

Explicit CX statistics, without materializing U:

```python
from ADP.solver.statistic_operator import factorized_statistics

operator, I, mass = factorized_statistics(
    X, y, weights, directions, normalized=True,
)
result = solve(B, operator, I, mass=mass, method="schur_gn")
```

With `normalized=False`, call `solve(..., mass=None)`; local mass is already
inside C and I. No automatic backend switch or persistent outer-loop cache is
installed. Rebuild C whenever the weights or directions change.

`method="metric"` applies the correctly whitened rank-limited direction.
`precondition` applies to Newton's CG only; Schur GN uses a small augmented
least-squares solve. `lambda_prox` is its GN damping and Newton's positive
preconditioner floor, with no penalty added to the profile objective.
`rank` bounds the left active dimension q; GN/Newton optimize a full q*m core.
The lower-level `SchurCore(..., A=A)` also supports inactive elimination and a
q*a chart, as checked separately. `local_ridge` is held fixed between compared
arms; it changes the statistical objective when made positive.

Reproduce the frozen suite and inspect available harness stages:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python \
  -m experiments.edr_structured_2026_10_15.bench --help
```

Failed attempts are retained in separate result directories. Only paired runs
meeting the same live full-gradient certificate support convergence-speed
comparisons. Budgeted objective and full-fit recovery are reported separately.
