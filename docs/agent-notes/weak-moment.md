# Weak-moment research routing

Completed isolated task: `benchmarks/results/verification/weak-moment-20261030/`.
Read THEORY.md for assumptions/proofs/remaining gaps, PROTOCOL.json for frozen
settings, REPORT.md and AUDIT.json for measured evidence when present.
Prototype: `benchmarks/experiments/weak_moments.py`; runner: `benchmarks/experiments/weak_moment_study.py`;
focused reference tests: `tests/test_weak_moments.py`. 576/576 fits completed;
residual quadratic validation gate passed; high-order/anisotropic limits remain.
Runner has one documented Pyright profiler argument-type diagnostic; core clean.
These paths are separate
from the historical WLS1/WLS2 and shared-polynomial studies; ADP unchanged.
