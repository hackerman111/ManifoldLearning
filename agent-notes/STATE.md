# Current agent state

Completed task: optimize the fixed-statistics truncated-SVD solver under
`PLAN.md` task `svd-hot-path-2026-09-28`. The preceding heavy SVD/HYBRID plan
is archived at
`agent-notes/history/plan-svd-vs-hybrid-heavy-30s-2026-09-28.md`.

Implemented exact hot-path changes in `ADP/solver/SVD.py`: contiguous flat U
GEMV, shifted damp solve, one final explicit certificate, cached `U @ V`,
factor objective, factor-rank reuse, small Cholesky solves, m-by-m initialization
eigensolve, and single public validation. Warm start uses an objective-preserving
augmented correction because SciPy LSMR applies damping to its initial
correction. The explicit final certificate remains authoritative.

The objective and change boundary are fixed: unhalved weighted SSE plus
`lambda_penalty * ||B-P||_F^2`, rank at most r, float64. Flat layout, shifted
`damp`, cached projected factors and algebraic factor objective are EXACT.
Adaptive Krylov tolerance and warm start are NUMERICAL and require separate
evidence. Direct/dual solves, block ALS, isotropic initialization, float32 and
GPU support are out of the active scope unless later evidence opens a new task.

Current relevant sources: `ADP/solver/SVD.py`, `tests/test_svd_solver.py`,
`SVD.tex` sections 1-8, `benchmarks/svd_vs_hybrid.py`, and
`agent-notes/ADP/multi-index.md`. The read-only Luna scout completed its
remaining-bottleneck and evidence scan; see the completed plan.

Last verified evidence: the user-shaped strict refactor took 0.6961 s and the
strict warm variant 0.6052 s versus 1.0372 s baseline. On the existing heavy
seeds 0-2, strict warm median was 40.85 s versus 47.56 s; seeds 0/1 reproduced
the baseline recovery and seed 2 moved from 0.09507 to 0.10289 after late-path
floating-point divergence. Adaptive tolerance reached 29.58 s median but moved
seed 2 to 0.12634, so its default gate failed and it remains opt-in. All final
v certificates passed.

The optimized profile remains dominated by v solves. A one-thread kernel probe
measured direct/LSMR time ratio 0.58 at the heavy `(Jp,d)=(8000,50)` geometry
and 1.10 at the user-shaped `(1200,400)` geometry. Positive-ridge d<=128 now
uses a 16 MiB bounded Cholesky path with explicit certificate and LSMR fallback.
On the heavy seeds 0/1/2, times were 24.92/25.10/25.20 s and projector
distances were 0.11831/0.08987/0.09291; all met 30 s, with zero fallbacks.
Old median was 47.56 s and distances 0.11831/0.08987/0.09507. Peak RSS rose
to 102424/102828/103312 KiB. The final d=400 strict LSMR micro median was
0.6263 s versus 1.0372 s old. Focused tests: 5 passed; Ruff passed. Pyright
was interrupted after several minutes without output; no result is claimed.

Task is done. Read `PLAN.md` for full evidence and stop conditions. Adaptive
Krylov tolerance remains opt-in because seed 2 recovery worsened. The default
HYBRID/HPAO estimator and its public behavior were not changed.
