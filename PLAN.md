task_id: grassman-code-optimization-2026-10-01
status: active
change_class: EXACT / NUMERICAL; original optimizer/settings preserved

# Goal / non-goals
Optimize CPU code in a separate ADP/solver/grassman_optim.py; compare with
unchanged grassman.py on heavy Spokoiny. No estimator, iteration-budget,
tolerance, default method, dependency, or outer-fit changes. Prior task evidence
archived in experiments/grassman_optim_2026_10_01/PREVIOUS_*.

# Evidence / hypotheses / invariants
Baseline default core_gn5 and Spokoiny n800 d50 m2 J800 p10,162 outer steps.
Hypotheses: batched local SVD and repeated core-coordinate actions dominate;
batched contractions, stable small local QR, frozen-step caching may help.
Preserve minimum-norm/rank cutoff, ridge augmented LS, full Jacobian cross
terms, Armijo/guards/status, orthonormality, live final certificate, float64.
No dense d*d matrices or unbounded caches. Cache invalidation: per solve or
per frozen basis; workspace guard must include new intermediates.

# Exact read set
AGENTS; PLAN/STATE/WORKFLOW; numerics/research/engineering contracts;
agent-notes/ADP/{solvers,multi-index}; ADP/solver/grassman.py:1-460;
LSMR input/refit helpers as needed; tests/test_grassman.py; pyproject.toml;
benchmarks/grassman_benchmark.py; saved heavy frozen/protocol artifacts.

# Bounded units / acceptance evidence
- [x] Luna profile baseline; frozen ~28ms, local profile/gradient dominate;
  reproducible BLAS1 paired protocol established in experiment artifacts.
- [x] Implement measured hot-path optimizations in separate solver; Luna17
  independent/parity/FD/rank/ridge/layout/chunk tests pass before final GEMM
  slice; final solver frozen (574 lines); existing optimized-module checks pending.
- [x] Luna final paired heavy full fits3seeds, alternating order, fixed BLAS;
  final ratio.94194, frozen2.52x, projector<=6.9e-10; raw runtime/RSS/trace/config
  and source/input hashes retained. Earlier ratio.9315 series also retained.
- [x] Diagnose seed2 call106 transient profile difference3.436 (~10.4%):
  captures/cross-solves show common-input objective delta<=1.66e-8 and
  projector<=2.30e-8; incoming basis diff3.60e-8, U8.60%/I12.99%/mass2.19%.
  Outer sensitivity measured; exact upstream amplification mechanism unisolated.
- [ ] Lead audit numerical formulas/raw evidence, configured lint/type/shared
  checks; affected notes, STATE/DECISIONS and final report updated.

# Verification / stop conditions
Reference original remains unchanged. Require finite results, orthogonality,
small-problem objective/coefficient/Jacobian agreement at scale-justified
float64 tolerances; heavy paired output and iteration differences recorded.
Retain failures. Stop after measured hot paths and heavy comparison completed;
reject changes that sacrifice correctness/stability or regress memory without
measured reason. No unrelated repository repair.
