# Current Grassmann GPU port and heavy MEANMAVE comparison

Task `grassman-gpu-2026-10-22`; status **active**.

Goal: efficiently port the live combined20 solver to GPU; paired recovery
on the level of actual R MEANMAVE and comparable/lower full fit time at
n=10000,d=50,m=5. Non-goals: implicit float32, changed CPU defaults, a changed
estimator disguised as a port, broad refactoring. Previous completed plan:
agent-notes/history/plans/grassman-combined-2026-10-22-complete.md.
CPU snapshot/hashes/prior STATE: experiments/grassman_gpu_2026_10_22/.

Evidence: defaults20/relativeTrue/quadratic3/zero-centerTrue. Past m2/n500
recovery lags MEANMAVE. GPUStatistics/resident U already exist; grassman
rejects CUDA U. RTX5060/8GiB, CuPy14.1.1 works outside sandbox. Requested
shape is unmeasured. Existing dirty/untracked research work must survive.

Design: one optimizer shared by NumPy/CuPy, backend chosen from U. Keep
profiles/gradient/Jacobian resident; download final HPAOResult arrays/scalars.
Reuse GPUStatistics and ADP_Config.gpu/gpu_solver; no duplicated optimizer
or new framework. Specialize stable batched QR/SVD or augmented LS only
with reference checks and profiler evidence.
Class EXACT backend operations; NUMERICAL reductions/SVD/stable QR if used.
Changed budget is APPROXIMATE and changed N_J/N_phi/N_loc is ESTIMATOR:
explicit experimental configurations frozen before independent validation.
Invariants: same finite-sketch objective, local cutoff/ridge, relative GN
damping, true-profile Armijo, exact constants/gradient scale, live final
certificate, conservative rank-boundary status. No J*n*d or dense d*d.
U(J10000,P12,d50) is48MB; GN scratch retains workspace_bytes bound. CUDA
library workspace/pools measured separately from explicit scratch.

Read set: AGENTS/STATE, numerics/research/engineering/WORKFLOW;
agent-notes/ADP/{README,solvers,multi-index,r-sdr-comparison};
ADP/{gpu,core/ADP_Config,core/ADP_Solver,solver/grassman,solver/LSMR}.py;
engine/common/{gpu_statistics,index_fit}; tests/{test_grassman*,test_gpu_index};
experiments/r_sdr/{design,worker,worker.R,runner,protocol}. Frozen
cpu_reference.py is auditable baseline; live CPU results must be preserved.

- [x] G1 Test first: local independent lstsq/Jacobian-FD, smooth/intrinsic/
  rank-deficient/exact-zero, unequal masses/ridge, memory fallback, invalid
  inputs, public full-fit CUDA path. Observe CUDA rejection fail; port
  shared kernels/optimizer. CPU reference/focused regression and CUDA pass.
  Evidence:134 focused CPU/CUDA tests pass; nine frozen CPU cases bitwise.
  Pyright4 files clean; full public GPU fit passes. Extreme ill-conditioning
  boundary documented in MATH.md, not hidden with a changed cutoff.
- [x] G2 Isolated reproducible pilot harness: CPU/GPU combined20 and actual
  R MEANMAVE; identical data, fit/process times, CUDA warm/cold, peak RSS/
  device pools, input/source/config hashes and solver stops/certificates.
  Exploration seed90800, additive model unless user specifies another.
  Profile at exact heavy shape before optimizing kernels.
  Harness/tests exist; exact-shape pilot saved (see STATE). R baseline
  82.156s/EDR.613168; original GPU warm5.686s/EDR.719328 fails quality.
  Initial N_J300 configuration failures are retained; valid minimum500.
- [ ] G3 Bounded optimization/config selection on consumed pilot seeds only.
  Test every algebraic/numerical change; record rejections. Freeze one
  candidate/gates before development109800..109809. Validation110800..110809
  untouched until development passes; no retuning to validation.
- [ ] G4 >=10 paired heavy seeds per independent stage. Interpretation of
  on-level/comparable: geometric EDR ratio<=1.05, bootstrap95% upper<=1.10,
  worst EDR increase<=0.05; median fit ratio<=1.10, including cold separately;
  no failures/nonfinite/nonorthogonal output. Report angles, worst seeds,
  rank/stops/certificates, RSS/device memory and uncertainty separately.
- [ ] G5 Relevant CPU/CUDA checks, Ruff/format/Pyright and whole-suite status
  with unrelated failures recorded; update routes/decisions/report/STATE.
  Mark done only when required heavy evidence exists, else preserve exact
  unachieved gates and next action.

Stop/revise after twice failing the same math assumption, rank/certificate
mismatch, OOM/unbounded allocation, pairing/hash drift, or decisive frozen
development failure (leave validation untouched). A port, microbenchmark,
one seed, low objective or scheduled run does not establish the user goal.
