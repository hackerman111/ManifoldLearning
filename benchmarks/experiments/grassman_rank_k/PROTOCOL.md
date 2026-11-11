# Frozen protocol, 2026-10-16

Goal: test simultaneous rank-k updates against actual EDR recovery and whole
fit time. APPROXIMATE optimizer of the same profile; no statistics or default
change. Read MATH.md for conditional theory and overlap with existing methods.

## Solver and ranks

Production grassman.solve remains byte-for-byte behavior compatible. Separate
grassman.block_rank_k callable and exp.solve(method="block_rank_k") explicitly
select the experiment. Spectral reuses existing shared-step exponential;
optimized reuses flat GN with two joint inner steps instead of eight. Fixed
rank_k selects A; None with energy_tol=.05 selects B. C starts from energy,
probes at most two prefix ranks (half and full energy rank) once per solve,
and caps subsequent energy ranks by the pilot winner. Pilot score is actual
positive objective decrease divided by measured trial time plus proportional
cached projection time. Failed Armijo pilots do not count as improvement.
The winner is reusable on fallback; all pilot time/evaluations count. This is
a bounded heuristic and can depend on timer noise, not an optimal-rank oracle.

Common settings: float64, tau=0, damping=.05, angle cap=.5, energy_tol=.05,
workspace16MiB (extra GN scratch; excludes inputs and baseline profile/cache),
full-gradient tolerance1e-7 frozen and1e-6 fullfit, Armijo1e-4. At rank boundary
retain existing value-based rank-one guard. Memory/no-descent fallback is
truncated negative-gradient spectral step. Record final full gradient and
stationarity_applicable; a rank-boundary endpoint is not smooth convergence.

## Independent inputs and splitting

Fresh development seeds81600..81602 and fullfit validation81700..81704.
No previous metric/flat seeds reused. Inner development81600 and validation
81700 at all m in2/3/5/8 and d in20/100/300; r=min(m,d-m). Synthetic frozen
statistics are planted linear operators with local slopes and observation
noise, J64,P=max(12,2m). Supplementary cost comparison J16/256 at d100,m5.
No tuning from validation. No solver warm-start advantage between paired arms.
Hash U/I/mass/Y0; preserve all failures and reference-quality diagnostics.

Inner arms: rank_one,spectral,core_gn,flat_gn, block fixed1/full, energy,cost,
and spectral block. Max100 iterations, tolerance1e-7. One-step k sweep all
k=1..r for both variants and k sequential rank-one updates with recomputed
gradient. Save trace prefixes for equal wall, evaluation and iteration budgets;
distinguish independent certificate work from actual update cost. Independent
multi-start high-accuracy reference uses direct local lstsq and independent
Riemannian solver, includes Y0 and planted truth among starts, bounded1000
iterations/start. F_ref is a numerical upper reference, never proven global.
Exclude near-zero/negative reference gaps from exponential fitting; report
counts. Fit c on development only, then report validation inequality violations
and prediction errors; also show raw ratio versus k/r. No post-hoc c tuning.

## Full ADP fit

Use exact hard Spokoiny _make_problem and PARAMS from
benchmarks/grassman_benchmark.py (n800,d50,m2,J800,P10), original scheduling,
kernel/noise/data/init/centers/directions; no outer-budget shortening. Also
use the existing mi-5 catalog point from experiments/multi.py: d25,n250,m3,
multi_additive, link_scale3,basis_pool_dim10, inherited noise/schedule/init;
serialize its exact configuration in manifest. All nine arms on3 development
seeds in both scenarios. Select candidate by paired median fit ratio subject
to no worse median normalized EDR and no extra completed-fit failures, ties
by EDR, then arm name. If none qualifies, select the lowest median EDR arm
for a negative-result validation without claiming a passed development gate.
Freeze selection before any heldout runs. Selected candidate, core and spectral
on all5 heldout seeds, both scenarios (30 validation fits). No retuning.

Report e_EDR=sqrt(sum sin(principal_angles)^2/m), equivalent to requested
projector norm without allocating d*d. Full fit, solver wall, peak process RSS,
outer stop, inner iteration distribution, profile calls/fallbacks, final
orthogonality, completed-fit fraction and inner certificate fraction.
Recovery success is predeclared e_EDR<=.1; also report completed-fit and
certificate fractions separately. Completed fit is not numerical convergence;
h_min is a schedule stop.
Keep cases with lower inner F and worse EDR. Last adaptive F values belong to
different statistics, so compare frozen F for that diagnosis, not fullfit F
as a common objective. Additional equal-outer-budget evidence from these
unshortened paired schedules; equal-time/evaluation prefixes only frozen.

## Execution and acceptance

Serial fresh subprocess workers; all BLAS/OMP/MKL/NUMEXPR thread limits1,
verify threadpoolctl. Record Python/NumPy/SciPy/BLAS, commit/dirty state,
source SHA256, seeds, configs, hashes, timing boundary, RSS method. Wall uses
perf_counter, RSS uses Linux ru_maxrss KiB. No timed processes concurrently.
Timeout180s fullfit/60s frozen arm; independent three-start reference180s.
Failed/timeouts remain raw JSONL.
Lead reviews harness before numerical runs. No overwrites of completed stages.

Success on heldout: median paired total fit ratio<=.8 versus core; no worse
median EDR, no material failure increase, orthonormality and frozen monotonicity.
Apply same checks versus spectral. Per-scenario and pooled paired summaries
both shown. This small finite seed sample cannot establish a universal speedup
or failure probability. No automatic promotion even after a passed gate.
