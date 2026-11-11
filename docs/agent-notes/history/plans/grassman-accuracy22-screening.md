# Grassmann accuracy with bounded runtime

Task `grassman-accuracy-2026-10-22`.
Status: **screening stopped at failed gate; requested accuracy gain unachieved**.
No production promotion. This task is not marked done as an accuracy improvement.

Goal: improve full-ADP recovery over live combined20 with <=10% median fit-time
penalty, assessed against actual R MEANMAVE. User requests critical SVD draft
use and Luna for routine scans/benchmarks/analysis. Initial clean HEAD4b06b98.
Prior GPU plan preserved: agent-notes/history/plans/
grassman-gpu-2026-10-22-suspended.md; its heavy quality goal remains unresolved.

Invariants: float64 finite-sketch profile, local cutoffs, all-center gradient,
Armijo/final live certificate, bounded memory, API and CPU/CuPy compatibility.
No dense d*d/J*n*d. Positive local ridge is an explicit ESTIMATOR experiment;
all other candidates are NUMERICAL or APPROXIMATE, not new localization.

Evidence/routes: experiments/grassman_accuracy_2026_10_22/
{reference.py,candidates.py,MATH.md,REPORT.md,analyze.py,benchmark.py};
benchmark_outputs/r_sdr/grassman-accuracy22-{pilot,gn-pilot,ridge-pilot,krylov-pilot}/.
Reference equals live source bytewise, SHA3a0807d0d31afded0705890a2478cdd5
39de1f1f553831dcf593861ddb3acda6; git diff ADP/solver/grassman.py empty.

Read set for resumption: AGENTS/STATE/WORKFLOW, numerics/research contracts;
ADP notes solvers/multi-index/r-sdr-comparison; MATH/REPORT above; SVD/
EDR_unified_theory.tex382-444, SVD_solver.tex680-774; grassman.py relevant
profiles96-215/gradient216-240/Jacobian317-355/intrinsic372-403/solve405-810;
tests/test_grassman_accuracy22.py; existing R-SDR runner/worker/protocol.

- [x] A Critical SVD/prior-evidence routing by Luna, immutable live reference,
  old plan/state preserved. Existing scalar/core/polar ideas already present.
- [x] B25 independent candidate checks: profile FD incl ridge/m1,2,4,
  80-digit stress, conditional scaling, horizontal descent, actual final
  certificate, intrinsic zero-padding J/J* equivalence and shared U storage.
  Luna independently audited N/S/P/K; Q true-profile acceptance checked.
- [x] C80/80 full fits, all processes ok, four unique datasets90800,
  n500,m2,d5/20/50/corr.9. Nine candidates versus contemporaneous baseline,
  actual R and relative10 controls. Source/input/method-seed integrity checked.
  N: no precision/recovery gain; S/P/Q: d50 worse; tau1e-6/4/2: d50 worse;
  K10 approximately neutral; K20 slower35..51% and d50+13%. No candidate.
- [ ] D GATED, NOT RUN: freeze one candidate before development111800..809;
  validation112800..809 only after dev passes. These seeds remain untouched.
  Frozen gates: EDRgeo<=.98/bootstrap95upper<1; time median<=1.10/p90<=1.25;
  RSS<=1.10, finite/orthogonal<=1e-10; no case median EDR increase>.005,
  no individual increase>.05; no failed runs. No retuning to heldout.
- [x] E Preserve production; maintain report/math/routes/decisions/state;
  targeted checks, Ruff/Pyright and whole-suite status recorded in REPORT.
  Whole suite759passed/53CUDA-skipped/7known failures/3known collection errors.

Stop condition reached: no candidate meets the joint pilot screen. One-seed
rejection is not a universal impossibility proof. No fresh-seed consumption
or default change is justified. A continuation needs a new mathematical or
statistical mechanism, not a rerun/retune of these rejected candidates.
