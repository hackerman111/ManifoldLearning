# Frozen relative intrinsic-GN comparison, 2026-10-22

Frozen prototype method ADP-Grassmann-relative calls relative_gn.solve. APPROXIMATE
optimizer; same all-center finite-sketch profile, rank cutoffs, ridge=0 and
live certificate semantics. Only eligible intrinsic regular-center GN uses
mu=.05*||J||F^2/(q*m). Original smooth GN remains unchanged; bad/zero scale
skips GN to the existing spectral fallback. Full-gradient true-profile Armijo.

Budget frozen before fresh data:10 steps for nonsmooth initial profiles,
smooth cap5,tol1e-6,lambda_prox=.05,energy_tol=.05,max_angle=.5,
workspace16MiB. Other settings unchanged. Original Grassmann5, candidate10,
actual R MAVE::mave.compute MEANMAVE (MAVE-package), same input/method seeds.
d-5/d-20/d-50,n500,m2,ten seeds each,one BLAS thread,300sec/4096MiB per fit.

Development105800..105809. Unchanged validation106800..106809 only after dev
passes. Reject missing/error/duplicate/mismatched pairs; EDRgeo>.95;
any-case median EDR delta>.005; paired time median/p90>1.10/1.25;
median RSS ratio>1.10; orthogonality>1e-10; any individual EDR ratio>2.
CI10000 paired bootstrap resamples seed908123. Report R accuracy/time, stops,
certificates, source/environment/config hashes and all failures. On failed gate
close candidate and preserve validation unused. No retuning on development.

R0:12eligible/14 frozen90800 calls, initial scale344..2797; no quality theorem.
Exploratory90800 only: budgets5/10/15 give EDRgeo1.06879/.95377/1.04254,
time medians.48757/.60728/.70264 (p90 .79258/.95780/1.13895).
Ten has the best pilot quality and cost headroom, justifying independent dev;
the pilot itself does not pass or establish the frozen .95 quality gate.
64 focused checks passed, Ruff/Pyright clean before pilot; sources saved beside
pilot_relative_gn.json. No default/API/production promotion from this evidence.

Integration correction: lambda_prox is supplied by ADP_Config.lambda_penalty
(default.05) through fit_current, so runner.relative_solver must not pass it
again. First launch stopped63rows/21candidate errors before any candidate fit.
Its raw artifacts remain separate; corrected same105800 stage changes only
runner forwarding, with unchanged optimizer/production hashes and settings.

Public port after both gates: grassman.solve(relative_gn=True,max_steps=10).
Current R-SDR relative dispatch uses that function/flag; the original manifests
keep their prototype snapshots.120 full-fit replays (60default/60relative)
match saved bases and all inner diagnostics exactly. A direct public/R timing
confirmation repeats106800 only; this adds no independent datasets. The cap5
is internal in the public option; no smooth_max_steps job argument is passed.
