# Frozen combined/budget protocol, 2026-10-22

Freeze before development107800. User requests integration and increased
compute even for earlier joint gains below historical adoption thresholds.

Data: R-SDR full-profile d-5,d-20,d-50, n500,m2; ten datasets/case.
Development seeds107800..107809; untouched validation108800..108809.
Same generated X/y/truth, method seed and ADP settings for each arm.
One BLAS thread, isolated processes, timeout300s, memory4096MiB.
Actual MAVE::mave.compute(method="MEANMAVE"), screen=d, max.dim=m.
Kernel, bandwidth/center/statistic procedure, rank cutoff and tau0 unchanged.

Five arms,150 fits per stage:

- ADP-Grassmann: old scalar/coreGN5, relativeFalse, quadratic0, zerosFalse.
- ADP-Grassmann-relative: accepted relative10, quadratic0, zerosFalse, smooth5.
- ADP-Grassmann-combined10: relativeTrue, quadratic3, zerosTrue,10/smooth5.
- ADP-Grassmann-combined: same components,20steps, smooth capNone.
- MAVE-package: actual R MEANMAVE.

Budget20 chosen only on reused90800 pilot, before these fresh datasets.
The matched10/smooth5 comparison isolates integration; combined20/combined10
isolates the larger budget including removal of the smooth cap. No retuning
between development and validation. source/ + sources.zip and manifest.json
freeze executed source/settings/environment; inputs and per-fit metadata kept.

Report each paired ratio (EDR geometric mean and bootstrap95, time median/p90
and bootstrap95, process peak-RSS ratio), per-case medians, input hash audit,
individual EDR outlier, final orthogonality, stop/certificate counts and actual
accepted iterations. Bootstrap10000, seed908123, using analyze.py.
Compare final20 to original5, relative10 and R, plus integration/budget arms.

Mandatory safety: complete paired data, zero solver errors, finite/orthogonal
outputs (1e-10), no workspace/rank/certificate violation. Pause and investigate
an individual EDR ratio>2 vs original or a case median EDR regression>.005.
Former EDR<=.95/time<=1.10 median<=1.25p90/RSS<=1.10 gates are reported as
historical reference, not a blanket veto of the user's new compute request.
More steps/lower profile loss do not imply better EDR or convergence.

Command (change seed and output only for validation):

    uv run --no-sync python -m experiments.r_sdr --profile full --cases d-5,d-20,d-50 --runs 10 --seed 107800 --methods ADP-Grassmann,ADP-Grassmann-relative,ADP-Grassmann-combined10,ADP-Grassmann-combined,MAVE-package --grassman-steps 5 --threads 1 --keep-inputs --output benchmark_outputs/r_sdr/grassman-combined-dev-2026-10-22

Analysis: python -m experiments.grassman_accuracy_2026_10_18.analyze_combined RUN.
Public default promotion after evidence: same combined settings, max_steps20;
explicit legacy settings retained for compatibility and reproducible controls.
