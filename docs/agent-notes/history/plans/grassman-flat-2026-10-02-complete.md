# Plan: joint angles in a cached Grassmann flat

- **Task ID:** `grassman-flat-2026-10-02`; **status:** done. Selection failed;
  heldout skipped. Explicit experimental method implemented and verified.
- **Goal:** glm.tex joint q-angle flat in grassman_optim.py; GPT-6 Luna owns
  independent testing and benchmarking. Existing core_gn default retained.
- **Non-goals:** estimator change, full tangent GN, global torus minimizer,
  preconditioned planes, tuning on results, promotion after failed selection.
- **Evidence/hypothesis:** expensive U[Y,V] cache permits repeated cheap
  angle refinement. Frozen objective/gradient improved; full adaptive ADP
  time and recovery gates failed. Prior full-GN plan archived in
  agent-notes/history/plans/glm-grassmann-integration-2026-10-02-complete.md.
- **Classification/invariants:** Exp/cache/Jacobian EXACT/NUMERICAL;
  bounded optimizer APPROXIMATE opt-in. Preserve float64, mass-weighted
  min_g(||I-UYg||²+tau||g||²), minimum-norm rank, HPAOResult row basis,
  true-profile Armijo. No dense d*d/full tangent Jacobian/normal equations.
  tau retains estimator meaning; lambda_prox is GN damping.
- **Read set:** SVD/glm.tex:323-388; earlier glm PROOF; relevant contracts,
  ADP/solver/grassman_optim.py; tests/test_grassman{,_optim,_flat}.py;
  benchmarks/grassman{,_optim,_flat}_benchmark.py; pyproject.toml;
  exact ADP multi driver/weights/config ranges for benchmark metadata.
- **F0/F1 done:** preserve dirty tree/archive prior plan; derive cached Exp
  and full dg; implement method=flat_gn, flat_max_steps8, one vector of q
  angles, up to8 refinements per flat, local box and augmented GN/Armijo.
  Unsafe rank ->rank_one; workspace/no descent ->spectral; final live check.
- **F2 done (Luna):**45 focused tests (ambient/refit, nonzero-angle ridge FD,
  gauge/pi, simultaneous q2, operator call count, rank/workspace/full-space,
  validation, baseline compatibility); Ruff/format/Pyright pass.
- **F3 done (Luna):**freeze hashes/config before runs; baseline cProfile;
  one BLAS thread, serial subprocess RSS,120s cap. Synthetic frozen
  J120,p10,m2/m3,d20/100/300/600, seeds74100-74102, four methods,max_steps5,
  one warmup/3 repeats:96 rows. Full ADP n400,d8/30,m2 same seeds:24 rows.
  All120 unique/error-free/hash-paired, no code/config retuning.
- **Selection gate (frozen before runs):**finite monotone orthonormal output,
  failure-free pairs; fullfit projector delta<=+1e-4 per pair, median time
  ratio<=1.10, RSS<=baseline+32MiB; >=10% median relative advantage in frozen
  objective OR gradient OR fullfit projector quality. Frozen advantage
  passes (objective ratio0.586,gradient0.606); frozen time ratio2.557.
  Fullfit time1.8555 and4/6 projector regressions fail (max+0.071155).
  RSS passes. Fullfit orthogonality not serialized, unverified in raw rows;
  separate solver tests verify orthonormality. All inner calls max_steps,
  no convergence certificate. Distinguish frozen fit from EDR recovery.
- **F4 skipped:**seeds74200-74202 untouched after failed selection. Retain
  experimental opt-in, no claim of preferred default or recovery benefit.
- **F5 done:**PROOF/REPORT, freeze/profiles/raw protocols and gates in
  experiments/grassman_flat_2026_10_02; notes/routes/STATE/DECISIONS refreshed.
- **Stop conditions:**two conceptual derivation failures require revision;
  rank/workspace guards as above; failed selection stops heldout/tuning.
  Mathematical checks required even if performance hypothesis fails.
