# Current state

Task `grassman-gpu-2026-10-22` active; PLAN.md is current record.
Goal: current combined20 GPU port; n10000,d50,m5 quality comparable to R
MEANMAVE and comparable/lower full fit time. Optional data-model question
pending; default existing R-SDR additive model.

Prior completed plan: history/plans/grassman-combined-2026-10-22-complete.md.
Prior STATE/live CPU snapshot/hashes: experiments/grassman_gpu_2026_10_22/.
Preserve existing dirty/untracked research work; no commit/push.

CPU defaults20/relativeTrue/quadratic3/constantTrue/smoothNone unchanged.
Past m2/n500 recovery lags MEANMAVE: GPU alone cannot promise improvement.
Any compute/statistical selection remains explicit experimental config.

RTX5060/8GiB, CuPy14.1.1; working CUDA outside sandbox. Sandbox reports no
device: CUDA calls need escalation. Project .venv has CuPy/pytest/psutil/
threadpoolctl; uv cache also needs escalation or /tmp cache.

Routes: grassman.py full806 lines, profile1-198/Jacobian273-309/intrinsic323-354/
solve355-760. Shared validation/normalization: LSMR340-425. GPUStatistics
holds X/Y and optional U/I on GPU; fit_index requires final HPAOResult host
arrays. R-SDR worker.run accepts explicit gpu config; main runner forbids
GPU flags. Use isolated experiment with existing workers/protocol utilities.

G1 verified:134 focused tests pass (CPU+CUDA, frozen reference, full API).
Pyright4 changed modules:0 errors/warnings. New routes: grassman1-822,
intrinsic337-368,solve370-776. API guards now allow LSMR and Grassmann.

Heavy seed90800 additive: pilot-original retained3 expected configuration
failures (N_J300<ceil(n/N_loc)500); actual R MEANMAVE EDR.61316764,
82.156s. pilot-valid uses N_J500:CPU EDR.70860093/4.407s; GPUcold
.71932762/12.688s; warm.71932762/5.686s. GPU pool164658688 bytes.
First3 objectives agree closely; outer localization amplifies tiny rounding
later. Frozen inputs preserve formulas; do not promise full-trajectory parity.
GPU warm solver2.914s vsCPU1.333, statistics.957vs1.170; current GPU
is not a CPU speedup. MATH.md documents objective/conditioning/memory.
No fresh development/validation seeds consumed.

Next: consumed-seed local-configuration exploration; pilot-loc80 in progress.
Improve recovery before freezing candidate; profile GPU kernels if justified.
