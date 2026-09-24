# M3: selection-only estimator probes

All runs use frozen selection seed 0–9, the same radial data and center streams,
and the unaltered full-center RMS sine threshold 0.2. Each directory named
`<variant>_selection/` contains the manifest, every center/phase, and all
failures. This table was fixed before running validation seed 100–119.

| Explicit variant | Recovered / 10 | Numerical failures | Final RMS range on successful fits |
|---|---:|---:|---:|
| Original baseline | 0 | 6 | 0.437–0.570 |
| Per-center mass floor 6 | 0 | 0 | 0.374–0.570 |
| Keep only each center's own graph edge | 0 | 4 | 0.227–0.281 |
| Self edge + mass floor 6 | 1 | 0 | 0.196–0.281 |
| Local quadratic pilot + self edge + floor 6 | 3 | 0 | 0.167–0.259 |
| Local quadratic moment correction + self edge + floor 6 | **10** | **0** | **0.034–0.076** |
| Same correction, self edge, floor 12 | 10 | 0 | 0.033–0.075 |
| Same correction, two graph neighbors, floor 6 | 1 | 0 | 0.198–0.408 |
| Same correction, three graph neighbors, floor 6 | 1 | 0 | 0.173–0.479 |

All listed changes are **ESTIMATOR** variants. A mass floor alone removes the
rank failures but does not cure graph bias. A self-only graph alone reduces bias
but leaves unsupported local statistics. Replacing the pilot without correcting
the directional response moment cannot reliably cross 0.2. The combination is
needed on these data. With oracle initial projectors and oracle moments, the
self-only graph returns exact projectors (numerical error `<1e-8`) after the
mass floor, confirming that the local B solver and recovery do not set the
observed accuracy floor.

Selected for implementation and the single validation run:
`local_quadratic` = 60 nearest observations for a full second-order pilot,
subtract its quadratic directional moment from `I_j`, use only the source
center `j=l` in each local update, and raise per-center observation bandwidth
only when kernel mass is below 6. The floor-12 variant has no material
selection advantage; floor 6 changes fewer local weights. This estimator is
for `m=1` only, while the existing manifold estimator remains available.

The selected candidate fits the radial link's quadratic structure. Passing
this scenario will establish recovery on that design, not general manifold
recovery. A separate noiseless varying-geometry check remains in M4.
