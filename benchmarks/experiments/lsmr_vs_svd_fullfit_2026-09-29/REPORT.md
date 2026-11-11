# LSMR vs current SVD: paired full-fit comparison

## Protocol and interpretation

Each seed uses identical generated data, truth basis, and initialization. Fits ran in isolated processes with float64, one BLAS thread, and no `tracemalloc`. RSS is process high-water memory, including interpreter and import overhead. Entries use median [Q1, Q3]. Paired 95% intervals are percentile bootstrap intervals for the median (20,000 resamples); treat them as descriptive, especially for the 25-seed large case.

LSMR optimizes the current HPAO correction-penalty objective. SVD uses the fixed-g rank-constrained matrix objective (`low_rank_target=matrix`, rank `m-1`); it alternates an approximate low-rank solve and completes the remaining public basis direction from the prior. Projector recovery therefore compares end-to-end estimators, not identical inner objectives. Convergence diagnostics also differ. Small/medium fits use three outer steps; the large saved schedule runs to `h_min`.
A positive paired error delta (SVD minus LSMR) means LSMR had the lower projector error. In the diagnostics table, SVD convergence means all rank-one inner solves converged; LSMR convergence is its HPAO stopping certificate. These flags are not directly equivalent.

## Results

| Workload | Successful pairs | SVD time s | LSMR time s | Median paired time ratio SVD/LSMR (95% CI) | SVD projector error | LSMR projector error | Median paired error delta SVD-LSMR (95% CI) | SVD RSS MiB | LSMR RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Малая: n=500, d=20, m=3 (100 пар) | 100/100 | 0.0648 [0.0628, 0.0672] | 0.568 [0.522, 0.616] | 0.117 [0.11, 0.12] | 0.519 [0.464, 0.599] | 0.565 [0.521, 0.609] | -0.0457 [-0.062, -0.0136] | 77.7 [77.6, 77.7] | 74.7 [74.6, 74.8] |
| Средняя: n=900, d=60, m=3 (50 пар) | 50/50 | 0.159 [0.155, 0.162] | 1.26 [1.23, 1.33] | 0.127 [0.122, 0.131] | 0.729 [0.702, 0.756] | 0.721 [0.698, 0.748] | +0.0066 [-0.00595, 0.0214] | 88.3 [88.2, 88.4] | 83.1 [83, 83.3] |
| Большая: Spokoiny, n=800, d=50, m=2, N_J=800 (25 пар) | 25/25 | 24.8 [24.4, 25.9] | 37.1 [36.6, 38.2] | 0.66 [0.648, 0.69] | 0.118 [0.0983, 0.659] | 0.0464 [0.045, 0.052] | +0.0659 [0.0533, 0.496] | 101 [101, 102] | 99 [98.6, 100] |

## Paired outcome counts

| Workload | SVD faster | Lower projector error: SVD / LSMR | SVD lower RSS |
|---|---:|---:|---:|
| Малая: n=500, d=20, m=3 (100 пар) | 100/100 | 69/100 / 31/100 | 0/100 |
| Средняя: n=900, d=60, m=3 (50 пар) | 50/50 | 20/50 / 30/50 | 0/50 |
| Большая: Spokoiny, n=800, d=50, m=2, N_J=800 (25 пар) | 25/25 | 1/25 / 24/25 | 5/25 |

## Solver diagnostics

| Workload | SVD inner-converged fits | Median SVD inner iterations / direct solves | LSMR converged fits | Median LSMR iterations | Outer stop reasons (SVD / LSMR) |
|---|---:|---:|---:|---:|---|
| Малая: n=500, d=20, m=3 (100 пар) | 82/100 | 42.5 / 42.5 | 0/100 | 1887.0 | {'outer_steps': 100} / {'outer_steps': 100} |
| Средняя: n=900, d=60, m=3 (50 пар) | 36/50 | 51.0 / 51.0 | 0/50 | 4166.5 | {'outer_steps': 50} / {'outer_steps': 50} |
| Большая: Spokoiny, n=800, d=50, m=2, N_J=800 (25 пар) | 7/25 | 1301 / 1301 | 0/25 | 25712 | {'h_min': 25} / {'h_min': 25} |


## Failures and incomplete pairs

No worker failures or incomplete fits were recorded.
