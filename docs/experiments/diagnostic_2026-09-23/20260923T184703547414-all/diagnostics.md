# ADP diagnostic benchmark

Параметры сравниваются на одинаковых данных и random streams внутри сценария.
Победитель выбирается только по selection seed; validation не участвует в выборе.
Отбор: максимум восстановлений, минимум численных ошибок; разницу качества до 0.01 считаем практическим равенством. При равенстве оставляем baseline, если парное ускорение меньше 10%.
Проверка: не менее 5 validation seed, recovery >= 0.8, без численных ошибок. Новый вариант рекомендуем при лучшем recovery либо практическом выигрыше качества/времени относительно baseline.
`baseline_retained` означает, что отобранный вариант не подтвердил преимущество или сам baseline был лучшим. Рекомендация предварительная.
Время — fit wall-clock, включая ошибочные fits в новых прогонах; память — tracemalloc внутри успешного fit, не полный RSS. Число измеренных fits указано в diagnostics.csv.
Quality сравнивается только внутри одного mode/scenario.

## single / base

Рекомендация: **baseline** (`baseline_retained`); выбран на selection: `lambda_penalty=0`.

| Вариант | Split | Recovery | Wilson 95% | Fail / Nonconv / Bad quality | Quality median | Time median, s | Peak, MiB | Δ recovery | Δ quality | Time ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | selection | 6/6 | [0.61, 1] | 0 / 0 / 0 | 0.999845 (n=6) | 0.251 | 1 | 0 | 0 | 1 |
| baseline | validation | 6/6 | [0.61, 1] | 0 / 0 / 0 | 0.999679 (n=6) | 0.251 | 1 | 0 | 0 | 1 |
| N_loc=10 | selection | 6/6 | [0.61, 1] | 0 / 0 / 0 | 0.999563 (n=6) | 0.322 | 0.3 | 0 | -0.000285 | 1.21 |
| N_loc=10 | validation | 6/6 | [0.61, 1] | 0 / 0 / 0 | 0.999629 (n=6) | 0.337 | 0.3 | 0 | -0.000158 | 1.22 |
| N_loc=30 | selection | 6/6 | [0.61, 1] | 0 / 0 / 0 | 0.999855 (n=6) | 0.245 | 1 | 0 | 1.07e-05 | 0.946 |
| N_loc=30 | validation | 6/6 | [0.61, 1] | 0 / 0 / 0 | 0.999787 (n=6) | 0.237 | 1 | 0 | 8.6e-05 | 1.01 |
| N_phi=4 | selection | 5/6 | [0.436, 0.97] | 0 / 1 / 0 | 0.999760 (n=6) | 0.336 | 0.643 | -0.167 | -8.97e-05 | 1.34 |
| N_phi=4 | validation | 6/6 | [0.61, 1] | 0 / 0 / 0 | 0.999825 (n=6) | 0.369 | 0.643 | 0 | 9.53e-05 | 1.46 |
| N_phi=16 | selection | 6/6 | [0.61, 1] | 0 / 0 / 0 | 0.999832 (n=6) | 0.253 | 1.72 | 0 | 2.43e-06 | 0.961 |
| N_phi=16 | validation | 6/6 | [0.61, 1] | 0 / 0 / 0 | 0.999815 (n=6) | 0.23 | 1.72 | 0 | 7.95e-05 | 0.942 |
| N_J=48 | selection | 6/6 | [0.61, 1] | 0 / 0 / 0 | 0.999818 (n=6) | 0.271 | 1.34 | 0 | -3.36e-05 | 1.02 |
| N_J=48 | validation | 6/6 | [0.61, 1] | 0 / 0 / 0 | 0.999841 (n=6) | 0.27 | 1.34 | 0 | 0.000154 | 1.09 |
| solver_max_steps=20 | selection | 6/6 | [0.61, 1] | 0 / 0 / 0 | 0.999845 (n=6) | 0.24 | 1 | 0 | 0 | 0.952 |
| solver_max_steps=20 | validation | 6/6 | [0.61, 1] | 0 / 0 / 0 | 0.999679 (n=6) | 0.255 | 1 | 0 | 0 | 1.01 |
| solver_max_steps=80 | selection | 6/6 | [0.61, 1] | 0 / 0 / 0 | 0.999845 (n=6) | 0.243 | 1 | 0 | 0 | 0.954 |
| solver_max_steps=80 | validation | 6/6 | [0.61, 1] | 0 / 0 / 0 | 0.999679 (n=6) | 0.25 | 1 | 0 | 0 | 0.995 |
| lambda_penalty=0 | selection | 6/6 | [0.61, 1] | 0 / 0 / 0 | 0.999845 (n=6) | 0.225 | 1 | 0 | 5.35e-14 | 0.887 |
| lambda_penalty=0 | validation | 5/6 | [0.436, 0.97] | 1 / 0 / 0 | 0.999696 (n=5) | 0.206 | 1 | -0.167 | -2.47e-13 | 0.871 |
| lambda_penalty=0.2 | selection | 6/6 | [0.61, 1] | 0 / 0 / 0 | 0.999845 (n=6) | 0.24 | 1 | 0 | -1.6e-13 | 0.951 |
| lambda_penalty=0.2 | validation | 6/6 | [0.61, 1] | 0 / 0 / 0 | 0.999679 (n=6) | 0.235 | 1 | 0 | 3.05e-13 | 0.953 |

Выбранный вариант на validation: initial quality=0.997, изменение от initial=0.00316, outer iterations=5.
Остановки: `{"outer_steps": 5, "numerical_failure": 1}`; ошибки: `{"RuntimeError: unregularized HPAO step failed the trust certificate": 1}`.

Подробности: `docs/experiments/diagnostic_2026-09-23/20260923T184703547414-all/series/diagnostic-single-base`.

## multi / base

Рекомендация: **solver_max_steps=80** (`preliminary_validation_pass`); выбран на selection: `solver_max_steps=80`.

| Вариант | Split | Recovery | Wilson 95% | Fail / Nonconv / Bad quality | Quality median | Time median, s | Peak, MiB | Δ recovery | Δ quality | Time ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | selection | 5/6 | [0.436, 0.97] | 0 / 1 / 0 | 0.997279 (n=6) | 0.746 | 0.96 | 0 | 0 | 1 |
| baseline | validation | 4/6 | [0.3, 0.903] | 0 / 2 / 0 | 0.997209 (n=6) | 0.852 | 0.96 | 0 | 0 | 1 |
| N_loc=10 | selection | 0/6 | [2.78e-17, 0.39] | 0 / 6 / 0 | 0.995501 (n=6) | 1.1 | 0.96 | -0.833 | -0.00157 | 1.39 |
| N_loc=10 | validation | 1/6 | [0.0301, 0.564] | 0 / 5 / 0 | 0.996126 (n=6) | 1.26 | 0.341 | -0.5 | -0.00132 | 1.37 |
| N_loc=30 | selection | 3/6 | [0.188, 0.812] | 0 / 3 / 0 | 0.997006 (n=6) | 0.575 | 0.96 | -0.333 | -0.000873 | 0.757 |
| N_loc=30 | validation | 5/6 | [0.436, 0.97] | 0 / 1 / 0 | 0.996052 (n=6) | 0.744 | 0.96 | 0.167 | -0.00115 | 0.856 |
| N_phi=4 | selection | 0/6 | [2.78e-17, 0.39] | 0 / 6 / 0 | 0.995895 (n=6) | 1.24 | 0.601 | -0.833 | -0.0012 | 1.64 |
| N_phi=4 | validation | 1/6 | [0.0301, 0.564] | 0 / 5 / 0 | 0.993495 (n=6) | 1.25 | 0.601 | -0.5 | -0.0021 | 1.5 |
| N_phi=16 | selection | 4/6 | [0.3, 0.903] | 0 / 2 / 0 | 0.996616 (n=6) | 0.663 | 1.68 | -0.167 | -3.21e-05 | 0.859 |
| N_phi=16 | validation | 3/6 | [0.188, 0.812] | 0 / 3 / 0 | 0.997712 (n=6) | 0.745 | 1.68 | -0.167 | -5.6e-05 | 0.759 |
| N_J=48 | selection | 5/6 | [0.436, 0.97] | 0 / 1 / 0 | 0.998291 (n=6) | 0.641 | 1.28 | 0 | 0.000279 | 0.876 |
| N_J=48 | validation | 6/6 | [0.61, 1] | 0 / 0 / 0 | 0.998036 (n=6) | 0.671 | 1.28 | 0.333 | 0.000565 | 0.691 |
| solver_max_steps=20 | selection | 0/6 | [2.78e-17, 0.39] | 0 / 6 / 0 | 0.997162 (n=6) | 0.472 | 0.96 | -0.833 | -5.77e-06 | 0.615 |
| solver_max_steps=20 | validation | 0/6 | [2.78e-17, 0.39] | 0 / 6 / 0 | 0.997230 (n=6) | 0.46 | 0.96 | -0.667 | 2.1e-05 | 0.548 |
| solver_max_steps=80 | selection | 6/6 | [0.61, 1] | 0 / 0 / 0 | 0.997280 (n=6) | 0.818 | 0.96 | 0.167 | 3.74e-08 | 1.1 |
| solver_max_steps=80 | validation | 5/6 | [0.436, 0.97] | 0 / 1 / 0 | 0.997226 (n=6) | 0.986 | 0.96 | 0.167 | 0 | 1.05 |
| lambda_penalty=0 | selection | 4/6 | [0.3, 0.903] | 1 / 1 / 0 | 0.997193 (n=5) | 0.627 | 0.96 | -0.167 | 8.29e-11 | 0.827 |
| lambda_penalty=0 | validation | 3/6 | [0.188, 0.812] | 2 / 1 / 0 | 0.998262 (n=4) | 0.694 | 0.96 | -0.167 | 1.53e-11 | 0.814 |
| lambda_penalty=0.2 | selection | 5/6 | [0.436, 0.97] | 0 / 1 / 0 | 0.997279 (n=6) | 0.726 | 0.96 | 0 | -1.16e-10 | 0.963 |
| lambda_penalty=0.2 | validation | 4/6 | [0.3, 0.903] | 0 / 2 / 0 | 0.997209 (n=6) | 0.816 | 0.96 | 0 | -9.08e-11 | 0.977 |

Выбранный вариант на validation: initial quality=0.981, изменение от initial=0.0187, outer iterations=5.
Остановки: `{"local_mass_limit": 2, "outer_steps": 4}`; ошибки: `{}`.

Подробности: `docs/experiments/diagnostic_2026-09-23/20260923T184703547414-all/series/diagnostic-multi-base`.

## manifold / base

Рекомендация: **нет надёжного варианта** (`no_reliable_candidate`); выбран на selection: `N_loc=30`.

| Вариант | Split | Recovery | Wilson 95% | Fail / Nonconv / Bad quality | Quality median | Time median, s | Peak, MiB | Δ recovery | Δ quality | Time ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | selection | 0/6 | [2.78e-17, 0.39] | 2 / 0 / 4 | 0.495600 (n=4) | 0.255 | 0.657 | 0 | 0 | 1 |
| baseline | validation | 0/6 | [2.78e-17, 0.39] | 4 / 0 / 2 | 0.492202 (n=2) | 0.239 | 0.645 | 0 | 0 | 1 |
| N_loc=10 | selection | 0/6 | [2.78e-17, 0.39] | 3 / 0 / 3 | 0.532375 (n=3) | 0.234 | 0.645 | 0 | -0.0221 | 0.892 |
| N_loc=10 | validation | 0/6 | [2.78e-17, 0.39] | 5 / 0 / 1 | 0.503513 (n=1) | 0.174 | 0.643 | 0 | -0.00938 | 0.725 |
| N_loc=30 | selection | 0/6 | [2.78e-17, 0.39] | 0 / 0 / 6 | 0.507124 (n=6) | 0.268 | 0.646 | 0 | -0.00967 | 1.12 |
| N_loc=30 | validation | 0/6 | [2.78e-17, 0.39] | 2 / 0 / 4 | 0.495877 (n=4) | 0.304 | 0.646 | 0 | -0.0055 | 1.14 |
| N_phi=4 | selection | 0/6 | [2.78e-17, 0.39] | 2 / 0 / 4 | 0.511157 (n=4) | 0.245 | 0.392 | 0 | -0.0212 | 0.962 |
| N_phi=4 | validation | 0/6 | [2.78e-17, 0.39] | 4 / 0 / 2 | 0.504343 (n=2) | 0.233 | 0.388 | 0 | -0.0121 | 0.974 |
| N_phi=16 | selection | 0/6 | [2.78e-17, 0.39] | 2 / 0 / 4 | 0.478739 (n=4) | 0.247 | 0.929 | 0 | 0.0172 | 0.971 |
| N_phi=16 | validation | 0/6 | [2.78e-17, 0.39] | 3 / 0 / 3 | 0.492994 (n=3) | 0.237 | 0.925 | 0 | 0.00512 | 0.999 |
| N_manifold=3 | selection | 0/6 | [2.78e-17, 0.39] | 2 / 0 / 4 | 0.455924 (n=4) | 0.225 | 0.644 | 0 | 0.0502 | 0.874 |
| N_manifold=3 | validation | 0/6 | [2.78e-17, 0.39] | 4 / 0 / 2 | 0.425133 (n=2) | 0.227 | 0.644 | 0 | 0.0671 | 0.951 |
| N_manifold=10 | selection | 0/6 | [2.78e-17, 0.39] | 0 / 0 / 6 | 0.584602 (n=6) | 0.254 | 0.649 | 0 | -0.103 | 0.996 |
| N_manifold=10 | validation | 0/6 | [2.78e-17, 0.39] | 3 / 0 / 3 | 0.551158 (n=3) | 0.252 | 0.646 | 0 | -0.0512 | 1.05 |
| lambda_manifold=0 | selection | 0/6 | [2.78e-17, 0.39] | 1 / 0 / 5 | 0.498414 (n=5) | 0.235 | 0.645 | 0 | -0.00627 | 0.915 |
| lambda_manifold=0 | validation | 0/6 | [2.78e-17, 0.39] | 3 / 0 / 3 | 0.496483 (n=3) | 0.239 | 0.645 | 0 | -0.0199 | 0.999 |
| lambda_manifold=2 | selection | 0/6 | [2.78e-17, 0.39] | 2 / 0 / 4 | 0.488911 (n=4) | 0.234 | 0.645 | 0 | 0.00669 | 0.919 |
| lambda_manifold=2 | validation | 0/6 | [2.78e-17, 0.39] | 4 / 0 / 2 | 0.482794 (n=2) | 0.234 | 0.645 | 0 | 0.00941 | 0.978 |
| sync_steps=3 | selection | 0/6 | [2.78e-17, 0.39] | 2 / 0 / 4 | 0.498545 (n=4) | 0.331 | 0.647 | 0 | 0.00203 | 1.29 |
| sync_steps=3 | validation | 0/6 | [2.78e-17, 0.39] | 3 / 0 / 3 | 0.505882 (n=3) | 0.332 | 0.647 | 0 | -0.0105 | 1.38 |

Выбранный вариант на validation: initial quality=—, изменение от initial=—, outer iterations=4.
Остановки: `{"numerical_failure": 2, "h_min": 4}`; ошибки: `{"RuntimeError: rank-deficient local slope for target 14: rank=0, required=1; increase N_phi or N_loc": 1, "RuntimeError: rank-deficient local slope for target 9: rank=0, required=1; increase N_phi or N_loc": 1}`.

Подробности: `docs/experiments/diagnostic_2026-09-23/20260923T184703547414-all/series/diagnostic-manifold-base`.
