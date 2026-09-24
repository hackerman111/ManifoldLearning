# Multi v2: исходный диагноз двух точек

Источник: `benchmark_outputs/experiments/20260924T012524585775-multiv2/`,
`multiv2-mi-d/runs.csv` и `multiv2-mi-n/runs.csv`. Это чтение сохранённых
результатов, без новых fits. В `suite.json` статус `running`: запланированы
22 серии, завершены только 7. Эти две серии полные (по 30 повторов для
каждой точки), но весь каталог нельзя считать завершённым.

| Точка | Условия | Медиана trace_score | quality pass | fit convergence | stop | HPAO convergence по 90 outer шагам |
|---|---|---:|---:|---:|---|---:|
| `multiv2-mi-d`, `d=10`, point 0 | `n=1000`, `m=2`, `N_lin=70`, `N_J=500`, `N_phi=40` | 0.997866 | 30/30 | 0/30 | `outer_steps`: 30/30 | 0/90 |
| `multiv2-mi-n`, `n_samples=1000`, point 2 | `d=100`, `m=2`, `N_lin=160`, `N_J=500`, `N_phi=40` | 0.575446 | 0/30 | 0/30 | `outer_steps`: 30/30 | 0/90 |

Обе точки: `multi_additive`, `link_scale=3`, `tau=0.4`, `sigma_x=1`,
`rho_corr=0`, `sigma_eps=0.2`, `N_loc=20`, `lambda_penalty=0.05`,
`index_init=local`, `direction_mode=isotropic`, `multi_tensor=orthogonal`,
`select_step=best`, `center_displacement=0.1`, `training_set=all`,
`redraw_directions=True`, `outer_steps=3`, `solver=lsmr`,
`solver_tol=1e-6`, `solver_max_steps=5`. `effective_config` подтверждает эти
значения на каждой точке (включая `estimator=new` и Epanechnikov); seed
инициализации меняется между повторениями. Каждый point использует seed
0–29. Для последующих парных прогонов нужны новые seed.

На всех 180 сохранённых outer шагах solver принял ровно 5 HPAO шагов и
вернул `converged=false`; `lsmr_stop=2` во всех шагах. Медиана последнего
`normal_residual_ratio` на шагах: 0.018878 у `d=10`, 0.000899 у `d=100`;
максимумы 0.093355 и 0.097186. Это свидетельство прохождения проверки
принятого linear correction (`theta=0.1`), но **не** сертификат полной
сходимости HPAO. Поэтому `outer_steps=3` действительно ограничивает fit,
однако отсутствие сходимости внутренней AO-процедуры уже при пяти шагах
не позволяет приписать причину только внешнему лимиту. У `d=10` выбран
outer шаг 2 в 28/30 повторах и 1 в 2/30; у `d=100` — шаг 2 в 28/30,
0 и 1 по одному разу. Численных ошибок в этих двух точках нет.

Сравнивать далее лимиты inner AO при фиксированных данных и `outer_steps=3`.
Только если inner HPAO сертифицируется, отдельно менять внешний лимит.
