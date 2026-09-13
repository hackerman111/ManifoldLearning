# ADP manifold learning из `manifold-ade.tex`

## Цель и границы

Добавить самодостаточный класс `ADP_Manifold` в `ADP/ADP_Manifold.py` для
оценивания семейства локальных EDR-подпространств. Все алгоритмические функции
реализуются приватными методами класса. NumPy и SciPy используются напрямую;
внешний ADP engine не является частью вычислительного пути.

Публичный API:

```python
model = ADP_Manifold(index_dim=m, seed=42).fit(X, Y)
model.centers_  # (J, d)
model.projectors_  # (J, m, d), P_j P_j.T = I_m
model.eigenvalues_  # (J, m), lambda_1 = 1
model.gradients_  # (J, d)
model.trace_  # только скалярная диагностика
coordinates = model.transform(X_new)  # (n_new, m), chart ближайшего центра
prediction = model.predict(X_new)  # (n_new,)
```

Совместимый исторический alias `ADP_manifold = ADP_Manifold` экспортируется
вместе с `ADP_Manifold`. `transform` использует локальный row-basis ближайшего
центра; координаты разных chart не считаются глобально синхронизированными.
`predict` использует ближайший local-linear pilot, градиент которого проецируется
в итоговое EDR-подпространство. Это отдельный `ESTIMATOR` для prediction: он не
меняет manifold-fit и не приписывается TeX.

Минимальный CLI запускается через `python -m ADP.cli --mode manifold`.
Воспроизводимый радиальный smoke/full-каталог доступен как
`python -m ADP.cli.experiment --experiment manifold`; он записывает локальные
principal-angle метрики и prediction RMSE.

Для базовой проверки всех однофакторных серий используется
`python -m ADP.cli.experiment --experiment manifold-basic`. Группа включает
совместную scaling-сетку и отдельные сетки по `n`, `d`, шуму, масштабу,
корреляции, `N_lin`, `N_loc`, `N_J`, `N_phi`, `N_manifold`,
`lambda_manifold` и `sync_steps`. Full-профиль использует пять общих seed на
уровнях каждого фактора; численные ошибки сохраняются в таблицах, а recovery
требует сходимости и `local_projector_distance <= 0.2`.

## Зафиксированная интерпретация TeX

Реализуется полный structure-adaptive алгоритм, а не только one-step.
Следующие неоднозначности согласованы явно:

1. Manifold bandwidth остаётся постоянным:
   `h_manifold,k = h_manifold,0`.
2. Начальный function bandwidth `h_0`, изотропные ADP-статистики и manifold
   weights вычисляются до synchronization.
3. В one-step используется финальная формула алгоритма
   `lambda_manifold * tr(B.T B (I - I_bar))` без дополнительного множителя
   `W_M,l`.
4. Ядро применяется буквально как в TeX:
   `K(q) = max(1 - q**2, 0)`, где `q` уже является квадратичной формой.
5. Внешним множителем data term служит локальная масса `c_j = sum_i w_ij`;
   `I_j` и `U_j` вычисляются с нормированными весами.

Пункты 1–4 меняют статистическую процедуру при другом выборе и поэтому не
скрываются за численной оптимизацией.

## Конфигурация

Конструктор содержит только параметры, используемые алгоритмом:

- обязательный `index_dim=m`;
- `N_loc=15`;
- `N_lin=None`, эффективное значение `max(2*d, d+2)`;
- `N_J=None`, эффективное значение
  `min(n, ceil(2*n/N_loc))` из выделенного значения `s=2` в TeX;
- `N_phi=None`, эффективное значение `N_loc`;
- `N_manifold=None`, эффективное значение
  `max(m+1, ceil(N_J/10))` из указанного в TeX начального предложения;
- `lambda_manifold=1.0`;
- `sync_steps=5`;
- `a=None`, эффективное значение `2**(1/m)`;
- `h_min=None`, эффективное значение
  `3 * mean(std(X, axis=0)) / sqrt(n)`;
- `batch_size=32`, `seed=42`, `cg_tol=1e-8`, `cg_maxiter=None`.

Скрытая ridge-регуляризация не добавляется. Локальные least-squares задачи
решаются через `np.linalg.lstsq`; потеря требуемого ранга является явным
`RuntimeError`.

## Формы и данные

Используется row-major layout:

- `X`: `(n, d)`, `Y`: `(n,)`;
- `centers`: `(J, d)`;
- `P`: `(J, m, d)`;
- `Lambda`: `(J, m)`;
- `I`: `(J, N_phi)`;
- `U`: `(J, N_phi, d)`;
- function mass: `(J,)`;
- manifold weights: CSR `(J, J)`, строки индексируют target `l`, столбцы —
  source `j`.

Не создаются `(J,n,d)`, `(J,N_phi,n,d)`, `(J,d,d)`, плотная normal matrix или
плотная постоянная `(J,J)` матрица. Временные веса имеют форму `(B,n)` либо
`(B,J)`, где `B <= batch_size`.

## Инициализация

1. Валидация `X`, `Y` и эффективной конфигурации; `n > d + 1` и
   `N_lin > d + 1` обязательны.
2. Центры выбираются без возвращения локальным `np.random.default_rng`.
3. Bandwidth search использует Gram-формулу расстояний и center chunks.
4. Для каждого центра локальный gradient определяется weighted linear
   least-squares с intercept. Нефинитная, пустая или rank-deficient система
   не маскируется.
5. Изотропные manifold weights строятся блоками и сохраняются как CSR.
6. Для каждого target `l` локальная gradient-PCA выполняется SVD матрицы
   `sqrt(c_j w_M,jl) * gradient_j`; первые `m` правых singular vectors задают
   `P_l`.
7. Вычисляются `h_0` и начальные нормированные `I/U/mass` с независимыми
   изотропными направлениями.
8. One-step повторяется `sync_steps` раз с фиксированными начальными
   statistics и manifold weights.

SVD weighted-gradient матрицы является `EXACT` low-rank формой локальной
матрицы `J_EDR`; матрица `d x d` не строится.

## One-step alternating optimization

Для каждого target `l` используются только ненулевые source-центры из строки
CSR manifold graph.

При текущем `P_l` коэффициенты `ell_jl` решают

```text
min_ell ||I_j - U_j P_l.T ell||^2.
```

Затем вычисляется действие subprojector без `d x d`:

```text
B (I - I_bar_l)
= B - sum_j normalized_w_jl * (B P_j.T) P_j.
```

Глобальная квадратичная задача для `B_l` решается SciPy CG через симметричный
`LinearOperator` размерности `m*d`. Forward и adjoint совпадают и тестируются.
Действие data Hessian вычисляется как сумма

```text
c_j w_M,jl * outer(ell_jl, U_j.T @ (U_j @ (ell_jl @ B))).
```

Penalty action равен `lambda_manifold * B (I - I_bar_l)`. Правая часть —

```text
sum_j c_j w_M,jl * outer(ell_jl, U_j.T @ I_j).
```

CG над normal operator является изменением класса `NUMERICAL`: конечная
квадратичная задача сохраняется, но обусловленность хуже, чем у augmented
LSMR. Выбор обусловлен bounded-memory требованием: augmented penalty operator
потребовал бы рабочий выход порядка `J*m*d`. Каждый CG solve обязан вернуть
finite solution, нулевой SciPy status и явный residual certificate.

После решения строится малая матрица

```text
M_l = sum_j c_j w_M,jl * ell_jl ell_jl.T       # (m, m)
```

и фактор `sqrt(M_l) @ B_l` формы `(m,d)`. Его SVD возвращает row basis `P_l`
и ненулевой спектр `Lambda_l`, нормированный на старшее значение. Это `EXACT`
эквивалентно eigendecomposition `B_l.T M_l B_l`, но не создаёт `d x d`.

Потеря ранга `m`, неортонормальность, превышение допуска CG residual или
невалидный спектр завершают шаг ошибкой. Objective записывается как
диагностика: проекция relaxed `B_l` обратно на rank-`m` подпространство не
гарантирует его монотонность.

## Structural adaptation

Для каждого scale генерируются новые независимые изотропные directions.
Function weights вычисляются без localization tensor:

```text
q_ij = (
    alpha**2 * (||X_i-x_j||**2 - ||P_j(X_i-x_j)||**2)
    + (P_j(X_i-x_j)).T Lambda_j (P_j(X_i-x_j))
) / h**2.
```

Manifold weights используют ту же формулу между центрами с постоянным
`h_manifold`. Наибольшие `alpha` и `alpha_manifold` в `[0,1]`, сохраняющие
среднюю массу не меньше `N_loc` и `N_manifold`, находятся бинарным поиском.
Если условие не выполняется даже при `alpha=0`, процедура завершается
`RuntimeError`, а не продолжает с пустыми окрестностями.

После пересчёта statistics и manifold graph выполняется one-step. Function
bandwidth уменьшается как `h <- h/a`; алгоритм заканчивается перед шагом,
который дал бы `h < h_min`. `h_manifold` не уменьшается.

## Диагностика и ошибки

`trace_` хранит по synchronization/scale только скаляры:

- phase, iteration, `h`, `h_manifold`, `alpha`, `alpha_manifold`;
- function/manifold support edges и mass quantiles;
- median/max projector change;
- objective и manifold penalty;
- суммарные CG iterations, максимальный relative residual;
- rank failures и stop reason.

Полные weights, directions, per-iteration `U` и промежуточные projectors в
trace не копируются.

Граница API отвергает неверные shapes, complex/non-numeric/non-finite данные,
невозможные neighborhood targets и неверные параметры. Численные failures
используют `RuntimeError`; deliberately unsupported prediction использует
`NotImplementedError` только если такой метод будет добавлен позже.

## Тестирование и измерение

Один файл `tests/test_manifold.py` покрывает:

1. Gram distances против явных differences.
2. Sparse/chunked weights против dense reference.
3. Low-rank penalty action против явного `I_bar`.
4. CG forward/adjoint identity и решение против малой dense system.
5. Low-rank `J_EDR` SVD против явной `d x d` eigendecomposition.
6. Orthonormality, basis/projector invariance и deterministic seed.
7. End-to-end synthetic varying-gradient smoke.
8. Rank-deficient и impossible-mass failure modes.

После реализации запускаются настроенные Ruff, Pyright, pytest,
`uv lock --check` и package import. На умеренной задаче отдельно измеряются
wall-clock и `tracemalloc` peak; результат не объявляется performance
преимуществом без сравнимого baseline.

## Файлы

Изменяются только:

- `ADP/ADP_Manifold.py` — класс и все его algorithm helpers;
- `ADP/__init__.py` — два публичных имени;
- `ADP/cli/main.py` — terminal smoke для manifold;
- `ADP/cli/experiment.py` — радиальный каталог и локальные метрики;
- `tests/test_manifold.py` — reference, invariant, smoke и failure checks.
