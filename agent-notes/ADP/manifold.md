# Manifold ADP: алгоритм локальных подпространств

## Модель и API

`ADP_Manifold(index_dim=m)` оценивает отдельный ортонормированный row-basis/projector размера `(m,d)` для каждого из `J` центров, то есть `projectors_=(J,m,d)`. Это structure-adaptive ветка: EDR-подпространство может изменяться по области X. Она не является multi-index с повторенным по центрам global basis. Источники: **SRC-MAN-API**, **SRC-MAN-FIT-ORCH**.

Публичный класс хранит конфигурацию, вызывает `engine.manifol_engine.fit`, затем публикует centers, projectors, eigenvalues, gradients, center response, bandwidths, scale factors, trace и `ADP_Manifold_result`. `transform` возвращает координаты в локальном chart ближайшего центра. `predict` применяет affine chart этого же ближайшего центра. Источники: **SRC-MAN-API**, **SRC-MAN-QUERY**.

## Эффективные параметры

При незаданных размерах live-код выводит:

- `N_lin = max(2*d, d+2)` — локальная полная linear regression;
- `N_J = min(n, ceil(2*n/N_loc))`;
- `N_phi = N_loc`;
- `N_manifold = max(m+1, ceil(N_J/10))`;
- `a = 2**(1/m)`;
- `h_min = 3*mean(std(X, axis=0))/sqrt(n)`.

Далее dimensions проверяются друг относительно друга; например требуется `n>d+1`, `N_lin` строго между `d+1` и n, а `m < N_manifold < N_J`. Solver по умолчанию `cg`, доступны `cg`/`hybrid`; `scale_boundary` принимает `raise`/`stop`. Источники: **SRC-MAN-CONFIG**, **SRC-MAN-CHECKS**, **SRC-MAN-API**.

## Fit в фазах

1. **Подготовка:** X/Y валидируются, затем единожды центрируются. `SeedSequence(seed).spawn(2)` отделяет seed центров от seed random directions. Выбираются `N_J` центров без replacement.
2. **Локальная линейная структура:** isotropic kernel bandwidth `h_lin` выбирается по целевой массе `N_lin`. В каждом центре решается weighted centered least squares через `lstsq` без ridge; требование — полный rank d. Сохраняются gradient, локальная масса и значение локальной линейной модели в центре.
3. **Начальная геометрия manifold:** по расстояниям между центрами выбирается `h_manifold` с целевой средней mass `N_manifold`; строится CSR-граф. Для каждого target локальные projectors и eigenvalues инициализируются SVD взвешенных градиентов соседних source centers.
4. **Первичная ADP-статистика:** по X и centers выбирается h для `N_loc`; случайные единичные Gaussian направления строят normalized I/U и function graph mass/effective sample size.
5. **Sync-фаза:** `sync_steps` раз обновляются projectors на фиксированном графе и стартовой ADP-статистике.
6. **Масштабный цикл:** пока `h/a >= h_min`, h уменьшается на a. По текущим локальным projectors ищется максимальный function `alpha`, пересчитываются directions/statistics; затем ищется `alpha_manifold`, обновляются graph edges и выполняется один проекторный шаг. В `scale_boundary="stop"` используется feasible mass boundary; иначе infeasible target поднимает явную ошибку.
7. **Результат:** trace содержит фазу/масштаб, bandwidths/alpha, число ребер, квантили mass и n_eff, solver итерации/остаток, objective и изменение projector.

## Локальный projector update

Для target l CSR-строка графа задает source neighbors j и веса. Сначала оцениваются slopes `s_j` как LS по `U_j P_l.T` к `I_j`; rank должен быть m. Затем обновляется фактор `B_l` `(m,d)` по data fit соседей с весами `gamma_j=mass_j*graph_weight_j` и manifold penalty, который притягивает row-space B к взвешенным source projectors. Шаг даёт новую локальную EDR-структуру:

1. строится rank-m матрица `M = sum_j gamma_j*s_j*s_j.T` размера `m x m`;
2. из M берется малый спектральный square root, умножается на B;
3. SVD только фактора `(m,d)` восстанавливает right singular row basis и normalized spectrum.

Ни один из этих этапов не требует d-by-d projector matrix. При `solver="cg"` B решается matrix-free preconditioned CG на normal operator и проверяется исходный residual. При `solver="hybrid"` используется ограниченный dense SVD или block-PCG; если block-PCG не сертифицируется, есть augmented LSMR ветка. Параметр `lambda_manifold` входит в геометрический penalty, а не в ADP statistic. Источники: **SRC-MAN-STEP**, **SRC-MAN-B-SYSTEM**, **SRC-MAN-RECOVER**, **SRC-MAN-HYBRID**.

Для `m=1` live-код применяет точное rank-one действие penalty `v−Σ_j normalized_weight_j (v·p_j)p_j` и восстанавливает projector нормированием `B` после прежних rank/finiteness checks; `m>1` сохраняет факторный SVD. Этот частный путь не меняет цель, CG или spectrum `[1]`. Источники: **SRC-MAN-PENALTY**, **SRC-MAN-RECOVER**.

## Transform / predict

Query привязывается к ближайшему center через пакетную Gram distance. Локальная coordinate равна row-projector, умноженному на смещение query-center. Прогноз — center response плюс скалярное произведение coordinate со slope `projector @ gradient`. Следовательно, transform/predict используют один chart и не интерполируют между соседними центрами.

## Локаторы

| ID | Фрагмент | Команда sed |
|---|---|---|
| SRC-MAN-API | config, fit publication, transform/predict, private hook facade | `rtk proxy sed -n '15,329p' ADP/core/manifold/ADP_Manifold.py` |
| SRC-MAN-CONFIG | effective sizes and boundary validations | `rtk proxy sed -n '49,101p' ADP/core/manifold/ADP_Manifold_utils.py` |
| SRC-MAN-FIT-ORCH | setup, init, sync/scale loops, result | `rtk proxy sed -n '37,213p' ADP/engine/manifol_engine/fit.py` |
| SRC-MAN-QUERY | nearest chart, local coordinates, prediction | `rtk proxy sed -n '214,234p' ADP/engine/manifol_engine/fit.py` |
| SRC-MAN-WEIGHTS | kernel block, bandwidth/alpha, full gradients, statistics | `rtk proxy sed -n '16,224p' ADP/engine/manifol_engine/weights.py` |
| SRC-MAN-GRAPH | CSR orientation and projectors initialization | `rtk proxy sed -n '14,67p' ADP/engine/manifol_engine/graphs.py` |
| SRC-MAN-PENALTY | exact rank-one penalty action and general low-rank path | `rtk proxy sed -n '16,35p' ADP/engine/manifol_engine/optimisation.py` |
| SRC-MAN-STEP | all-target synchronous update / solver dispatch | `rtk proxy sed -n '43,136p' ADP/engine/manifol_engine/optimisation.py` |
| SRC-MAN-B-SYSTEM | matrix-free B normal operator and preconditioned CG | `rtk proxy sed -n '158,257p' ADP/engine/manifol_engine/optimisation.py` |
| SRC-MAN-RECOVER | rank-one normalization and general m-by-m factor SVD recovery | `rtk proxy sed -n '258,319p' ADP/engine/manifol_engine/optimisation.py` |
| SRC-MAN-UTIL-ENGINE | feasible boundary, trace, chart coordinate math | `rtk proxy sed -n '15,127p' ADP/engine/manifol_engine/utils.py` |
| SRC-MAN-HYBRID | manifold dense/PCG/augmented LSMR | `rtk proxy sed -n '435,703p' ADP/solver/HYBRID.py` |
| SRC-MAN-CHECKS | rank, residual, projector and config failure conditions | `rtk proxy sed -n '125,269p' ADP/core/manifold/ADP_Manifold_utils.py` |

Полные файлы и дополнительные модули — в [README.md](README.md#каталог-исходников). Не переносите в эту ветку автоматические ожидания глобального basis из multi-index.
