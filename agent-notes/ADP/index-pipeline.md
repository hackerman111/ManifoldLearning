# Общий pipeline single/multi

Этот документ описывает реальный shared fit-loop, используемый публичными single-index и multi-index классами (source locators: **SRC-IFIT-ENTRY**, **SRC-IFIT-LOOP**, **SRC-IFIT-STOP**). Он не является отдельным алгоритмом от этих моделей: различия режима передаются в один `fit_index`.

## Контракт и размеры

- Входы: `X=(n,d)`, `Y=(n,)`.
- Центры `centers=(J,d)`, выбираются из X без replacement (если J=n — все наблюдения). В опциональном протоколе можно сместить центры нормальными возмущениями; `training_set=exclude_centers` исключает исходные центр-индексы из обучающих наблюдений, но их `Y` остаются holdout-ответами для выбора шага.
- Направления random sketch: `Phi=(J,P,d)`.
- Статистики: `I=(J,P)`, `U=(J,P,d)`; `mass`, `mean`, `n_eff` имеют leading dimension J.
- Внутренний индекс: single `(d,)`, multi `(m,d)`. Публичный multi basis транспонирован и хранится `(d,m)`.

## Исполнение одного масштаба

1. `fit_index` валидирует массивы и размеры; при `gpu=True` требует CuPy до затратной инициализации.
2. `SeedSequence(seed).spawn(3)` разделяет потоки выбора центров, инициализации и направлений.
3. Выбираются центры и формируется train/test protocol; заранее вычисляется `distance2=(J,n_train)` через Gram-формулу.
4. Инициализируется индекс (local/local-cv/pilot/random), подбирается исходный `h` так, чтобы средняя сумма весов удовлетворяла `N_loc` с нижней границей `h_min`.
5. Генерируются isotropic либо localized направления. В режиме `auto` isotropic назначается только normalized multi; остальные случаи localized.
6. Блоками по центрам вычисляются kernel weights, затем CPU или GPU statistics строят `I/U`. Для estimator `new` статистики нормированы по mass, а `mass` передается решателю внешним весом. Для `legacy` `I/U` не нормируются и solver получает `mass=None`.
7. Solver пересчитывает индекс и локальные коэффициенты. Индекс канонизуется; trace содержит выбранную геометрию, ошибку на `test_Y - statistics.S`, качество, спектр и диагностику solver.
8. Outer-loop уменьшает `h` на `a`. Для следующего масштаба подбирает single `rho` или multi `alpha`; если целевая масса недостижима, завершает с `local_mass_limit`. Иные остановки: `h_min` и `outer_steps`.
9. При `select_step=best` возвращается индекс шага с минимальной holdout SSE; при `last` — последний. Это не solver loss: criterion outer selection — сумма квадратов на значениях `Y` в центрах.

`fit_index` возвращает `IndexFitResult` (начальный и конечный индекс, коэффициенты, собственные значения и metadata). Публичные классы копируют нужные поля в result/state модели (source: **SRC-IFIT-STOP**, **SRC-SI**, **SRC-MI**).

## Из чего складывается локальная статистика

Пусть `A_ji = w_ji / mass_j`, `mu_j = Σ_i A_ji X_i`, `ybar_j = Σ_i A_ji Y_i`, `q_jpi = phi_jp · (X_i - mu_j)`, `r_jp = Σ_i A_ji q_jpi`. Код центрирует X до проекций, затем исключает weighted residual mean. После этого строит скалярную связь `I_jp` с Y и векторную `U_jp`; normalized режим оставляет моменты на единицу массы, иначе домножает их обратно на mass. `n_eff=1/Σ A_ji²`; он не равен числу ненулевых весов.

Для плотных строк moments вычисляются матричными произведениями батча; при малом support CPU переключается на exact-neighbor loop. Это ветвление статистики не меняет estimator. Максимальный support threshold в live-коде — `4*k_max <= n` (source: **SRC-STAT-CPU**).

## Локаторы

| ID | Что проверять | Команда sed |
|---|---|---|
| SRC-IFIT-ENTRY | валидация, seed streams, центры, train/test protocol | `rtk proxy sed -n '187,280p' ADP/engine/common/index_fit.py` |
| SRC-IFIT-INIT | ветвление режимов инициализации | `rtk proxy sed -n '45,171p' ADP/engine/common/index_fit.py` |
| SRC-IFIT-LOOP | bandwidth, направления, веса, статистики и solver шаг | `rtk proxy sed -n '280,423p' ADP/engine/common/index_fit.py` |
| SRC-IFIT-STOP | trace, step selection и metadata | `rtk proxy sed -n '423,540p' ADP/engine/common/index_fit.py` |
| SRC-CALC-DIST-H | устойчивые расстояния и поиск bandwidth | `rtk proxy sed -n '16,109p' ADP/engine/common/calculus.py` |
| SRC-CALC-RHO-DIR | rho и single/isotropic направления | `rtk proxy sed -n '136,258p' ADP/engine/common/calculus.py` |
| SRC-CALC-ALPHA-DIR | multi alpha, exact support compaction и directions | `rtk proxy sed -n '259,422p' ADP/engine/common/calculus.py` |
| SRC-W-SINGLE | single localization weights | `rtk proxy sed -n '13,86p' ADP/engine/common/weights.py` |
| SRC-W-MULTI | multi principal/residual localization weights | `rtk proxy sed -n '88,184p' ADP/engine/common/weights.py` |
| SRC-STAT-CPU | текущие плотные и sparse-support moments CPU | `rtk proxy sed -n '29,229p' ADP/engine/common/statistic.py` |
| SRC-STAT-GPU | GPUStatistics batching, support и transfer policy | `rtk proxy sed -n '51,167p' ADP/engine/common/gpu_statistics.py` |
| SRC-INIT-LOCAL | локальные ridge gradients / weighted gradient-PCA | `rtk proxy sed -n '14,184p' ADP/engine/common/initialize.py` |
| SRC-INIT-CV | стабилизированный ridge и weighted leave-one-out выбор | `rtk proxy sed -n '185,284p' ADP/engine/common/initialize.py` |
| SRC-INIT-OTHER | pilot/random init и канонизация базиса | `rtk proxy sed -n '285,424p' ADP/engine/common/initialize.py` |
| SRC-CFG | значения конфигурации и допустимые estimator/options | `rtk proxy sed -n '10,112p' ADP/core/ADP_Config.py` |
| SRC-STAT-REC | поля результата статистики | `rtk proxy sed -n '1,37p' ADP/core/ADP_Statistic.py` |

Полный индекс каждого ID с диапазоном модуля находится в [README.md](README.md#каталог-исходников). В текущем `fit_index` профиль kernel передается как callable; для Epanechnikov `search_bandwidth` и `calculate_alpha_k` могут точно удалять уже невозможный нулевой support. См. `calculus.py`, а не считать это приближенным pruning.
