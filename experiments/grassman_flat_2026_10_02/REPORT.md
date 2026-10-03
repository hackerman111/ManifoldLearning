# Результаты проверки `flat_gn` — 2026-10-02

## Вывод

Независимые проверки формул и геометрии прошли. На замороженной синтетической выборке `flat_gn` существенно улучшил конечный профильный objective и норму градиента относительно `core_gn`, в том числе при `d=600`, но потребовал примерно в 2.56 раза больше времени. В полном ADP fit он оказался примерно в 1.86 раза медленнее по медиане и дал больший projector error, чем `core_gn`, в четырех из шести пар. Полный fit тем самым провалил заранее заданные критерии качества и времени. Held-out seeds не запускались; оснований рекомендовать `flat_gn` как предпочтительный метод нет. Метод остается явной экспериментальной опцией.

Опция вызывается явно как `solve(..., method="flat_gn", flat_max_steps=8)`; стандартный метод не менялся.

## Независимая численная проверка

`tests/test_grassman_flat.py` проверяет равенство плоского поворота независимой ambient-экспоненте и повторной least-squares подгонке, центральные конечные разности полного ridge Jacobian для `q=2`, периодичность проектора и профиля, масштабирование масс, ridge/gauge инвариантности и ковариантность по ортогональному повороту строкового базиса. Solver-level проверки подтверждают ненулевое одновременное обновление двух различных углов, монотонную историю и ортонормальность базиса, fallback для rank/workspace ограничений, случай `m=d` и валидацию опций. Инструментирование `_project` подтверждает, что число вызовов не зависит от `flat_max_steps` и не превышает числа принятых внешних шагов плюс два.

Команда проверки:

```text
PYTHONPATH=. uv run --no-sync pytest -q tests/test_grassman_flat.py tests/test_grassman.py tests/test_grassman_optim.py
```

Результат: 45 тестов прошли. Ruff check/format и Pyright для измененных тестов и benchmark harness прошли.

## Замороженный benchmark

Использованы tau=0 synthetic cases `m∈{2,3}`, `d∈{20,100,300,600}`, `J=120`, `p=10`, seeds `74100–74102`; методы `core_gn`, `spectral`, `rank_one`, `flat_gn`. Для каждого метода и входа был отдельный последовательный subprocess, один warmup и три измерения; BLAS/OpenMP потоки зафиксированы в 1, timeout — 120 секунд на worker. Objective и градиент сравниваются для одинакового input hash. Перед timed workers были сохранены freeze manifest и профиль одного `core_gn` вызова при `m=2,d=100,seed=74100`.

```text
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rtk proxy python -m benchmarks.grassman_flat_benchmark --stage frozen
```

Получено 96/96 успешных строк и 24/24 совпадающих наборов input hashes. Solver SHA-256 неизменен: `7c48ca081a0023dd6cff36defdaec5b45137bb112ce7a6cb727b8d15467893e3`. Для всех 24 `flat_gn` случаев были одновременные угловые шаги `q>1`; зарегистрировано 741 принятое внутреннее обновление, без rank/workspace/no-descent fallback.

Медианы 24 попарных отношений `flat_gn/core_gn`:

| Метрика | Отношение | Изменение |
|---|---:|---:|
| Время | 2.557 | в 2.56 раза дольше |
| Конечный profile objective | 0.586 | лучше на 41.4% |
| Норма градиента | 0.606 | лучше на 39.4% |

При `d=600` objective ratio составил 0.391–0.455 для `m=2` и 0.387–0.465 для `m=3`; время ratio — 1.80–1.83 и 2.07–2.12 соответственно. При `m=3,d=600,J=120,p=10` имеется лишь 1200 скалярных наблюдений sketch против 1791 параметров Grassmann-базиса и 360 локальных коэффициентов. Эти underdetermined finite-sketch cases служат диагностикой поведения solver, а не свидетельством recovery или идентифицируемости. Таким образом, замороженный objective/gradient gain gate пройден, но эти короткие solver runs сами по себе не устанавливают качество полного estimator fit.

## Полный ADP fit и gate

Применен существующий `grassman_benchmark._make_problem` и его Spokoiny `mi-spokoini-m2-dhigh` генератор: `X=2*Beta(1,tau=1)-1`, `sigma_x=1`, `sigma_eps=0.1` с Gaussian noise, `n=400`, `m=2`, `d∈{8,30}`. Использованы `N_J=120`, `N_phi=10`, `N_lin=80`, `N_loc=10`, seeds `74100–74102`; `a=1.010050167084168`, `h_min=1`, `select_step=last`, `index_init=local`, `outer_steps=None`, `lambda_penalty=0.05`. Начальная ширина `h_0` и последующий schedule получаются штатным `calculate_h0_from_adp`, затем `h` делится на `a`; все fit остановились с `h_min`. Не заданные в точке поля взяты из `ADP_Config`: Epanechnikov kernel, `direction_mode=auto`, `multi_tensor=orthogonal`, `center_displacement=0`, `training_set=all`, `redraw_directions=true`, `batch_size=32`, `estimator=new`; rho начинается с 1, далее вычисляется штатным `calculate_rho_k_from_adp`. Центры, начальное направление и данные детерминированы seed bundle и model seed исходного генератора. Профильный solver получил `max_steps=5`, `tol=1e-6`, `energy_tol=0.05`, `local_ridge=0`, `max_angle=0.5`, `workspace_bytes=16 MiB`, `flat_max_steps=8`; `local_ridge` здесь относится к профильному solver, local ridge статистик остается на default конфигурации. По три повтора не проводились: это один измеренный полный fit на метод и вход. Все четыре метода были перемешаны по порядку внутри case/seed и запускались в отдельном subprocess.

```text
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rtk proxy python -m benchmarks.grassman_flat_benchmark --stage fullfit
```

Все 24/24 строки имеют уникальный ключ (case, seed, method), завершились без worker errors, input hashes совпадают между методами для каждого case/seed, solver hash до и после совпадает. Profile histories конечны и монотонны с заранее записанным scale-aware допуском. RSS получен через `resource.getrusage(RUSAGE_SELF).ru_maxrss` в отдельном Linux worker, единица — KiB.

Уточнение live-семантики: `direction_mode=auto` при normalized multi
разрешается в `isotropic` — нормированные независимые Gaussian векторы,
равномерные на сфере, перерисовываемые на каждом outer шаге. Они не зависят
от rho/спектра. Live kernel равен `max(1-argument**2,0)`, где
`argument=(alpha**2*residual2+principal2)/h**2` уже содержит squared distances;
это точная используемая формула, без замены на `max(1-argument,0)`.
Local ridge статистик `1e-8` отличается от solver `local_ridge=0`.

Шесть попарных сравнений `flat_gn` с `core_gn`:

| Размерность | Seed | Время ratio | Разность projector error (`flat-core`) | Разность RSS, KiB |
|---:|---:|---:|---:|---:|
| 8 | 74100 | 1.853 | +0.003109 | -104 |
| 8 | 74101 | 1.219 | -0.000128 | +112 |
| 8 | 74102 | 1.488 | +0.071155 | 0 |
| 30 | 74100 | 1.872 | +0.002302 | +672 |
| 30 | 74101 | 1.858 | +0.002075 | +100 |
| 30 | 74102 | 1.922 | -0.028848 | -228 |

Медианное полное-fit время ratio равно 1.855 (предел 1.10); медианная разность projector error равна +0.002188. По размерности медианы времени/error delta: `d=8` — 1.488 / +0.003109; `d=30` — 1.872 / +0.002075. Условие `delta <= 1e-4` нарушено в четырех парах (74100 и 74102 при `d=8`; 74100 и 74101 при `d=30`). Медианная разность RSS +50 KiB проходит предел +32768 KiB. Во всех solver calls были достигнуты `max_steps`, `converged_calls=0`. В `d=8` с seeds 74101 и 74102 сработали соответственно 115 и 25 rank guards; остальные `flat_gn` fits имели ноль rank/workspace/flat fallback. Следовательно, часть последовательности действительно использовала прежний rank-one fallback.

Полный-fit harness не сохранял численную ошибку ортонормальности конечного базиса в JSONL, поэтому соответствующая часть gate для этих шести fit не сертифицирована артефактами. Ортонормальность проверялась независимыми solver-level тестами. Независимо от этого пробела, gate уже однозначно провален по projector error и median runtime; заранее установленные held-out seeds `74200–74202` не запускались.

## Артефакты и воспроизводимость

- `PROOF.md` — математическое обоснование (подготовлено отдельно).
- `freeze.json` — зафиксированные сетка, настройки, gate, условия запуска и хеши до timed runs.
- `baseline_profile.txt` — cProfile baseline `core_gn` до benchmark timing.
- `frozen_runs.jsonl`, `frozen_protocol.json` — сырые frozen результаты и протокол.
- `fullfit_runs.jsonl`, `fullfit_protocol.json` — сырые fit результаты и протокол.
- `gates.json` — итоговый аудит 120 строк, попарные метрики и проваленные
  условия selection (собран из Luna результатов, без новых solver запусков).

В замороженной конфигурации зафиксированы Python 3.14.7, NumPy 2.5.2, SciPy 1.18.1, float64, Linux и один поток для `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`; подробные BLAS сведения записаны в protocol JSON. SHA-256: solver `7c48ca081a0023dd6cff36defdaec5b45137bb112ce7a6cb727b8d15467893e3`, harness `54f9aaae316ed78c45f680e3f4fe64c63716c07068f38a378b64731adbed1aae`, full-fit generator `77e1beec7e94496c0690df4737b4ea387423695cf79dcb87db55e284bf583647`; freeze manifest `8bac03a0ffcb16a18914681fe5ff465f3b1c35b30cc61c2503f93dea42e64a39`, baseline profile `5137034e369bc7c7733a0c7cf28fc9e527c04ff2cd775eccd7046f6a523ccaef`. Суммарно проверены 120 уникальных строк (96 frozen + 24 full-fit), без ошибок и с совпадающими парными input hashes. `git_dirty=true` отражает совместную работу в рабочей копии; на протяжении обоих benchmark stages код solver и harness имел тот же записанный SHA-256. Перенастройка seed или параметров после просмотра результатов не выполнялась.
