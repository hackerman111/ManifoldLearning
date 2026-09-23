# Точные локальные задачи и кандидаты ускорения

Дата: 2026-09-23. Рабочий план: [`PLAN.md`](../PLAN.md). Все времена ниже относятся к CPU, одному потоку BLAS и текущему HEAD `2bf9ff809a54f39f6d87fa2a126a9c6802a6652f`. В момент M1 рабочее дерево было чистым. Исходные JSON и локальные входы лежат в [`docs/experiments/manifold_hpao_opt_2026-09-23/`](experiments/manifold_hpao_opt_2026-09-23/); команды и среда записаны в JSON.

## M1: исходная цель и проверка reference

Manifold CSR имеет строку target `l`, столбцы source `j`. Для одной строки графа код сначала решает `s_j = argmin_s ||I_j-U_j P_l^T s||²` и требует rank `m`. При фиксированных slopes он решает по `B∈R^{m×d}`

\[
F(B)=\sum_j\gamma_j\|I_j-U_jB^Ts_j\|^2+\lambda\operatorname{tr}\{B(I-E)B^T\},\quad
\gamma_j=\mathrm{mass}_j\omega_j,\quad
E=\sum_j\frac{\omega_j}{\sum_k\omega_k}P_j^TP_j.
\]

Если `D` имеет строку `(j,p)` равную `sqrt(gamma_j) (s_j⊗U_{jp})`, а `b_(j,p)=sqrt(gamma_j) I_jp`, то normal system есть `(DᵀD+λ[I_m⊗(I-E)]) vec(B)=Dᵀb`. Код применяет этот operator без плотной `(md)²` матрицы и проверяет исходный residual CG. Восстановление EDR идёт через SVD малого фактора `M^{1/2}B`, где `M=Σ_j γ_j s_j s_jᵀ`; сравнение projector нужно делать по `PᵀP`. **Отличие от `tex/manifold-ade.tex:216-232`:** рукописная relaxation умножает penalty на `Σω_j`, текущий код этого множителя не имеет. Оптимизация сохраняет текущую цель; расхождение здесь не исправляется.

HPAO при фиксированных refit coefficients `c_j` строит `A vec(δ)` со строкой `(j,p)`, равной `sqrt(mass_j) (c_j⊗U_jp)`, и `r_jp=sqrt(mass_j)(I_jp-predicted_jp)`. Для single `c_j` скаляр; для multi — строка размера `m`. Correction решает `min_δ ||Aδ-r||²+λ_prox||δ||²` посредством LSMR `damp=sqrt(λ_prox)`. Отдельный исходный certificate использует `g=Aᵀr-(AᵀA+λI)δ`; отношение `||g||/(λ||δ||)` при положительном `λ`. Gauge, trust radius и acceptance выполняются после correction.

| Серия | Форма `(n,d,J,P,m)` | Fit median, ms | Peak RSS median, MiB | Stop / solver |
| --- | --- | ---: | ---: | --- |
| multi_base | `(600,8,48,16,2)` | 69.815 | 78.28 | `outer_steps`, оба HPAO шага сошлись |
| multi_J2 | `(600,8,96,16,2)` | 98.671 | 79.46 | `outer_steps`, оба HPAO шага сошлись |
| manifold_base | `(600,6,40,12,1)` | 178.747 | 75.48 | `h_min`, CG certified |
| manifold_J2 | `(600,6,80,12,1)` | 342.202 | 76.06 | `h_min`, CG certified |

Каждая серия имеет отдельный прогрев и 10 измерений в отдельных процессах. Для `multi_base` извлечена первая реальная correction: `A` совпала с плотной матрицей до `4.44e-16`, adjoint identity до `6.94e-18`, относительная ошибка от прямого augmented LS `8.50e-11`, исходное certificate ratio `3.93e-7`. Для `manifold_base` первая реальная строка имеет 8 рёбер: normal operator и RHS совпали с плотной задачей до `3.55e-15` и `2.66e-15`; относительная ошибка B от прямого solve `7.40e-16`, residual `6.56e-16`. Входы сохранены как `reference_{multi,manifold}_base.npz`; воспроизведение — `python -m benchmarks.optimization_reference --case multi_base|manifold_base --output NEW_PATH`.

Профилирование исходной серии: `_one_step` manifold занял 99.77/196.65 мс при `J=40/80`; HPAO solver phase — см. исходные JSON. Production-код до математического gate G не меняется.

## M2: manifold кандидаты до production-правки

**Кандидат M-A, EXACT — специализированное восстановление для `m=1`.** Здесь `M=Σ_j γ_j s_j²` — скаляр. У единственной ненулевой сингулярной компоненты `sqrt(M) B` правое направление равно `B/||B||` с точностью до знака; normalized spectrum всегда `[1]`. Достаточно прежних проверок `M>tolerance`, `singular=sqrt(M)||B||>rank_cutoff`, конечности и ортонормированности, затем `orient_rows(B/||B||)`. Для `m>1` остается старый факторный SVD. Work и storage у `m=1` — `O(K+d)`/`O(d)`; исчезают eigen/SVD и их временные массивы. На реальной строке `(K,P,d,m)=(8,12,6,1)` projector distance `1.18e-16`, старое/новое время 91.43/21.64 µs на 1000 вызовах (один процесс, прогрев 50). При 120 target-solves это даёт верхнюю оценку экономии около 8.4 ms на base fit; полный fit надо измерить отдельно.

**Кандидат M-B, EXACT — прямое rank-one действие penalty при `m=1`.** Если `p_j` — единственная строка source projector и `v=B[0]`, то `B(I-E)=v-Σ_j(ω_j/Σω)(v·p_j)p_j`. Реализация через один `(K,d) @ (d,)`, взвешивание `(K,)`, затем `(K,) @ (K,d)` использует вид исходного `(K,1,d)` массива и `O(K+d)` временных элементов; нет `d×d` либо `K×d` нового массива. На захваченной строке максимальная ошибка действия 0, время 2.43/1.91 µs на 1000 вызовах. Выигрыш на одном действии мал, поэтому принятие зависит от полного шага; этот кандидат можно применять вместе с M-A только если общий benchmark подтвердит выигрыш.

Отклоненный в качестве основного пути вариант — плотное `(md)²` normal solve: на текущем `m=1,d=6` он мал, но масштабируется по памяти как `O(m²d²)` и вступает в конфликт с установленным бюджетом для больших `d`. Он остается только auditable reference в `benchmarks/optimization_reference.py`. Формулы M-A/M-B не меняют `γ`, `E`, λ, CG tolerance, порядок target и локальные slopes. Малые прямые проверки и microtiming воспроизводятся `python -m benchmarks.optimization_candidates --manifold reference_manifold_base.npz --hpao reference_multi_base.npz` из указанного каталога артефактов.

## M3: HPAO кандидаты до production-правки

**Кандидат H-A, EXACT — mass-scaled coefficients.** Обозначим `a_j=√mass_j c_j`. Тогда `Aδ` по строке `(j,p)` равно `U_jp(a_jᵀδ)`, а `Aᵀy=Σ_j a_j(U_jᵀy_j)`. Это та же матрица A, поскольку скаляр `√mass_j` коммутирует с локальным произведением. Предварительно хранится только `a=(J,m)` (`(J,)` для single), то есть `8Jm` байт; каждое действие больше не создает дополнительное масштабированное `(J,P)` полотно. Цена основных contractions остается `O(JPd+Jmd)` на действие. На сохранённой реальной correction `(J,P,d,m)=(48,16,8,2)` максимальные ошибки forward/adjoint `8.88e-16`/`5.33e-15`, adjoint identity `1.42e-14`; при том же damp и допуске LSMR прошёл 16 итераций в обоих вариантах. Разность corrections `1.31e-11`, исходное certificate ratio нового варианта `4.52e-7`, прямой augmented reference исходного solve — в M1. Три независимых micro-прогона показали 513/471, 550/505 и 586/502 µs на solve (старый/новый); это локальная оценка, полный fit ещё нужен.

**Кандидат H-B, NUMERICAL — точная замена координат для ridge.** Для `λ>0` положим `S=diag(diag(AᵀA)+λI)^{-1/2}` и `δ=Sz`. Тогда исходная задача *точно* равна `min_z ||[AS;√λ S]z-[r;0]||²`, после решения нужен обратный переход `δ=Sz` и прежний normal certificate в исходных координатах. Цена диагонали `O(JPd+Jmd)` и `O(md)` памяти; одно Krylov действие требует старые A/A* плюс `O(md)` масштабирование и augmented vector длиной `JP+md`. Для `λ=0` нулевые элементы диагонали и minimum-norm поведение требуют отдельного режима; поэтому кандидат рассматривался только при положительном λ. На сохранённой correction число итераций уменьшилось 16→12, но исходное certificate ratio `2.39e-7`, а время трёх прогонов weighted/unscaled против weighted/scaled составило 471/579, 505/546 и 502/664 µs. На этом режиме цена augmented действий превысила экономию итераций; кандидат отклонён для текущего default.

Дополнительное EXACT сокращение: в исходном certificate `Aᵀr` вычисляется дважды; его можно сохранить один раз в пределах correction, не меняя формулу. Оно устраняет ровно один adjoint на пробу, поэтому включается в H-A только после проверки полного fit. Альтернатива dense SVD уже существует в явно выбранном HYBRID и ограничена memory budget; в default HPAO/LSMR она не переносится без отдельного выигрыша и проверки всех режимов. Результаты micro-прогонов сохранены как `candidates_micro*.json` рядом с захваченными входами.

## C1: manifold implementation и парные измерения

В `optimisation.py` сохранён старый `m>1` путь; для `m=1` реализованы M-A/M-B. На захваченном реальном target старое/новое действие penalty совпало точно, расстояние проекторов `1.18e-16`. Проверка нулевого `B` и нулевых slopes сохраняет явный отказ по rank. `tests/test_manifold.py`: 12 passed; `tests/test_hybrid.py`: 14 passed; Ruff check/format для затронутых файлов прошли.

| Manifold | Fit median old→C1, ms | `_one_step` median old→C1, ms | Peak RSS old→C1, MiB | max projector distance | CG iterations C1 |
| --- | ---: | ---: | ---: | ---: | --- |
| `J=40` | 178.747→167.827 | 99.77→88.00 | 75.48→75.47 | `1.12e-15` | 240+239+240 |
| `J=80` | 342.202→317.379 | 196.65→172.52 | 76.06→76.14 | `3.22e-15` | 478+473+480 |

У всех 20 новых измерений `stop_reason=h_min`, ошибок нет. Максимальный исходный CG residual по trace: `6.52e-9`/`9.86e-9`; максимальная разница objective по трём шагам `5.33e-15`/`2.66e-15`. JSON: `c1_manifold_{base,J2}.json`. Один поток BLAS и по 10 отдельных процессов, как в M1. Измеренный выигрыш полного fit 6.1%/7.3%; RSS менялся меньше 0.1 MiB между сериями.

## C2: HPAO implementation и парные измерения

`LSMR._linear_operator` один раз вычисляет `sqrt(mass)*coefficients`; forward/adjoint используют тот же `A`, что сохранён в `benchmarks/optimization_candidates.py::hpao_operator_reference`. Внутри одной correction `Aᵀr` теперь берётся один раз для исходного normal certificate и его знаменателя. Ridge penalty, damp, LSMR tolerances, trust/gauge/acceptance и публичные поля оставлены прежними. На сохранённой реальной задаче HPAO forward/adjoint отличаются от reference не более чем на `8.88e-16`/`5.33e-15`, adjoint identity `1.42e-14`; новая correction имеет исходный ratio `4.52e-7`. Малый CPU direct ridge для single/multi при `λ=0,0.2` и неравных массах проверен в `tests/test_lsmr.py`.

| HPAO fit | Fit median reference→C2, ms | Peak RSS, MiB | Subspace distance | LSMR iterations old→new | Stop/certificate |
| --- | ---: | ---: | ---: | --- | --- |
| single / `J=48` | 46.316→42.676 | 78.32→78.29 | `5.12e-17` | 96+69→96+69 | оба converged; max ratio `0.0672` |
| multi / `J=48` | 69.815→67.116 | 78.28→78.34 | `2.02e-9` | 256+568→256+562 | оба converged; max ratio `0.0631` |
| multi / `J=96` | 98.671→94.096 | 79.46→79.42 | `3.17e-10` | 218+578→218+574 | оба converged; max ratio `0.0536` |

Для multi subspace distance — Frobenius норма разности проекторов. Разница final outer `err` не выше `1.78e-15`; для single final loss отличается не более чем на `2.78e-16`. Все 30 новых fit runs завершились `outer_steps` без ошибок. Multi baseline и C2 использовали по 10 отдельных процессов; single reference/current запускались на одном кодовом checkout с monkeypatch только старого operator и одинаковым профилем без `tracemalloc`, по 10 процессов. JSON: `c2_multi_{base,J2}.json`, `single_{reference,current}_converged.json`. Первоначальный `single_reference.json` содержит только ошибки сериализации callable `kernel` в benchmark harness и не входит в сравнение. GPU тесты CPU/API набора пропущены средой (`cudaErrorNoDevice`), поэтому устройство и GPU timing не проверены.

## V: совместная проверка на одном checkout

`benchmarks.fit_bottlenecks --variant reference|current` подставляет сохранённые прежние функции только в benchmark-процессе. Для каждого варианта и формы выполнены прогрев и 10 отдельных процессов. Все 18 серий (`v_*` и `v2_multi_*`, 180 измерений) имеют один и тот же SHA-256 набор для всех `ADP/**/*.py` и benchmark runner до/после серии, не содержат ошибок, сохраняют `h_min` у manifold и `outer_steps` у multi. Ниже время полного `fit` в ms: медиана `[10%,90%]`; peak RSS — медиана абсолютного `ru_maxrss` процесса в MiB.

| Форма `(n,d,J,P,m)` | Reference, ms | Current, ms | Снижение времени | Peak RSS ref→current | Projector distance |
| --- | ---: | ---: | ---: | ---: | ---: |
| manifold `(600,6,40,12,1)` | 181.934 `[179.525,185.992]` | 168.377 `[165.217,174.554]` | 7.45% | 75.65→75.38 | `1.11e-15` |
| manifold `(600,6,80,12,1)` | 341.965 `[338.934,356.932]` | 319.807 `[313.346,324.555]` | 6.48% | 76.26→76.11 | `3.22e-15` |
| manifold `(300,4,20,8,1)` | 52.603 `[50.438,53.584]` | 47.917 `[46.537,50.216]` | 8.91% | 73.22→73.15 | `9.01e-16` |
| multi `(600,8,48,16,2)` | 70.895 `[69.279,71.745]` | 68.890 `[66.854,72.412]` | 2.83% | 78.42→78.33 | `2.02e-9` |
| multi `(600,8,96,16,2)` | 98.494 `[96.561,102.634]` | 93.209 `[92.640,96.745]` | 5.37% | 79.46→79.42 | `3.17e-10` |
| multi `(300,6,24,8,2)` | 44.723 `[43.123,49.270]` | 43.462 `[42.494,44.530]` | 2.82% | 73.71→73.70 | `2.17e-8` |

Порядок первых multi серий был reference→current. При обратном порядке current→reference медианы reference→current: 70.054→67.247 ms (`J=48`), 98.721→93.640 ms (`J=96`), 44.429→43.922 ms (control). На control разница 1.14% во второй серии и лежит внутри разброса; устойчивого выигрыша для этого малого размера не утверждаем. Для основных `J=48/96` обе серии дают положительный эффект. В manifold медиана `_one_step` 102.82→88.45, 198.86→174.22 и 32.56→28.38 ms соответственно. Все trace objective manifold различаются не более чем на `5.33e-15`; максимальный CG residual `9.86e-9`. Для multi внешний `err` различается не более чем на `1.78e-15`, все HPAO шаги сошлись и прошли исходный normal certificate `ratio≤0.1`. В этих режимах заметного роста RSS нет.

Проверки текущего дерева: `pytest -q -p no:cacheprovider` — 305 passed, 28 GPU skipped (`cudaErrorNoDevice`); Ruff check и format для затронутых файлов прошли; Pyright — 0 errors, одно существующее предупреждение в `legacy_lsmr.py` об unnecessary ignore; `git diff --check` чист. Runtime GPU не измерен. Benchmark reference-код находится в `benchmarks/optimization_candidates.py` (SHA-256 `18618d5225cd439d550d490730da49de70fcfc7970b1619eae2b3f7a1f918aee`); runner `fit_bottlenecks.py` — `9e3cb63286032038db726f58946d886f508cb3dcf0deeee21dc17172e7318802`. Малые reference и single runner — `268ca13fadd1e755fabe4a08e2f5744b4fc3e79de03efe404dc85a3bcb1744da` и `e0871cae4ab8591418aa7e2d076d11a8d5647ca235f0f7415b8c883d06f4acab` соответственно. Эти хэши дополняют `source_sha256_*` в JSON для незакоммиченных benchmark-модулей.
