# Анализ glm.tex и интеграция в Grassmann

Исследование завершено 2026-10-02. **Новый production-солвер сейчас не
обоснован:** большинство основных идей уже реализовано; проверенный вариант
полного матрично-свободного GN не прошёл заранее заданный отбор. Практический
следующий кандидат — предобусловленный выбор оболочки для существующего core,
после выяснения причин локальных rank guards.

## Соответствие текста текущему коду

Использованы `SVD/glm.tex`, live `ADP/solver/grassman.py` и
`ADP/solver/grassman_optim.py`. Пользовательский TeX не изменялся.

| Идея из glm.tex | Текущее состояние | Что реально остаётся |
|---|---|---|
| Профиль по g, инвариантность базиса | `_profile`, SVD/minimum-norm; ускоренные QR в optim | Сохранить rank/cutoff и mass, не подменять fixed-g целью |
| И1: rank-q вместо rank-one | `spectral`, энергетический q; default `core_gn` | Не писать ещё один такой же решатель |
| Точный rank-one Schur-профиль | `_AngleProfile`: cached шесть скаляров, ортогональное исключение | Полярная рациональная параметризация — маленькая отдельная абляция |
| Полный производный GN в малом core | `_core_jacobian`, включая T^T r и dg | Core полный только внутри Z=VK, не во всём tangent |
| Polar, истинный профиль/Armijo | `_polar` и acceptance уже есть | TR потребует отдельной модели/ratio, сама смена ретракции не обещает выигрыш |
| И2: предобусловленные плоскости | Нет: V берётся из обычного горизонтального градиента | Наиболее компактное продолжение исследования |
| И5: доводка вне оболочки | Нет; проверен H1 full GN | Без предобусловливания и обработки rank domain не внедрять |
| И4: OPG, kappa, GLS, chordal, tau>0 | Local-gradient PCA initialization уже есть, гомотопии/GLS/chordal нет | Отдельный ESTIMATOR, нужны covariance/rank/recovery проверки |
| И6: QR-сжатие/факторы | Cached U[Y,V], memory guard; exact compression уже исследовалось | При p<d+1 QR [U,I] ничего не сокращает; streaming требует нового operator API |
| И7/И8: PSD-тензор/поле | Это другой слой/модель, не frozen global solver | Не смешивать с текущим multi-index обновлением |

Статистическая цель при tau=0 совпадает с локально профилированной
finite-sketch целью HPAO. Она отличается от fixed-g rank-constrained SVD.
`lambda_prox` в Grassmann — damping GN, а не lambda*d_ch^2 из glm.tex.

## Что требует исправления в теории

Подробный аудит и вывод J/J* находятся в [PROOF.md](PROOF.md).
Наиболее существенные пункты:

- Постоянный ранг не разрешает обычное обращение A^T A: нужен полный
  столбцовый ранг либо pseudoinverse (`prop:profile`, строка 125).
- Скалярные пути polar(tZ) и Exp(tZ) совпадают по параметризации при
  rank-one. При q>1 это разные кривые: углы atan(t*s) не пропорциональны
  t*s (`prop:retr`, строка 157). Направление Ньютона и конечное обновление
  также надо различать.
- Утверждение «шар Ky Fan всегда даёт rank-one» неверно. k=1 означает
  spectral norm: для G=I_2 оптимален X=I_2, rank=2 (`rem:kyfan`, 261).
- Производная одного рационального центра без chordal penalty имеет
  числитель степени <=2, а не необходимую кубику (`prop:rational`, 304).
  Сумма центров всё равно не имеет универсального закрытого argmin.
- Геодезическая L-bound не переносится с той же L на произвольную ретракцию;
  сходимость всей траектории/конечная длина требуют дополнительных условий
  на шаги, а не одного decrease (`prop:conv`, 348).
- В `eq:Lop` (418) штраф с H требует lambda*XH и lambda*PH; написанное
  lambda*X/lambda*P относится к H=I либо согласованным whitened координатам.
- Для идентифицируемости нужны наблюдаемость U и covariance assumptions,
  а не только span(g). При U=0 GN information нулевая при любых g.
  Геометрия fixed-rank PSD также требует выбранной метрики (`prop:kappa`, 524).

Эти замечания не опровергают полезные profile/Schur/polar тождества.
Они ограничивают перенос заявленных гарантий на текущий код. Литературная
новизна и ссылки из незавершённой библиографии здесь не верифицировались;
TeX обрывается на строке 763, сборка документа не выполнялась.

## Проверенный новый вариант H1

Реализован [full_gn.py](full_gn.py), отдельный CPU research solver с
HPAOResult. Он использует существующие локальные SVD/QR-факторы, полный
горизонтальный J/J*, LSMR augmented ridge и polar/Armijo по истинному профилю.
Масса, моменты, tau и target objective сохраняются. Лимит итераций и другой
optimizer — APPROXIMATE. Нет dense d*d, явного complement или dense Jacobian
в кандидате. Память сверх входа O(J*p*m+J*m*m+d*m), но каждый Krylov-шаг
снова читает U, поэтому стоимость O(k*J*p*d*m).

Почему идея математически имеет смысл: текущий `core_gn` решает q*m
координат, q<=m, тогда как полный tangent имеет m*(d-m). Даже q=m не
делает его полным. Анизотропный Hessian может вывести H^-1 G из span(G).
Полный GN имеет не худшую квадратичную модель; выигрыш по истинному
профилю/восстановлению/времени из этого не следует.

Независимые FD, adjoint identity, envelope gradient и augmented dense ridge
прошли. Также проверены ridge=0/0.2/1e-9, неравные массы, ортонормальность,
смена базиса, монотонность, Krylov cap и rank boundary. Все 40 новых и
существующих Grassmann-тестов проходят; Ruff/targeted Pyright проходят.
Первый FD-reference в избыточных ambient координатах усиливал численный шум
нулевых gauge-направлений; reference исправлен переходом к независимым
горизонтальным координатам, без ослабления допуска. Алгебра кандидата
проверена отдельно через дифференцирование normal equation и перенос
скалярного произведения; отдельный второй рецензент не привлекался.

## Замороженное сравнение R3

Протокол: [protocol.json](protocol.json), seeds 73100–73102. Всего 30
уникальных успешных строк: 18 frozen solves и 12 полных fits, hash-парность
входа подтверждена. Один BLAS thread, float64, max_steps=5,
lambda_prox=0.05, solver tau=0; H1: maxiter=30, LSMR tol=1e-9,
normal certificate <=1e-6. Core сравнения — оптимизированная реализация.

Frozen: J=120,p=10,m=2, анизотропные колонки U с масштабом 1..10;
известный true subspace, близкая perturbed initialization и noise=0.1.
Медианы по трём seeds, время — медиана пяти повторов после warmup:

| d | core, s | full H1, s | Медиана paired time ratio | Profile ratio full/core |
|---:|---:|---:|---:|---:|
| 20 | 0.002414 | 0.028078 | 11.63 | ~1 |
| 100 | 0.003194 | 0.048658 | 15.48 | ~1 |
| 300 | 0.004819 | 0.097389 | 20.21 | ~1 |

**Все 45/45 full GN solves остановились с LSMR stop=7 и не прошли
certificate.** Solver каждый раз выполнил один core fallback. Поэтому
нет измеренного преимущества полного GN по objective, gradient или recovery;
это измерение стоимости неуспешной доводки, не принятого full GN обновления.
Максимальные normal certificates по d: 9.16e-5 / 8.23e-3 / 1.49e-2.
Во всех frozen финальных состояниях smooth=true, но converged=false.

Полные ADP fits: X Gaussian, y=sin(z1)+0.5*z2^2+noise(0,0.15), m=2;
n=300 при d=8, n=600 при d=30. N_loc=8, N_lin=N_J=100, N_phi=12,
outer_steps<=10, local initialization, last selection, batch32, h_min=1.
Код адаптивно нашёл h, затем уменьшал h; оба метода остановились по h_min
после 3/6 outer steps соответственно. Подробная конфигурация — в protocol.

| (n,d) | core/full, s | core/full projector distance | Медиана paired time ratio |
|---|---|---|---:|
| (300,8) | 0.22581 / 0.27145 | 0.05482178 / 0.05482184 | 1.209 |
| (600,30) | 0.47572 / 0.54943 | 0.08816701 / 0.08816704 | 1.220 |

Нормированная projector distance = ||(I-Qhat Qhat^T)Qtrue||_F/sqrt(m).
27 inner calls на метод; **во всех 27 есть local rank loss (2–11 центров)**,
и H1 выполнил 135 rank-guard fallbacks. Полный GN в adaptive fits не
применялся. Ни одного применимого stationarity certificate, все inner
calls max_steps/unconverged. Общая медиана paired time ratio=1.2143.
Максимальная регрессия distance=5.51e-8 — практически округление, но заранее
заданный строгий gate 1e-10 всё равно не пройден; порог не менялся.

RSS измерен Linux ru_maxrss свежих serial workers, включая native память,
импорты и вход. Максимальный peak 86.95 MiB; прирост нигде не превысил
32 MiB. Производительности/качества за пределами этих размеров не заявляем.
Frozen synthetic U не является снимком actual ADP covariance: это отдельно
обозначенный anisotropic diagnostic.

[gates.json](gates.json): complete/failure-free и RSS pass; quality, time и
advantage fail. **R4/R5 не запускались, seeds 73200–73204 не использованы.**
Это отрицательный результат именно bounded unpreconditioned H1, а не
опровержение полной GN-геометрии. Code/config hashes, native environment,
сырые diagnostics и per-pair данные сохранены в protocol/runs/pairs.

Уточнение описательной строки frozen protocol: `direction_mode="auto"`
при normalized multi в hashed `index_fit.py:303–305` означает **isotropic**,
а не localized; направления — независимый Gaussian vector с нормировкой
на единичную сферу, redraw на каждом outer step. Численная конфигурация
не менялась. Kernel K(q)=max(1-q^2,0) получает анизотропное **squared**
расстояние q=(alpha^2*orthogonal2+principal2)/h^2. Initial h ищется
удвоением+бисекцией по mean mass>=N_loc; anisotropy alpha — текущая
бисекция той же mass цели. См. [protocol_clarifications.json](protocol_clarifications.json).

## Что интегрировать следующим исследованием

1. **Разобрать rank domain существующего core.** Сейчас один unsafe центр
   блокирует smooth GN для всей задачи. Отдельно проверить mass=0: такой
   центр не влияет на objective и не должен диктовать его дифференцируемость.
   Ограничение derivative guard активными mass>0 центрами может быть EXACT,
   сохраняя shapes и minimum-norm output. Это гипотеза о возможной причине,
   а не измеренное объяснение всех 2–11 rank losses. Положительно взвешенные
   deficient центры нельзя просто проигнорировать. Добавление tau>0 — ESTIMATOR.
2. **Предобусловленные плоскости в существующем core.** Например SPD diagonal
   surrogate s_l=delta+sum_j mass_j*||g_j||^2*sum_p U_jpl^2, delta>0;
   направление H=-horizontal(diag(s)^-1 G). Тогда
   <G,H>=-sum_lk G_lk^2/s_l<0 для ненулевого horizontal G.
   Это точная гарантия descent direction, не exact profiled Hessian и не
   гарантия скорости/recovery. SVD(H) даёт новую V; остальные cached
   U[Y,V], `_core_jacobian`, damping и Armijo переиспользуются. Это маленький
   APPROXIMATE вариант, с отдельным новым планом и свежими seeds.
3. **Рациональный rank-one backend.** Eta=tan(theta), bounds
   [0,tan(max_angle)] задают ту же rank-one кривую. Оставить существующие
   denominator/cancellation/rank guards и final refit. Изменение bounded
   scalar-search координаты может изменить finite-budget результат; его
   следует проверять как отдельный backend. Преимущество времени не доказано:
   прежний Schur уже O(J), тригонометрия — всего два scalar вызова.
4. **kappa/GLS — отдельный statistical experiment.** Текущий solver получает
   только U/I/mass. Из U=S Sigma при p<d нельзя восстановить Sigma и c
   однозначно: U не содержит нужной информации для covariance powers.
   Требуется upstream local covariance/operator или weighted-design factor.
   Локальная gradient-PCA уже есть, она не означает реализованную гомотопию.
   При singular neighborhoods нужны pseudoinverse/ridge и identifiability
   условия; нельзя «улучшить solver» заменой случайного S на whitening молча.

Прямое добавление ещё одного full-GN mode сейчас не рекомендуется.
Прототип оставлен как проверяемый reference и основа нового исследования
предобусловливания; существующие `ADP/solver/*.py` не изменены.

## Воспроизведение

Из корня репозитория:

```sh
rtk proxy env UV_CACHE_DIR=/tmp/adp-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync pytest -q tests/test_glm_full_gn.py tests/test_grassman.py tests/test_grassman_optim.py
rtk proxy env UV_CACHE_DIR=/tmp/adp-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python -m experiments.glm_grassmann_2026_10_02.benchmark --worker --stage frozen --d 300 --seed 73100 --method full --output /tmp/glm-recheck.json
```

Driver без `--worker` сохраняет полный protocol до selection и отказывается
перезаписывать существующий protocol. Исходные results сохранены; для
повторной проверки доступны независимые workers с output под /tmp.
