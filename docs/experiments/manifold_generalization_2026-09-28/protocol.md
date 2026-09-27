# Протокол общего manifold-эксперимента

Дата: 2026-09-28. Изменений estimator/solver нет.

## Модель и цель

X~N(0,I_8), u=XQ, Q — независимая ортогональная матрица.
z_k=u_k+c*u_(m+k)^2/2, k=1..m; f(X)=sum(z_k²)/2.
Y=(f-mean(f))/std(f)+sigma*epsilon. Истинное локальное EDR-подпространство
задаётся row(Dz(X)), а не одним gradient скалярного отклика.
Dz имеет единичный блок в первых m координатах u, поэтому ранг m всюду.
Для c=0 это обычное общее подпространство; для c>0 оно меняется по X.
Точная формула gradient — Dz.T z. QR даёт ортонормированный row-basis.
Случайный Q исключает совпадение геометрии с осями наблюдаемых признаков.

Один скалярный отклик не идентифицирует произвольное меняющееся rank-m
распределение единственным образом. Это проверка заданной структуры
генератора, а не доказательство универсальной идентифицируемости. Здесь
данные имеют полную внешнюю размерность: проверяется local EDR, не
восстановление embedded data manifold.

## Замороженная сетка

n=600,d=8,m=1/2/3,c=0/0.35/0.8,sigma=0/0.1.
Seeds 81000..81004; 512 независимых queries на каждый профиль.
SeedSequence(seed).spawn(5): X,Q,epsilon,queries,model seed.
По одному seed X/Q/epsilon/queries одинаковы для сравниваемых профилей.
В production-модели центр/directions streams независимы. Truth не
передаётся в fit и не используется для выбора параметров.

Режимы поддержки:

| Режим | N_lin | N_loc | N_manifold |
|---|---:|---:|---:|
| local | 200 | 80 | 10 |
| broad | 300 | 300 | 30 |

J=40,P=40,sync_steps=3,lambda_manifold=0.5,cg_tol=1e-6,
a=2**(1/m), h_min=3*mean(std(X))/sqrt(n), batch_size=32.
Явный estimator=manifold, solver=cg, float64, один BLAS thread.
Kernel max(1-(distance_squared/h²)²,0); остальные правила production.
Основная серия — 180 fits; порядок перемешан seed=20260928.
Бюджет 600 s проверяется между fits, поэтому один fit может выйти за бюджет.
Incomplete-run не является полной серией. Каталог вывода должен быть новым.
Manifest записывается до fits, затем JSONL/summary после каждого fit.

## Критерии

В каждом центре вычисляются все m principal angles между truth и estimate.
RMS = sqrt(mean(sin²(theta))) по всем центрам и углам.
Максимум = max(sin(theta)) по всем центрам и углам.
Center recovery: обе ошибки <=0.2 без удаления каких-либо центров.
Query recovery: те же проверки по 512 независимым точкам, estimate —
chart ближайшего центра, как в публичном transform.
Итоговый recovery требует center AND query recovery. Это отдельный
более строгий экспериментальный показатель, общий runner не меняется.

Oracle query error использует точные center bases вместо оценённых: он
показывает ошибку nearest-chart дискретизации. Finite queries не доказывают
равномерное восстановление на непрерывной/неограниченной области.
Fit exceptions остаются в знаменателе как failure; partial/initial basis
не объявляются завершённым восстановлением. Solver residual, stop reason,
trace, min mass/n_eff/graph degree — диагностика.
RSS — cumulative process ru_maxrss, не независимый per-fit peak.

## Изменения протокола и границы выводов

Два preflight smoke на seed=80000 сохранены отдельно: исходные
N_lin=40,N_loc=30 дали 18/18 ошибок linear rank; N_lin=200,N_loc=80
дали 18/18 ошибок graph/slope rank. Это не выбор по recovery-метрике.
До main seeds зафиксированы два режима local/broad; дальнейшего подбора
этих параметров не было.

Первичная main/ использует production scale_boundary=raise. После её
180/180 ошибок (149 mass feasibility, 31 rank) запланировано отдельное
парное main_stop/ с существующим production scale_boundary=stop.
Эта серия exploratory: вариант выбран после исхода основной серии;
она не является untouched validation или основанием смены defaults.
Остановка на mass boundary — явная существующая estimator option;
её нельзя смешивать с первоначальным raise-протоколом.
