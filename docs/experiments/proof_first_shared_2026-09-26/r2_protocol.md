# R2: reference, ограничения и замороженный paired-протокол H1

Дата: 2026-09-26. Production `ADP/` не изменялся. Изолированный
экспериментальный код: `experiments/proof_first_h1.py`, SHA-256
`f43e658305f8b8f9ee881abff424625080d5ab23c23388d2abf7c2829ab63666`.
Fingerprint кода `ADP/`, `experiments/`, `pyproject.toml`:
`97f76bb8d9b7a5dc383638882ddee7a1ced86ac8fa8f11ea88d09e8ea75c408a`.
Дальнейшая правка этих `.py` до R3/R4 запрещена; validation сверяет fingerprint.

## Проверка R2

- `tests/test_proof_first_h1.py`: Haar-ортогональность, равенство Gram
  прямого `P=2d+r` скетча и compact-фактора, `I=Φc`, `U=ΦC` при разной
  массе центров, общая взвешенная квадратичная цель для нескольких targets,
  guard 256 MiB на массив направлений и вырожденная self-only окрестность.
  Последняя всё ещё даёт `rank-deficient local slope`; H1 не маскирует её.
- 5 тестов прошли. Ruff check/format и Pyright для прототипа/теста прошли.
  `git diff --check` прошёл. Численная эквивалентность compact и полного
  скетча установлена только для квадратичной цели; float64 полный fit
  остаётся предметом R3.
- Стресс **только генератора**, seed 17001, один BLAS-поток:
  `(n=10000,d=1000,J=200,P=40)` дал `(200,40,1000)`, 64,000,000 bytes,
  0.373 с, process RSS 106.1 МиБ, finite=True. Полный fit при этом размере
  не запускался. При `P=d=1000` guard отклоняет 1.6 GB тензор;
  это не область кандидата.

## Коррекция baseline и диагностические seed

R0 CLI-строка manifold `0.356 с / RMS sine 0.0455` использовала
`local_quadratic` default, а не исходный `manifold`: это отдельный estimator,
не baseline H1. Оба manifold-сценария R3 явно создают
`ADP_Manifold(..., estimator="manifold", scale_boundary="stop")`.
`stop` — одинаковая настройка baseline/кандидата; completion по допустимой
границе массы не означает outer stationarity.

На **диагностических** seed 60001–60003 до кандидата исходный m=1 при
`N_loc=20` дал rank-0 slope failure; с зафиксированными `N_loc=40,
N_lin=80,N_manifold=8` все 3 fit завершились, RMS sine
`0.4684/0.5018/0.5357`, ни один не recovered. Для m=2 увеличение
`N_lin` до 120 исправило один local-linear failure, но при `N_loc=60`
seed 60003 всё ещё дал rank-0 slope failure; seed 60001/60002 завершились
с RMS sine `0.4691/0.4887`. Повторная rank-проблема фиксируется как
граница исходного метода: конфигурацию далее не подбирать, m=2 сохранить
в selection с failure accounting.

## Схема до full fits

Все fit: CPU float64, `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`,
отдельный дочерний процесс на вариант/seed для `ru_maxrss`.
`fit_seconds` охватывает generation/patch/fit/metric без создания данных;
данные создаются одинаковым seed-потоком до таймера. Порядок вариантов
чередуется по чётности seed. Только изотропный redraw;
`multi` — `estimator="new"`, `direction_mode="isotropic"`.

| Случай | `(n,d,J,P,m)` | Дополнительная конфигурация |
|---|---|---|
| `multi_d10` | `(1000,10,500,40,2)` | `N_loc=20,N_lin=70,outer=3,HPAO max=5,lambda=0.05,tau=0.4,noise=0.2` |
| `multi_d100` | `(1000,100,500,40,2)` | как d10, `N_lin=160` |
| `manifold_m1` | `(240,4,24,10,1)` | radial, `N_loc=40,N_lin=80,N_man=8,sync=1,lambda=0.5,noise=0.05,h_min=10/sqrt(n)` |
| `manifold_m2` | `(240,4,24,12,2)` | `sin(x0)+0.5*(x1²+x2²)`, истинный span `e0` и radial `(x1,x2)`, `N_loc=60,N_lin=120,N_man=8`, прочее как m1 |

Варианты: `iid` с номинальным `P`; `h1` с независимыми Haar-блоками
и compact `sqrt(k)I` для полных блоков; `iid_budget` — независимые
сферические направления при том же числе **фактических** строк, умноженные
на `sqrt(P/rows)`, так что ожидаемый Gram равен номинальному. Контроль
`iid_budget` запускается только при сжатии: d10 имеет 10 строк, m1 6,
m2 4. Он интерпретируется отдельно; основной gate сравнивает `h1` с
`iid`, потому что исходная полная ADP-задача использует номинальный `P`.

Новые selection seed **61000–61005**: 66 fit (три варианта для d10/m1/m2,
два для d100). Предпроверка точного целочисленного поиска по сохранённым
документам/скриптам не нашла их прежнего использования. Held-out seed
**62000–62019** не запускать до прохождения selection gate; ровно одна
прошедшая ветка/точка, выбранная в порядке таблицы, получает 40 validation
fit `iid/h1`. Код требует R3 manifest и неизменный fingerprint.

## Неизменяемый gate

Для каждой точки R3 нужны **6 полных iid/H1 пар** без ошибок обеих сторон.
Для каждой пары качество H1 не ниже baseline больше чем на `1e-10`:
`trace_score` для multi (выше лучше), RMS principal sine по **всем**
центрам для manifold (ниже лучше). Запрещены новая numerical failure,
потеря HPAO convergence, manifold completion или recovery. Multi recovery
требует `convergence=True` и `trace_score>=0.95`; manifold recovery требует
completion, максимальный inner linear residual `<=1e-5` и RMS sine `<=0.2`.
Manifold completion — `h_min` или остановка у одной из границ массы с тем же
inner-residual cutoff; outer stationarity из него не следует.

После этих условий достаточно **одного** из двух путей:

1. `median(H1 wall / iid wall) <= 0.8` и
   `median(H1 process RSS / iid process RSS) <= 1.05`;
2. H1 восстанавливает минимум на `2/6` seed больше baseline и обе
   медианы wall/RSS `<=1.1`.

Положительные результаты не переносятся на другие точки. Для R4 на
20 пар сохраняются те же per-fit условия и отношения времени/памяти;
recovery-порог второго пути — `ceil(20/3)=7` дополнительных seed.
Отчёт даёт все paired differences, failure accounting и bootstrap 95%
интервал медианы отношения wall/RSS (2000 ресэмплов, RNG 70000).
Отрицательный R3 означает остановку без R4 и без production-опции.

Команды R3 из корня репозитория:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.proof_first_h1 --stage selection \
  --output-dir docs/experiments/proof_first_shared_2026-09-26/selection
```
