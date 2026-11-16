# e-means-inspired anisotropic ADP experiment

Дополнительный режим `N_J=n`:400/400fit, mean EDR original20=.332018
против bounded-J=.395326 при paired time ratio3.169533. Mini+EMA=.358669
при time ratio.843886 к all-center original20; quality gates не пройдены,
validation не запускалась. Полный [ALL_CENTERS_REPORT.md](ALL_CENTERS_REPORT.md),
доказательство и затраты [ALL_CENTERS_MATH.md](ALL_CENTERS_MATH.md).
Контроль `original20_bounded` использует те же свежие данные. Предыдущий
отрицательный эксперимент и его heldout сохраняются. Запуск в новом output:

```sh
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -m experiments.e_means_adp.study run /tmp/e-means-all-centers-replay --all-centers
```

Selection1134400..409, conditional validation1135400..409. Настройки candidate
и gates прежние. Аудит сверяет `N_J` и в job, и в фактическом trace, а равенство
инициализации проверяет отдельно внутри каждой группы J. Public `N_J=None`
уже означает n; CLI-флаг меняет только этот экспериментальный протокол.

Исследовательская модификация исходного ADP-Grassmann, не реализация
кластеризации e-means. Исходный пакет ADP, API и defaults не изменены.
**Frozen selection отрицательный:**320/320 fit, mini+EMA time ratio.8864,
EDRmean.4080->.4374; quality gates не пройдены, validation не запускалась.
Полные результаты: [REPORT.md](REPORT.md). Вариант не выбран для применения.
Математика, четыре семейства подходов, доказательства, контрпример и точные
ограничения: [MATH.md](MATH.md).

Вариант меняет только bounded optimizer одного масштаба: два прохода по
четырём минибатчам центров, по одному исходному Grassmann-шагу на батч,
затем четыре исходных шага на всех центрах. Дополнительный старт строится
через cosine EMA проекторов без плотной ambient covariance. Full-profile
guard включает входной базис; итоговый refit и сертификаты используют все
центры. Nested `full_finalization` — диагностика попытки финального solve;
при rollback она не описывает возвращённый базис. Итоговые поля верхнего
уровня всегда пересчитываются для возвращённого результата.

```python
from ADP import ADP_Config, ADP_multi_index, ADP_solver
from experiments.e_means_adp.candidate import MiniBatchSolver

model = ADP_multi_index(
    3,
    ADP_Config(seed=42, N_J=200, N_phi=12, N_loc=20),
    ADP_solver(MiniBatchSolver(7000042, ema=True)),
).fit(X, y)
```

Новый MiniBatchSolver нужен для КАЖДОГО fit: его RNG меняется между масштабами.
Веса, центры, случайные sketch-направления, локальные I/U, alpha/h schedule и
outer best-step criterion остаются в исходном public fit_index. Изменённые
оценки подпространства естественно меняют последующую адаптивную траекторию.
Вариант экспериментальный; пример не является рекомендацией его применять.

Воспроизведение в новом, ещё не существующем output:

```sh
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -m experiments.e_means_adp.study run /tmp/e-means-adp-replay
rtk proxy python -m experiments.e_means_adp.study audit /tmp/e-means-adp-replay
```

В output сохраняются source snapshot/SHA256, окружение, frozen PROTOCOL,
все данные и их хеши до первого fit этапа, jobs, базисы, коэффициенты,
trace/profile, stdout/error и raw runs.jsonl. Workers запускаются из snapshot.
Контроли original20 и full6 отделяют вклад минибатчей/EMA от сокращения бюджета.
Selection:8cases x10seeds x4arms=320fits. Validation10 новых seeds запускается
только после заранее заданных gates и независимого аудита selection.
Ошибки сохраняются; автоматического resume нет, занятый output отклоняется.

RSS — максимум Linux VmHWM/VmRSS с периодом25ms и worker ru_maxrss; включает
Python/imports. Ограничение2048MiB опрашивается, не hard OS cap. Timeout120s.
Cold fit time отделён от process time. Один BLAS thread, порядок методов
перемешан, но другие задания хоста могут влиять на абсолютное время.
Bootstrap CI — численная оценка на10 seed-кластерах, не строгая гарантия.

Проверки:

```sh
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests/test_e_means_adp.py tests/test_grassman.py tests/test_r_sdr.py -q
rtk proxy ruff check benchmarks/experiments/e_means_adp tests/test_e_means_adp.py
rtk proxy uv run --no-project --with pyright pyright --pythonpath /usr/bin/python benchmarks/experiments/e_means_adp
```

Ни снижение frozen profile, ни успешный refit не доказывают улучшения EDR.
Исследовательская попытка завершена отрицательным отбором; defaults сохранены.
