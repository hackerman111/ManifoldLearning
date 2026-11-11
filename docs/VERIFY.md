# Проверка экспорта — 2026-10-23

Проверено на Linux, Python 3.14.7. Исходная папка скопирована в
`/tmp/adp-export-verification-20261023`; `PYTHONPATH` исходного репозитория
удалён. Импорты ADP/experiments проверены по фактическим путям файлов.
BLAS: `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`.

- `python -m pytest -q`: **139 passed, 2 xfailed**. Лог:
  [benchmarks/results/verification/pytest.log](../benchmarks/results/verification/pytest.log).
- Grassmann CLI с default20 и явным бюджетом2 побитово совпадает с публичной
  `ADP_multi_index(..., ADP_solver(grassman.solve))` на одинаковых данных/config.
- CLI каталогов, r_sdr dry-run и одиночный multi/Grassmann запуск работают
  без импорта родительского репозитория.
- `benchmarks.grassman_benchmark --stage fullfit --seeds 0`: **2/2 ok**,
  SVD и текущий core_gn, n800/d50/m2. Результаты и протокол:
  [benchmarks/results/verification/svd-grassman-smoke/](../benchmarks/results/verification/svd-grassman-smoke/).
- Фактическое сравнение с R: `experiments.r_sdr --profile smoke --cases tiny-multi
  --runs 10 --methods ADP-Grassmann-combined,MAVE-package`: **20/20 ok**,
  seeds20000–20009, n80/d5/m2. Результаты и протокол:
  [benchmarks/results/verification/r-grassman-smoke/](../benchmarks/results/verification/r-grassman-smoke/).
- R reference, загрузка пакета MAVE и возобновление прерванного R benchmark
  также входят в прошедшие тесты.
- Wheel и sdist собраны локально через `setuptools.build_meta` без скачивания
  зависимостей. Wheel установлен через `uv pip install --no-deps` в отдельный
  venv с системными NumPy/SciPy. Из `/tmp` проверены import origins, наличие
  `worker.R`, source-snapshot inputs, одиночный Grassmann CLI, experiment list,
  benchmark help и r_sdr dry-run. Постороннего checkout в PYTHONPATH нет.

Это проверка работоспособности упаковки. Smoke-результаты не являются новым
исследованием качества/скорости; оба полноразмерных fit остановлены по h_min,
что не является сертификатом сходимости. CUDA в этой задаче не проверялась.

Известные унаследованные ограничения:

- Необязательный `experiments.multiv2` исключён: исходный модуль падает при
  импорте с `KeyError: mi-spokoini-m2-n`. Основной каталог multi содержит эти
  серии без изменений. Связанный test_experiment.py не включён.
- `test_cli_manifold_smoke[cg/hybrid]` — strict xfail: исходный CLI не передаёт
  estimator в manifold-модель, чей нынешний default local_quadratic запрещает
  index_dim=2. Ошибка воспроизведена и в корневом репозитории.
- Корневой полный pytest запускался и остановился на трёх ошибках collection:
  test_experiment.py, test_multi_center_diagnostic.py, test_multiv2_quality.py —
  все из-за указанного исходного multiv2 KeyError. Корневые численные исходники
  не менялись; зелёным является проверенный набор экспортированного комплекта.

`SOURCE_MANIFEST.json` фиксирует SHA256 комплекта и соответствующих оригиналов.
Все ADP-исходники кроме CLI adapter совпадают с исходным рабочим деревом.
