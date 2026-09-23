GPU single-index и multi-index (текущий HPAO)
==========================================

Реализованы две явные границы CPU/GPU. CPU остаётся стандартным backend.
`gpu=True` вычисляет статистики на GPU и решает задачу на CPU;
`gpu=True, gpu_solver=True` оставляет I/U на GPU и выполняет там HPAO-LSMR.
Это частичный перенос: инициализация, направления, weights, bandwidth,
rho/alpha и канонизация конечного индекса остаются CPU в обоих режимах.
Manifold в эту реализацию не входит.

Установка и использование
------------------------

Нужны NVIDIA GPU, совместимый драйвер и CUDA Toolkit 13.x:

```bash
uv sync --extra gpu --group bench
uv run --extra gpu python -m ADP.cli --mode single --gpu
uv run --extra gpu python -m ADP.cli --mode multi --index-dim 2 --gpu
uv run --extra gpu python -m ADP.cli --mode multi --index-dim 2 --gpu --gpu-solver
uv run --extra gpu python -m ADP.cli --mode multi --index-dim 2 --gpu --solver hybrid
```

`gpu` extra устанавливает CuPy CUDA 13, не дублируя системный CUDA Toolkit.
Если Toolkit отсутствует, его нужно установить отдельно. Проверены CuPy 14.1.1 и 14.2.0; итоговый benchmark и полный набор тестов
выполнены с 14.2.0 из uv.lock. Допустимый диапазон задан в pyproject.toml.

```python
from ADP import ADP_Config, ADP_single_index, ADP_multi_index

config = ADP_Config(gpu=True, N_phi=10, outer_steps=3)
single = ADP_single_index(config).fit(X, y)
multi = ADP_multi_index(2, config).fit(X, y)
```

Для GPU-солвера дополнительно задайте `gpu_solver=True`. Он поддерживает
встроенный `ADP.solver.LSMR.solve`. CPU CG/HYBRID и custom HPAO-solvers
по-прежнему доступны при `gpu=True, gpu_solver=False`.
Без CuPy/рабочего CUDA-устройства выдаётся RuntimeError; переключения на CPU
при ошибке нет. Все публичные результаты возвращаются как NumPy-массивы.
Прямой `LSMR.solve(index, U_gpu, I_gpu, ...)` выбирает backend по массиву U;
он также возвращает NumPy index/coefficients.

Сохранённая математика
---------------------

Классификация переноса **NUMERICAL**: тот же конечный random sketch,
целевой функционал, float64, cutoff локальной SVD, ridge, допуски и trust
certificate; меняются библиотеки линейной алгебры и порядок редукций.
Нет FP32, усечения sketch, приближённой поддержки или изменения estimator.

Для каждого центра A_j = W_j / mass_j, mu_j = A_j Xc, S_j = A_j y:

```
Q_j = Phi_j (Xc - mu_j)^T
r_j = Q_j A_j
H_j = (Q_j - r_j[:,None]) * A_j[None,:]
I_j = H_j y - sum(H_j,axis=1) S_j
U_j = H_j Xc - sum(H_j,axis=1)[:,None] mu_j
```

Для `estimator="legacy"` I/U умножаются на mass; для `new` mass используется
внешним весом loss = 1/2 sum_j mass_j ||I_j - U_j B^T l_j||^2.
Single — тот же функционал со скаляром l_j и вектором beta.
Чередование local refit / global correction сохранено. Коррекция решается
через augmented matrix-free LSMR, без построения нормальной матрицы.
CuPy имеет [LSMR с LinearOperator и damp](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.lsmr.html);
local multi refit использует [пакетную SVD](https://docs.cupy.dev/en/stable/reference/generated/cupy.linalg.svd.html).
Нормальный residual пересчитывается исходным оператором после каждой пробы.
Диагностики включают stop, последний и суммарный iteration count,
число solves, loss history, certificate, stationarity и orthogonality.

Xc/Y загружаются один раз на fit. Центрирование X выполнено тем же CPU
reference, после чего Xc хранится на GPU. Направления генерируются прежними
NumPy SeedSequence-потоками и передаются блоками. Это сохраняет сравнимость
seed между CPU/GPU. Для GPU-LSMR большие I/U не скачиваются между внутренними
итерациями; на CPU приходят скаляры управления и конечные index/coefficients.

Память и вычисления
------------------

Формы: X=(n,d), Phi/U=(J,P,d), I=(J,P), weights block=(B,n).
Dense-статистики требуют O(J P n d) работы, временно O(B P n + B P d).
При k ненулевых соседях пакетная ветка требует O(B k d + B P k + B P d)
временной памяти. Соседи упаковываются без удаления ненулевых весов; cap
на k отсутствует. Тензор (J,n,d) и полный dense weights не создаются.
Порог 25% поддержки унаследован от CPU для выбора локальной формулы;
это не утверждение об оптимальном GPU crossover для всех размерностей.

B ограничен `batch_size` и оценкой рабочих массивов в 64 MiB (минимум один
центр; если один центр больше бюджета, его массивы могут превысить бюджет).
CUDA/BLAS workspace в эту оценку не входит. Полный Phi остаётся на CPU.
В statistics-only режиме I/U скачиваются по блокам, поэтому полный U
не занимает VRAM. В GPU-solver режиме VRAM дополнительно содержит U:
8*J*P*d байт. Статистики прошлого outer-шага освобождаются до создания новых.
Multi forward использует l @ B формы (J,d), затем batched U @ local;
adjoint — batched U.T @ residual, затем l.T @ pulled.

При n=10000,d=1000,J=1000,P=20 данные Xc занимают 80 MB, U — 160 MB,
CPU Phi — 160 MB, существующий CPU distance2 — 80 MB (десятичные MB).
При J=10000 U/Phi уже по 1.6 GB, distance2 — 800 MB. Эта реализация
не устраняет старый CPU distance2; перенос всего fit на GPU потребовал бы
отдельной работы с инициализацией и tiled geometry/search.

Проверка и воспроизведение
-------------------------

`tests/test_gpu_index.py` сравнивает GPU с CPU/reference: normalized/legacy,
dense/sparse, разные batch_size, offsets, коррелированные признаки,
неравные массы, пустые соседства, локальная потеря ранга, adjoint,
augmented dense least-squares для разных ridge, сертификат и лимит итераций,
полный fit single/multi, совместимость CPU CG/HYBRID с GPU statistics.
Без CUDA аппаратные тесты явно пропускаются; no-device проверяется отдельно.

```bash
uv run --extra gpu pytest -q tests/test_gpu_index.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --extra gpu --group bench python -m benchmarks.index_gpu \
  --output /tmp/index_gpu_results.json
```

Benchmark сохраняет seed, конфигурацию, формулу kernel (аргумент — квадрат
локализованного расстояния/h^2), окружение, commit/dirty state и хеш ADP
исходников. Таймер включает host/device transfers и synchronize; прогрев
каждого backend/размера/режима записан отдельно и исключён из измерений.
Один BLAS thread, одинаковые данные и направления, 3 независимых seed.
Фактические overrides относительно config_defaults в первых JSON: N_loc=20,
N_lin=2*d, N_J=J, N_phi=P, outer_steps=2, select_step=last, seed=model_seed,
gpu=(backend != cpu), gpu_solver=(backend == gpu_solver). Batch_size=32,
a=sqrt(2), h_min=1, lambda_penalty=0.05, local_ridge=1e-8.
Скрипт дополнительно сохраняет эти overrides явно в новых запусках.
Ошибочные fit остаются в JSON с error. Два outer и три HPAO шага — ограниченный
протокол производительности, а не доказательство статистической сходимости.

Память измеряется отдельно: host tracemalloc high-water по стадиям и CUDA
allocator peak used/reserved. Это не полный процессный пик VRAM: workspace
библиотек и CUDA context не учитываются. RSS high-water всего benchmark
процесса записан отдельно и не приписывается отдельным fit.
Предварительный запуск в `index_gpu_pilot_results.json` прерван после
медленной large CPU-инициализации. `index_gpu_initial_results.json` — первая
полная серия в системном Python/CuPy 14.1.1 с прежним CLI seed-протоколом.
Оба файла сохранены как история измерений; итоговая таблица использует
`index_gpu_results.json` (окружение проекта, CuPy 14.2.0, раздельные
SeedSequence-потоки data/method, общие method streams для пар CPU/GPU).

Проверки инструментов: полный `uv run --no-sync pytest -q` с CUDA —
328 passed; Ruff check/format всех изменённых Python-файлов — чисто;
`uv lock --check` и `git diff --check` — чисто. Pyright — 0 ошибок,
1 предупреждение о старом suppression в `solver/legacy_lsmr.py`, которое
стало лишним после установки CuPy. Общие Ruff check/format репозитория
показывают прежние замечания в архивном `test/` и старых документах;
посторонние файлы не переформатированы.

Измерения RTX 5060, float64, local initialization
------------------------------------------------

Медианы трёх seed; секунды полного fit, включая передачи данных.

| Режим | (n,d,J,P) | CPU | GPU statistics + CPU LSMR | GPU statistics + GPU LSMR |
|---|---|---:|---:|---:|
| single | (1000, 32, 100, 12) | 0.158 | 0.157 | 0.406 |
| single | (2000, 100, 200, 20) | 1.031 | 0.935 | 1.215 |
| single | (4000, 200, 200, 20) | 4.297 | 4.058 | 4.337 |
| multi, m=3 | (1000, 32, 100, 12) | 0.368 | 0.363 | 1.529 |
| multi, m=3 | (2000, 100, 200, 20) | 2.051 | 1.935 | 4.531 |
| multi, m=3 | (4000, 200, 200, 20) | 6.548 | 6.242 | 9.680 |

Все 54 fit завершились без ошибок. Максимальный residual подпространства
||B_gpu - (B_gpu B_cpu.T) B_cpu||_F/sqrt(m) = 3.21e-09.
Сравнение loss, quality, h/factor, числа итераций и сертификатов сохранено
в JSON по каждому запуску. Число Krylov-итераций может немного различаться
из-за округлений, при неизменных допусках и проверке сертификата.

| (n,d,J,P) | Peak GPU pool, statistics / solver, MiB | Peak host traced, CPU / statistics / solver, MiB |
|---|---:|---:|
| (1000, 32, 100, 12) | 6.7 / 7.0 | 9.1 / 5.7 / 5.7 |
| (2000, 100, 200, 20) | 39.9 / 43.0 | 34.1 / 24.6 / 24.6 |
| (4000, 200, 200, 20) | 65.9 / 72.0 | 71.2 / 49.2 / 49.2 |

Это максимумы среди двух моделей и трёх seed, а не сумма пиков стадий.
Рекомендуемая исходная настройка для этих режимов — `gpu=True` без
`gpu_solver`. На малом размере разница CPU / GPU statistics сопоставима
с шумом тайминга; полезность GPU не гарантируется. Инициализация local
остаётся главным ограничителем больших full-fit: полный GPU-resident fit
не реализован и его скорость этим сравнением не установлена.
Следующий отдельный кандидат на перенос — пакетная локальная инициализация,
с сохранением её ridge и rank semantics; перенос одного LSMR этот участок
не ускоряет.

Стрессовый полный протокол (2 outer / 3 HPAO, random init) был прерван:
первый измеренный CPU single fit занял 63.313 s; парное сравнение не закончено.
Частичный JSON был перезаписан при запуске ограниченной проверки; сохранена
сводка наблюдавшегося stdout в `index_gpu_stress_attempt.json`. Этот запуск
не используется в выводах об ускорении. Ограниченная проверка ниже использует
отдельно указанные 1 outer / 1 HPAO и один seed; это диагностика, а не
подтверждение устойчивого выигрыша на стрессовом размере.
