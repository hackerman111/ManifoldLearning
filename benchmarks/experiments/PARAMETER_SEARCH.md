# Однофакторные эксперименты multi-index, 25.10.2026

Текущий CPU Grassmann, без изменений ADP/, API и defaults.
Меняется только один параметр. Для каждого численного параметра — 10 значений,
по 10 парных запусков на значение и сценарий. Команды из Code.

## Сценарии

| Имя | n | d | m | noise SD | Связь |
|---|---:|---:|---:|---:|---|
| easy | 1000 | 10 | 2 | .1 | additive |
| base | 1000 | 25 | 2 | .3 | additive |
| m3 | 1000 | 25 | 3 | .3 | additive |
| m5 | 1000 | 25 | 5 | .3 | additive |
| noisy | 1000 | 25 | 2 | 1 | additive |
| larger | 2000 | 25 | 3 | .3 | additive |
| high_d | 1000 | 50 | 3 | .3 | additive |
| balanced_m5 | 1000 | 25 | 5 | .3 | sine |

X~N(0,I), truth=QR Gaussian, response noise Gaussian. Сигнал центрируется
и нормируется до выборочной дисперсии 1 перед добавлением шума.
Additive: sin(z1)+.5*z2²+.35*sum(tanh(zj),j>=3). Sine: sum(sin(zj)).
Существующий генератор experiments.mave_grassmann_grid.arrays.
Нет корреляций признаков, выбросов или измерительного шума.

## Defaults и сетки

База — действительный ADP_Config(): N_J=None (J=n), N_loc=10,
N_lin=None (2d), N_phi=None (min(N_loc,d)), h_min=1, direction_mode=auto
(для multi/new это isotropic), redraw_directions=True, index_init=local,
select_step=best, a=sqrt(2), lambda_penalty=.05. Solver — публичный default
max_steps=20,tol=1e-6. Меняется только указанное поле, остальные — defaults.

| Параметр | 10 значений | Шаг |
|---|---|---|
| J / N_J | n/10,2n/10,...,n | n/10 |
| N_loc | 10,20,...,100 | 10 |
| h_min | .2,.4,...,2.0 | .2 |
| N_phi | 10,20,...,100 | 10 |

Категориальные эксперименты: direction_mode=auto/localized и
redraw_directions=False/True; в них только два доступных варианта.
Нет декартова произведения или комбинаций изменённых параметров.
Совпадающие defaults переиспользуются; при явных N_J=n и N_phi=10 сохраняются
отдельные fit, поскольку requested config отличается от None.
41 уникальная requested configuration на сценарий, всего 3280 fit.

Сохранение defaults включает стандартную автоматическую логику: при
изменении N_loc и N_phi=None фактическое число направлений min(N_loc,d).
Эффективные N_J,N_phi,N_lin,direction_mode сохраняются. Не приписывать весь
эффект N_loc только локальной массе. N_phi>=d — случайные направления,
не Phi=I. Все J допустимы при N_loc=10, автоматических замен J нет.

## Повторы и измерения

Seeds=792100..792109. В сценарии X,y,truth и algorithm seed совпадают
между настройками; вложенные исходные массивы между совместимыми сценариями.
Изменение J меняет центры и локальную инициализацию. Порядок всех задач
перемешан RNG=20261025, schedule сохраняется до fit. Истинный базис
алгоритму не передаётся; best выбирается по обучающей ошибке.

EDR=||(I-P_true)B_hat||F/sqrt(m), trace score=1-EDR², max principal
sine/angle, initial EDR, trace и solver diagnostics сохраняются.
Условный порог восстановления EDR<=.1. Сбои сохраняются в полном знаменателе.
Низкий loss, h_min или convergence не заменяют EDR. Однофакторная сетка не
устанавливает лучшее совместное сочетание и не даёт независимой validation.

Расчёт последовательный, отдельный subprocess на fit, BLAS/OMP/MKL=1,
timeout=180s. fit_seconds охватывает полный публичный fit: инициализация,
статистики, solver, выбор шага, встроенный profiler. Импорты, чтение данных,
внешние метрики и JSON вне fit_seconds. process_seconds включает весь запуск
Python, импорты, чтение, fit, метрики и JSON. Первый fit в свежем процессе;
холодный дисковый кэш и повторный warm fit не измеряются.
Peak RSS — OS high watermark отдельного subprocess. Внешняя системная
нагрузка может влиять на время.

## Артефакты и анализ

Каталог verification/parameter-search-20261025/:
manifest.json — сценарии, сетки и ссылки на общие baseline, seeds,
source hashes, окружение, fingerprint; sources.zip и source/ — снимок кода;
explore/schedule.json — задачи; inputs/*.npz — данные; fits/*.json —
результаты, requested/effective parameters, input/initial hashes,
initial/final bases, trace, время и RSS; *.job.json — задания.
state.json/run.log — прогресс. complete.json появляется после всех задач,
включая неудачные (число указано явно). Resume не перезаписывает результаты.

Анализ после завершения по следующему запросу пользователя выполняет
Luna (gpt-6-luna). Независимо пересчитать EDR из basis/truth, проверить hashes,
coverage, pairing, дубликаты и единичность изменённого поля. Показать
качество/время/RSS по каждому параметру и сценарию, сбои, uncertainty по
10 seed и ограничения отбора на тех же данных. Новые стадии отбора или
совместные сочетания автоматически не запускать.

Команды:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/adp-search-mpl python -m experiments.parameter_search dry-run
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/adp-search-mpl python -m experiments.parameter_search run --workers 1
