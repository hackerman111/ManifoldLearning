# ADP CLI: быстрый запуск

CLI запускает single- и multi-index эксперименты вручную или из Python-файла
с переменной `experiment = ADP_Experiment(...)`.

```bash
python ADP/cli.py --help
```

## Быстрые команды

### Single-index, только вывод в терминал

Ничего не сохраняет:

```bash
python ADP/cli.py \
  --terminal-only --no-progress \
  --mode single --solver lsmr --runs 2 \
  --n 40 --d 3 --N_loc 5 --N_lin 8 --N_J 8 --N_phi 3 \
  --h_min 1000000
```

### Multi-index, только вывод в терминал

```bash
python ADP/cli.py \
  --terminal-only --no-progress \
  --mode multi --index-dim 2 --solver lsmr --runs 2 \
  --n 48 --d 4 --N_loc 6 --N_lin 10 --N_J 8 --N_phi 3 \
  --h_min 1000000 --index-init pilot
```

Для ручного multi-index запуска по умолчанию используются
`--index-init pilot` и `--lambda-penalty 10000`. Режимы `local` и `random`
можно выбрать явно.

### Точный screening и умный выбор локализации

`--smart-weights` включает exact screening заведомо нулевых весов и выбирает
максимальный допустимый `rho` (single) или `alpha` (multi) по breakpoint-ам:

```bash
python ADP/cli.py \
  --terminal-only --no-progress --smart-weights \
  --mode multi --index-dim 2 --solver lsmr \
  --n 1000 --d 100 --N_loc 5 --N_J 200 --N_phi 3
```

Режим работает с `--kernel epanechnikov`. Без флага используется прежний
расчёт; `--no-smart-weights` отключает его явно. Параметр `--a` по-прежнему
задаёт геометрическое уменьшение bandwidth и не является `alpha`.

### Sparse box и plateau ядра

Оба режима используют общий exact neighborhood backend без плотных матриц
расстояний и весов `J×n`:

```bash
python ADP/cli.py --terminal-only --mode single --kernel box
python ADP/cli.py --terminal-only --mode multi --index-dim 2 \
  --kernel plateau --kernel-tau 0.5
```

Для plateau по умолчанию `tau=0.5`, допустимо только `0 < tau < 1`.
Автоматического переключения `box → plateau` нет. Эти режимы уже выполняют
exact screening и поэтому не сочетаются с `--smart-weights`.

### Статистики и LSMR на GPU

`--gpu` переносит статистики и встроенный LSMR на CUDA через CuPy:

```bash
python ADP/cli.py \
  --terminal-only --no-progress --gpu --smart-weights \
  --mode multi --index-dim 2 --solver lsmr \
  --n 10000 --d 1000 --N_loc 20 --N_J 10000 --N_phi 3
```

CuPy и доступная NVIDIA GPU обязательны: CPU fallback отсутствует. Генерация
данных, направлений и весов остаётся на CPU. `--gpu` несовместим с VarPro.

### Single-index с VarPro

```bash
python ADP/cli.py \
  --terminal-only --no-progress \
  --mode single --solver varpro --solver-max-steps 20 --runs 1 \
  --n 40 --d 3 --N_loc 5 --N_lin 8 --N_J 8 --N_phi 3 \
  --h_min 1000000
```

### Обычный запуск с сохранением результатов

Создаёт NPZ, commits, CSV и графики в `ADP/experiment_outputs/`:

```bash
python ADP/cli.py \
  --mode single --solver lsmr --runs 3 \
  --n 240 --d 3 --N_loc 10 --N_J 64 \
  --output-dir ADP/experiment_outputs
```

### Проверка experiment-файла без вычислений

```bash
python ADP/cli.py \
  --experiment-file ADP/examples/experiment_compare.py \
  --dry-run
```

### Experiment-файл без сохранения

```bash
python ADP/cli.py \
  --experiment-file ADP/examples/experiment_compare.py \
  --terminal-only --no-progress
```

### Experiment-файл с сохранением

```bash
python ADP/cli.py \
  --experiment-file ADP/examples/experiment_compare.py \
  --output-dir ADP/experiment_outputs
```

При `--experiment-file` параметры модели и данных берутся из файла. Флаги
`--output-dir`, `--resume`, `--dry-run`, `--terminal-only`,
`--no-save-models` и `--no-progress` остаются управляющими.

### Продолжить существующую серию

Нужно передать тот же experiment-файл и точный каталог серии:

```bash
python ADP/cli.py \
  --experiment-file ADP/examples/experiment_compare.py \
  --resume ADP/experiment_outputs/lsmr_vs_varpro/<TIMESTAMP> \
  --no-progress
```

### Только пересобрать графики и `artifacts.csv`

```bash
python ADP/cli.py \
  --reports-only ADP/experiment_outputs/lsmr_vs_varpro/<TIMESTAMP>
```

## Мастер-команда

Все перечисленные флаги совместимы и реально участвуют в ручном persistent
запуске. Команда покрывает multi-index, LSMR, solver settings, параметры данных,
ADP-конфигурацию, kernel, память и управление выводом:

```bash
python ADP/cli.py \
  --mode multi \
  --index-dim 2 \
  --solver lsmr \
  --solver-tol 1e-8 \
  --solver-max-steps 20 \
  --runs 2 \
  --output-dir ADP/experiment_outputs \
  --no-save-models \
  --no-progress \
  --n 96 \
  --d 6 \
  --noise 0.05 \
  --seed 7 \
  --N_loc 8 \
  --N_lin 16 \
  --N_J 16 \
  --N_phi 4 \
  --outer_steps 4 \
  --lambda_penalty 100 \
  --local_ridge 1e-8 \
  --kernel epanechnikov \
  --smart-weights \
  --gpu \
  --a 1.4142135623730951 \
  --h_min 1000000 \
  --batch_size 16 \
  --index_init random
```

Не включены альтернативные режимы: `--experiment-file` игнорировал бы ручные
параметры, `--resume` требует готовую серию, `--reports-only` только строит
отчёты, `--dry-run` не выполняет fit, а `--terminal-only` отключает сохранение.

## Что сохраняется

Обычный запуск создаёт каталог
`<output-dir>/<experiment-name>/<timestamp>/` с исходными данными, commits,
CSV, optional model NPZ, `plots/` и `artifacts.csv`. `--no-save-models`
отключает только model NPZ. `--terminal-only` не создаёт ничего.

Коды завершения: `0` — все jobs успешны; `1` — есть `nonconverged` или
`numerical_failure`; `2` — ошибка аргументов/конфигурации; `130` — `Ctrl-C`.
