# ADP / EDR

Исследовательский код Average Derivative Procedure: single-index, multi-index,
manifold, Grassmann, LSMR и другие решатели. Этот репозиторий объединяет
`EDR_code` с экспериментами из `Vlad_EDR_M/Code`.

При конфликте приоритет имеет `EDR_code`. Его численные реализации,
LSMR по умолчанию, CG/HYBRID и GPU-возможности сохранены. Экспериментальные
полные моменты, штрафы и метрики доступны отдельными модулями;
они не включаются автоматически.

## Навигация

- [Эксперименты](benchmarks/experiments/README.md): основной каталог;
  [добавленные исследования из M](benchmarks/experiments/IMPORTED_FROM_M.md).
- [Бенчмарки и сохранённые результаты](benchmarks/README.md), включая
  [проверки и аудиты](benchmarks/results/verification/README.md).
- [TeX и PDF](docs/tex/README.md): алгоритмы, результаты, теория, исходные идеи и архив.
- [Бенчмарки с R](benchmarks/experiments/r_sdr/README.md).
- [Карта реализации](docs/agent-notes/ADP/README.md).
- [Документы и карта переезда](docs/README.md).
- [Протокол объединения и проверок](docs/integration/vlad_edr_m_20261026/README.md).

## Запуск

Из корня `EDR_code`, Python >= 3.11:

```bash
uv sync --group bench
uv run python -m ADP.cli --help
uv run python -m ADP.cli.experiment --list
uv run python -m experiments.multi --list
```

Grassmann можно выбрать явно; прежний CLI по умолчанию использует LSMR:

```bash
uv run python -m ADP.cli --mode multi --solver grassman --n 48 --d 4 --index-dim 2 --N_J 8 --N_phi 3 --N_loc 6 --N_lin 10 --outer_steps 1 --solver-max-steps 2 --index_init random --h_min 1000000
```

Для нового явного `--solver grassman` бюджет без `--solver-max-steps` равен
20. Бюджеты существующих вариантов CLI сохранены: 3 для single, 5 для multi.
Python-модели `ADP_single_index` и `ADP_multi_index` сохраняют прежний LSMR.

Проверки:

```bash
uv run pytest
```

Сохранённые исследовательские результаты относятся к зафиксированным в них
версиям кода. В старых сериях из M имя `ADP` могло обозначать Grassmann;
в действующем `experiments.r_sdr` оно по-прежнему обозначает LSMR.
Для воспроизведения исторического результата используйте его протокол,
манифест и снимок исходников, а не текущие настройки по умолчанию.
