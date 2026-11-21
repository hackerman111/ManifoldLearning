# Benchmarks and experiment results

- [`experiments/`](experiments/README.md) contains experiment modules and their
  protocols. The modules remain importable as `experiments.*`; run them with
  `python -m experiments.<module>` from the repository root.
- The scripts directly in this directory contain focused benchmark and audit
  drivers.
- [`results/`](results/README.md) stores generated benchmark outputs and frozen
  verification campaigns.

The live estimator and solver implementation remains in [`../ADP/`](../ADP/).

## Очистка локальных NPZ

```bash
python benchmarks/clean_npz.py
python benchmarks/clean_npz.py benchmarks/results/outputs/r_sdr/имя-запуска
```

По умолчанию открывается `benchmarks/results/outputs`. Можно указать каталог
внутри Git-репозитория. Папки и файлы отсортированы по объёму NPZ.
Стрелки перемещают курсор, `Enter` открывает папку, `←` возвращает на уровень
выше. `Space` отмечает файл или папку целиком, `a` отмечает всё в текущей папке.
Отметки сохраняются при навигации. `d` показывает число файлов и объём перед
безвозвратным удалением, `y` подтверждает, `Esc` отменяет. `r` обновляет список
и сбрасывает отметки, `q` закрывает скрипт.

Под удаление попадают только отмеченные обычные `.npz` вне индекса Git.
Символьные ссылки, отчёты и файлы остальных форматов исключены. NPZ из
`seven-families-nj85-20261118/*/inputs/` служат эталоном при запуске замеров
`seven-families-lambda-20261119`.
