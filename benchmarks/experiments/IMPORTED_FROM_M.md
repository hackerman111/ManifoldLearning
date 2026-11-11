# Эксперименты

Основной каталог `EDR_code` сохранён. Из `Vlad_EDR_M/Code` добавлены отдельные
исследовательские модули и все сохранённые результаты. Новые варианты остаются
экспериментальными. Таблица показывает расположение материалов; перенос сам
по себе не является повторной проверкой научных выводов.

| Тема | Модули / описание | Сохранённые результаты |
|---|---|---|
| Базовые single/multi/manifold серии | [single.py](single.py), [multi.py](multi.py), [manifold.py](manifold.py), [suite.py](suite.py) | [Основной архив](../results/outputs/) |
| Сравнение с R MAVE / MEANMAVE | [r_sdr/README.md](r_sdr/README.md) | [r_sdr](../results/outputs/r_sdr/), [verification](../results/verification/README.md) |
| Диагностика начального ранга и QR | [init_rank_diagnosis.py](init_rank_diagnosis.py), [init_rank_checks.py](init_rank_checks.py), [qr_profile.py](qr_profile.py), [qr_highm_study.py](qr_highm_study.py) | [init-rank](../results/verification/init-rank-20261023/), [qr-highm](../results/verification/qr-highm-20261023/) |
| Начальная LLS / полные моменты / локальные полиномы | [LLS_PILOT.md](LLS_PILOT.md), [local_polynomial.py](local_polynomial.py), [moment_quadratic_study.py](moment_quadratic_study.py) | [moment-quadratic](../results/verification/moment-quadratic-20261023/) |
| Геометрия и сравнение Grassmann/MAVE | [mave_grassmann_grid.py](mave_grassmann_grid.py), [mave_grassmann_analysis.py](mave_grassmann_analysis.py), [mave_grassmann_geometry.py](mave_grassmann_geometry.py), [mave_grassmann_figures.py](mave_grassmann_figures.py) | [mave-grassmann-grid](../results/verification/mave-grassmann-grid-20261025/) |
| OFAT-поиск параметров | [PARAMETER_SEARCH.md](PARAMETER_SEARCH.md), [parameter_search.py](parameter_search.py) | [parameter-search](../results/verification/parameter-search-20261025/) |
| Моментные метрики и сертификаты | [edr_metric_corrections.py](edr_metric_corrections.py), [edr_metric_study.py](edr_metric_study.py) | [edr-metric](../results/verification/edr-metric-20261025/), [TeX](../../docs/tex/result/edr_metric_corrections.tex) |
| Связанные и локальные штрафы | [coupled_penalties.py](coupled_penalties.py), [light_penalties.py](light_penalties.py), [curvature_penalties.py](curvature_penalties.py), [curvature_correction.py](curvature_correction.py) | [coupled](../results/verification/coupled-penalties-20261025/), [light](../results/verification/light-penalties-20261025/), [curvature](../results/verification/curvature-penalties-20261025/) |
| Полные моменты с ridge и ускоренный профиль | [FULL_RIDGE_FAST.md](FULL_RIDGE_FAST.md), [full_ridge_fast.py](full_ridge_fast.py), [full_ridge_batched.py](full_ridge_batched.py) | [batched](../results/verification/full-ridge-batched-20261025/), [high-dim](../results/verification/full-ridge-batched-hd-r2-20261025/), [TeX](../../docs/tex/result/full_moments_ridge.tex) |
| Варианты полной моментной задачи | [full_moment_variants.py](full_moment_variants.py), [full_moment_variant_study.py](full_moment_variant_study.py), [full_moments_meanmave.py](full_moments_meanmave.py) | [variants](../results/verification/full-moment-variants-20261026/), [vs MEANMAVE](../results/verification/full-moments-vs-meanmave-20261025/), [development](../results/verification/full-moments-development-20261026/) |
| Fourier-моменты гауссовского дизайна | [GAUSSIAN_FOURIER.md](GAUSSIAN_FOURIER.md), [gaussian_fourier.py](gaussian_fourier.py) | [gaussian-fourier](../results/verification/gaussian-fourier-20261025/), [исходная идея](../../docs/tex/data/gaussian_fourier_solver.tex) |
| Анизотропные метрики и геометрия | [ANISOTROPIC_METRICS.md](ANISOTROPIC_METRICS.md), [anisotropic_metrics.py](anisotropic_metrics.py), [anisotropic_fixed_windows.py](anisotropic_fixed_windows.py), [anisotropic_geometry_study.py](anisotropic_geometry_study.py) | [anisotropic-metrics](../results/verification/anisotropic-metrics-20261026/), [TeX](../../docs/tex/result/anisotropic_full_moments.tex) |
| Облегчённая метрика | [light_metric.py](light_metric.py), [light_metric_study.py](light_metric_study.py) | [light-metric](../results/verification/light-metric-edr-20261026/) |
| Двусторонняя метрика и MEANMAVE | [two_sided_meanmave.py](two_sided_meanmave.py) | [two-sided](../results/verification/two-sided-meanmave-20261026/), [nloc30 mini](../results/verification/two-sided-nloc30-mini-20261026/) |

Все 36 перенесённых исследовательских каталогов, включая неудачные и неполные
серии, перечислены в [verification/README.md](../results/verification/README.md).
Ранее существовавшие исследования `grassman_*`, `svd_*`, manifold и другие
каталоги остаются на прежних местах; карта кода — [agent-notes](../../docs/agent-notes/ADP/README.md).

Короткая проверка каталога R без расчётов и без запуска R:

```bash
python -m experiments.r_sdr --profile high-dim-dm --runs 10 --methods ADP-Grassmann-combined,MAVE-package --dry-run
python -m experiments.r_sdr --profile d25-n5000 --runs 10 --methods ADP-Grassmann-combined,MAVE-package --dry-run
```

Для запуска серии следуйте её протоколу: некоторые runner-ы используют
фиксированные каталоги результатов. Исторические измерения и манифесты при
объединении не переписывались. `ADP` в действующем R-каталоге остаётся LSMR;
для Grassmann выбирайте явное имя метода из `r_sdr/README.md`.


## Дополнение от 2026-10-29

Материалы, добавленные в исходной M-копии после первоначальной интеграции: shared-polynomial refinement, Gaussian projector localization и local-score spectral study. Карта и протоколы находятся в [отчёте о переносе](../../docs/integration/vlad_edr_m_20261029/README.md); отличавшиеся исходные версии сохранены рядом в `source-conflicts/`. Сохранение файлов не означает повторного запуска или подтверждения гипотез.
