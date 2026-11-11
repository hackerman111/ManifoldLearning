# Перенос исследований Vlad_EDR_M, 2026-10-29

Исследовательское содержимое из `Vlad_EDR_M/Code` отделено от комплекта кода и рукописи. Файлы перенесены в соответствующие каталоги `EDR_code`; существовавшие идентичные файлы не дублировались, а отличавшиеся версии исходника сохранены как provenance-снимки. Перенос не переоценивает научные результаты.

## Состав

- `experiments/`, `tests/`, `benchmarks/` и их материалы — в одноимённых каталогах `EDR_code`.
- `benchmarks/results/outputs/` и `verification/` — в архиве результатов и протоколов `EDR_code`.
- `tex/archive/`, `tex/data/`, `tex/result/` и история Grassmann — в соответствующих исследовательских разделах `EDR_code`.
- Экспериментальные CLI-помощники `ADP/cli/experiment*.py` сохранены там же в EDR_code.
- Различающиеся версии с M лежат в [`source-conflicts/`](source-conflicts/). Старые описания перенесены в [`README-M-research.md`](README-M-research.md).

## Недавние ветки исследования

- shared-polynomial refinement: [`experiments/shared_polynomial.py`](../../../benchmarks/experiments/shared_polynomial.py), [`verification/shared-polynomial-20261028/`](../../../benchmarks/results/verification/shared-polynomial-20261028/).
- Gaussian projector localization: [`experiments/projector_gaussian.py`](../../../benchmarks/experiments/projector_gaussian.py), [`verification/weak-localization-20261028/`](../../../benchmarks/results/verification/weak-localization-20261028/).
- local-score spectral work: [`experiments/local_score_spectral.py`](../../../benchmarks/experiments/local_score_spectral.py), [`verification/local-score-spectral-20261029/`](../../../benchmarks/results/verification/local-score-spectral-20261029/).
- anisotropic EDR guard: [`verification/anisotropic-edr-guard-20261029/`](../../../benchmarks/results/verification/anisotropic-edr-guard-20261029/).

## Аудит

`MOVE_MANIFEST.json` перечисляет 26,957 файлов: 1 archived-document, 1 archived-provenance, 23 conflict-snapshot, 3,689 copied, 23,243 identical-existing.
Для каждого файла сохранён исходный SHA-256 и маршрут: копия, совпадающий файл, conflict-снимок или архивированное описание. Тесты и численные серии в рамках этой структурной миграции не запускались.
