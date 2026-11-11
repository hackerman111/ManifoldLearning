# Различающиеся файлы

По прямому указанию пользователя EDR_code имеет приоритет. Все 38 отличавшихся
файлов M сохранены здесь как оригинальные снимки; рабочие версии EDR_code
сохранены. Из M выборочно добавлены явный CLI Grassmann, два новых R-профиля
и fallback для базового метода парного отчёта. Исторические данные не изменены.

Численные реализации ADP, default LSMR и все остальные солверы EDR_code
не заменены урезанной CPU-версией M. Различавшиеся тесты также сохранены
как снимки; рабочие тесты EDR_code не перезаписывались. Новый импорт
`ADP/core/solver_common.py` адаптирован к существующему классу результата;
его оригинал находится в source-adapted/ADP/core/solver_common.py.

| Рабочая версия EDR_code | Исходная версия M |
|---|---|
| [.gitignore](../../../.gitignore) | [версия M](source-conflicts/.gitignore) |
| [ADP/cli/experiment.py](../../../ADP/cli/experiment.py) | [версия M](source-conflicts/ADP/cli/experiment.py) |
| [ADP/cli/experiment_utils.py](../../../ADP/cli/experiment_utils.py) | [версия M](source-conflicts/ADP/cli/experiment_utils.py) |
| [ADP/cli/main.py](../../../ADP/cli/main.py) | [версия M](source-conflicts/ADP/cli/main.py) |
| [ADP/core/ADP_Config.py](../../../ADP/core/ADP_Config.py) | [версия M](source-conflicts/ADP/core/ADP_Config.py) |
| [ADP/core/ADP_Solver.py](../../../ADP/core/ADP_Solver.py) | [версия M](source-conflicts/ADP/core/ADP_Solver.py) |
| [ADP/core/manifold/ADP_Manifold.py](../../../ADP/core/manifold/ADP_Manifold.py) | [версия M](source-conflicts/ADP/core/manifold/ADP_Manifold.py) |
| [ADP/core/manifold/ADP_Manifold_utils.py](../../../ADP/core/manifold/ADP_Manifold_utils.py) | [версия M](source-conflicts/ADP/core/manifold/ADP_Manifold_utils.py) |
| [ADP/core/multi/ADP_multi_index.py](../../../ADP/core/multi/ADP_multi_index.py) | [версия M](source-conflicts/ADP/core/multi/ADP_multi_index.py) |
| [ADP/core/multi/ADP_multi_index_utils.py](../../../ADP/core/multi/ADP_multi_index_utils.py) | [версия M](source-conflicts/ADP/core/multi/ADP_multi_index_utils.py) |
| [ADP/core/single/ADP_single_index.py](../../../ADP/core/single/ADP_single_index.py) | [версия M](source-conflicts/ADP/core/single/ADP_single_index.py) |
| [ADP/core/single/ADP_single_index_utils.py](../../../ADP/core/single/ADP_single_index_utils.py) | [версия M](source-conflicts/ADP/core/single/ADP_single_index_utils.py) |
| [ADP/engine/common/index_fit.py](../../../ADP/engine/common/index_fit.py) | [версия M](source-conflicts/ADP/engine/common/index_fit.py) |
| [ADP/engine/manifol_engine/optimisation.py](../../../ADP/engine/manifol_engine/optimisation.py) | [версия M](source-conflicts/ADP/engine/manifol_engine/optimisation.py) |
| [ADP/solver/LSMR.py](../../../ADP/solver/LSMR.py) | [версия M](source-conflicts/ADP/solver/LSMR.py) |
| [ADP/solver/grassman.py](../../../ADP/solver/grassman.py) | [версия M](source-conflicts/ADP/solver/grassman.py) |
| [benchmarks/grassman_benchmark.py](../../../benchmarks/grassman_benchmark.py) | [версия M](source-conflicts/benchmarks/grassman_benchmark.py) |
| [experiments/diagnostic.py](../../../benchmarks/experiments/diagnostic.py) | [версия M](source-conflicts/experiments/diagnostic.py) |
| [experiments/grassman_accuracy_2026_10_18/backtracking.py](../../../benchmarks/experiments/grassman_accuracy_2026_10_18/backtracking.py) | [версия M](source-conflicts/experiments/grassman_accuracy_2026_10_18/backtracking.py) |
| [experiments/grassman_accuracy_2026_10_18/candidate.py](../../../benchmarks/experiments/grassman_accuracy_2026_10_18/candidate.py) | [версия M](source-conflicts/experiments/grassman_accuracy_2026_10_18/candidate.py) |
| [experiments/grassman_accuracy_2026_10_18/constant_centers.py](../../../benchmarks/experiments/grassman_accuracy_2026_10_18/constant_centers.py) | [версия M](source-conflicts/experiments/grassman_accuracy_2026_10_18/constant_centers.py) |
| [experiments/grassman_accuracy_2026_10_18/relative_gn.py](../../../benchmarks/experiments/grassman_accuracy_2026_10_18/relative_gn.py) | [версия M](source-conflicts/experiments/grassman_accuracy_2026_10_18/relative_gn.py) |
| [experiments/models.py](../../../benchmarks/experiments/models.py) | [версия M](source-conflicts/experiments/models.py) |
| [experiments/multi.py](../../../benchmarks/experiments/multi.py) | [версия M](source-conflicts/experiments/multi.py) |
| [experiments/r_sdr/README.md](../../../benchmarks/experiments/r_sdr/README.md) | [версия M](source-conflicts/experiments/r_sdr/README.md) |
| [experiments/r_sdr/__main__.py](../../../benchmarks/experiments/r_sdr/__main__.py) | [версия M](source-conflicts/experiments/r_sdr/__main__.py) |
| [experiments/r_sdr/design.py](../../../benchmarks/experiments/r_sdr/design.py) | [версия M](source-conflicts/experiments/r_sdr/design.py) |
| [experiments/r_sdr/report.py](../../../benchmarks/experiments/r_sdr/report.py) | [версия M](source-conflicts/experiments/r_sdr/report.py) |
| [experiments/r_sdr/runner.py](../../../benchmarks/experiments/r_sdr/runner.py) | [версия M](source-conflicts/experiments/r_sdr/runner.py) |
| [experiments/r_sdr/worker.py](../../../benchmarks/experiments/r_sdr/worker.py) | [версия M](source-conflicts/experiments/r_sdr/worker.py) |
| [experiments/single.py](../../../benchmarks/experiments/single.py) | [версия M](source-conflicts/experiments/single.py) |
| [experiments/suite.py](../../../benchmarks/experiments/suite.py) | [версия M](source-conflicts/experiments/suite.py) |
| [pyproject.toml](../../../pyproject.toml) | [версия M](source-conflicts/pyproject.toml) |
| [tests/test_cli.py](../../../tests/test_cli.py) | [версия M](source-conflicts/tests/test_cli.py) |
| [tests/test_experiment_suites.py](../../../tests/test_experiment_suites.py) | [версия M](source-conflicts/tests/test_experiment_suites.py) |
| [tests/test_grassman.py](../../../tests/test_grassman.py) | [версия M](source-conflicts/tests/test_grassman.py) |
| [tests/test_index_models.py](../../../tests/test_index_models.py) | [версия M](source-conflicts/tests/test_index_models.py) |
| [tests/test_r_sdr.py](../../../tests/test_r_sdr.py) | [версия M](source-conflicts/tests/test_r_sdr.py) |
