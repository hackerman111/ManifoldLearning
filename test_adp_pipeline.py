"""
CLI-клиент для ручного запуска учебного single-index ADP пайплайна.

Типовой запуск:
    python test_adp_pipeline.py
    python test_adp_pipeline.py --n 500 --d 20 --n-j 180 --summary-json run.json

Для ручных экспериментов можно менять EDITABLE_CONFIG ниже, а точечные
изменения передавать через аргументы командной строки.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
ADP_DIR = ROOT_DIR / "ADP"
if str(ADP_DIR) not in sys.path:
    sys.path.insert(0, str(ADP_DIR))

from algorithm import step0, stepk
from data import generation as data
from diagnostics import trace as trace_tools
from facades import myadp
from models.single_index import ADP_single_index
from pipeline import main as adp_pipeline
from runtime import monitoring as runtime_tools


# Основные настройки ручного запуска. Обычно достаточно менять этот блок.
EDITABLE_CONFIG = {
    "n": 1000,
    "d": 100,
    "seed": 12,
    "function": "sin",
    "data_type": "normal",
    "noise_std": 0.02,
    "n_J": 700,
    "n_directions": 100,
    "n_min": 100,
    "min_cosine": 0.75,
    "trace_output_dir": "adp_trace_test_outputs",
    "summary_json": None,
    "save_diagnostics": True,
    "show_progress": False,
    "use_rich": False,
    "log_runtime": False,
    "runtime_log_path": None,
    "compatibility_smoke": True,
    "class_smoke": True,
    "layout_check": True,
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("значение должно быть положительным")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("значение должно быть неотрицательным")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Запускает single-index ADP пайплайн на синтетических данных.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data_group = parser.add_argument_group("данные")
    data_group.add_argument("--n", type=positive_int, help="число наблюдений")
    data_group.add_argument("--d", type=positive_int, help="число признаков")
    data_group.add_argument("--seed", type=int, help="seed генератора")
    data_group.add_argument(
        "--function",
        choices=("linear", "quadratic", "cubic", "sin", "tanh", "exp", "step", "abs"),
        help="одномерная функция отклика",
    )
    data_group.add_argument(
        "--data-type",
        choices=("normal", "uniform", "correlated_normal", "sphere", "student", "mixture"),
        help="тип распределения X",
    )
    data_group.add_argument("--noise-std", type=nonnegative_float, help="std шума Y")

    adp_group = parser.add_argument_group("ADP")
    adp_group.add_argument("--n-j", dest="n_J", type=positive_int, help="число центров J")
    adp_group.add_argument(
        "--n-directions",
        type=positive_int,
        help="число случайных направлений на центр",
    )
    adp_group.add_argument(
        "--n-min",
        type=positive_int,
        help="минимальное эффективное число соседей",
    )
    adp_group.add_argument(
        "--min-cosine",
        type=float,
        help="порог успешности abs(cos(beta_hat, beta_true))",
    )

    output_group = parser.add_argument_group("вывод")
    output_group.add_argument(
        "--trace-output-dir",
        help="каталог для графиков и trace_summary.csv",
    )
    output_group.add_argument(
        "--summary-json",
        help="путь для сохранения JSON summary",
    )
    output_group.add_argument(
        "--dry-run",
        action="store_true",
        help="только показать итоговую конфигурацию",
    )
    output_group.add_argument(
        "--quiet",
        action="store_true",
        help="не печатать подробный отчет",
    )

    switches = parser.add_argument_group("переключатели")
    switches.add_argument(
        "--diagnostics",
        dest="save_diagnostics",
        action="store_true",
        help="сохранять графики и trace_summary.csv",
    )
    switches.add_argument(
        "--no-diagnostics",
        dest="save_diagnostics",
        action="store_false",
        help="не сохранять графики и trace_summary.csv",
    )
    switches.add_argument(
        "--progress",
        dest="show_progress",
        action="store_true",
        help="показывать tqdm progress",
    )
    switches.add_argument(
        "--no-progress",
        dest="show_progress",
        action="store_false",
        help="отключить progress",
    )
    switches.add_argument("--rich", dest="use_rich", action="store_true", help="использовать rich")
    switches.add_argument(
        "--runtime-log",
        dest="runtime_log_path",
        help="путь для runtime log; включает log_runtime",
    )
    switches.add_argument(
        "--compatibility-smoke",
        dest="compatibility_smoke",
        action="store_true",
        help="проверять совместимый facade myadp",
    )
    switches.add_argument(
        "--no-compatibility-smoke",
        dest="compatibility_smoke",
        action="store_false",
        help="не запускать smoke facade myadp",
    )
    switches.add_argument(
        "--class-smoke",
        dest="class_smoke",
        action="store_true",
        help="проверять ADP_single_index class API",
    )
    switches.add_argument(
        "--no-class-smoke",
        dest="class_smoke",
        action="store_false",
        help="не запускать smoke class API",
    )
    switches.add_argument(
        "--layout-check",
        dest="layout_check",
        action="store_true",
        help="проверять текущую структуру ADP/",
    )
    switches.add_argument(
        "--no-layout-check",
        dest="layout_check",
        action="store_false",
        help="не проверять структуру ADP/",
    )

    parser.set_defaults(
        save_diagnostics=None,
        show_progress=None,
        use_rich=None,
        compatibility_smoke=None,
        class_smoke=None,
        layout_check=None,
    )
    return parser


def merge_config(args: argparse.Namespace) -> dict:
    config = dict(EDITABLE_CONFIG)

    for name in (
        "n",
        "d",
        "seed",
        "function",
        "data_type",
        "noise_std",
        "n_J",
        "n_directions",
        "n_min",
        "min_cosine",
        "trace_output_dir",
        "summary_json",
        "save_diagnostics",
        "show_progress",
        "use_rich",
        "runtime_log_path",
        "compatibility_smoke",
        "class_smoke",
        "layout_check",
    ):
        value = getattr(args, name, None)
        if value is not None:
            config[name] = value

    if config["runtime_log_path"]:
        config["log_runtime"] = True

    return config


def assert_shape(name: str, value: np.ndarray, expected_shape: tuple[int, ...]) -> None:
    actual_shape = value.shape
    if actual_shape != expected_shape:
        raise AssertionError(
            f"{name}: ожидалась форма {expected_shape}, получена {actual_shape}"
        )


def make_test_data(config: dict):
    X, Y, beta, info = data.MakeData(
        n=config["n"],
        d=config["d"],
        function=config["function"],
        data_type=config["data_type"],
        noise_std=config["noise_std"],
        seed=config["seed"],
        return_info=True,
    )
    return X, Y, beta, info


def check_data_generation(config: dict):
    X, Y, beta, info = make_test_data(config)

    assert_shape("X", X, (config["n"], config["d"]))
    assert_shape("Y", Y, (config["n"],))
    assert_shape("beta", beta, (config["d"],))
    if not np.isclose(np.linalg.norm(beta), 1.0):
        raise AssertionError("beta должен быть нормирован")
    if info["function"] != config["function"]:
        raise AssertionError("info['function'] не совпадает с config['function']")
    if info["data_type"] != config["data_type"]:
        raise AssertionError("info['data_type'] не совпадает с config['data_type']")

    return X, Y, beta


def check_step0(config: dict, X: np.ndarray):
    state = step0.PrepareADPInitialState(
        X,
        n_J=config["n_J"],
        n_directions=config["n_directions"],
        n_min=config["n_min"],
        seed=config["seed"],
    )

    assert_shape("J", state["J"], (config["n_J"],))
    assert_shape("x_j", state["x_j"], (config["n_J"], config["d"]))
    assert_shape(
        "directions",
        state["directions"],
        (config["n_J"], config["n_directions"], config["d"]),
    )
    assert_shape("weights", state["weights"], (config["n_J"], config["n"]))
    if state["h0"] <= 0:
        raise AssertionError("h0 должен быть положительным")
    if state["weights"].sum(axis=1).mean() < config["n_min"] - 1e-6:
        raise AssertionError("средняя масса весов меньше n_min")

    return state


def check_stepk(config: dict, X: np.ndarray, Y: np.ndarray, state: dict):
    gradients = stepk.EstimateLocalGradients(
        X,
        Y,
        state["x_j"],
        state["h0"],
        weights=state["weights"],
    )

    assert_shape("local_gradients", gradients, (config["n_J"], config["d"]))
    if not np.all(np.isfinite(gradients)):
        raise AssertionError("градиенты должны быть конечными")

    return gradients


def check_main_pipeline(config: dict, X: np.ndarray, Y: np.ndarray, beta: np.ndarray):
    trace = trace_tools.CreateTrace(store_arrays=False)
    runtime_monitor = runtime_tools.CreateRuntimeMonitor(
        enabled=True,
        use_tqdm=config["show_progress"],
        use_rich=config["use_rich"],
        log_runtime=config["log_runtime"],
        log_path=config["runtime_log_path"],
    )
    result = adp_pipeline.RunADP(
        X,
        Y,
        n_J=config["n_J"],
        n_min=config["n_min"],
        n_directions=config["n_directions"],
        seed=config["seed"],
        trace=trace,
        runtime_monitor=runtime_monitor,
    )

    assert_shape("result beta", result["beta"], (config["d"],))
    assert_shape("result weights", result["weights"], (config["n_J"], config["n"]))
    assert_shape(
        "result local_gradients",
        result["local_gradients"],
        (config["n_J"], config["d"]),
    )

    cosine = stepk.CosineSimilarity(result["beta"], beta)
    if "trace" not in result:
        raise AssertionError("результат должен содержать trace")
    if len(result["trace"]["steps"]) < 5:
        raise AssertionError("trace должен содержать шаги процесса")
    if not any(step["name"] == "final_result" for step in result["trace"]["steps"]):
        raise AssertionError("trace должен содержать final_result")
    if "runtime" not in result:
        raise AssertionError("результат должен содержать runtime")
    if "total_pipeline" not in result["runtime"]["summary"]:
        raise AssertionError("runtime summary должен содержать total_pipeline")

    diagnostics = {}
    if config["save_diagnostics"]:
        diagnostics = trace_tools.SaveADPDiagnostics(
            result,
            output_dir=config["trace_output_dir"],
        )
        if not diagnostics.get("plots"):
            raise AssertionError("должен быть создан хотя бы один график")
        if "trace_summary" not in diagnostics:
            raise AssertionError("должен быть создан CSV summary трассировки")

    return result, cosine, diagnostics


def check_compatibility_facade(config: dict, X: np.ndarray, Y: np.ndarray) -> None:
    result = myadp.RunADP(
        X,
        Y,
        n_J=min(config["n_J"], 60),
        n_min=min(config["n_min"], 18),
        seed=config["seed"] + 1,
    )
    assert_shape("myadp beta", result["beta"], (config["d"],))


def check_single_index_class_api() -> None:
    config = {
        "n": 180,
        "d": 6,
        "seed": 21,
        "function": "sin",
        "data_type": "normal",
        "noise_std": 0.03,
        "n_J": 70,
        "n_directions": 6,
        "n_min": 18,
        "min_cosine": 0.75,
    }
    X, Y, beta, _ = data.MakeData(
        n=config["n"],
        d=config["d"],
        function=config["function"],
        data_type=config["data_type"],
        noise_std=config["noise_std"],
        seed=config["seed"],
        return_info=True,
    )

    model = ADP_single_index(
        n_J=config["n_J"],
        n_min=config["n_min"],
        n_directions=config["n_directions"],
        seed=config["seed"],
        trace_enabled=True,
        show_progress=False,
    )
    fitted = model.fit(X, Y)

    if fitted is not model:
        raise AssertionError("fit должен возвращать self")
    assert_shape("model beta_", model.beta_, (config["d"],))
    if model.h0_ <= 0:
        raise AssertionError("model.h0_ должен быть положительным")
    if not model.result_["trace"]["steps"]:
        raise AssertionError("класс должен сохранять trace при trace_enabled=True")

    projected = model.transform(X[:5])
    assert_shape("projected", projected, (5, 1))
    if not np.allclose(projected[:, 0], model.predict_index(X[:5])):
        raise AssertionError("transform и predict_index должны совпадать")

    cosine = model.score_direction(beta)
    if cosine < config["min_cosine"]:
        raise AssertionError(f"class API cosine={cosine:.3f}")

    generated_X, generated_Y, generated_beta = model.make_data(n=40, d=4, seed=3)
    assert_shape("generated_X", generated_X, (40, 4))
    assert_shape("generated_Y", generated_Y, (40,))
    assert_shape("generated_beta", generated_beta, (4,))


def check_layout() -> None:
    expected_files = [
        ADP_DIR / "data" / "generation.py",
        ADP_DIR / "algorithm" / "step0.py",
        ADP_DIR / "algorithm" / "stepk.py",
        ADP_DIR / "runtime" / "monitoring.py",
        ADP_DIR / "diagnostics" / "trace.py",
        ADP_DIR / "pipeline" / "main.py",
        ADP_DIR / "models" / "single_index.py",
        ADP_DIR / "facades" / "myadp.py",
    ]
    old_files = [
        "ADP_Data_Gen.py",
        "ADP_step0.py",
        "ADP_stepk.py",
        "ADP_Runtime.py",
        "ADP_Trace.py",
        "Main_ADP.py",
        "ADP_single_index.py",
        "MyADP.py",
    ]

    for path in expected_files:
        if not path.exists():
            raise AssertionError(f"ожидался файл {path.relative_to(ADP_DIR)}")
    for filename in old_files:
        if (ADP_DIR / filename).exists():
            raise AssertionError(f"старый фасад {filename} должен быть удален")


def jsonable_path_map(value):
    if isinstance(value, dict):
        return {key: jsonable_path_map(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable_path_map(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def run_experiment(config: dict) -> dict:
    X, Y, beta = check_data_generation(config)
    state = check_step0(config, X)
    check_stepk(config, X, Y, state)
    result, cosine, diagnostics = check_main_pipeline(config, X, Y, beta)

    checks = {
        "data_generation": True,
        "step0": True,
        "stepk": True,
        "main_pipeline": True,
        "diagnostics": bool(config["save_diagnostics"]),
        "compatibility_facade": False,
        "class_api": False,
        "layout": False,
    }

    if config["compatibility_smoke"]:
        check_compatibility_facade(config, X, Y)
        checks["compatibility_facade"] = True
    if config["class_smoke"]:
        check_single_index_class_api()
        checks["class_api"] = True
    if config["layout_check"]:
        check_layout()
        checks["layout"] = True

    ok = cosine >= config["min_cosine"]
    runtime_summary = result.get("runtime", {}).get("summary", {})
    return {
        "ok": bool(ok),
        "cosine": float(cosine),
        "min_cosine": float(config["min_cosine"]),
        "h0": float(result["h0"]),
        "beta_true": np.asarray(beta, dtype=float).tolist(),
        "beta_hat": np.asarray(result["beta"], dtype=float).tolist(),
        "runtime": {name: float(value) for name, value in runtime_summary.items()},
        "diagnostics": jsonable_path_map(diagnostics),
        "checks": checks,
        "config": jsonable_path_map(config),
    }


def format_vector(vector, max_items=8) -> str:
    vector = np.asarray(vector)
    if vector.size <= 2 * max_items:
        return str(np.round(vector, 4))

    head = np.round(vector[:max_items], 4)
    tail = np.round(vector[-max_items:], 4)
    return f"{head} ... {tail}  shape={vector.shape}"


def print_config(config: dict) -> None:
    for key in EDITABLE_CONFIG:
        print(f"{key} = {config[key]}")


def print_report(summary: dict) -> None:
    status = "OK" if summary["ok"] else "FAIL"
    print(f"ADP pipeline CLI: {status}")
    print(
        "cosine(beta_hat, beta_true) = "
        f"{summary['cosine']:.3f} / threshold {summary['min_cosine']:.3f}"
    )
    print(f"h0 = {summary['h0']:.6f}")
    print("beta_true =", format_vector(summary["beta_true"]))
    print("beta_hat  =", format_vector(summary["beta_hat"]))

    if summary["runtime"]:
        print("runtime:")
        for name, value in summary["runtime"].items():
            print(f"  {name}: {value:.4f}s")

    if summary["diagnostics"]:
        print("diagnostics:")
        for name, value in summary["diagnostics"].items():
            print(f"  {name}: {value}")


def write_summary(path: str, summary: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = merge_config(args)

    if args.dry_run:
        print_config(config)
        return 0

    try:
        summary = run_experiment(config)
    except Exception as exc:
        print(f"ADP pipeline CLI: ERROR: {exc}", file=sys.stderr)
        return 1

    if config["summary_json"]:
        write_summary(config["summary_json"], summary)

    if not args.quiet:
        print_report(summary)

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
