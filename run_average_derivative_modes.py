import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from adp.edr_adp_single_index import EDR_ADP_single_index


MODE_CONFIGS = {
    "small": {
        "n_samples": 64,
        "n_features": 5,
        "n_centers": 8,
        "n_directions": 4,
        "min_neighbors": 2,
        "max_outer_steps": 2,
        "random_starts": 2,
        "max_iter": 12,
        "h_search_steps": 8,
        "rho_search_steps": 8,
    },
    "medium": {
        "n_samples": 180,
        "n_features": 40,
        "n_centers": 18,
        "n_directions": 5,
        "min_neighbors": 4,
        "max_outer_steps": 2,
        "random_starts": 3,
        "max_iter": 10,
        "h_search_steps": 8,
        "rho_search_steps": 8,
    },
    "large": {
        "n_samples": 400,
        "n_features": 300,
        "n_centers": 24,
        "n_directions": 4,
        "min_neighbors": 5,
        "max_outer_steps": 2,
        "random_starts": 2,
        "max_iter": 7,
        "h_search_steps": 7,
        "rho_search_steps": 7,
    },
}


def Run_Folder_Create(output_dir, run_name, overwrite):
    # Вход:
    # - output_dir: базовая папка для всех прогонов;
    # - run_name: имя текущего запуска;
    # - overwrite: можно ли переиспользовать существующую папку.
    #
    # Выход:
    # - path к папке текущего запуска.
    #
    # Что делает:
    # - создает корневую папку, куда будут сохранены все режимы.
    #
    # Реализация:
    # - при overwrite=False добавляет числовой суффикс, чтобы не потереть старые данные.

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_folder = output_path / run_name

    if overwrite:
        run_folder.mkdir(parents=True, exist_ok=True)

        return run_folder

    if not run_folder.exists():
        run_folder.mkdir(parents=True)

        return run_folder

    # Ищем свободное имя запуска рядом с уже существующей папкой.
    for suffix_index in range(1, 1000):
        candidate_folder = output_path / f"{run_name}_{suffix_index:03d}"

        if not candidate_folder.exists():
            candidate_folder.mkdir(parents=True)

            return candidate_folder

    raise RuntimeError("Could not create a unique run folder.")


def Modes_Select(mode):
    # Вход:
    # - mode: один режим или строка "all".
    #
    # Выход:
    # - список режимов для запуска.
    #
    # Что делает:
    # - переводит CLI-выбор в конкретный порядок прогонов.
    #
    # Реализация:
    # - порядок small -> medium -> large фиксирован для воспроизводимости.

    if mode == "all":
        return ["small", "medium", "large"]

    return [mode]


def Model_Build(mode_name, mode_config, seed, dtype):
    # Вход:
    # - имя режима, словарь параметров, seed и dtype.
    #
    # Выход:
    # - объект EDR_ADP_single_index.
    #
    # Что делает:
    # - собирает полный average derivative алгоритм для выбранного масштаба.
    #
    # Реализация:
    # - все режимы используют одинаковые параметры устойчивости и отличаются только размером.

    mode_seed = int(seed) + list(MODE_CONFIGS).index(mode_name)

    model = EDR_ADP_single_index(
        **mode_config,
        h_decay=1.6,
        h_min=0.0,
        rho_min=1e-2,
        ridge=1e-6,
        tol=1e-6,
        noise_scale=0.05,
        dtype=dtype,
        random_state=mode_seed,
    )

    return model


def Cosine_Calculate(beta, beta_hat):
    # Вход:
    # - beta: истинное направление;
    # - beta_hat: оцененное направление.
    #
    # Выход:
    # - absolute cosine между beta и beta_hat или None.
    #
    # Что делает:
    # - считает качество восстановления направления без зависимости от знака.
    #
    # Реализация:
    # - использует обычную евклидову норму и защищается от вырожденных векторов.

    if beta is None or beta_hat is None:
        return None

    beta = np.asarray(beta, dtype=float).reshape(-1)
    beta_hat = np.asarray(beta_hat, dtype=float).reshape(-1)

    if beta.shape != beta_hat.shape:
        return None

    beta_norm = np.linalg.norm(beta)
    beta_hat_norm = np.linalg.norm(beta_hat)

    if beta_norm <= 0 or beta_hat_norm <= 0:
        return None

    cosine = float(beta @ beta_hat / (beta_norm * beta_hat_norm))

    return abs(cosine)


def JSON_Save(path, data):
    # Вход:
    # - path: файл для сохранения;
    # - data: JSON-compatible структура.
    #
    # Выход:
    # - path к записанному файлу.
    #
    # Что делает:
    # - сохраняет словари и списки в читаемом JSON.
    #
    # Реализация:
    # - ensure_ascii=False оставляет русские подписи читаемыми.

    path = Path(path)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return path


def Data_Save(mode_folder, model, beta_hat):
    # Вход:
    # - папка режима;
    # - обученная модель;
    # - beta_hat после полного алгоритма.
    #
    # Выход:
    # - path к data.npz.
    #
    # Что делает:
    # - сохраняет все основные массивы запуска для дальнейшего анализа.
    #
    # Реализация:
    # - np.savez_compressed хранит raw данные генератора и массивы, с которыми работал алгоритм.

    data_path = mode_folder / "data.npz"

    np.savez_compressed(
        data_path,
        raw_X=model.Data.X,
        raw_Y=model.Data.Y,
        raw_beta=model.Data.beta,
        raw_centers=model.Data.centers,
        algorithm_X=model.X,
        algorithm_Y=model.Y,
        algorithm_beta=model.beta,
        algorithm_centers=model.centers,
        beta_hat=beta_hat,
        mean=model.mean,
        scale=model.scale,
    )

    return data_path


def Summary_CSV_Save(path, records):
    # Вход:
    # - path: файл summary.csv;
    # - records: список словарей по режимам.
    #
    # Выход:
    # - path к записанному CSV.
    #
    # Что делает:
    # - сохраняет табличную сводку всех прогонов.
    #
    # Реализация:
    # - поля берутся из первого record, потому что все режимы пишут один формат.

    path = Path(path)

    if not records:
        path.write_text("", encoding="utf-8")

        return path

    fieldnames = list(records[0].keys())

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Записываем по одной строке на каждый завершенный режим.
        for record in records:
            writer.writerow(record)

    return path


def Mode_Run(mode_name, mode_config, run_folder, seed, dtype, use_latex):
    # Вход:
    # - mode_name и mode_config для выбранного масштаба;
    # - run_folder: корневая папка запуска;
    # - seed, dtype, use_latex из CLI.
    #
    # Выход:
    # - summary record по завершенному режиму.
    #
    # Что делает:
    # - запускает полный average derivative алгоритм и сохраняет все артефакты режима.
    #
    # Реализация:
    # - модель сама генерирует данные, считает beta_hat и пишет характеристики через Run_Save.

    mode_folder = run_folder / mode_name
    mode_folder.mkdir(parents=True, exist_ok=True)

    model = Model_Build(mode_name, mode_config, seed, dtype)

    start_time = time.perf_counter()
    beta_hat = model.run_average_derivative()
    elapsed_seconds = time.perf_counter() - start_time

    cosine = Cosine_Calculate(model.beta, beta_hat)
    characteristics = model.Characteristics.Characteristics_Get()

    data_path = Data_Save(mode_folder, model, beta_hat)

    characteristics_folder = model.Characteristics.Run_Save(
        mode_folder,
        run_name="characteristics",
        use_latex=use_latex,
        overwrite=True,
    )

    config_path = JSON_Save(
        mode_folder / "config.json",
        {
            "mode": mode_name,
            "seed": int(seed) + list(MODE_CONFIGS).index(mode_name),
            "dtype": str(np.dtype(dtype)),
            "parameters": mode_config,
        },
    )

    result = {
        "mode": mode_name,
        "n_samples": model.n_samples,
        "n_features": model.n_features,
        "n_centers": model.n_centers,
        "n_directions": model.n_directions,
        "outer_steps": len(characteristics["h_k_values"]),
        "elapsed_seconds": elapsed_seconds,
        "cosine_abs": cosine,
        "beta_norm": float(np.linalg.norm(model.beta)),
        "beta_hat_norm": float(np.linalg.norm(beta_hat)),
        "last_objective": model.last_objective,
        "data_path": str(data_path),
        "config_path": str(config_path),
        "characteristics_path": str(characteristics_folder),
    }

    JSON_Save(mode_folder / "result.json", result)

    return result


def Manifest_Save(run_folder, run_name, modes, seed, dtype, use_latex, records):
    # Вход:
    # - параметры всего запуска и records по режимам.
    #
    # Выход:
    # - path к manifest.json.
    #
    # Что делает:
    # - сохраняет машинно-читаемое описание полного запуска.
    #
    # Реализация:
    # - manifest связывает summary и папки отдельных режимов.

    manifest = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "modes": modes,
        "seed": seed,
        "dtype": str(np.dtype(dtype)),
        "use_latex": use_latex,
        "records_count": len(records),
        "records": records,
    }

    return JSON_Save(run_folder / "manifest.json", manifest)


def Arguments_Parse(argv=None):
    # Вход:
    # - аргументы командной строки.
    #
    # Выход:
    # - argparse.Namespace.
    #
    # Что делает:
    # - задает CLI для полного запуска average derivative в трех режимах.
    #
    # Реализация:
    # - по умолчанию запускает все режимы и сохраняет результат в average_derivative_runs.

    parser = argparse.ArgumentParser(
        description="Run full average derivative algorithm in small, medium and large modes.",
    )

    parser.add_argument(
        "--mode",
        choices=["all", "small", "medium", "large"],
        default="all",
        help="Which mode to run.",
    )
    parser.add_argument(
        "--output",
        default="average_derivative_runs",
        help="Directory where run artifacts will be saved.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run folder name. If omitted, a timestamped name is used.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260709,
        help="Base random seed.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float64",
        help="Numeric dtype for generated data and calculations.",
    )
    parser.add_argument(
        "--no-latex",
        action="store_true",
        help="Disable LaTeX rendering for characteristic plots.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow reusing an existing run folder.",
    )

    return parser.parse_args(argv)


def main(argv=None):
    # Вход:
    # - argv: optional список CLI-аргументов для программного запуска.
    #
    # Выход:
    # - integer exit code.
    #
    # Что делает:
    # - запускает выбранные режимы average derivative и сохраняет данные.
    #
    # Реализация:
    # - каждый режим считается независимо, затем пишется общий summary и manifest.

    arguments = Arguments_Parse(argv)

    run_name = arguments.run_name

    if run_name is None:
        run_name = "average_derivative_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    selected_modes = Modes_Select(arguments.mode)
    dtype = np.dtype(arguments.dtype)
    use_latex = not arguments.no_latex

    run_folder = Run_Folder_Create(
        arguments.output,
        run_name,
        arguments.overwrite,
    )

    records = []

    # Последовательно запускаем выбранные режимы, чтобы пик памяти был только от одного режима.
    for mode_name in selected_modes:
        print(f"Running {mode_name} mode...")

        record = Mode_Run(
            mode_name,
            MODE_CONFIGS[mode_name],
            run_folder,
            arguments.seed,
            dtype,
            use_latex,
        )
        records.append(record)

        print(
            f"{mode_name}: "
            f"cosine_abs={record['cosine_abs']:.6f}, "
            f"time={record['elapsed_seconds']:.3f}s"
        )

    Summary_CSV_Save(run_folder / "summary.csv", records)
    JSON_Save(run_folder / "summary.json", records)
    Manifest_Save(
        run_folder,
        run_folder.name,
        selected_modes,
        arguments.seed,
        dtype,
        use_latex,
        records,
    )

    print(f"Saved run to {run_folder}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
