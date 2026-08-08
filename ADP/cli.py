import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ADP import ADP_Config, ADP_single_index
from ADP.ADP_Config import epanechnikov
from ADP.engine.logger import format_profile


def parse_kernel(value: str):
    if value == "epanechnikov":
        return epanechnikov
    try:
        module_name, function_name = value.rsplit(":", 1)
        kernel = getattr(importlib.import_module(module_name), function_name)
    except (AttributeError, ImportError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "kernel должен быть 'epanechnikov' или 'module:function'"
        ) from error
    if not callable(kernel):
        raise argparse.ArgumentTypeError("kernel должен быть функцией")
    return kernel

def parse_solver(value:str):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Консольная проверка ADP single-index",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = parser.add_argument_group("тестовые данные")
    data.add_argument("--n", type=int, default=240)
    data.add_argument("--d", type=int, default=3)
    data.add_argument("--noise", type=float, default=0.05)

    adp = parser.add_argument_group("ADP_Config")
    adp.add_argument("--seed", type=int, default=7)
    adp.add_argument("--N_loc", "--n-loc", dest="N_loc", type=int, default=10)
    adp.add_argument("--N_lin", "--n-lin", dest="N_lin", type=int)
    adp.add_argument("--N_J", "--J", dest="N_J", type=int, default=64)
    adp.add_argument("--N_phi", "--n-phi", dest="N_phi", type=int)
    adp.add_argument("--outer_steps", "--outer-steps", dest="outer_steps", type=int)
    adp.add_argument(
        "--lambda_penalty",
        "--lambda-penalty",
        dest="lambda_penalty",
        type=float,
        default=100.0,
    )
    adp.add_argument(
        "--local_ridge",
        "--local-ridge",
        dest="local_ridge",
        type=float,
        default=1e-8,
    )
    adp.add_argument("--kernel", type=parse_kernel, default="epanechnikov")
    adp.add_argument("--a", type=float, default=np.sqrt(2))
    adp.add_argument("--h_min", "--h-min", dest="h_min", type=float)
    adp.add_argument(
        "--batch_size",
        "--batch-size",
        dest="batch_size",
        type=int,
        default=32,
    )
    adp.add_argument(
        "--index_init",
        "--index-init",
        dest="index_init",
        choices=("local", "random"),
        default="local",
    )
    args = parser.parse_args()

    if args.d < 1 or args.n <= args.d + 1:
        parser.error("требуется d >= 1 и n > d + 1")
    if not np.isfinite(args.noise) or args.noise < 0:
        parser.error("noise должен быть конечным и неотрицательным")
    if args.seed < 0:
        parser.error("seed не может быть отрицательным")

    rng = np.random.default_rng(args.seed)
    X = rng.normal(size=(args.n, args.d))
    beta_true = rng.normal(size=args.d)
    beta_true /= np.linalg.norm(beta_true)
    Y = np.sin(X @ beta_true) + args.noise * rng.normal(size=args.n)

    try:
        config = ADP_Config(
            seed=args.seed,
            N_loc=args.N_loc,
            N_lin=args.N_lin,
            N_J=args.N_J,
            N_phi=args.N_phi,
            outer_steps=args.outer_steps,
            lambda_penalty=args.lambda_penalty,
            local_ridge=args.local_ridge,
            kernel=args.kernel,
            a=args.a,
            h_min=args.h_min,
            batch_size=args.batch_size,
            index_init=args.index_init,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    model = ADP_single_index(config).fit(X, Y)
    result = model.result_
    result.Set_beta_true(beta_true)
    result.Calculate_cosine()

    print(f"косинус в начале: {result.cosine_init:.6f}")
    print(f"косинус в конце: {result.cosine_final:.6f}")
    print(f"количество центров N_J: {model.effective_parameters_['N_J']}")
    print(f"количество итераций: {len(result.trace)}")
    print(f"причина остановки: {result.stop_reason}")
    print()
    print(format_profile(model.profile_))


if __name__ == "__main__":
    main()
