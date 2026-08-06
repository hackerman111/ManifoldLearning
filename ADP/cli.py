import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ADP import ADP_Config, ADP_single_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Консольная проверка ADP single-index")
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
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

    model = ADP_single_index(ADP_Config(seed=args.seed, lambda_penalty=100.0)).fit(X, Y)

    print("истинное beta: ", np.array2string(beta_true, precision=6))
    print("полученное бета:  ", np.array2string(model.beta_, precision=6))
    print(f"косинус: {model.score_direction(beta_true):.6f}")
    print(f"количество итераций: {len(model.trace_)}")
    print(f"причина остнановки: {model.stop_reason_}")


if __name__ == "__main__":
    main()
