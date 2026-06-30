from __future__ import annotations

import argparse

from adp import benchmark_summary, default_scenarios, run_benchmark_suite, save_benchmark_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ADP benchmark scenarios against ready EDR baselines.")
    parser.add_argument("--output", default="benchmark_reports", help="Directory for CSV and PNG reports.")
    parser.add_argument("--quick", action="store_true", help="Use a smaller smoke benchmark.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["adp_new", "adp_old", "statsmodels_sir", "statsmodels_save", "statsmodels_phd", "sklearn_pls"],
        help="Methods to compare.",
    )
    args = parser.parse_args()

    scenarios = default_scenarios(quick=args.quick)
    frame = run_benchmark_suite(scenarios, methods=args.methods, random_state=args.seed, show_progress=True)
    saved = save_benchmark_report(frame, args.output)

    print(benchmark_summary(frame).to_string(index=False))
    print("\nSaved:")
    for name, path in saved.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
