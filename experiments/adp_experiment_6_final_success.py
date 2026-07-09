from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.adp_confirmatory_common import run_experiment_cli


def main() -> int:
    return run_experiment_cli(
        description="ADP experiment 6: final success protocol from Tests.md.",
        default_out=Path("outputs/adp_experiment_6_final_success"),
        experiments=("6",),
        output_prefix="experiment_6_final_success",
        experiment_label="ADP experiment 6: final success protocol",
    )


if __name__ == "__main__":
    raise SystemExit(main())
