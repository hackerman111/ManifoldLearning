from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.adp_confirmatory_common import run_experiment_cli


def main() -> int:
    return run_experiment_cli(
        description="ADP experiment 5: cosine growth and ablations from Tests.md.",
        default_out=Path("outputs/adp_experiment_5_cos_growth"),
        experiments=("5",),
        output_prefix="experiment_5_cos_growth",
        experiment_label="ADP experiment 5: cosine growth and ablations",
    )


if __name__ == "__main__":
    raise SystemExit(main())
