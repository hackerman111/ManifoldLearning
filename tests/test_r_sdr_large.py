from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import pytest
from experiments.r_sdr.design import catalog


@pytest.mark.parametrize(
    ("profile", "n", "d"),
    [("fourteen-d50-n4000", 4000, 50), ("fourteen-d100-n8000", 8000, 100)],
)
def test_large_profiles_preserve_all_fourteen_cases(profile, n, d):
    originals = catalog("seven-families") + catalog("oscillation-seven")
    cases = catalog(profile)
    assert len(cases) == len({c.name for c in cases}) == 14
    assert cases == [replace(c, n=n, d=d, m=3) for c in originals]
    assert all((c.n, c.d, c.m) == (n, d, 3) for c in cases)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.r_sdr",
            "--profile",
            profile,
            "--runs",
            "15",
            "--seed",
            "0",
            "--methods",
            "ADP-Grassmann-combined,MAVE-package",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "14 cases x 15 repetitions x 2 methods = 420 fits" in result.stdout
