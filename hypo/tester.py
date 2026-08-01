from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HYPO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = HYPO_ROOT.parent


def discover_testers() -> dict[str, Path]:
    return {
        path.parent.name: path
        for path in sorted(HYPO_ROOT.glob("*/tester.py"))
    }


def root_parser(testers: dict[str, Path]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Запуск проверок гипотез ADP.",
        usage="%(prog)s [--list] [tester [args ...]]",
    )
    parser.add_argument("--list", action="store_true", help="показать доступные проверки")
    parser.epilog = f"Доступные проверки: {', '.join(testers) or 'нет'}"
    return parser


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    testers = discover_testers()
    parser = root_parser(testers)
    if args and not args[0].startswith("-"):
        name, *forwarded = args
        path = testers.get(name)
        if path is None:
            parser.error(
                f"unknown tester {name!r}; available: {', '.join(testers) or 'none'}"
            )
        return subprocess.run(
            [sys.executable, path, *forwarded], cwd=REPO_ROOT, check=False
        ).returncode
    if parser.parse_args(args).list:
        print("\n".join(testers))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
