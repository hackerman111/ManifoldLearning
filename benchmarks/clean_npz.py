"""Интерактивная очистка локальных NPZ экспериментов."""

from __future__ import annotations

import argparse
import curses
import os
import stat
import subprocess
import sys
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent / "results" / "outputs"


def tracked_npz(root: Path) -> set[Path]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--cached", "--", ":(icase)*.npz"],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        detail = os.fsdecode(result.stderr).strip()
        raise RuntimeError(f"Ошибка чтения индекса Git. {detail}")
    return {root / os.fsdecode(name) for name in result.stdout.split(b"\0") if name}


def scan(root: Path, tracked: set[Path]) -> dict[Path, os.stat_result]:
    def report_error(error: OSError) -> None:
        raise error

    files = {}
    for current, dirs, names in os.walk(root, onerror=report_error):
        directory = Path(current)
        if directory != root and (".git" in dirs or ".git" in names):
            dirs.clear()
            continue
        if ".git" in dirs:
            dirs.remove(".git")
        for name in names:
            path = directory / name
            if path.suffix.lower() != ".npz" or path in tracked:
                continue
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISREG(info.st_mode):
                files[path] = info
    return files


def delete_selected(
    selected: set[Path], snapshots: dict[Path, os.stat_result], root: Path
) -> tuple[int, list[str]]:
    tracked = tracked_npz(root)
    removed = 0
    skipped = []
    for path in sorted(selected):
        name = path.relative_to(root)
        try:
            before = snapshots[path]
            now = path.lstat()
            if path in tracked:
                skipped.append(f"{name} теперь под Git")
            elif (
                not stat.S_ISREG(now.st_mode)
                or not os.path.samestat(before, now)
                or before.st_size != now.st_size
                or before.st_mtime_ns != now.st_mtime_ns
                or path.resolve(strict=True) != path
            ):
                skipped.append(f"{name} изменился после сканирования")
            else:
                path.unlink()
                removed += 1
        except (OSError, RuntimeError) as error:
            skipped.append(f"{name} — {error}")
    return removed, skipped


@dataclass(frozen=True, slots=True)
class Row:
    path: Path
    files: tuple[Path, ...]
    size: int


def rows_at(directory: Path, files: dict[Path, os.stat_result]) -> list[Row]:
    groups: dict[Path, list[Path]] = defaultdict(list)
    for path in files:
        if path.is_relative_to(directory):
            child = directory / path.relative_to(directory).parts[0]
            groups[child].append(path)
    rows = [
        Row(child, tuple(paths), sum(files[path].st_size for path in paths))
        for child, paths in groups.items()
    ]
    return sorted(rows, key=lambda row: (-row.size, row.path.name))


def size_label(size: int) -> str:
    amount = float(size)
    for unit in ("Б", "КиБ", "МиБ"):
        if amount < 1024:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} ГиБ"


def draw(screen: curses.window, line: int, text: str, style: int = 0) -> None:
    clean = "".join(char if char.isprintable() else "?" for char in text)
    with suppress(curses.error):
        screen.addnstr(line, 0, clean, screen.getmaxyx()[1] - 1, style)


def browse(screen: curses.window, root: Path) -> None:
    with suppress(curses.error):
        curses.curs_set(0)
    curses.set_escdelay(50)
    files = scan(root, tracked_npz(root))
    directory = root
    rows = rows_at(directory, files)
    selected: set[Path] = set()
    cursor = start = 0
    confirming = False
    status = ""

    while True:
        height, width = screen.getmaxyx()
        screen.erase()
        if height < 10 or width < 50:
            draw(screen, 0, "Увеличьте окно. q — выход")
            if screen.getch() == ord("q"):
                return
            continue

        page = height - 6
        movement = {
            curses.KEY_UP: -1,
            curses.KEY_PPAGE: -page,
            curses.KEY_DOWN: 1,
            curses.KEY_NPAGE: page,
        }
        start = min(max(start, cursor - page + 1), cursor)
        total = size_label(sum(files[path].st_size for path in selected))
        draw(screen, 0, f"NPZ  {directory}", curses.A_BOLD)
        draw(screen, 1, f"Выбрано · {len(selected)} шт. · {total}")
        if not rows:
            draw(screen, 2, "Нет локальных NPZ для очистки")
        for index, row in enumerate(rows[start : start + page], start):
            marked = sum(path in selected for path in row.files)
            mark = "x" if marked == len(row.files) else "~" if marked else " "
            name = row.path.name
            if row.path not in files:
                name += f"/ ({len(row.files)})"
            draw(
                screen,
                index - start + 2,
                f"[{mark}] {size_label(row.size):>12}  {name}",
                curses.A_REVERSE if index == cursor else 0,
            )

        draw(screen, height - 4, status)
        if confirming:
            prompt = f"Удалить {len(selected)} шт. · {total}?"
            draw(screen, height - 3, prompt, curses.A_BOLD)
            draw(screen, height - 2, "y Удалить безвозвратно    Esc Отмена")
        else:
            draw(screen, height - 3, "↑↓ Выбор  Enter Открыть  ← Назад  Space Отметить")
            draw(screen, height - 2, "a Отметить папку  d Удалить  r Обновить  q Выход")
        key = screen.getch()
        if key == ord("q"):
            return
        if confirming and key != ord("y"):
            if key in (27, ord("n")):
                confirming = False
            continue
        if confirming or key == ord("r"):
            if confirming:
                count, skipped = delete_selected(selected, files, root)
                status = f"Удалено {count} · пропущено {len(skipped)}"
                if skipped:
                    status += f" · {skipped[0]}"
            else:
                status = "Список обновлён"
            confirming = False
            selected.clear()
            files = scan(root, tracked_npz(root))
            if not directory.is_dir():
                directory = root
            rows = rows_at(directory, files)
            cursor = start = 0
        elif key in movement:
            cursor = max(0, min(cursor + movement[key], len(rows) - 1))
        elif key in (curses.KEY_LEFT, curses.KEY_BACKSPACE, 127) and directory != root:
            previous = directory
            directory = directory.parent
            rows = rows_at(directory, files)
            cursor = next((i for i, row in enumerate(rows) if row.path == previous), 0)
            start = 0
        elif rows and key in (10, 13, curses.KEY_ENTER, curses.KEY_RIGHT):
            target = rows[cursor].path
            if target not in files:
                directory = target
                rows = rows_at(directory, files)
                cursor = start = 0
        elif rows and key in (ord(" "), ord("a")):
            targets = (
                set(rows[cursor].files)
                if key == ord(" ")
                else {path for row in rows for path in row.files}
            )
            if targets <= selected:
                selected.difference_update(targets)
            else:
                selected.update(targets)
        elif key == ord("d") and selected:
            confirming = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help="каталог внутри Git-репозитория (по умолчанию benchmarks/results/outputs)",
    )
    root = parser.parse_args().directory.resolve()
    if not root.is_dir() or ".git" in root.parts:
        parser.error(f"Ожидается каталог результатов, получен {root}")
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        parser.error("Запустите скрипт в интерактивном терминале")
    try:
        curses.wrapper(browse, root)
    except (OSError, RuntimeError, curses.error) as error:
        parser.exit(1, f"Ошибка очистки NPZ. {error}\n")
    except KeyboardInterrupt:
        parser.exit(130)


if __name__ == "__main__":
    main()
