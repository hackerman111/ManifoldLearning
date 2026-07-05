import logging
import time
from contextlib import contextmanager
from pathlib import Path


def _optional_import_tqdm():
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None

    return tqdm


def _optional_import_rich_console():
    try:
        from rich.console import Console
    except Exception:
        return None

    return Console


def _make_logger(name, log_path=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def CreateRuntimeMonitor(
    enabled=True,
    use_tqdm=True,
    use_rich=False,
    log_runtime=False,
    log_path=None,
    name="ADP",
):
    """
    Создает монитор выполнения ADP.

    use_tqdm включает progress-bar для длинных циклов.
    use_rich включает красивый console status, если библиотека установлена.
    log_runtime включает сообщения logging.
    """
    tqdm = _optional_import_tqdm() if use_tqdm else None
    Console = _optional_import_rich_console() if use_rich else None
    console = Console() if Console is not None else None
    logger = _make_logger(name, log_path=log_path) if log_runtime else None

    return {
        "enabled": enabled,
        "use_tqdm": use_tqdm and tqdm is not None,
        "use_rich": use_rich and console is not None,
        "tqdm": tqdm,
        "console": console,
        "logger": logger,
        "events": [],
        "active": [],
    }


def LogRuntimeEvent(monitor, name, event_type="event", **payload):
    """
    Записывает runtime-событие и при необходимости пишет его в logger/rich.
    """
    if monitor is None or not monitor.get("enabled", True):
        return monitor

    event = {
        "name": name,
        "type": event_type,
        "time": time.perf_counter(),
        "payload": payload,
    }
    monitor["events"].append(event)

    logger = monitor.get("logger")
    if logger is not None:
        logger.info("%s | %s | %s", event_type, name, payload)

    console = monitor.get("console")
    if console is not None and event_type in {"start", "finish"}:
        console.log(f"{event_type}: {name} {payload}")

    return monitor


@contextmanager
def RuntimeStage(monitor, name, **payload):
    """
    Контекстный менеджер для измерения длительности этапа.
    """
    if monitor is None or not monitor.get("enabled", True):
        yield monitor
        return

    started_at = time.perf_counter()
    LogRuntimeEvent(monitor, name, event_type="start", **payload)

    try:
        yield monitor
    finally:
        duration = time.perf_counter() - started_at
        LogRuntimeEvent(monitor, name, event_type="finish", duration=duration)


def IterateWithProgress(iterable, monitor=None, total=None, description=None):
    """
    Оборачивает iterable в tqdm, если монитор включен и tqdm доступен.
    """
    if monitor is None or not monitor.get("enabled", True):
        return iterable

    tqdm = monitor.get("tqdm")
    if tqdm is None:
        return iterable

    return tqdm(iterable, total=total, desc=description, leave=False)


def RuntimeSummary(monitor):
    """
    Возвращает компактную сводку длительностей по finish-событиям.
    """
    if monitor is None:
        return {}

    summary = {}

    for event in monitor.get("events", []):
        if event.get("type") != "finish":
            continue

        duration = event.get("payload", {}).get("duration")
        if duration is not None:
            summary[event["name"]] = duration

    return summary
