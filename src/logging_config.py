from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from models import AppConfig
from utils.runtime_logging import (
    RuntimeContextFilter,
    NormalizedMessageFilter,
    _build_compact_formatter,
    _build_formatter,
)

_THIRD_PARTY_LOGGERS = [
    "httpx",
    "httpcore",
    "telegram",
    "telegram.ext",
    "telegram.request",
    "telegram.ext.Application",
    "faster_whisper",
    "huggingface_hub",
    "urllib3",
]


class ConsoleNoiseFilter(logging.Filter):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self._last_seen: dict[tuple[str, int, str, str], float] = {}
        self._lock = threading.Lock()

    def filter(self, record: logging.LogRecord) -> bool:
        if not _passes_third_party_threshold(record, self.config.logging.third_party_level):
            return False
        if not _passes_segment_caption_mode(record, self.config):
            return False
        if not _passes_repeated_suppression(record, self.config, self._last_seen, self._lock):
            return False
        return True


def configure_logging(log_dir: Path, *, config: AppConfig | None = None, debug: bool | None = None) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    job_log_dir = log_dir / "jobs"
    job_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "video_processing.log"
    error_path = log_dir / "errors.log"
    current_config = config or AppConfig()

    console_level = _resolve_level(
        current_config.logging.console_level or current_config.logging.level,
        fallback=logging.INFO if debug is None else (logging.DEBUG if debug else logging.INFO),
    )
    file_level = _resolve_level(
        current_config.logging.file_level or current_config.logging.level,
        fallback=logging.DEBUG if debug is None else (logging.DEBUG if debug else logging.INFO),
    )
    third_party_level = _resolve_level(current_config.logging.third_party_level, fallback=logging.WARNING)

    root = logging.getLogger()
    root.setLevel(min(console_level, file_level, third_party_level))
    root.handlers.clear()
    root.filters.clear()
    root.addFilter(RuntimeContextFilter())
    root.addFilter(NormalizedMessageFilter())

    for name in _THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(third_party_level)

    detailed_formatter = _build_formatter()
    compact_formatter = _build_compact_formatter()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(compact_formatter if current_config.logging.compact_console else detailed_formatter)
    console_handler.addFilter(RuntimeContextFilter())
    console_handler.addFilter(NormalizedMessageFilter())
    console_handler.addFilter(ConsoleNoiseFilter(current_config))
    root.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(detailed_formatter)
    file_handler.addFilter(RuntimeContextFilter())
    file_handler.addFilter(NormalizedMessageFilter())
    root.addHandler(file_handler)

    error_handler = logging.FileHandler(error_path, encoding="utf-8")
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(detailed_formatter)
    error_handler.addFilter(RuntimeContextFilter())
    error_handler.addFilter(NormalizedMessageFilter())
    root.addHandler(error_handler)


def _resolve_level(level_name: str, fallback: int) -> int:
    if not level_name:
        return fallback
    return getattr(logging, level_name.upper(), fallback)


def _passes_third_party_threshold(record: logging.LogRecord, configured_level: str) -> bool:
    if record.levelno >= logging.WARNING:
        return True
    threshold = _resolve_level(configured_level, logging.WARNING)
    if record.name.startswith("httpx") or record.name.startswith("httpcore") or record.name.startswith("telegram"):
        return record.levelno >= threshold
    return True


def _passes_segment_caption_mode(record: logging.LogRecord, config: AppConfig) -> bool:
    stage = str(getattr(record, "stage", "") or "")
    mode = config.logging.segment_log_mode.lower().strip()
    caption_mode = config.logging.caption_log_mode.lower().strip()
    if stage.startswith("SEGMENT ") and mode in {"summary", "none"}:
        return record.levelno >= logging.ERROR
    if stage in {"BUILD_CAPTION_IMAGES"} and caption_mode in {"summary", "none"}:
        return record.levelno >= logging.ERROR
    if stage.startswith("CAPTION_IMAGE ") and caption_mode in {"summary", "none"}:
        return record.levelno >= logging.ERROR
    return True


def _passes_repeated_suppression(
    record: logging.LogRecord,
    config: AppConfig,
    last_seen: dict[tuple[str, int, str, str], float],
    lock: threading.Lock,
) -> bool:
    interval = max(0.0, float(config.logging.suppress_repeated_messages_seconds))
    if interval <= 0:
        return True
    message = str(record.getMessage())
    key = (record.name, record.levelno, str(getattr(record, "job_id", "-")), message)
    now = time.monotonic()
    with lock:
        last = last_seen.get(key)
        if last is not None and now - last < interval:
            return False
        last_seen[key] = now
    return True
