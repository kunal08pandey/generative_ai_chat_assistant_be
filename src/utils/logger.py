"""
Production-Grade Logging System
================================

Dual-output logging with:
  • Terminal: Rich-formatted, color-coded output via ``RichHandler``
  • File: JSON-structured, daily-rotated logs via ``TimedRotatingFileHandler``

Usage::

    from src.utils.logger import get_logger, setup_logging

    # Call once at app startup
    setup_logging()

    # Then in any module
    logger = get_logger(__name__)
    logger.info("Server started")
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import json
from datetime import datetime, timezone
from pathlib import Path

from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

from src.utils.request_context import get_request_id
from src.settings import LOG_PATH

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LOG_DIR = LOG_PATH
_LOG_FILE = _LOG_DIR / "app.log"
_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
_BACKUP_COUNT = 30  # keep 30 rotated files

_CONFIGURED = False  # guard against double-init

# ---------------------------------------------------------------------------
# Custom Rich theme for terminal colors
# ---------------------------------------------------------------------------
_RICH_THEME = Theme({
    "logging.level.debug": "dim white",
    "logging.level.info": "bold cyan",
    "logging.level.warning": "bold yellow",
    "logging.level.error": "bold red",
    "logging.level.critical": "bold white on red",
})

_console = Console(theme=_RICH_THEME, stderr=True)


# ---------------------------------------------------------------------------
# JSON formatter for file logs
# ---------------------------------------------------------------------------
class JSONFormatter(logging.Formatter):
    """
    Formats each log record as a single JSON line for machine parsing.

    Fields emitted:
        timestamp, level, logger, file, line, request_id, message, exc_info
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
            "request_id": get_request_id(),
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


# ---------------------------------------------------------------------------
# Setup function — call once at startup
# ---------------------------------------------------------------------------
def setup_logging(
    level: str | None = None,
    log_dir: Path | str | None = None,
) -> None:
    """
    Configure the root logger with two handlers:

    1. **RichHandler** → colourful, human-readable terminal output.
    2. **TimedRotatingFileHandler** → JSON-structured, daily-rotated files.

    Args:
        level: Override log level (default: ``LOG_LEVEL`` env var or ``DEBUG``).
        log_dir: Override log directory (default: ``LOG_DIR`` env var or ``./logs``).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    effective_level = getattr(logging, (level or _LOG_LEVEL), logging.DEBUG)
    effective_dir = Path(log_dir) if log_dir else _LOG_DIR
    effective_dir.mkdir(parents=True, exist_ok=True)
    log_file = effective_dir / "app.log"

    # ── Root logger ──────────────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(effective_level)

    # Remove any pre-existing handlers (prevents duplicates on reload)
    root.handlers.clear()

    # ── Terminal handler (Rich) ──────────────────────────────────────────
    rich_handler = RichHandler(
        console=_console,
        show_time=True,
        show_level=True,
        show_path=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        markup=True,
        log_time_format="[%Y-%m-%d %H:%M:%S]",
    )
    rich_handler.setLevel(effective_level)
    rich_fmt = logging.Formatter("%(message)s", datefmt="[%X]")
    rich_handler.setFormatter(rich_fmt)
    root.addHandler(rich_handler)

    # ── File handler (JSON, rotating) ────────────────────────────────────
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=str(log_file),
        when="midnight",
        interval=1,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
        utc=True,
    )
    file_handler.setLevel(effective_level)
    file_handler.setFormatter(JSONFormatter())
    file_handler.suffix = "%Y-%m-%d"
    root.addHandler(file_handler)

    # ── Silence noisy third-party loggers ────────────────────────────────
    for noisy in ("httpx", "httpcore", "chromadb", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root.info(
        "Logging initialised  |  level=%s  |  log_dir=%s  |  file=%s",
        logging.getLevelName(effective_level),
        effective_dir,
        log_file,
    )


# ---------------------------------------------------------------------------
# Module-level helper — returns a child logger
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    If ``setup_logging()`` hasn't been called yet, it is called automatically
    with default settings so that imports at module-top-level always work.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)


# Convenience default logger
logger = get_logger("app")
