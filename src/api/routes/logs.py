"""
Logs Endpoint
==============

``GET /logs`` — retrieve recent application logs for admin monitoring.

Features:
  • Reads from the configured log directory (not hardcoded).
  • Efficient tail-read (seeks from end of file).
  • Optional ``level`` filter (e.g. ``?level=ERROR``).
  • Async file I/O via ``asyncio.to_thread``.
"""
from __future__ import annotations

import secrets
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from src.api.schemas import ok
from src.utils.logger import get_logger
from src.settings import LOG_PATH, admin_api_key

logger = get_logger(__name__)
router = APIRouter(prefix="/logs", tags=["Logs"])

_LOG_DIR = LOG_PATH
_LOG_FILE = _LOG_DIR / "app.log"


async def require_admin_key(x_admin_key: Optional[str] = Header(None)) -> None:
    configured = admin_api_key()
    if not configured:
        raise HTTPException(status_code=503, detail="Log access is disabled until ADMIN_API_KEY is configured")
    if not x_admin_key or not secrets.compare_digest(x_admin_key, configured):
        raise HTTPException(status_code=401, detail="Invalid or missing admin API key")


def _tail_lines(filepath: Path, n: int = 100) -> list[str]:
    """
    Efficiently read the last *n* lines of a file by seeking from the end.

    Falls back to reading the full file if it's small enough.
    """
    if not filepath.exists():
        return []

    with open(filepath, "rb") as f:
        # Seek to end to get file size
        f.seek(0, 2)
        file_size = f.tell()

        if file_size == 0:
            return []

        # Read up to 1 MB from the end — sufficient for ~10k lines
        read_size = min(file_size, 1024 * 1024)
        f.seek(-read_size, 2)
        data = f.read().decode("utf-8", errors="replace")

    lines = data.splitlines()
    return lines[-n:]


@router.get("/", dependencies=[Depends(require_admin_key)])
async def get_recent_logs(
    lines: int = Query(100, ge=1, le=5000, description="Number of log lines to return"),
    level: Optional[str] = Query(None, description="Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"),
):
    """
    Return the last *lines* from the application log file.

    Optionally filter by log level (case-insensitive).

    Args:
        lines: How many lines to return from the tail of the log.
        level: If provided, only lines containing this level are returned.

    Returns:
        Standard API envelope with a list of log line strings.
    """
    if not _LOG_FILE.exists():
        logger.info("Log file not found at %s — returning empty", _LOG_FILE)
        return ok(data={"lines": [], "file": str(_LOG_FILE), "message": "No log file found yet."})

    try:
        import asyncio
        raw_lines = await asyncio.to_thread(_tail_lines, _LOG_FILE, lines * 3 if level else lines)

        if level:
            upper = level.upper()
            filtered = [ln for ln in raw_lines if f'"level": "{upper}"' in ln or f"[{upper}]" in ln]
            raw_lines = filtered[-lines:]

        logger.debug("Served %d log lines (filter=%s)", len(raw_lines), level)
        return ok(data={
            "lines": raw_lines,
            "count": len(raw_lines),
            "file": str(_LOG_FILE),
            "level_filter": level,
        })
    except Exception as e:
        logger.error("Failed to read logs: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {e}")
