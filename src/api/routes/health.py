"""
Health Check Endpoint
======================

``GET /health`` — returns a detailed status report covering:
  • Database connectivity
  • System metrics (CPU, memory, disk)
  • Application uptime
"""
import asyncio
import time
from datetime import datetime, timezone

import psutil
import httpx
from fastapi import APIRouter, Depends
import aiosqlite

from src.api.schemas import HealthResponse, ServiceStatus, ok
from src.db.database import get_db
from src.utils.logger import get_logger
from src.settings import OLLAMA_BASE_URL
from src.rag.vector_store import collection

logger = get_logger(__name__)
router = APIRouter(tags=["Health"])

_START_TIME = time.monotonic()


@router.get("/health")
async def health_check(db: aiosqlite.Connection = Depends(get_db)):
    """
    Detailed health check that probes database connectivity and
    reports system resource metrics.

    Returns:
        A ``HealthResponse`` wrapped in the standard API envelope.
    """
    services: list[ServiceStatus] = []
    overall = "healthy"

    # ── Database check ───────────────────────────────────────────────────
    try:
        t0 = time.monotonic()
        await db.execute("SELECT 1")
        latency = round((time.monotonic() - t0) * 1000, 2)
        services.append(ServiceStatus(name="sqlite", status="ok", latency_ms=latency))
    except Exception as e:
        logger.error("Health check: database probe failed — %s", e)
        services.append(ServiceStatus(name="sqlite", status="error", message=str(e)))
        overall = "degraded"

    # ── Ollama check ─────────────────────────────────────────────────────
    try:
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
        latency = round((time.monotonic() - t0) * 1000, 2)
        services.append(ServiceStatus(name="ollama", status="ok", latency_ms=latency))
    except Exception as e:
        logger.warning("Health check: Ollama probe failed — %s", e)
        services.append(ServiceStatus(name="ollama", status="error", message=str(e)))
        overall = "degraded"

    # ── Persistent vector store check ────────────────────────────────────
    try:
        t0 = time.monotonic()
        document_count = await asyncio.to_thread(collection.count)
        latency = round((time.monotonic() - t0) * 1000, 2)
        services.append(
            ServiceStatus(
                name="chroma",
                status="ok",
                latency_ms=latency,
                message=f"{document_count} indexed records",
            )
        )
    except Exception as e:
        logger.error("Health check: vector-store probe failed — %s", e)
        services.append(ServiceStatus(name="chroma", status="error", message=str(e)))
        overall = "degraded"

    # ── System metrics ───────────────────────────────────────────────────
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    system_info = {
        "cpu_percent": psutil.cpu_percent(interval=0),
        "memory": {
            "total_gb": round(mem.total / (1024 ** 3), 2),
            "used_gb": round(mem.used / (1024 ** 3), 2),
            "percent": mem.percent,
        },
        "disk": {
            "total_gb": round(disk.total / (1024 ** 3), 2),
            "used_gb": round(disk.used / (1024 ** 3), 2),
            "percent": disk.percent,
        },
    }

    uptime = round(time.monotonic() - _START_TIME, 2)
    response = HealthResponse(
        status=overall,
        uptime_seconds=uptime,
        timestamp=datetime.now(timezone.utc).isoformat(),
        services=services,
        system=system_info,
    )
    logger.debug("Health check completed — status=%s", overall)
    return ok(data=response.model_dump())
