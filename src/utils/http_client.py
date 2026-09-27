"""
Shared Async HTTP Client Utility
=================================

Provides a singleton ``httpx.AsyncClient`` instance configured with connection pooling.
This avoids the significant overhead of DNS lookup and TCP handshakes on every
individual request or stream token connection.
"""
import httpx
from typing import Optional

_shared_client: Optional[httpx.AsyncClient] = None


def get_http_client() -> httpx.AsyncClient:
    """
    Return a thread-safe, shared httpx.AsyncClient with optimized pooling limits.
    """
    global _shared_client
    if _shared_client is None:
        # Configure pooling limits: keep-alive up to 20 connections, max 100 concurrent sockets
        limits = httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=30.0,
        )
        timeout = httpx.Timeout(120.0, connect=10.0, read=120.0)
        _shared_client = httpx.AsyncClient(limits=limits, timeout=timeout)
    return _shared_client


async def close_http_client() -> None:
    """
    Gracefully shut down the shared HTTP client, closing all active connections.
    Should be called during application teardown.
    """
    global _shared_client
    if _shared_client is not None:
        await _shared_client.aclose()
        _shared_client = None
