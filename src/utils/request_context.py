"""
Request context utilities using contextvars.

Provides a per-request correlation ID that is automatically injected
into every log line by the logger and set by the HTTP middleware.
"""
from __future__ import annotations

import uuid
from contextvars import ContextVar

# Stores a unique request ID for the current async context (per-request).
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def get_request_id() -> str:
    """Return the current request's correlation ID."""
    return request_id_var.get()


def set_request_id(rid: str | None = None) -> str:
    """
    Set (or generate) a request ID for the current context.

    Args:
        rid: An explicit ID to use. If None, a new UUID4 is generated.

    Returns:
        The request ID that was set.
    """
    value = rid or uuid.uuid4().hex[:12]
    request_id_var.set(value)
    return value
