"""
Centralized Pydantic Schemas & API Response Envelope
=====================================================

All request/response models live here so that routes stay thin
and validation is consistent across the entire API surface.
"""
from __future__ import annotations

import math
from typing import Any, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field, model_validator

T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════
# Standard API Response Envelope
# ═══════════════════════════════════════════════════════════════════════════

class PaginationMeta(BaseModel):
    """Pagination metadata included in collection responses."""
    page: int = Field(..., ge=1, description="Current page number (1-indexed)")
    page_size: int = Field(..., ge=1, le=200, description="Items per page")
    total_count: int = Field(..., ge=0, description="Total items across all pages")
    total_pages: int = Field(..., ge=0, description="Total number of pages")


class ErrorDetail(BaseModel):
    """Structured error payload."""
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Any] = Field(None, description="Additional error context")


class APIResponse(BaseModel, Generic[T]):
    """
    Standardized JSON response envelope.

    Every endpoint returns this shape so consumers have a single,
    predictable contract.

    Example success::

        {
            "success": true,
            "data": { ... },
            "meta": { "page": 1, ... },
            "error": null
        }

    Example error::

        {
            "success": false,
            "data": null,
            "meta": null,
            "error": { "code": "NOT_FOUND", "message": "..." }
        }
    """
    success: bool = True
    data: Optional[T] = None
    meta: Optional[PaginationMeta] = None
    error: Optional[ErrorDetail] = None


def ok(data: Any = None, meta: PaginationMeta | None = None) -> dict:
    """Shortcut to build a success envelope dict."""
    return APIResponse(success=True, data=data, meta=meta).model_dump(exclude_none=True)


def paginated_ok(
    items: list,
    total_count: int,
    page: int,
    page_size: int,
) -> dict:
    """Shortcut to build a success envelope with pagination metadata."""
    total_pages = max(1, math.ceil(total_count / page_size)) if page_size else 1
    meta = PaginationMeta(
        page=page,
        page_size=page_size,
        total_count=total_count,
        total_pages=total_pages,
    )
    return APIResponse(success=True, data=items, meta=meta).model_dump(exclude_none=True)


def error_response(
    code: str,
    message: str,
    details: Any = None,
) -> dict:
    """Shortcut to build an error envelope dict."""
    return APIResponse(
        success=False,
        error=ErrorDetail(code=code, message=message, details=details),
    ).model_dump(exclude_none=True)


# ═══════════════════════════════════════════════════════════════════════════
# Chat & RAG Request Schemas
# ═══════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """Request body for ``POST /chat``."""
    query: str = Field("", description="User message text")
    model: str = Field(..., min_length=1, description="LLM model name")
    conversation_id: str = Field(..., min_length=1, description="Conversation thread ID")
    images: Optional[List[str]] = Field(None, description="Base64-encoded images for vision models")
    provider: Optional[str] = Field(None, description="LLM provider override (e.g. ollama, llama_cpp)")

    @model_validator(mode="after")
    def require_text_or_image(self):
        if not self.query.strip() and not self.images:
            raise ValueError("Either query text or at least one image is required")
        return self


class RagRequest(BaseModel):
    """Request body for ``POST /rag``."""
    query: str = Field("", description="User query for RAG")
    model: str = Field(..., min_length=1, description="LLM model name")
    conversation_id: str = Field(..., min_length=1, description="Conversation thread ID")
    images: Optional[List[str]] = Field(None, description="Base64-encoded images")
    provider: Optional[str] = Field(None, description="LLM provider override")


# ═══════════════════════════════════════════════════════════════════════════
# Conversation Schemas
# ═══════════════════════════════════════════════════════════════════════════

class ConversationCreate(BaseModel):
    """Request body for ``POST /conversations``."""
    title: str = Field("New Chat", min_length=1, max_length=200, description="Conversation title")


class ConversationUpdate(BaseModel):
    """Request body for ``PATCH /conversations/{conv_id}``."""
    title: str = Field(..., min_length=1, max_length=200, description="New conversation title")


class ConversationOut(BaseModel):
    """Response representation of a conversation."""
    id: str
    title: str
    created_at: str


# ═══════════════════════════════════════════════════════════════════════════
# Message Schemas
# ═══════════════════════════════════════════════════════════════════════════

class MessageCreate(BaseModel):
    """Request body for ``POST /messages``."""
    conversation_id: str = Field(..., min_length=1)
    role: str = Field(..., pattern="^(user|assistant|system)$", description="Message role")
    content: str = Field(..., min_length=1, description="Message content")


class MessageOut(BaseModel):
    """Response representation of a message."""
    id: str
    conversation_id: str
    role: str
    content: str
    created_at: str


# ═══════════════════════════════════════════════════════════════════════════
# Health Schemas
# ═══════════════════════════════════════════════════════════════════════════

class ServiceStatus(BaseModel):
    """Status of an individual service dependency."""
    name: str
    status: str  # "ok" | "error"
    latency_ms: Optional[float] = None
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Response for ``GET /health``."""
    status: str  # "healthy" | "degraded" | "unhealthy"
    uptime_seconds: float
    timestamp: str
    services: List[ServiceStatus]
    system: Optional[dict] = None
