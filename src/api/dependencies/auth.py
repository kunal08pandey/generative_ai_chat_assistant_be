"""
Authentication Dependencies
=============================

FastAPI dependencies for extracting and validating the current user
from a secure HttpOnly session cookie.

Usage in routes::

    from src.api.dependencies.auth import get_current_user

    @router.get("/protected")
    async def protected_endpoint(user: dict = Depends(get_current_user)):
        ...
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import aiosqlite
from fastapi import Cookie, Depends, HTTPException, Request

from src.db.database import get_db
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def get_current_user(
    request: Request,
    session_token: Optional[str] = Cookie(None, alias="session_token"),
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    """
    Extract and validate the current authenticated user from the session cookie.

    Checks:
      1. Session cookie is present.
      2. Session exists in the database.
      3. Session is not revoked.
      4. Session has not expired.
      5. The associated user exists.

    Returns a dict with user fields: id, email, name, picture, provider.
    Raises HTTP 401 on any failure.
    """
    # Also check Authorization header as fallback for API clients
    if not session_token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            session_token = auth_header[7:].strip()

    if not session_token:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please sign in.",
        )

    # Look up the session
    async with db.execute(
        "SELECT token, user_id, expires_at, revoked FROM sessions WHERE token = ?",
        (session_token,),
    ) as cursor:
        session_row = await cursor.fetchone()

    if not session_row:
        raise HTTPException(
            status_code=401,
            detail="Invalid session. Please sign in again.",
        )

    if session_row["revoked"]:
        raise HTTPException(
            status_code=401,
            detail="Session has been revoked. Please sign in again.",
        )

    # Check expiration
    try:
        expires_at = datetime.fromisoformat(session_row["expires_at"])
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) > expires_at:
            raise HTTPException(
                status_code=401,
                detail="Session expired. Please sign in again.",
            )
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=401,
            detail="Invalid session data. Please sign in again.",
        )

    # Look up the user
    user_id = session_row["user_id"]
    async with db.execute(
        "SELECT id, email, name, picture, provider FROM users WHERE id = ?",
        (user_id,),
    ) as cursor:
        user_row = await cursor.fetchone()

    if not user_row:
        raise HTTPException(
            status_code=401,
            detail="User account not found. Please sign in again.",
        )

    return {
        "id": user_row["id"],
        "email": user_row["email"],
        "name": user_row["name"],
        "picture": user_row["picture"],
        "provider": user_row["provider"],
    }


async def get_optional_user(
    request: Request,
    session_token: Optional[str] = Cookie(None, alias="session_token"),
    db: aiosqlite.Connection = Depends(get_db),
) -> Optional[dict]:
    """
    Like get_current_user but returns None instead of raising 401
    when no valid session is found. Useful for endpoints that work
    differently for authenticated vs anonymous users (e.g., health).
    """
    try:
        return await get_current_user(request, session_token, db)
    except HTTPException:
        return None
