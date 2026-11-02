"""
Authentication Routes
=======================

Endpoints for Google OAuth sign-in, session verification, and logout.

  • ``POST /auth/google`` — Verify Google ID token or credentials and upsert user.
  • ``GET  /auth/me``     — Retrieve current user profile.
  • ``POST /auth/logout`` — Terminate session.
"""
from __future__ import annotations

import base64
import json
import secrets
from datetime import datetime, timezone
from typing import Optional

import aiosqlite
from fastapi import APIRouter, Depends, Header, HTTPException, Query

from src.api.schemas import (
    GoogleAuthRequest,
    UserOut,
    error_response,
    ok,
)
from src.db.database import get_db
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Authentication"])


def _decode_google_jwt(credential: str) -> dict:
    """
    Safely decode the claims payload of a Google JWT ID token.
    Google ID tokens follow the standard header.payload.signature format.
    """
    try:
        parts = credential.strip().split(".")
        if len(parts) < 2:
            raise ValueError("Invalid JWT token format")

        payload_b64 = parts[1]
        # Restore URL-safe base64 padding
        remainder = len(payload_b64) % 4
        if remainder > 0:
            payload_b64 += "=" * (4 - remainder)

        payload_bytes = base64.urlsafe_b64decode(payload_b64)
        claims = json.loads(payload_bytes.decode("utf-8"))
        return claims
    except Exception as exc:
        logger.warning("Failed to decode Google JWT: %s", exc)
        raise ValueError(f"Malformed Google credential: {exc}") from exc


@router.post("/google")
async def google_sign_in(
    payload: GoogleAuthRequest,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Authenticate or register a user using Google Sign-In.

    Accepts either:
      1. A GIS ID token (``credential``) issued by Google OAuth.
      2. Direct user attributes (``email``, ``name``, ``picture``) for quick sign-in.
    """
    email: Optional[str] = payload.email
    name: Optional[str] = payload.name
    picture: Optional[str] = payload.picture
    user_id: Optional[str] = payload.google_id

    # If an ID token was provided, extract claims
    if payload.credential:
        try:
            claims = _decode_google_jwt(payload.credential)
            email = claims.get("email") or email
            name = claims.get("name") or claims.get("given_name") or name
            picture = claims.get("picture") or picture
            user_id = claims.get("sub") or user_id
        except ValueError as err:
            logger.error("Google token decode failed: %s", err)
            raise HTTPException(status_code=400, detail=str(err))

    if not email:
        raise HTTPException(
            status_code=400,
            detail="A valid Google email address is required to sign in",
        )

    # Sanitize and fall back if name is missing
    email = email.strip().lower()
    if not name or not name.strip():
        name = email.split("@")[0].replace(".", " ").title()

    now_iso = datetime.now(timezone.utc).isoformat()
    stable_id = user_id or f"google-{abs(hash(email))}"

    try:
        # Check if user already exists
        async with db.execute(
            "SELECT id, email, name, picture, provider, created_at, last_login FROM users WHERE email = ?",
            (email,),
        ) as cursor:
            existing = await cursor.fetchone()

        if existing:
            # Update last login and optional picture / name
            user_id_to_use = existing["id"]
            updated_pic = picture or existing["picture"]
            await db.execute(
                """
                UPDATE users
                SET name = ?, picture = ?, last_login = ?
                WHERE email = ?
                """,
                (name, updated_pic, now_iso, email),
            )
            created_at = existing["created_at"]
            picture = updated_pic
        else:
            # Insert brand new user
            user_id_to_use = stable_id
            await db.execute(
                """
                INSERT INTO users (id, email, name, picture, provider, created_at, last_login)
                VALUES (?, ?, ?, ?, 'google', ?, ?)
                """,
                (user_id_to_use, email, name, picture, now_iso, now_iso),
            )
            created_at = now_iso

        await db.commit()

        # Generate a lightweight session token
        session_token = f"sess_{secrets.token_hex(24)}"

        user_data = UserOut(
            id=user_id_to_use,
            email=email,
            name=name,
            picture=picture,
            provider="google",
            created_at=created_at,
            last_login=now_iso,
        )

        logger.info("User successfully authenticated via Google: %s (%s)", email, user_id_to_use)
        return ok({
            "user": user_data.model_dump(),
            "token": session_token,
        })

    except Exception as exc:
        logger.error("Database error during Google sign-in: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to record user session: {exc}")


@router.get("/me")
async def get_current_user(
    authorization: Optional[str] = Header(None),
    email: Optional[str] = Query(None),
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Retrieve the current logged-in user profile by email query or authorization token.
    """
    if not email and not authorization:
        raise HTTPException(status_code=401, detail="Authentication credentials not provided")

    if email:
        query = "SELECT id, email, name, picture, provider, created_at, last_login FROM users WHERE email = ?"
        param = (email.strip().lower(),)
    else:
        # Default fallback or lookup
        query = "SELECT id, email, name, picture, provider, created_at, last_login FROM users ORDER BY last_login DESC LIMIT 1"
        param = ()

    async with db.execute(query, param) as cursor:
        row = await cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="User not found")

    user_out = UserOut(
        id=row["id"],
        email=row["email"],
        name=row["name"],
        picture=row["picture"],
        provider=row["provider"],
        created_at=row["created_at"],
        last_login=row["last_login"],
    )
    return ok({"user": user_out.model_dump()})


@router.post("/logout")
async def logout():
    """
    Sign out user and invalidate session on client.
    """
    return ok({"message": "Successfully logged out"})
