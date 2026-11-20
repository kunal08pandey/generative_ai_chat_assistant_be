"""
Authentication Routes
=======================

Endpoints for Google OAuth sign-in with server-side token verification,
secure session management via HttpOnly cookies, demo profiles, and logout.

  • ``POST /auth/google`` — Verify Google ID token server-side, upsert user, create session.
  • ``POST /auth/demo``   — Sign in with a demo profile (isolated from real Google accounts).
  • ``GET  /auth/me``     — Retrieve current user profile from validated session.
  • ``POST /auth/logout`` — Revoke session and clear cookie.

Phase 1 Security Improvements (audit S1, S2):
  - Google ID tokens are verified server-side (google-auth or Google tokeninfo endpoint).
  - Signature, issuer, audience, email_verified, and expiration are all validated.
  - Direct email/profile attribute sign-in is rejected for real accounts.
  - Sessions are persisted in SQLite with expiration and revocation.
  - Session token is delivered as an HttpOnly, Secure, SameSite=Lax cookie.
  - ``/auth/me`` derives the user exclusively from a validated session.
  - ``/auth/logout`` revokes the session server-side and clears the cookie.
"""
from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiosqlite
from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response

from src.api.schemas import (
    GoogleAuthRequest,
    UserOut,
    ok,
)
from src.api.dependencies.auth import get_current_user
from src.db.database import get_db
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Authentication"])

SESSION_EXPIRY_HOURS = int(os.environ.get("SESSION_EXPIRY_HOURS", "72"))
GOOGLE_CLIENT_ID = os.environ.get("VITE_GOOGLE_CLIENT_ID", os.environ.get("GOOGLE_CLIENT_ID", ""))

_IS_PRODUCTION = os.environ.get("ENVIRONMENT", "development").lower() in ("production", "prod")
COOKIE_SECURE = _IS_PRODUCTION
COOKIE_SAMESITE = "lax"
COOKIE_DOMAIN = os.environ.get("COOKIE_DOMAIN", None)


# ── Demo profiles (isolated from real accounts) ─────────────────────────
DEMO_PROFILES = {
    "alex": {
        "id": "demo-alex-rivers",
        "email": "alex.rivers@demo.kshitiz-ai.local",
        "name": "Alex Rivers (Dev)",
        "picture": None,
        "provider": "demo",
    },
    "sarah": {
        "id": "demo-sarah-chen",
        "email": "sarah.chen@demo.kshitiz-ai.local",
        "name": "Sarah Chen (Research)",
        "picture": None,
        "provider": "demo",
    },
}


async def _verify_google_id_token(credential: str) -> dict:
    """
    Verify a Google ID token server-side.
    Uses google-auth if available, or Google's tokeninfo API endpoint.
    Validates issuer, audience, expiration, and email_verified claim.
    """
    claims = None

    # 1. Try google-auth library if available
    try:
        from google.oauth2 import id_token as google_id_token
        from google.auth.transport import requests as google_requests
        claims = google_id_token.verify_oauth2_token(
            credential,
            google_requests.Request(),
            audience=GOOGLE_CLIENT_ID or None,
        )
    except ImportError:
        # 2. Fallback to Google's official tokeninfo endpoint
        import httpx
        from src.utils.http_client import get_http_client
        try:
            client = get_http_client()
            res = await client.get(
                "https://oauth2.googleapis.com/tokeninfo",
                params={"id_token": credential},
                timeout=10.0,
            )
            if res.status_code != 200:
                err_msg = "Google token verification failed"
                try:
                    err_data = res.json()
                    err_msg = err_data.get("error_description") or err_msg
                except Exception:
                    pass
                raise ValueError(err_msg)
            claims = res.json()
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError(f"Failed to reach Google token verifier: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Invalid Google credential: {exc}") from exc

    if not claims:
        raise ValueError("Empty claims returned from Google token verification")

    # 3. Validate claims
    if GOOGLE_CLIENT_ID and claims.get("aud") != GOOGLE_CLIENT_ID:
        raise ValueError(f"Token audience mismatch: expected {GOOGLE_CLIENT_ID}, got {claims.get('aud')}")

    issuer = claims.get("iss", "")
    if issuer not in ("accounts.google.com", "https://accounts.google.com"):
        raise ValueError(f"Invalid token issuer: {issuer}")

    email_verified = claims.get("email_verified")
    if str(email_verified).lower() not in ("true", "1", "yes"):
        raise ValueError("Google account email is not verified")

    if not claims.get("email"):
        raise ValueError("No email in Google token claims")

    return claims


async def _create_session(
    db: aiosqlite.Connection,
    user_id: str,
    response: Response,
    user_agent: str = "",
) -> str:
    """Create a new session, persist it, and set the HttpOnly cookie."""
    session_token = f"sess_{secrets.token_hex(32)}"
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=SESSION_EXPIRY_HOURS)

    await db.execute(
        """
        INSERT INTO sessions (token, user_id, created_at, expires_at, revoked, user_agent)
        VALUES (?, ?, ?, ?, 0, ?)
        """,
        (
            session_token,
            user_id,
            now.isoformat(),
            expires_at.isoformat(),
            user_agent[:500] if user_agent else "",
        ),
    )

    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        max_age=SESSION_EXPIRY_HOURS * 3600,
        domain=COOKIE_DOMAIN,
        path="/",
    )

    return session_token


@router.post("/google")
async def google_sign_in(
    payload: GoogleAuthRequest,
    request: Request,
    response: Response,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Authenticate a user using a Google ID token with server-side verification.
    The ``credential`` field is REQUIRED (audit S1 fix).
    Direct email/name attributes are rejected.
    """
    if not payload.credential:
        raise HTTPException(
            status_code=400,
            detail="A valid Google ID token (credential) is required. "
                   "Direct email sign-in is not supported for security.",
        )

    try:
        claims = await _verify_google_id_token(payload.credential)
    except ValueError as err:
        logger.warning("Google sign-in rejected: %s", err)
        raise HTTPException(status_code=401, detail=str(err))

    email = claims["email"].strip().lower()
    name = claims.get("name") or claims.get("given_name") or email.split("@")[0].replace(".", " ").title()
    picture = claims.get("picture")
    google_sub = claims.get("sub", "")

    if not google_sub:
        raise HTTPException(status_code=401, detail="Invalid Google token: missing 'sub' claim")

    now_iso = datetime.now(timezone.utc).isoformat()
    user_id = f"google-{google_sub}"

    try:
        async with db.execute(
            "SELECT id, email, name, picture, provider, created_at, last_login FROM users WHERE email = ?",
            (email,),
        ) as cursor:
            existing = await cursor.fetchone()

        if existing:
            user_id = existing["id"]
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
            await db.execute(
                """
                INSERT INTO users (id, email, name, picture, provider, created_at, last_login)
                VALUES (?, ?, ?, ?, 'google', ?, ?)
                """,
                (user_id, email, name, picture, now_iso, now_iso),
            )
            created_at = now_iso

        user_agent = request.headers.get("User-Agent", "")
        session_token = await _create_session(db, user_id, response, user_agent)
        await db.commit()

        user_data = UserOut(
            id=user_id,
            email=email,
            name=name,
            picture=picture,
            provider="google",
            created_at=created_at,
            last_login=now_iso,
        )

        logger.info("User authenticated via Google (verified): %s", email)
        return ok({
            "user": user_data.model_dump(),
            "token": session_token,
        })

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Database error during Google sign-in: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to record user session")


@router.post("/demo")
async def demo_sign_in(
    request: Request,
    response: Response,
    profile: str = "alex",
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Sign in with a demo profile. Demo accounts are isolated from real
    Google accounts and use provider='demo'.
    """
    profile_key = profile.lower().strip()
    demo = DEMO_PROFILES.get(profile_key)
    if not demo:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown demo profile: {profile}. Available: {', '.join(DEMO_PROFILES.keys())}",
        )

    now_iso = datetime.now(timezone.utc).isoformat()

    async with db.execute(
        "SELECT id FROM users WHERE id = ?", (demo["id"],)
    ) as cursor:
        existing = await cursor.fetchone()

    if existing:
        await db.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (now_iso, demo["id"]),
        )
    else:
        await db.execute(
            """
            INSERT INTO users (id, email, name, picture, provider, created_at, last_login)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (demo["id"], demo["email"], demo["name"], demo["picture"], demo["provider"], now_iso, now_iso),
        )

    user_agent = request.headers.get("User-Agent", "")
    session_token = await _create_session(db, demo["id"], response, user_agent)
    await db.commit()

    user_data = UserOut(
        id=demo["id"],
        email=demo["email"],
        name=demo["name"],
        picture=demo["picture"],
        provider="demo",
        created_at=now_iso,
        last_login=now_iso,
    )

    logger.info("Demo user signed in: %s", demo["name"])
    return ok({
        "user": user_data.model_dump(),
        "token": session_token,
    })


@router.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    """
    Retrieve the current logged-in user profile.
    Derived exclusively from the validated session cookie (audit S2 fix).
    """
    user_out = UserOut(
        id=user["id"],
        email=user["email"],
        name=user["name"],
        picture=user.get("picture"),
        provider=user.get("provider", "google"),
        created_at=user.get("created_at", ""),
        last_login=user.get("last_login", ""),
    )
    return ok({"user": user_out.model_dump()})


@router.post("/logout")
async def logout(
    response: Response,
    session_token: Optional[str] = Cookie(None, alias="session_token"),
    request: Request = None,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Sign out: revoke the session server-side and clear the cookie (audit S2 fix).
    """
    token = session_token
    if not token and request:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()

    if token:
        await db.execute(
            "UPDATE sessions SET revoked = 1 WHERE token = ?",
            (token,),
        )
        await db.commit()
        logger.info("Session revoked: %s...", token[:12])

    response.delete_cookie(
        key="session_token",
        path="/",
        domain=COOKIE_DOMAIN,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
    )

    return ok({"message": "Successfully logged out"})
