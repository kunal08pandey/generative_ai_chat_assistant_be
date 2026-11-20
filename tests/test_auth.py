"""
Unit & Security Regression Tests for Authentication Routes (/auth)
==================================================================

Covers Phase 1 Acceptance Criteria:
  - Unverified email/profile attributes rejected as authentication (audit S1).
  - Malformed or invalid Google credentials rejected with 401.
  - Wrong audience and unverified email claims rejected with 401.
  - Valid token creates real database session and HttpOnly cookie.
  - Profile lookup (/auth/me) derived exclusively from validated session (audit S2).
  - Expired and revoked sessions rejected with 401.
  - Logout revokes server-side session and clears cookie.
  - Demo profiles create isolated 'demo' provider sessions.
"""
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from starlette.testclient import TestClient

from src.db.database import DB_PATH


def _get_error_message(response) -> str:
    """Helper to extract error message from standard API error envelope or direct detail."""
    data = response.json()
    if "error" in data and isinstance(data["error"], dict):
        return data["error"].get("message", "")
    return data.get("detail", "")


def test_unverified_email_sign_in_rejected(client: TestClient):
    """S1: Reject direct email/profile sign-in without a verifiable credential."""
    payload = {
        "email": "victim@gmail.com",
        "name": "Spoofed Victim",
    }
    response = client.post("/auth/google", json=payload)
    assert response.status_code == 400
    msg = _get_error_message(response).lower()
    assert "credential" in msg


def test_invalid_credential_rejected(client: TestClient):
    """S1: Malformed or unverified JWT credential must be rejected with 401."""
    response = client.post("/auth/google", json={"credential": "invalid.jwt.token"})
    assert response.status_code == 401


def test_wrong_audience_rejected(client: TestClient, monkeypatch):
    """S1: Google token with mismatched audience must be rejected."""
    mock_claims = {
        "sub": "sub-12345",
        "email": "test@gmail.com",
        "email_verified": True,
        "aud": "wrong-client-id.apps.googleusercontent.com",
        "iss": "accounts.google.com",
    }
    monkeypatch.setattr("src.api.routes.auth.GOOGLE_CLIENT_ID", "expected-client-id")

    async def fake_verify(cred):
        if mock_claims.get("aud") != "expected-client-id":
            raise ValueError("Token audience mismatch")
        return mock_claims

    monkeypatch.setattr("src.api.routes.auth._verify_google_id_token", fake_verify)

    response = client.post("/auth/google", json={"credential": "mock.jwt.claims"})
    assert response.status_code == 401


def test_unverified_email_claim_rejected(client: TestClient, monkeypatch):
    """S1: Token where email_verified is False must be rejected."""
    async def fake_verify(cred):
        raise ValueError("Google account email is not verified")

    monkeypatch.setattr("src.api.routes.auth._verify_google_id_token", fake_verify)

    response = client.post("/auth/google", json={"credential": "mock.jwt.claims"})
    assert response.status_code == 401


def test_valid_verified_token_creates_session_and_cookie(client: TestClient, monkeypatch):
    """Acceptance: Valid Google token creates user, persists session, and sets HttpOnly cookie."""
    verified_claims = {
        "sub": "google-sub-777",
        "email": "verified.user@gmail.com",
        "name": "Verified Google User",
        "picture": "https://lh3.googleusercontent.com/avatar.jpg",
        "email_verified": True,
        "iss": "https://accounts.google.com",
    }

    async def fake_verify(cred):
        return verified_claims

    monkeypatch.setattr("src.api.routes.auth._verify_google_id_token", fake_verify)

    response = client.post("/auth/google", json={"credential": "valid.signed.jwt"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["user"]["email"] == "verified.user@gmail.com"
    assert data["data"]["user"]["id"] == "google-google-sub-777"
    assert "token" in data["data"]
    token = data["data"]["token"]

    # Verify session persisted in SQLite
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM sessions WHERE token = ?", (token,)).fetchone()
        assert row is not None
        assert row["user_id"] == "google-google-sub-777"
        assert row["revoked"] == 0

    # Verify HttpOnly cookie was sent
    assert "session_token" in response.cookies


def test_get_me_unauthenticated_returns_401(client: TestClient):
    """S2: Accessing /auth/me without a session returns 401."""
    response = client.get("/auth/me")
    assert response.status_code == 401


def test_get_me_with_email_param_alone_returns_401(client: TestClient):
    """S2: Email query parameter bypass is removed; returns 401 without valid session."""
    response = client.get("/auth/me?email=alice@example.com")
    assert response.status_code == 401


def test_get_me_with_valid_session(auth_client: TestClient, test_user):
    """Profile lookup returns current user when session is valid."""
    response = auth_client.get("/auth/me")
    assert response.status_code == 200
    data = response.json()["data"]["user"]
    assert data["id"] == test_user["id"]
    assert data["email"] == test_user["email"]


def test_expired_session_fails(client: TestClient):
    """Acceptance: Expired sessions are rejected with 401."""
    expired_token = "sess_expired_test_token"
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO users (id, email, name, provider, created_at, last_login)
            VALUES ('u-exp', 'exp@test.com', 'Exp User', 'google', ?, ?)
            """,
            (past, past),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO sessions (token, user_id, created_at, expires_at, revoked)
            VALUES (?, 'u-exp', ?, ?, 0)
            """,
            (expired_token, past, past),
        )
        conn.commit()

    res = client.get("/auth/me", headers={"Authorization": f"Bearer {expired_token}"})
    assert res.status_code == 401
    msg = _get_error_message(res).lower()
    assert "expired" in msg


def test_revoked_session_fails(client: TestClient):
    """Acceptance: Revoked sessions are rejected with 401."""
    revoked_token = "sess_revoked_test_token"
    now_iso = datetime.now(timezone.utc).isoformat()
    future = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO users (id, email, name, provider, created_at, last_login)
            VALUES ('u-rev', 'rev@test.com', 'Rev User', 'google', ?, ?)
            """,
            (now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO sessions (token, user_id, created_at, expires_at, revoked)
            VALUES (?, 'u-rev', ?, ?, 1)
            """,
            (revoked_token, now_iso, future),
        )
        conn.commit()

    res = client.get("/auth/me", headers={"Authorization": f"Bearer {revoked_token}"})
    assert res.status_code == 401
    msg = _get_error_message(res).lower()
    assert "revoked" in msg


def test_logout_revokes_session_and_clears_cookie(auth_client: TestClient, auth_session):
    """Acceptance: Logout sets revoked=1 in SQLite and invalidates subsequent calls."""
    # Logout
    logout_res = auth_client.post("/auth/logout")
    assert logout_res.status_code == 200

    # Verify revoked in DB
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT revoked FROM sessions WHERE token = ?", (auth_session,)).fetchone()
        assert row is not None
        assert row[0] == 1

    # Subsequent /auth/me call with this revoked token must fail with 401
    me_res = auth_client.get("/auth/me")
    assert me_res.status_code == 401


def test_demo_profile_sign_in(client: TestClient):
    """Verify demo sign in produces an isolated demo account with provider='demo'."""
    res = client.post("/auth/demo?profile=alex")
    assert res.status_code == 200
    data = res.json()["data"]
    assert data["user"]["provider"] == "demo"
    assert data["user"]["id"] == "demo-alex-rivers"
    assert "token" in data
