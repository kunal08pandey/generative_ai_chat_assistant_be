import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

TEST_ROOT = Path(tempfile.mkdtemp(prefix="gen-ai-tests-"))
os.environ["DATABASE_URL"] = str(TEST_ROOT / "chat.db")
os.environ["CHROMA_PATH"] = str(TEST_ROOT / "chroma")
os.environ["LOG_DIR"] = str(TEST_ROOT / "logs")
os.environ["ADMIN_API_KEY"] = "test-admin-key"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["GOOGLE_CLIENT_ID"] = "test-google-client-id"

from src.api.main import app  # noqa: E402
from src.db.database import DB_PATH, init_db  # noqa: E402


@pytest.fixture
def client():
    """Unauthenticated client for testing security boundaries and public routes."""
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def clean_database():
    init_db()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("DELETE FROM messages")
        connection.execute("DELETE FROM conversations")
        connection.execute("DELETE FROM sessions")
        connection.execute("DELETE FROM documents")
        connection.execute("DELETE FROM users")
        connection.commit()
    yield


@pytest.fixture
def test_user():
    """Create a primary test user in the isolated test database."""
    now_iso = datetime.now(timezone.utc).isoformat()
    user_id = "user-alice-123"
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO users (id, email, name, picture, provider, created_at, last_login)
            VALUES (?, ?, ?, ?, 'google', ?, ?)
            """,
            (user_id, "alice@example.com", "Alice Tester", None, now_iso, now_iso),
        )
        conn.commit()
    return {"id": user_id, "email": "alice@example.com", "name": "Alice Tester"}


@pytest.fixture
def auth_session(test_user):
    """Create an active session for the primary test user."""
    token = "sess_alice_valid_token_001"
    expires = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
    now_iso = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO sessions (token, user_id, created_at, expires_at, revoked, user_agent)
            VALUES (?, ?, ?, ?, 0, 'test-agent')
            """,
            (token, test_user["id"], now_iso, expires),
        )
        conn.commit()
    return token


@pytest.fixture
def auth_client(auth_session):
    """Authenticated TestClient carrying Alice's session cookie and Bearer header."""
    with TestClient(
        app,
        cookies={"session_token": auth_session},
        headers={"Authorization": f"Bearer {auth_session}"},
        raise_server_exceptions=False,
    ) as test_client:
        yield test_client


@pytest.fixture
def user_bob():
    """Create a secondary test user for isolation testing."""
    now_iso = datetime.now(timezone.utc).isoformat()
    user_id = "user-bob-456"
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO users (id, email, name, picture, provider, created_at, last_login)
            VALUES (?, ?, ?, ?, 'google', ?, ?)
            """,
            (user_id, "bob@example.com", "Bob Tester", None, now_iso, now_iso),
        )
        conn.commit()
    return {"id": user_id, "email": "bob@example.com", "name": "Bob Tester"}


@pytest.fixture
def user_bob_session(user_bob):
    """Create an active session for the secondary test user Bob."""
    token = "sess_bob_valid_token_002"
    expires = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
    now_iso = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO sessions (token, user_id, created_at, expires_at, revoked, user_agent)
            VALUES (?, ?, ?, ?, 0, 'test-agent')
            """,
            (token, user_bob["id"], now_iso, expires),
        )
        conn.commit()
    return token


@pytest.fixture
def bob_client(user_bob_session):
    """Authenticated TestClient carrying Bob's session cookie."""
    with TestClient(
        app,
        cookies={"session_token": user_bob_session},
        headers={"Authorization": f"Bearer {user_bob_session}"},
        raise_server_exceptions=False,
    ) as test_client:
        yield test_client


@pytest.fixture
def conversation(auth_client):
    """Create a conversation owned by Alice."""
    response = auth_client.post("/conversations", json={"title": "Test conversation"})
    assert response.status_code == 201
    return response.json()["data"]
