import os
import sqlite3
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


TEST_ROOT = Path(tempfile.mkdtemp(prefix="gen-ai-tests-"))
os.environ["DATABASE_URL"] = str(TEST_ROOT / "chat.db")
os.environ["CHROMA_PATH"] = str(TEST_ROOT / "chroma")
os.environ["LOG_DIR"] = str(TEST_ROOT / "logs")
os.environ["ADMIN_API_KEY"] = "test-admin-key"
os.environ["LOG_LEVEL"] = "WARNING"

from src.api.main import app  # noqa: E402
from src.db.database import DB_PATH, init_db  # noqa: E402


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def clean_database():
    init_db()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("DELETE FROM messages")
        connection.execute("DELETE FROM conversations")
        connection.commit()
    yield


@pytest.fixture
def conversation(client):
    response = client.post("/conversations", json={"title": "Test conversation"})
    assert response.status_code == 201
    return response.json()["data"]
