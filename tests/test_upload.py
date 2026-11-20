"""
Upload Route Tests
===================

Tests:
  - Document upload, text extraction, chunking, and vector indexing.
  - Rejection of unauthenticated uploads with 401.
  - Rejection of legacy/unsupported document formats with 415.
  - Rejection of oversized files exceeding MAX_UPLOAD_BYTES with 413 (audit S6).
  - Document ownership isolation (chunks and SQLite records assigned to owner_id).
"""
import sqlite3
import pytest

from src.db.database import DB_PATH


def test_upload_unauthenticated_rejected(client):
    """Verify unauthenticated uploads return 401."""
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"content", "text/plain")},
    )
    assert response.status_code == 401


def test_upload_success(auth_client, test_user, monkeypatch):
    captured = {}
    monkeypatch.setattr("src.api.routes.upload.extract_text_from_file", lambda file: "Sample document content for testing")

    async def add_documents(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("src.api.routes.upload.add_documents", add_documents)
    response = auth_client.post(
        "/upload",
        files={"file": ("notes.txt", b"Sample document content for testing", "text/plain")},
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["filename"] == "notes.txt"
    assert data["status"] == "success"
    assert "id" in data
    doc_id = data["id"]

    # Verify owner_id was passed to vector store
    assert captured.get("owner_id") == test_user["id"]
    for meta in captured.get("metadatas", []):
        assert meta["owner_id"] == test_user["id"]
        assert meta["doc_id"] == doc_id

    # Verify document recorded in SQLite database
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        assert row is not None
        assert row["owner_id"] == test_user["id"]
        assert row["filename"] == "notes.txt"


def test_upload_rejects_legacy_doc_files(auth_client):
    """Legacy .doc formats must return 415 Unsupported Media Type."""
    response = auth_client.post(
        "/upload",
        files={"file": ("notes.doc", b"content", "application/msword")},
    )
    assert response.status_code == 415


def test_upload_rejects_oversized_files(auth_client, monkeypatch):
    """S6: Uploads exceeding MAX_UPLOAD_BYTES must be rejected early with 413."""
    # Temporarily lower limit to 100 bytes for test
    monkeypatch.setattr("src.api.routes.upload.MAX_UPLOAD_BYTES", 100)

    large_content = b"X" * 150
    response = auth_client.post(
        "/upload",
        files={"file": ("large.txt", large_content, "text/plain")},
    )
    assert response.status_code == 413
    err_msg = response.json().get("error", {}).get("message", "") or response.json().get("detail", "")
    assert "maximum allowed upload size" in err_msg.lower()


def test_upload_preserves_extraction_error_status(auth_client, monkeypatch):
    def fail(_file):
        raise ValueError("Corrupt file content cannot be parsed")

    monkeypatch.setattr("src.api.routes.upload.extract_text_from_file", fail)
    response = auth_client.post(
        "/upload",
        files={"file": ("notes.txt", b"corrupt", "text/plain")},
    )
    assert response.status_code == 400
