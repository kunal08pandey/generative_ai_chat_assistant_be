"""
Documents Management & Citations Tests
======================================

Tests:
  - Document listing and deletion endpoints (GET /documents, DELETE /documents/{id}).
  - Cross-user data isolation (User B cannot list or delete User A's documents).
  - Clean vector store chunk cleanup upon deletion.
  - RAG structured citation metadata extraction.
"""
import sqlite3
import pytest

from src.db.database import DB_PATH


def test_documents_unauthenticated_rejected(client):
    """Unauthenticated requests to /documents must return 401."""
    res = client.get("/documents")
    assert res.status_code == 401

    res = client.delete("/documents/some-doc-id")
    assert res.status_code == 401


def test_documents_list_and_delete_lifecycle(auth_client, bob_client, test_user, monkeypatch):
    """
    Verify full lifecycle:
    1. Upload a document as test_user.
    2. List documents as test_user -> document is returned.
    3. List documents as bob_client -> document is NOT returned (isolation).
    4. Bob attempts to delete test_user's document -> 404 Not Found.
    5. Test_user deletes document -> success, deleted from DB and Chroma.
    """
    deleted_calls = []

    async def mock_delete_by_id(doc_id: str, owner_id: str):
        deleted_calls.append({"doc_id": doc_id, "owner_id": owner_id})

    monkeypatch.setattr("src.api.routes.documents.delete_document_by_id", mock_delete_by_id)
    monkeypatch.setattr("src.api.routes.upload.extract_text_from_file", lambda f: "Line 1\nLine 2\nLine 3")

    async def mock_add_docs(**kwargs):
        pass

    monkeypatch.setattr("src.api.routes.upload.add_documents", mock_add_docs)

    # 1. Upload doc as test_user
    upload_res = auth_client.post(
        "/upload",
        files={"file": ("project_spec.txt", b"Line 1\nLine 2\nLine 3", "text/plain")},
    )
    assert upload_res.status_code == 200
    doc_id = upload_res.json()["data"]["id"]

    # 2. List documents as test_user
    list_res = auth_client.get("/documents")
    assert list_res.status_code == 200
    docs = list_res.json()["data"]
    assert len(docs) >= 1
    matching = [d for d in docs if d["id"] == doc_id]
    assert len(matching) == 1
    assert matching[0]["filename"] == "project_spec.txt"

    # 3. List documents as Bob -> should NOT see test_user's doc
    bob_list = bob_client.get("/documents")
    assert bob_list.status_code == 200
    bob_docs = bob_list.json()["data"]
    assert not any(d["id"] == doc_id for d in bob_docs)

    # 4. Bob attempts to delete test_user's doc -> 404
    bob_del = bob_client.delete(f"/documents/{doc_id}")
    assert bob_del.status_code == 404

    # 5. Test_user deletes own doc -> 200
    del_res = auth_client.delete(f"/documents/{doc_id}")
    assert del_res.status_code == 200
    assert len(deleted_calls) == 1
    assert deleted_calls[0] == {"doc_id": doc_id, "owner_id": test_user["id"]}

    # Verify removed from database
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT id FROM documents WHERE id = ?", (doc_id,)).fetchone()
        assert row is None


@pytest.mark.anyio
async def test_retriever_citations_extraction(monkeypatch):
    """Verify retrieve_context_and_citations returns structured citations."""
    from src.rag.retriever import retrieve_context_and_citations

    async def mock_search_with_meta(query, owner_id=None, n_results=3):
        return [
            {
                "text": "This is chunk 0 of system overview.",
                "metadata": {"source": "overview.pdf", "chunk": 0, "doc_id": "doc-123"},
                "id": "doc-123_chunk_0",
            },
            {
                "text": "This is chunk 1 of system overview.",
                "metadata": {"source": "overview.pdf", "chunk": 1, "doc_id": "doc-123"},
                "id": "doc-123_chunk_1",
            },
        ]

    monkeypatch.setattr("src.rag.retriever.search_with_metadata", mock_search_with_meta)

    context, citations = await retrieve_context_and_citations("overview", owner_id="user-1")
    assert "system overview" in context
    assert len(citations) == 2
    assert citations[0]["source"] == "overview.pdf"
    assert citations[0]["chunk"] == 0
    assert citations[0]["doc_id"] == "doc-123"
    assert citations[1]["chunk"] == 1
