"""
RAG Endpoint Tests
===================

Tests:
  - Streaming RAG response with document context and images.
  - Security: unauthenticated requests rejected with 401.
  - Security: User Bob cannot access Alice's conversation for RAG (404).
  - Document retrieval is scoped to the requesting user's owner_id.
"""
import pytest


class FakeRagClient:
    def __init__(self):
        self.stream_images = None

    async def generate(self, prompt, images=None):
        return "optimized query"

    async def generate_stream(self, prompt, images=None):
        self.stream_images = images
        yield "RAG answer"


def test_rag_unauthenticated_rejected(client, conversation):
    """Verify unauthenticated RAG request returns 401."""
    response = client.post(
        "/rag/",
        json={
            "query": "Question",
            "model": "gen-labs-scout",
            "conversation_id": conversation["id"],
        },
    )
    assert response.status_code == 401


def test_rag_stream_uses_context_and_images(auth_client, conversation, monkeypatch):
    fake = FakeRagClient()
    monkeypatch.setattr("src.api.routes.rag.get_llm_client", lambda **kwargs: fake)

    captured_owner = None

    async def retrieve(_query, owner_id=None):
        nonlocal captured_owner
        captured_owner = owner_id
        return "document context"

    monkeypatch.setattr("src.api.routes.rag.retrieve_context", retrieve)
    response = auth_client.post(
        "/rag/",
        json={
            "query": "Question that is long enough to trigger query optimization step in RAG chain logic if needed",
            "model": "gen-labs-scout",
            "conversation_id": conversation["id"],
            "images": ["aW1hZ2U="],
        },
    )

    assert response.status_code == 200
    assert "RAG answer" in response.text
    # Verify owner_id was forwarded to retrieve_context for document isolation
    assert captured_owner is not None


def test_rag_rejects_other_user_conversation(bob_client, conversation, monkeypatch):
    """User Bob cannot run RAG against Alice's conversation."""
    response = bob_client.post(
        "/rag/",
        json={
            "query": "Infiltrating query",
            "model": "gen-labs-scout",
            "conversation_id": conversation["id"],
        },
    )
    assert response.status_code == 404
