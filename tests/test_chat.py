"""
Chat Route Tests
=================

Tests:
  - Streaming SSE chat response and message persistence.
  - Image-only chat with vision model.
  - Verification that chat requires an existing, owned conversation.
  - Validation of message content and registered models.
  - Security: unauthenticated requests rejected with 401.
  - Security: User Bob cannot send messages to Alice's conversation.
"""
import pytest


class FakeChatClient:
    def __init__(self):
        self.prompt = None
        self.images = None

    async def generate_stream(self, prompt, images=None):
        self.prompt = prompt
        self.images = images
        for token in ("Hello", " world"):
            yield token


def test_chat_unauthenticated_rejected(client, conversation):
    """Verify unauthenticated chat requests return 401."""
    response = client.post(
        "/chat/",
        json={
            "query": "Hello",
            "model": "gen-labs-scout",
            "conversation_id": conversation["id"],
        },
    )
    assert response.status_code == 401


def test_chat_stream_and_persistence(auth_client, conversation, monkeypatch):
    fake = FakeChatClient()
    monkeypatch.setattr("src.api.routes.chat.get_llm_client", lambda **kwargs: fake)

    response = auth_client.post(
        "/chat/",
        json={
            "query": "Hello",
            "model": "gen-labs-scout",
            "conversation_id": conversation["id"],
        },
    )
    assert response.status_code == 200
    assert '"token": "Hello"' in response.text
    assert '"done": true' in response.text

    history = auth_client.get(f"/messages/{conversation['id']}").json()["data"]
    assert [message["role"] for message in history] == ["user", "assistant"]
    assert history[-1]["content"] == "Hello world"


def test_image_only_chat_uses_default_prompt(auth_client, conversation, monkeypatch):
    fake = FakeChatClient()
    monkeypatch.setattr("src.api.routes.chat.get_llm_client", lambda **kwargs: fake)

    response = auth_client.post(
        "/chat/",
        json={
            "query": "",
            "model": "gen-labs-scout",
            "conversation_id": conversation["id"],
            "images": ["aW1hZ2U="],
        },
    )
    assert response.status_code == 200
    assert fake.images == ["aW1hZ2U="]
    assert "Describe the attached image" in fake.prompt


def test_chat_requires_existing_conversation(auth_client, monkeypatch):
    monkeypatch.setattr("src.api.routes.chat.get_llm_client", lambda **kwargs: FakeChatClient())
    response = auth_client.post(
        "/chat/",
        json={"query": "Hello", "model": "gen-labs-scout", "conversation_id": "nonexistent-conv-id"},
    )
    assert response.status_code == 404


def test_chat_rejects_other_user_conversation(bob_client, conversation, monkeypatch):
    """User Bob cannot chat into Alice's conversation."""
    monkeypatch.setattr("src.api.routes.chat.get_llm_client", lambda **kwargs: FakeChatClient())
    response = bob_client.post(
        "/chat/",
        json={
            "query": "Attacking conversation",
            "model": "gen-labs-scout",
            "conversation_id": conversation["id"],
        },
    )
    assert response.status_code == 404


def test_chat_requires_text_or_an_image(auth_client, conversation):
    response = auth_client.post(
        "/chat/",
        json={
            "query": "",
            "model": "gen-labs-scout",
            "conversation_id": conversation["id"],
            "images": [],
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_chat_rejects_unregistered_model(auth_client, conversation):
    response = auth_client.post(
        "/chat/",
        json={
            "query": "Hello",
            "model": "not-registered",
            "conversation_id": conversation["id"],
        },
    )
    assert response.status_code == 400
