"""
Conversations and Messages Tests
=================================

Tests:
  - Full CRUD lifecycle for conversations and messages (using auth_client).
  - Unauthenticated access returns 401.
  - Multi-user isolation: User Bob cannot view, modify, or delete User Alice's resources.
  - Owner-scoped "clear all" deletes only requesting user's data.
"""
import pytest


def test_unauthenticated_requests_are_rejected(client):
    """Verify unauthenticated requests cannot access conversations or messages (audit S3)."""
    assert client.get("/conversations").status_code == 401
    assert client.post("/conversations", json={"title": "Test"}).status_code == 401
    assert client.get("/messages/any-id").status_code == 401
    assert client.delete("/conversations").status_code == 401


def test_conversation_and_message_lifecycle(auth_client):
    created = auth_client.post("/conversations", json={"title": "Initial"})
    assert created.status_code == 201
    conversation = created.json()["data"]

    updated = auth_client.patch(
        f"/conversations/{conversation['id']}",
        json={"title": "Updated"},
    )
    assert updated.status_code == 200
    assert updated.json()["data"]["title"] == "Updated"

    message = auth_client.post(
        "/messages",
        json={
            "conversation_id": conversation["id"],
            "role": "user",
            "content": "Hello",
        },
    )
    assert message.status_code == 201

    history = auth_client.get(f"/messages/{conversation['id']}")
    assert history.status_code == 200
    assert history.json()["data"][0]["content"] == "Hello"
    assert history.json()["meta"]["total_count"] == 1

    deleted = auth_client.delete(f"/conversations/{conversation['id']}")
    assert deleted.status_code == 204
    assert deleted.content == b""


def test_pagination_reports_all_records(auth_client):
    for number in range(3):
        res = auth_client.post("/conversations", json={"title": f"Conversation {number}"})
        # Add a message so auto-purge doesn't prune it
        conv_id = res.json()["data"]["id"]
        auth_client.post("/messages", json={"conversation_id": conv_id, "role": "user", "content": "hi"})

    response = auth_client.get("/conversations?page=2&page_size=2")
    assert response.status_code == 200
    assert len(response.json()["data"]) == 1
    assert response.json()["meta"]["total_pages"] == 2


def test_missing_resources_keep_404_status(auth_client):
    response = auth_client.get("/messages/missing-conv-id-999")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_message_history_accepts_maximum_page_size(auth_client, conversation):
    response = auth_client.get(
        f"/messages/{conversation['id']}?page=1&page_size=200"
    )
    assert response.status_code == 200
    assert response.json()["meta"]["page_size"] == 200


def test_delete_message_lifecycle(auth_client, conversation):
    created = auth_client.post(
        "/messages",
        json={
            "conversation_id": conversation["id"],
            "role": "user",
            "content": "Message to delete",
        },
    )
    assert created.status_code == 201
    message_id = created.json()["data"]["id"]

    deleted = auth_client.delete(f"/messages/{message_id}")
    assert deleted.status_code == 204

    history = auth_client.get(f"/messages/{conversation['id']}")
    assert history.status_code == 200
    assert history.json()["meta"]["total_count"] == 0


def test_delete_conversation_message_scoped(auth_client, conversation):
    created = auth_client.post(
        "/messages",
        json={
            "conversation_id": conversation["id"],
            "role": "user",
            "content": "Scoped message",
        },
    )
    assert created.status_code == 201
    message_id = created.json()["data"]["id"]

    deleted = auth_client.delete(f"/conversations/{conversation['id']}/messages/{message_id}")
    assert deleted.status_code == 204


def test_delete_nonexistent_message_returns_404(auth_client):
    response = auth_client.delete("/messages/nonexistent-id-999")
    assert response.status_code == 404


# ── Data Isolation Tests (Phase 1 Acceptance Criteria) ───────────────────
def test_user_b_cannot_access_user_a_conversation(auth_client, bob_client, conversation):
    """
    Acceptance Test:
    User A (Alice) owns `conversation`.
    User B (Bob) must NOT be able to view, update, add messages to, or delete Alice's conversation.
    """
    alice_conv_id = conversation["id"]

    # Bob tries to read Alice's messages -> 404
    res = bob_client.get(f"/messages/{alice_conv_id}")
    assert res.status_code == 404

    # Bob tries to post message to Alice's conversation -> 404
    res = bob_client.post(
        "/messages",
        json={"conversation_id": alice_conv_id, "role": "user", "content": "Bob eavesdropping"},
    )
    assert res.status_code == 404

    # Bob tries to rename Alice's conversation -> 404
    res = bob_client.patch(f"/conversations/{alice_conv_id}", json={"title": "Hacked Title"})
    assert res.status_code == 404

    # Bob tries to delete Alice's conversation -> 404
    res = bob_client.delete(f"/conversations/{alice_conv_id}")
    assert res.status_code == 404

    # Bob lists conversations -> Alice's conversation is NOT returned
    res = bob_client.get("/conversations")
    assert res.status_code == 200
    bob_conv_ids = [c["id"] for c in res.json()["data"]]
    assert alice_conv_id not in bob_conv_ids


def test_clear_all_is_owner_scoped(auth_client, bob_client):
    """
    Acceptance Test:
    Alice clears all her conversations.
    Bob's conversations MUST NOT be touched (audit S3 fix).
    """
    # Create conversation for Alice
    alice_res = auth_client.post("/conversations", json={"title": "Alice Chat"})
    alice_id = alice_res.json()["data"]["id"]
    auth_client.post("/messages", json={"conversation_id": alice_id, "role": "user", "content": "A"})

    # Create conversation for Bob
    bob_res = bob_client.post("/conversations", json={"title": "Bob Chat"})
    bob_id = bob_res.json()["data"]["id"]
    bob_client.post("/messages", json={"conversation_id": bob_id, "role": "user", "content": "B"})

    # Alice calls DELETE /conversations
    del_res = auth_client.delete("/conversations")
    assert del_res.status_code == 200

    # Alice's conversation list is empty
    alice_list = auth_client.get("/conversations").json()["data"]
    assert len(alice_list) == 0

    # Bob's conversation still exists!
    bob_list = bob_client.get("/conversations").json()["data"]
    assert any(c["id"] == bob_id for c in bob_list)
