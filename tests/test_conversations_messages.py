def test_conversation_and_message_lifecycle(client):
    created = client.post("/conversations", json={"title": "Initial"})
    assert created.status_code == 201
    conversation = created.json()["data"]

    updated = client.patch(
        f"/conversations/{conversation['id']}",
        json={"title": "Updated"},
    )
    assert updated.status_code == 200
    assert updated.json()["data"]["title"] == "Updated"

    message = client.post(
        "/messages",
        json={
            "conversation_id": conversation["id"],
            "role": "user",
            "content": "Hello",
        },
    )
    assert message.status_code == 201

    history = client.get(f"/messages/{conversation['id']}")
    assert history.status_code == 200
    assert history.json()["data"][0]["content"] == "Hello"
    assert history.json()["meta"]["total_count"] == 1

    deleted = client.delete(f"/conversations/{conversation['id']}")
    assert deleted.status_code == 204
    assert deleted.content == b""


def test_pagination_reports_all_records(client):
    for number in range(3):
        client.post("/conversations", json={"title": f"Conversation {number}"})

    response = client.get("/conversations?page=2&page_size=2")
    assert response.status_code == 200
    assert len(response.json()["data"]) == 1
    assert response.json()["meta"]["total_pages"] == 2


def test_missing_resources_keep_404_status(client):
    response = client.get("/messages/missing")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_message_history_accepts_maximum_page_size(client, conversation):
    response = client.get(
        f"/messages/{conversation['id']}?page=1&page_size=200"
    )
    assert response.status_code == 200
    assert response.json()["meta"]["page_size"] == 200
