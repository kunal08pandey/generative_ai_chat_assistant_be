class FakeChatClient:
    def __init__(self):
        self.prompt = None
        self.images = None

    async def generate_stream(self, prompt, images=None):
        self.prompt = prompt
        self.images = images
        for token in ("Hello", " world"):
            yield token


def test_chat_stream_and_persistence(client, conversation, monkeypatch):
    fake = FakeChatClient()
    monkeypatch.setattr("src.api.routes.chat.get_llm_client", lambda **kwargs: fake)

    response = client.post(
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

    history = client.get(f"/messages/{conversation['id']}").json()["data"]
    assert [message["role"] for message in history] == ["user", "assistant"]
    assert history[-1]["content"] == "Hello world"


def test_image_only_chat_uses_default_prompt(client, conversation, monkeypatch):
    fake = FakeChatClient()
    monkeypatch.setattr("src.api.routes.chat.get_llm_client", lambda **kwargs: fake)

    response = client.post(
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


def test_chat_requires_existing_conversation(client, monkeypatch):
    monkeypatch.setattr("src.api.routes.chat.get_llm_client", lambda **kwargs: FakeChatClient())
    response = client.post(
        "/chat/",
        json={"query": "Hello", "model": "model", "conversation_id": "missing"},
    )
    assert response.status_code == 404


def test_chat_requires_text_or_an_image(client, conversation):
    response = client.post(
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


def test_chat_rejects_unregistered_model(client, conversation):
    response = client.post(
        "/chat/",
        json={
            "query": "Hello",
            "model": "not-registered",
            "conversation_id": conversation["id"],
        },
    )
    assert response.status_code == 400
