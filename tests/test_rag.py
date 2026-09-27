class FakeRagClient:
    def __init__(self):
        self.stream_images = None

    async def generate(self, prompt, images=None):
        return "optimized query"

    async def generate_stream(self, prompt, images=None):
        self.stream_images = images
        yield "RAG answer"


def test_rag_stream_uses_context_and_images(client, conversation, monkeypatch):
    fake = FakeRagClient()
    monkeypatch.setattr("src.api.routes.rag.get_llm_client", lambda **kwargs: fake)

    async def retrieve(_query):
        return "document context"

    monkeypatch.setattr("src.api.routes.rag.retrieve_context", retrieve)
    response = client.post(
        "/rag/",
        json={
            "query": "Question",
            "model": "gen-labs-scout",
            "conversation_id": conversation["id"],
            "images": ["aW1hZ2U="],
        },
    )

    assert response.status_code == 200
    assert "Optimizing search query" in response.text
    assert "RAG answer" in response.text
    assert fake.stream_images == ["aW1hZ2U="]
