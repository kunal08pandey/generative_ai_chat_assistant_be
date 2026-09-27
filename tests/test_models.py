from src.llm.registry import ModelRegistry


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "models": [
                {"name": "chat-model:latest", "capabilities": ["completion"]},
                {"name": "embed-model:latest", "capabilities": ["embedding"]},
            ]
        }


class FakeHTTPClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get(self, _url):
        return FakeResponse()


def test_models_only_advertises_installed_generation_models(client, monkeypatch):
    monkeypatch.setattr(
        ModelRegistry,
        "_models",
        {provider: list(models) for provider, models in ModelRegistry._models.items()},
    )
    monkeypatch.setattr(
        "src.api.routes.models.httpx.AsyncClient",
        lambda **_kwargs: FakeHTTPClient(),
    )

    response = client.get("/models/")

    assert response.status_code == 200
    models = response.json()["data"]["models_by_provider"]["ollama"]
    assert models == ["chat-model:latest"]
