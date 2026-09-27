class FakeResponse:
    def raise_for_status(self):
        return None


class FakeHTTPClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get(self, _url):
        return FakeResponse()


def test_health_checks_core_services(client, monkeypatch):
    monkeypatch.setattr(
        "src.api.routes.health.httpx.AsyncClient",
        lambda **_kwargs: FakeHTTPClient(),
    )
    response = client.get("/health")
    assert response.status_code == 200
    health = response.json()["data"]
    assert health["status"] == "healthy"
    assert {service["name"] for service in health["services"]} == {
        "sqlite",
        "ollama",
        "chroma",
    }
