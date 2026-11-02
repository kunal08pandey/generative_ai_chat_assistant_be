"""
Tests for GroqClient and Ollama localhost restriction
"""
import pytest
from starlette.testclient import TestClient

from src.llm.client_factory import get_llm_client
from src.llm.groq_client import GroqClient
from src.llm.registry import ModelRegistry
from src.utils.host_resolver import is_localhost, is_localhost_request


# ── GroqClient Model Listing Tests ──────────────────────────────────────────

def test_groq_client_list_models_fallback():
    """Verify list_models() returns curated models including screenshot categories."""
    models = GroqClient.list_models()
    assert isinstance(models, list)
    assert len(models) > 0

    # Check key models from user screenshot
    assert "qwen/qwen3.8-27b" in models
    assert "canopylabs/orpheus-v1-english" in models
    assert "canopylabs/orpheus-arabic-saudi" in models
    assert "meta-llama/llama-prompt-guard-2-22m" in models
    assert "meta-llama/llama-prompt-guard-2-86m" in models
    assert "openai/gpt-oss-120b" in models
    assert "openai/gpt-oss-20b" in models
    assert "whisper-large-v3" in models
    assert "whisper-large-v3-turbo" in models
    assert "openai/gpt-oss-safeguard-20b" in models


def test_groq_client_list_models_categorized():
    """Verify list_models_categorized() groups models under expected vendors."""
    categorized = GroqClient.list_models_categorized()
    assert "Alibaba Cloud" in categorized
    assert "Canopy Labs" in categorized
    assert "Meta" in categorized
    assert "OpenAI" in categorized

    assert "qwen/qwen3.8-27b" in categorized["Alibaba Cloud"]
    assert "canopylabs/orpheus-v1-english" in categorized["Canopy Labs"]
    assert "meta-llama/llama-prompt-guard-2-22m" in categorized["Meta"]
    assert "openai/gpt-oss-120b" in categorized["OpenAI"]


def test_groq_client_live_api_mock(monkeypatch):
    """Test dynamic live model discovery when GROQ_API_KEY is configured."""
    class FakeResponse:
        status_code = 200
        def json(self):
            return {
                "data": [
                    {"id": "llama-3.3-70b-versatile", "active": True},
                    {"id": "custom-groq-model", "active": True},
                    {"id": "inactive-model", "active": False},
                ]
            }

    class FakeClient:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def get(self, url, headers=None):
            return FakeResponse()

    monkeypatch.setattr("src.llm.groq_client.httpx.Client", lambda **kwargs: FakeClient())
    models = GroqClient.list_models(api_key="gsk_test_key_mock")
    assert "llama-3.3-70b-versatile" in models
    assert "custom-groq-model" in models
    assert "inactive-model" not in models


def test_client_factory_groq():
    """Verify client factory creates a GroqClient instance."""
    client = get_llm_client(provider="groq", model="llama-3.3-70b-versatile")
    assert isinstance(client, GroqClient)
    assert client.model == "llama-3.3-70b-versatile"


# ── Host Resolver Tests ─────────────────────────────────────────────────────

def test_is_localhost():
    assert is_localhost("127.0.0.1") is True
    assert is_localhost("localhost") is True
    assert is_localhost("::1") is True
    assert is_localhost("testclient") is True
    assert is_localhost("127.0.0.5") is True

    assert is_localhost("192.168.1.100") is False
    assert is_localhost("10.0.0.1") is False
    assert is_localhost("example.com") is False
    assert is_localhost(None) is False


# ── Localhost Ollama Availability Tests ─────────────────────────────────────

def test_models_endpoint_on_localhost(client: TestClient):
    """Localhost requests to /models/ include Ollama."""
    res = client.get("/models/")
    assert res.status_code == 200
    data = res.json()["data"]
    assert "groq" in data["models_by_provider"]
    assert "ollama" in data["models_by_provider"]
    assert data["is_localhost"] is True


def test_models_endpoint_on_remote_client(client: TestClient):
    """Remote requests to /models/ must NOT include Ollama models."""
    res = client.get("/models/", headers={"X-Forwarded-For": "203.0.113.195"})
    assert res.status_code == 200
    data = res.json()["data"]
    assert "groq" in data["models_by_provider"]
    assert "ollama" not in data["models_by_provider"]
    assert data["is_localhost"] is False


def test_chat_rejects_ollama_for_remote_client(client: TestClient, conversation):
    """Remote requests attempting to call Ollama via /chat/ are forbidden (403)."""
    res = client.post(
        "/chat/",
        headers={"X-Forwarded-For": "203.0.113.195"},
        json={
            "query": "Hello",
            "model": "gemma4:e4b",
            "provider": "ollama",
            "conversation_id": conversation["id"],
        },
    )
    assert res.status_code == 403
    err_body = res.json()
    err_msg = err_body.get("error", {}).get("message") or err_body.get("detail", "")
    assert "Ollama models are only available on localhost" in err_msg


def test_rag_rejects_ollama_for_remote_client(client: TestClient, conversation):
    """Remote requests attempting to call Ollama via /rag/ are forbidden (403)."""
    res = client.post(
        "/rag/",
        headers={"X-Forwarded-For": "203.0.113.195"},
        json={
            "query": "Hello",
            "model": "gemma4:e4b",
            "provider": "ollama",
            "conversation_id": conversation["id"],
        },
    )
    assert res.status_code == 403
    err_body = res.json()
    err_msg = err_body.get("error", {}).get("message") or err_body.get("detail", "")
    assert "Ollama models are only available on localhost" in err_msg


def test_groq_categorized_endpoint(client: TestClient):
    """Verify /models/groq returns categorized models."""
    res = client.get("/models/groq")
    assert res.status_code == 200
    data = res.json()["data"]
    assert "categories" in data
    assert "Meta" in data["categories"]
    assert "OpenAI" in data["categories"]
