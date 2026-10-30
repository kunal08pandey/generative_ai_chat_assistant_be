import pytest

from src.llm.client_factory import get_llm_client
from src.llm.llama_cpp_client import LlamaCppClient
from src.llm.ollama_client import OllamaClient
from src.settings import OLLAMA_BASE_URL


def test_factory_builds_supported_clients():
    assert isinstance(get_llm_client("ollama", "model"), OllamaClient)
    assert isinstance(get_llm_client("llama_cpp", "model"), LlamaCppClient)


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_llm_client("unknown", "model")


def test_ollama_uses_shared_base_url():
    assert OllamaClient("model").base_url == f"{OLLAMA_BASE_URL}/api/generate"


def test_registry_resolves_gemini_models():
    from src.llm.registry import ModelRegistry
    assert ModelRegistry.get_provider_for_model("gemini-pro") == "gemini"
    assert ModelRegistry.get_provider_for_model("gemini-argon") == "gemini"
    assert ModelRegistry.get_provider_for_model("gemini-4-argon") == "gemini"
    assert ModelRegistry.get_provider_for_model("gemini-2.5-pro") == "gemini"


def test_gemini_alias_resolution(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-1234567890")
    from src.llm.gemini_client import GeminiClient
    client_pro = GeminiClient("gemini-pro")
    assert client_pro.model_name == "gemini-3.1-pro-preview"

    client_argon = GeminiClient("gemini-argon")
    assert client_argon.model_name == "gemini-4-argon"
