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
