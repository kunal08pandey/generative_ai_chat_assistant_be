from src.llm.llama_cpp_client import LlamaCppClient


def test_llama_cpp_environment_configuration(monkeypatch):
    monkeypatch.setenv("LLAMA_CPP_HOST", "192.0.2.10")
    monkeypatch.setenv("LLAMA_CPP_PORT", "9000")
    client = LlamaCppClient("local-gguf")
    assert client.base_url == "http://192.0.2.10:9000"
