from src.llm.base import BaseLLM
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_llm_client(provider: str, model: str) -> BaseLLM:
    """
    Factory function that returns the appropriate LLM client
    based on the provider string.

    Supported providers:
        - "ollama"    → OllamaClient
        - "llama_cpp" → LlamaCppClient

    :param provider: The backend provider identifier.
    :param model: The model name to pass to the client.
    :returns: An instance of BaseLLM.
    :raises ValueError: If the provider is not recognized.
    """
    if provider == "ollama":
        from src.llm.ollama_client import OllamaClient
        logger.info(f"Creating OllamaClient for model={model}")
        return OllamaClient(model=model)

    elif provider == "llama_cpp":
        from src.llm.llama_cpp_client import LlamaCppClient
        logger.info(f"Creating LlamaCppClient for model={model}")
        return LlamaCppClient(model=model)

    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            f"Supported providers: ollama, llama_cpp"
        )
