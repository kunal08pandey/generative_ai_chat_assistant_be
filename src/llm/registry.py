"""
LLM Model Registry
===================

Central registry for advertised and available LLM models.
Supports dynamic discovery of local Ollama models via /api/tags.
"""
from __future__ import annotations

import httpx

from src.utils.host_resolver import get_ollama_host
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    _models = {
        "ollama": ["gemma4:e4b", "qwen2.5-coder:14b", "gemma4:12b", "x/z-image-turbo:latest"],
        "llama_cpp": ["local-gguf"],
        "gemini": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.5-flash-8b"],
    }

    @classmethod
    def refresh_ollama_models(cls) -> None:
        """
        Dynamically query local Ollama daemon to discover installed models.
        """
        host = get_ollama_host()
        url = f"http://{host}:11434/api/tags"
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(url)
                if res.status_code == 200:
                    data = res.json()
                    discovered = [
                        m["name"] for m in data.get("models", []) if "name" in m
                    ]
                    if discovered:
                        cls._models["ollama"] = list(dict.fromkeys(discovered))
                        logger.info("Discovered %d Ollama models: %s", len(discovered), discovered)
        except Exception as e:
            logger.debug("Could not query Ollama models from %s: %s", url, e)

    @classmethod
    def get_models(cls, provider: str | None = None) -> dict | list:
        """Get available models for a specific provider or all providers."""
        cls.refresh_ollama_models()
        if provider:
            return cls._models.get(provider, [])
        return cls._models

    @classmethod
    def get_all_models(cls) -> list[str]:
        """Get a flat list of all available models across providers."""
        cls.refresh_ollama_models()
        all_models = []
        for models in cls._models.values():
            all_models.extend(models)
        return list(dict.fromkeys(all_models))

    @classmethod
    def get_providers(cls) -> list[str]:
        """Get list of available providers."""
        return list(cls._models.keys())

    @classmethod
    def set_models(cls, provider: str, models: list[str]) -> None:
        """Replace a provider's advertised models with a custom list."""
        cls._models[provider] = list(dict.fromkeys(models))

    @classmethod
    def is_model_available(cls, model_name: str) -> bool:
        """Check if a specific model is non-empty and valid."""
        if not model_name or not model_name.strip():
            return False
        return True

    @classmethod
    def get_provider_for_model(cls, model_name: str) -> str:
        """Look up which provider owns a given model name."""
        for provider, models in cls._models.items():
            if model_name in models:
                return provider
        return "ollama"
