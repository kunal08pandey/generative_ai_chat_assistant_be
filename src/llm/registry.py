"""
LLM Model Registry
===================

Central registry for advertised and available LLM models.
Supports capability awareness and dynamic discovery of local Ollama models and Groq cloud models.
"""
from __future__ import annotations

import time
import httpx

from src.settings import OLLAMA_BASE_URL
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    _last_ollama_check: float = 0.0
    _last_groq_check: float = 0.0
    _cache_ttl: float = 60.0

    _models = {
        "groq": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "canopylabs/orpheus-v1-english",
            "canopylabs/orpheus-arabic-saudi",
            "meta-llama/llama-prompt-guard-2-22m",
            "meta-llama/llama-prompt-guard-2-86m",
            "whisper-large-v3",
            "whisper-large-v3-turbo",
            "openai/gpt-oss-safeguard-20b",
            "deepseek-r1-distill-llama-70b",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        "gemini": [
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-3-flash-preview",
            "gemini-3.1-pro-preview",
            "gemini-4-argon",
        ],
        "ollama": [
            "gemma4:e4b",
            "qwen2.5-coder:14b",
            "gemma4:12b",
            "x/flux2-klein",
            "gen-labs-scout",
        ],
        "llama_cpp": [
            "local-gguf",
        ],
    }

    @classmethod
    def refresh_ollama_models(cls, force: bool = False) -> None:
        """
        Dynamically query local Ollama daemon to discover installed models.
        Cached with TTL to prevent blocking hot paths (Phase 4).
        """
        now = time.monotonic()
        if not force and (now - cls._last_ollama_check < cls._cache_ttl):
            return
        cls._last_ollama_check = now

        url = f"{OLLAMA_BASE_URL}/api/tags"
        try:
            with httpx.Client(timeout=0.8) as client:
                res = client.get(url)
                if res.status_code == 200:
                    data = res.json()
                    discovered = [
                        m["name"] for m in data.get("models", []) if "name" in m
                    ]
                    if discovered:
                        if "x/flux2-klein" not in discovered:
                            discovered.append("x/flux2-klein")
                        cls._models["ollama"] = list(dict.fromkeys(discovered))
                        logger.info("Discovered %d Ollama models: %s", len(discovered), discovered)
        except Exception as e:
            logger.debug("Could not query Ollama models from %s: %s", url, e)

    @classmethod
    def refresh_groq_models(cls, force: bool = False) -> None:
        """
        Dynamically query Groq API for available models if GROQ_API_KEY is configured.
        Cached with TTL to avoid synchronous blocking (Phase 4).
        """
        now = time.monotonic()
        if not force and (now - cls._last_groq_check < cls._cache_ttl):
            return
        cls._last_groq_check = now

        try:
            from src.llm.groq_client import GroqClient
            live_models = GroqClient.list_models()
            if live_models:
                cls._models["groq"] = live_models
                logger.info("Registered %d Groq models", len(live_models))
        except Exception as e:
            logger.debug("Could not refresh Groq models: %s", e)

    @classmethod
    def get_capabilities(cls, model_name: str) -> list[str]:
        """
        Return declared capabilities for a model:
        'chat', 'vision', 'speech', 'code', 'image_generation'
        """
        caps = ["chat"]
        name = (model_name or "").lower().strip()

        # Vision capabilities
        if any(v in name for v in ["gemini", "argon", "flux", "vision", "vl", "pixtral"]):
            caps.append("vision")

        # Speech capabilities
        if any(s in name for s in ["whisper", "orpheus", "tts", "asr"]):
            caps.append("speech")

        # Code capabilities
        if any(c in name for c in ["coder", "code", "deepseek-r1"]):
            caps.append("code")

        # Image generation capabilities
        if any(img in name for img in ["flux", "imagen"]):
            caps.append("image_generation")

        return list(dict.fromkeys(caps))

    @classmethod
    def get_models_by_capability(cls, capability: str, include_ollama: bool = True) -> list[str]:
        """Return all models advertising a specific capability."""
        all_models = cls.get_all_models(include_ollama=include_ollama)
        return [m for m in all_models if capability in cls.get_capabilities(m)]

    @classmethod
    def get_models(cls, provider: str | None = None, include_ollama: bool = True) -> dict | list:
        """
        Get available models for a specific provider or all providers.
        If include_ollama is False, excludes Ollama models (for non-localhost clients).
        """
        if include_ollama:
            cls.refresh_ollama_models()

        cls.refresh_groq_models()

        if provider:
            if provider == "ollama" and not include_ollama:
                return []
            return cls._models.get(provider, [])

        filtered = {}
        for prov, models in cls._models.items():
            if prov == "ollama" and not include_ollama:
                continue
            filtered[prov] = models
        return filtered

    @classmethod
    def get_all_models(cls, include_ollama: bool = True) -> list[str]:
        """
        Get a flat list of all available models across providers.
        """
        models_dict = cls.get_models(include_ollama=include_ollama)
        all_models = []
        for models in models_dict.values():
            all_models.extend(models)
        return list(dict.fromkeys(all_models))

    @classmethod
    def get_providers(cls, include_ollama: bool = True) -> list[str]:
        """
        Get list of available providers.
        """
        return list(cls.get_models(include_ollama=include_ollama).keys())

    @classmethod
    def set_models(cls, provider: str, models: list[str]) -> None:
        """Replace a provider's advertised models with a custom list."""
        cls._models[provider] = list(dict.fromkeys(models))

    @classmethod
    def is_model_available(cls, model_name: str, include_ollama: bool = True) -> bool:
        """
        Check if a model is registered and available (audit finding 49 & Phase 4).
        Rejects arbitrary unregistered models.
        """
        if not model_name or not model_name.strip():
            return False

        name = model_name.strip()
        all_models = cls.get_all_models(include_ollama=include_ollama)

        # Exact match
        if name in all_models:
            return True

        # Check provider models directly without dynamic discovery if offline
        for prov, mlist in cls._models.items():
            if prov == "ollama" and not include_ollama:
                continue
            if name in mlist:
                return True
            # Match without tag if user passed tag or base
            if any(name == m.split(":")[0] or m == name.split(":")[0] for m in mlist):
                return True

        return False

    @classmethod
    def get_provider_for_model(cls, model_name: str) -> str:
        """Look up which provider owns a given model name."""
        for provider, models in cls._models.items():
            if model_name in models:
                return provider

        normalized = (model_name or "").lower().strip()
        if "gemini" in normalized or "argon" in normalized:
            return "gemini"
        if "gguf" in normalized:
            return "llama_cpp"
        if any(k in normalized for k in [
            "groq", "llama-3", "qwen", "canopylabs", "gpt-oss",
            "whisper", "deepseek", "mixtral", "orpheus", "prompt-guard"
        ]):
            return "groq"
        return "ollama"
