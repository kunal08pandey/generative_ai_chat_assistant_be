"""
Groq LLM Client
===============

High-performance LLM client leveraging Groq's LPU Inference Engine.
Uses Groq's OpenAI-compatible HTTP API for both streaming and non-streaming generation,
and supports dynamic model discovery as well as categorized model listing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

import httpx
from dotenv import load_dotenv

from src.llm.base import BaseLLM
from src.utils.cache import llm_cache
from src.utils.http_client import get_http_client
from src.utils.logger import get_logger
from src.utils.token_counter import count_tokens

logger = get_logger(__name__)


class GroqClient(BaseLLM):
    """
    Groq LPU Inference client implementing BaseLLM.
    Supports chat completion, streaming SSE tokens, vision inputs,
    and querying available models on Groq.
    """
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"

    # Default categorized model registry matching Groq's active catalog
    DEFAULT_CATEGORIZED_MODELS: Dict[str, List[str]] = {
        "Alibaba Cloud": [
            "qwen/qwen3.8-27b",
            "qwen-2.5-32b",
            "qwen-2.5-coder-32b",
        ],
        "Canopy Labs": [
            "canopylabs/orpheus-v1-english",
            "canopylabs/orpheus-arabic-saudi",
        ],
        "Meta": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "meta-llama/llama-prompt-guard-2-22m",
            "meta-llama/llama-prompt-guard-2-86m",
            "llama-guard-3-8b",
            "llama-3.2-11b-vision-preview",
            "llama-3.2-90b-vision-preview",
            "llama-3.2-3b-preview",
            "llama-3.2-1b-preview",
        ],
        "OpenAI": [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "whisper-large-v3",
            "whisper-large-v3-turbo",
            "openai/gpt-oss-safeguard-20b",
        ],
        "Mistral AI": [
            "mixtral-8x7b-32768",
        ],
        "Google": [
            "gemma2-9b-it",
        ],
        "DeepSeek": [
            "deepseek-r1-distill-llama-70b",
        ],
    }

    @classmethod
    def get_api_key(cls, explicit_key: Optional[str] = None) -> str:
        """
        Resolve the GROQ_API_KEY from parameter, environment, or .env files.
        """
        if explicit_key and explicit_key.strip():
            return explicit_key.strip()

        # Check existing environment variable
        env_key = os.environ.get("GROQ_API_KEY")
        if env_key and env_key.strip():
            return env_key.strip()

        # Search parent candidate paths for .env
        backend_dir = Path(__file__).resolve().parent.parent.parent
        candidates = [
            backend_dir / ".env",
            backend_dir.parent / ".env",
        ]
        for path in candidates:
            if path.is_file():
                load_dotenv(path, override=True)
                key = os.environ.get("GROQ_API_KEY")
                if key and key.strip():
                    return key.strip()

        return ""

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> List[str]:
        """
        List all models available for Groq.

        If a GROQ_API_KEY is available, dynamically queries the live Groq
        GET /models API. Falls back cleanly to the curated model list.
        """
        key = cls.get_api_key(api_key)
        if key:
            try:
                headers = {"Authorization": f"Bearer {key}"}
                with httpx.Client(timeout=5.0) as client:
                    resp = client.get(f"{cls.GROQ_BASE_URL}/models", headers=headers)
                    if resp.status_code == 200:
                        data = resp.json().get("data", [])
                        # Filter to active chat/completion models
                        model_ids = [
                            m["id"] for m in data
                            if m.get("id") and m.get("active", True)
                        ]
                        if model_ids:
                            logger.info("Successfully fetched %d models live from Groq API", len(model_ids))
                            return sorted(list(dict.fromkeys(model_ids)))
            except Exception as exc:
                logger.debug("Live Groq models query failed (%s); using curated models", exc)

        # Fallback to curated models across all categories
        flat_models: List[str] = []
        for models in cls.DEFAULT_CATEGORIZED_MODELS.values():
            flat_models.extend(models)
        return list(dict.fromkeys(flat_models))

    @classmethod
    def list_models_categorized(cls, api_key: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Return available Groq models grouped by vendor/organization
        (Alibaba Cloud, Canopy Labs, Meta, OpenAI, Mistral AI, etc.).
        """
        available = set(cls.list_models(api_key))
        categorized: Dict[str, List[str]] = {}

        for category, models in cls.DEFAULT_CATEGORIZED_MODELS.items():
            matching = [m for m in models if m in available]
            if matching:
                categorized[category] = matching

        # Collect any unassigned live models under "Other"
        assigned = {m for group in categorized.values() for m in group}
        unassigned = [m for m in available if m not in assigned]
        if unassigned:
            categorized["Other"] = unassigned

        return categorized

    def __init__(self, model: str, api_key: Optional[str] = None):
        """
        Initialize GroqClient for a specific model.
        """
        self.model = model.strip()
        self.api_key = self.get_api_key(api_key)
        self.base_url = f"{self.GROQ_BASE_URL}/chat/completions"
        logger.info("Initialized GroqClient for model=%s", self.model)

    def _ensure_api_key(self) -> str:
        key = self.api_key or self.get_api_key()
        if not key:
            raise ValueError(
                "No Groq API key found. Please set GROQ_API_KEY in your backend/.env "
                "or environment variables to use Groq models."
            )
        self.api_key = key
        return key

    def _build_payload(self, prompt: str, images: Optional[List[str]] = None, stream: bool = False) -> dict:
        """
        Build an OpenAI-compatible chat completion payload.
        """
        if images and len(images) > 0:
            content: List[dict] = [{"type": "text", "text": prompt}]
            for img in images:
                # Ensure data URI formatting
                img_url = img if img.startswith("data:") else f"data:image/jpeg;base64,{img}"
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img_url},
                })
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": prompt}]

        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "temperature": 0.7,
        }

    async def generate(self, prompt: str, images: Optional[List[str]] = None) -> str:
        """
        Full non-streaming generation via Groq API.
        """
        key = self._ensure_api_key()
        image_info = f"img:{len(images)}" if images else "no-img"
        cache_key = f"groq:{self.model}:{prompt}:{image_info}"

        cached = llm_cache.get(cache_key)
        if cached:
            logger.info("Serving Groq response from cache.")
            return cached

        prompt_tokens = count_tokens(prompt)
        logger.info(
            "Sending Groq request to %s (Est tokens: %d, Images: %d)",
            self.model,
            prompt_tokens,
            len(images) if images else 0,
        )

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        payload = self._build_payload(prompt, images=images, stream=False)

        client = get_http_client()
        try:
            response = await client.post(self.base_url, headers=headers, json=payload, timeout=60.0)
            if response.status_code != 200:
                err_detail = response.text
                try:
                    err_json = response.json()
                    err_detail = err_json.get("error", {}).get("message", err_detail)
                except Exception:
                    pass
                raise Exception(f"Groq API error ({response.status_code}): {err_detail}")

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            llm_cache.set(cache_key, content)
            return content

        except Exception as exc:
            logger.error("Groq generation failed for %s: %s", self.model, exc)
            raise Exception(f"Groq error: {str(exc)}") from exc

    async def generate_stream(self, prompt: str, images: Optional[List[str]] = None) -> AsyncIterator[str]:
        """
        Stream response tokens from Groq via Server-Sent Events (SSE).
        """
        key = self._ensure_api_key()
        prompt_tokens = count_tokens(prompt)
        logger.info("Streaming Groq request to %s (Est prompt tokens: %d)", self.model, prompt_tokens)

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        payload = self._build_payload(prompt, images=images, stream=True)

        client = get_http_client()
        try:
            async with client.stream("POST", self.base_url, headers=headers, json=payload, timeout=60.0) as response:
                if response.status_code != 200:
                    err_bytes = await response.aread()
                    err_msg = err_bytes.decode("utf-8", errors="replace")
                    try:
                        err_json = json.loads(err_msg)
                        err_msg = err_json.get("error", {}).get("message", err_msg)
                    except Exception:
                        pass
                    raise Exception(f"Groq streaming error ({response.status_code}): {err_msg}")

                async for line in response.aiter_lines():
                    trimmed = line.strip()
                    if not trimmed or not trimmed.startswith("data: "):
                        continue

                    data_str = trimmed[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            token = delta.get("content", "")
                            if token:
                                yield token
                    except json.JSONDecodeError:
                        continue

        except Exception as exc:
            logger.error("Groq streaming failed for %s: %s", self.model, exc)
            raise Exception(f"Groq stream error: {str(exc)}") from exc