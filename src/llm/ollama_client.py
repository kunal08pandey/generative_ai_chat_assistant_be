"""
Ollama LLM Client
==================

Integrates with local Ollama HTTP API.
Reuses the shared HTTP connection pool client and resolves host environment dynamically.
"""
import json
from typing import AsyncIterator

import httpx

from .base import BaseLLM
from src.utils.cache import llm_cache
from src.utils.host_resolver import get_ollama_host
from src.utils.http_client import get_http_client
from src.utils.logger import get_logger
from src.utils.token_counter import count_tokens

logger = get_logger(__name__)


class OllamaClient(BaseLLM):

    def __init__(self, model: str):
        host = get_ollama_host()
        self.base_url = f"http://{host}:11434/api/generate"
        self.model = model
        logger.info("Initialized OllamaClient with model=%s, host=%s", model, host)

    async def generate(self, prompt: str, images: list[str] | None = None) -> str:
        """Full (non-streaming) generation. Used by RAG sub-chains."""
        image_info = f"img:{len(images)}" if images else "no-img"
        cache_key = f"{self.model}:{prompt}:{image_info}"

        cached_response = llm_cache.get(cache_key)
        if cached_response:
            logger.info("Serving LLM response from cache.")
            return cached_response

        prompt_tokens = count_tokens(prompt)
        logger.info(
            "Sending LLM request to %s (Estimated prompt tokens: %d, Images: %d)",
            self.model,
            prompt_tokens,
            len(images) if images else 0,
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if images:
            payload["images"] = images

        client = get_http_client()
        try:
            response = await client.post(self.base_url, json=payload, timeout=120.0)
            response.raise_for_status()

            resp_data = response.json()
            generated_text = resp_data.get("response", "")

            eval_count = resp_data.get("eval_count", count_tokens(generated_text))
            logger.info("LLM request successful. Tokens generated: %d", eval_count)

            llm_cache.set(cache_key, generated_text)
            return generated_text
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            raise Exception(f"LLM error: {str(e)}")

    async def generate_stream(self, prompt: str, images: list[str] | None = None) -> AsyncIterator[str]:
        """Streams tokens from Ollama one at a time via NDJSON."""
        prompt_tokens = count_tokens(prompt)
        logger.info("Streaming LLM request to %s (Estimated prompt tokens: %d)", self.model, prompt_tokens)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
        }
        if images:
            payload["images"] = images

        client = get_http_client()
        try:
            async with client.stream("POST", self.base_url, json=payload, timeout=120.0) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done", False):
                            return
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error("LLM streaming failed: %s", e)
            raise Exception(f"LLM stream error: {str(e)}")
