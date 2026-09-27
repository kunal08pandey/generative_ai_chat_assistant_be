"""
llama.cpp LLM Client
=====================

Integrates with local llama-server HTTP completion API.
Reuses the shared HTTP connection pool client for improved request performance.
"""
import json
import os
from typing import AsyncIterator

from .base import BaseLLM
from src.utils.cache import llm_cache
from src.utils.http_client import get_http_client
from src.utils.logger import get_logger
from src.utils.token_counter import count_tokens

logger = get_logger(__name__)


class LlamaCppClient(BaseLLM):
    """
    Client for the llama.cpp HTTP server.
    Communicates with the /completion endpoint exposed by llama-server.

    Environment variables:
        LLAMA_CPP_HOST: Server hostname (default: localhost)
        LLAMA_CPP_PORT: Server port (default: 8080)
    """

    def __init__(self, model: str):
        host = os.environ.get("LLAMA_CPP_HOST", "localhost")
        port = os.environ.get("LLAMA_CPP_PORT", "8080")
        self.base_url = f"http://{host}:{port}"
        self.model = model
        logger.info("Initialized LlamaCppClient with model=%s, server=%s:%s", model, host, port)

    async def generate(self, prompt: str, images: list[str] | None = None) -> str:
        """Full (non-streaming) generation."""
        image_info = f"img:{len(images)}" if images else "no-img"
        cache_key = f"llama_cpp:{self.model}:{prompt}:{image_info}"

        cached_response = llm_cache.get(cache_key)
        if cached_response:
            logger.info("Serving llama.cpp response from cache.")
            return cached_response

        prompt_tokens = count_tokens(prompt)
        logger.info(
            "Sending llama.cpp request to %s (Estimated prompt tokens: %d, Images: %d)",
            self.model,
            prompt_tokens,
            len(images) if images else 0,
        )

        payload = {
            "prompt": prompt,
            "n_predict": 512,
            "stream": False,
            "temperature": 0.7,
            "stop": ["\nUser:", "\n###"],
        }

        if images:
            payload["image_data"] = [
                {"data": img, "id": idx}
                for idx, img in enumerate(images)
            ]

        client = get_http_client()
        try:
            response = await client.post(
                f"{self.base_url}/completion",
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()

            resp_data = response.json()
            generated_text = resp_data.get("content", "")

            tokens_predicted = resp_data.get("tokens_predicted", count_tokens(generated_text))
            tokens_evaluated = resp_data.get("tokens_evaluated", prompt_tokens)
            logger.info(
                "llama.cpp request successful. Tokens evaluated: %d, Tokens predicted: %d",
                tokens_evaluated,
                tokens_predicted,
            )

            llm_cache.set(cache_key, generated_text)
            return generated_text
        except Exception as e:
            logger.error("llama.cpp generation failed: %s", e)
            raise Exception(f"LLM error: {str(e)}")

    async def generate_stream(self, prompt: str, images: list[str] | None = None) -> AsyncIterator[str]:
        """Streams tokens from the llama.cpp server via NDJSON chunks."""
        prompt_tokens = count_tokens(prompt)
        logger.info("Streaming llama.cpp request to %s (Estimated prompt tokens: %d)", self.model, prompt_tokens)

        payload = {
            "prompt": prompt,
            "n_predict": 512,
            "stream": True,
            "temperature": 0.7,
            "stop": ["\nUser:", "\n###"],
        }

        if images:
            payload["image_data"] = [
                {"data": img, "id": idx}
                for idx, img in enumerate(images)
            ]

        client = get_http_client()
        try:
            async with client.stream("POST", f"{self.base_url}/completion", json=payload, timeout=120.0) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    # llama.cpp streams lines like "data: {...}"
                    text = line.removeprefix("data: ").strip()
                    if not text:
                        continue
                    try:
                        chunk = json.loads(text)
                        token = chunk.get("content", "")
                        if token:
                            yield token
                        if chunk.get("stop", False):
                            return
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error("llama.cpp streaming failed: %s", e)
            raise Exception(f"LLM stream error: {str(e)}")

    async def health(self) -> dict:
        """Check if the llama.cpp server is reachable and ready."""
        client = get_http_client()
        try:
            resp = await client.get(f"{self.base_url}/health", timeout=5.0)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error("llama.cpp health check failed: %s", e)
            return {"status": "error", "message": str(e)}
