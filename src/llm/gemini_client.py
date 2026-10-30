import asyncio
import base64
import os
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.llm.base import BaseLLM
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GeminiClient(BaseLLM):
    # Canonical mappings for friendly or alias model names
    MODEL_ALIASES = {
        "gemini-argon": "gemini-4-argon",
        "gemini argon": "gemini-4-argon",
        "argon": "gemini-4-argon",
        "gemini-pro": "gemini-3.1-pro-preview",
        "gemini pro": "gemini-3.1-pro-preview",
        "gemini-2.5-pro": "gemini-3.1-pro-preview",
    }

    def __init__(self, model: str):
        raw_model = model.strip()
        if raw_model.endswith("-latest"):
            raw_model = raw_model.replace("-latest", "")
        self.model = model
        self.model_name = self.MODEL_ALIASES.get(raw_model.lower(), raw_model)

        # Locate .env in backend directory or project root
        backend_dir = Path(__file__).resolve().parent.parent.parent
        candidates = [
            backend_dir / ".env",
            backend_dir.parent / ".env",
        ]
        loaded_from = None
        for path in candidates:
            if path.is_file():
                load_dotenv(path, override=True)
                loaded_from = path
                break
        else:
            load_dotenv(override=True)

        logger.info(f"Loaded environment from: {loaded_from or 'system environment'}")
        api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
        if not api_key:
            checked = [str(p) for p in candidates]
            raise ValueError(
                f"No API key provided. GEMINI_API_KEY was not found in environment or .env files ({checked}). "
                "Please ensure GEMINI_API_KEY is defined in backend/.env and restart the server."
            )

        logger.info(
            "Gemini API Key detected (prefix: %s..., length: %d). Model: %s (resolved: %s)",
            api_key[:6],
            len(api_key),
            self.model,
            self.model_name,
        )
        self.client = genai.Client(api_key=api_key)

    def _prepare_contents(self, prompt: str, images: list[str] | None = None) -> list:
        """Format text prompt and optional base64 vision images into Google GenAI content parts."""
        contents = []
        if images:
            for img in images:
                mime = "image/jpeg"
                raw_b64 = img
                if "," in img and "data:" in img:
                    header, raw_b64 = img.split(",", 1)
                    if ";" in header:
                        mime = header.split(";")[0].replace("data:", "")
                try:
                    img_bytes = base64.b64decode(raw_b64)
                    contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
                except Exception as e:
                    logger.warning("Failed to decode base64 image: %s", e)
        contents.append(prompt)
        return contents

    async def generate(self, prompt: str, images: list[str] | None = None) -> str:
        contents = self._prepare_contents(prompt, images)
        max_retries = 3
        backoff = 1.5
        for attempt in range(max_retries):
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                )
                return response.text or ""
            except Exception as e:
                err_msg = str(e).lower()
                is_transient = any(k in err_msg for k in ["503", "unavailable", "high demand", "resource_exhausted", "429"])
                if is_transient and attempt < max_retries - 1:
                    wait_time = backoff ** (attempt + 1)
                    logger.warning(
                        "Gemini generate transient error (%s) on attempt %d/%d for model=%s. Retrying in %.1fs...",
                        e, attempt + 1, max_retries, self.model_name, wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue
                logger.error("Gemini generation failed for model=%s: %s", self.model_name, e)
                raise Exception(f"Gemini error ({self.model_name}): {str(e)}") from e

    async def generate_stream(self, prompt: str, images: list[str] | None = None) -> AsyncIterator[str]:
        contents = self._prepare_contents(prompt, images)
        max_retries = 3
        backoff = 1.5
        for attempt in range(max_retries):
            yielded_any = False
            try:
                response_stream = await self.client.aio.models.generate_content_stream(
                    model=self.model_name,
                    contents=contents,
                )
                async for chunk in response_stream:
                    if chunk.text:
                        yielded_any = True
                        yield chunk.text
                return
            except Exception as e:
                err_msg = str(e).lower()
                is_transient = any(k in err_msg for k in ["503", "unavailable", "high demand", "resource_exhausted", "429"])
                if is_transient and not yielded_any and attempt < max_retries - 1:
                    wait_time = backoff ** (attempt + 1)
                    logger.warning(
                        "Gemini stream transient error (%s) on attempt %d/%d for model=%s. Retrying in %.1fs...",
                        e, attempt + 1, max_retries, self.model_name, wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue

                logger.error("Gemini streaming failed for model=%s: %s", self.model_name, e)
                raise Exception(f"Gemini stream error ({self.model_name}): {str(e)}") from e
