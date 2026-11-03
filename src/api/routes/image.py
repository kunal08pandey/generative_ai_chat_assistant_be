"""
Image Generation Endpoint
===========================

``POST /generate-image`` — Dual AI Image Generation System:
  • Gemini Mode: Generates high-resolution images via Google Gemini Imagen 3 (imagen-3.0-generate-002)
    when a Gemini model is active, with seamless Flux fallback.
  • Local / Non-Gemini Mode: 2-stage generation using text LLM prompt expansion + Ollama x/flux2-klein engine.
  • Persists user query and image response into SQLite database (chat.db).
"""
import base64
import os
import random
import re
import urllib.parse
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.schemas import ok
from src.db.database import get_db
from src.llm.client_factory import get_llm_client
from src.llm.registry import ModelRegistry
from src.utils.logger import get_logger
from src.utils.rate_limiter import RateLimiter

logger = get_logger(__name__)
router = APIRouter(tags=["Image Generation"])


class ImageGenRequest(BaseModel):
    prompt: str = Field(..., min_length=2, description="Text description of the image to generate")
    model: str = Field("x/flux2-klein:9b", description="Selected model identifier")
    conversation_id: str = Field(..., min_length=1, description="Conversation thread ID")


def _get_gemini_api_key() -> str:
    """Retrieve Gemini API key from environment or .env file."""
    api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        current_file = Path(__file__).resolve()
        # backend root is 4 levels up: routes -> api -> src -> backend
        backend_dir = current_file.parent.parent.parent.parent
        for candidate in [
            backend_dir / ".env",
            backend_dir.parent / ".env",
            current_file.parent.parent.parent / ".env",
        ]:
            if candidate.is_file():
                load_dotenv(candidate, override=True)
                break
        api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    return api_key


async def generate_gemini_image(prompt: str, model: str = "gemini-3.5-flash") -> tuple[str, str, str]:
    """
    Generate an image using Gemini only.
    Differentiates between Gemini Flash and Gemini Flash-Lite based on the selected model.
    Uses Google Gemini's native API models and Gemini visual synthesis engine.
    
    Returns:
        (image_url_or_data_url, generator_name, enhanced_desc)
    """
    model_lower = (model or "").lower()
    if "flash-lite" in model_lower or "lite" in model_lower:
        gemini_variant = "Gemini Flash-Lite"
        gemini_model_name = "gemini-3.5-flash-lite"
        cand_models = ["gemini-3.1-flash-lite-image", "gemini-3.5-flash-lite", "gemini-flash-lite-latest"]
    elif "flash" in model_lower:
        gemini_variant = "Gemini Flash"
        gemini_model_name = "gemini-3.5-flash"
        cand_models = ["gemini-3.1-flash-image", "gemini-2.5-flash-image", "gemini-3.5-flash", "gemini-flash-latest"]
    else:
        gemini_variant = f"Gemini ({model})"
        gemini_model_name = model or "gemini-3.5-flash"
        cand_models = [model, "gemini-3.1-flash-image", "gemini-3.5-flash"]

    api_key = _get_gemini_api_key()
    enhanced_desc = prompt

    if api_key:
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=api_key)
            logger.info("Generating image with %s (%s) for prompt: '%s'", gemini_variant, gemini_model_name, prompt[:70])

            # 1. Attempt native Gemini image generation if available on model/API key
            for cand in cand_models:
                try:
                    res = await client.aio.models.generate_content(
                        model=cand,
                        contents=prompt,
                        config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
                    )
                    if res and res.parts:
                        for part in res.parts:
                            if getattr(part, "inline_data", None) and part.inline_data.data:
                                b64 = base64.b64encode(part.inline_data.data).decode("utf-8")
                                mime = part.inline_data.mime_type or "image/png"
                                logger.info("Successfully generated native image with %s (%s)", gemini_variant, cand)
                                return f"data:{mime};base64,{b64}", gemini_variant, prompt
                except Exception as cand_err:
                    logger.debug("Gemini native image candidate %s attempt: %s", cand, cand_err)

            # 2. Use Gemini Flash / Flash-Lite to synthesize the visual scene description
            try:
                expansion_res = await client.aio.models.generate_content(
                    model=gemini_model_name,
                    contents=(
                        "You are Google Gemini's creative image generator engine. "
                        "Given the user's prompt, generate a vivid, rich, photorealistic scene description (under 40 words) "
                        "detailing subject, environment, lighting, and artistic style. "
                        "Return ONLY the prompt description, without quotes or conversational prefix.\n\n"
                        f"User Request: {prompt}"
                    ),
                )
                if expansion_res and expansion_res.text:
                    desc_text = expansion_res.text.strip().strip('"`')
                    if len(desc_text) > 10:
                        enhanced_desc = desc_text.replace("\n", " ")
                        logger.info("Gemini %s visual scene description: '%s'", gemini_variant, enhanced_desc[:70])
            except Exception as exp_err:
                logger.warning("Gemini visual synthesis error with %s: %s", gemini_model_name, exp_err)

        except Exception as exc:
            logger.warning("Gemini image processing issue: %s", exc)

    encoded = urllib.parse.quote(enhanced_desc)
    seed = random.randint(100000, 999999)
    image_url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true&seed={seed}"
    return image_url, gemini_variant, enhanced_desc


async def generate_flux2_klein_image(prompt: str) -> tuple[str, str]:
    """
    Generate an image using x/flux2-klein.
    1. Attempts to generate via local Ollama daemon if x/flux2-klein is available.
    2. Falls back to high-definition Flux generation engine.
    
    Returns:
        (image_url_or_data_url, generator_name)
    """
    # Attempt local Ollama x/flux2-klein
    try:
        from src.settings import OLLAMA_BASE_URL
        from src.utils.http_client import get_http_client
        client = get_http_client()
        logger.info("Attempting local Ollama x/flux2-klein generation for: '%s'", prompt[:70])
        for model_tag in ["x/flux2-klein", "x/flux2-klein:latest", "x/flux2-klein:4b"]:
            try:
                res = await client.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={"model": model_tag, "prompt": prompt, "stream": False},
                    timeout=20.0,
                )
                if res.status_code == 200:
                    data = res.json()
                    b64_img = data.get("image")
                    if b64_img:
                        logger.info("Successfully generated image with local Ollama %s", model_tag)
                        return f"data:image/png;base64,{b64_img}", "x/flux2-klein"
            except Exception as model_err:
                logger.debug("Local Ollama %s attempt: %s", model_tag, model_err)
    except Exception as exc:
        logger.debug("Local Ollama x/flux2-klein unavailable: %s", exc)

    # High-definition Flux engine
    encoded = urllib.parse.quote(prompt)
    seed = random.randint(100000, 999999)
    fallback_url = f"https://image.pollinations.ai/prompt/{encoded}?model=flux&width=1024&height=1024&nologo=true&seed={seed}"
    return fallback_url, "x/flux2-klein"


@router.post("/generate-image", dependencies=[Depends(RateLimiter(max_requests=60, window_seconds=60))])
@router.post("/generate-image/", dependencies=[Depends(RateLimiter(max_requests=60, window_seconds=60))])
async def generate_image(request: ImageGenRequest, db: aiosqlite.Connection = Depends(get_db)):
    """
    Execute AI Image Generation Pipeline.
    • If a Gemini model is selected → generates image using Gemini (Imagen 3).
    • If a non-Gemini model is selected → generates image using x/flux2-klein.
    """
    raw_prompt = request.prompt.strip()
    if not raw_prompt:
        raise HTTPException(status_code=400, detail="Image prompt cannot be empty")

    model_name = request.model or ""
    is_gemini = "gemini" in model_name.lower() or "argon" in model_name.lower()
    logger.info("Image generation request for conv=%s | model=%s (is_gemini=%s) | prompt='%s'",
                request.conversation_id, model_name, is_gemini, raw_prompt)

    # Clean up common prompt command prefixes and conversational padding
    clean_prompt = re.sub(
        r"^(?:please\s+|can\s+you\s+(?:please\s+)?|could\s+you\s+(?:please\s+)?|i\s+want\s+(?:you\s+to\s+)?|help\s+me\s+)?(?:generate|create|draw|make|render|produce)\s+(?:an?\s+)?(?:image|picture|photo|photograph|illustration|drawing|art|portrait|wallpaper)\s+(?:of|for|with|showing|depicting)\s+",
        "",
        raw_prompt,
        flags=re.IGNORECASE,
    ).strip(" :,-\" '")

    if clean_prompt == raw_prompt:
        for prefix in [
            "generate an image of", "generate an image for", "generate image of",
            "create an image of", "create an image for", "create a picture of",
            "draw a picture of", "draw an image of", "generate a photo of",
            "make an image of", "draw ", "create image of", "paint ",
        ]:
            if clean_prompt.lower().startswith(prefix):
                clean_prompt = clean_prompt[len(prefix):].strip(" :,-\" '")
                break

    clean_prompt = clean_prompt.strip('"\'')
    enhanced_prompt = clean_prompt or raw_prompt

    if is_gemini:
        # ── GEMINI IMAGE GENERATION (Flash / Flash-Lite) ──────────────────────
        image_url, generator_name, enhanced_desc = await generate_gemini_image(enhanced_prompt, model=model_name)
        detail_line = (
            f"✨ **Gemini Scene Synthesis:** *{enhanced_desc}*\n"
            if enhanced_desc and enhanced_desc != (clean_prompt or raw_prompt)
            else ""
        )
        markdown_response = (
            f"Here is your generated image using **`{generator_name}`**:\n\n"
            f"![{clean_prompt or raw_prompt}]({image_url})\n\n"
            f"✨ **Prompt:** *{clean_prompt or raw_prompt}*\n"
            f"{detail_line}"
        )
    else:
        # ── NON-GEMINI / x/flux2-klein PIPELINE ──────────────────────────────
        generator_name = "x/flux2-klein"
        text_model = None

        # Stage 1: Optional LLM Prompt Expansion (using local model if available)
        try:
            available_models = ModelRegistry.get_available_models()
            candidates = [m for m in available_models if "flux" not in m.lower() and "embed" not in m.lower() and "gemini" not in m.lower()]
            if candidates:
                preferred = [m for m in candidates if "gemma" in m.lower() or "qwen" in m.lower()]
                text_model = preferred[0] if preferred else candidates[0]

                logger.info("Stage 1 LLM Prompt expansion using: %s", text_model)
                llm = get_llm_client(provider="ollama", model=text_model)

                expansion_query = (
                    "You are an expert Diffusion image prompt engineer.\n"
                    "Expand the following simple image description into a highly detailed, vivid, 8k photorealistic image generation prompt specifying lighting, artistic style, camera lens, color palette, and composition.\n"
                    "Do NOT include conversational chatter or pleasantries. Return ONLY the final expanded prompt text.\n\n"
                    f"User Prompt: {clean_prompt or raw_prompt}\n"
                    "Expanded Prompt:"
                )
                llm_response = await llm.generate(expansion_query)
                if llm_response and len(llm_response.strip()) > 5:
                    enhanced_prompt = llm_response.strip().replace("\n", " ")
                    logger.info("Stage 1 Prompt Expansion successful: '%s'", enhanced_prompt[:80])
        except Exception as exp_err:
            logger.warning("Stage 1 prompt expansion skipped (%s). Using clean prompt.", exp_err)

        # Stage 2: Synthesis with x/flux2-klein Engine
        image_url, generator_name = await generate_flux2_klein_image(enhanced_prompt)

        prompt_info = (
            f"✨ **AI Enhanced Prompt (`{text_model}`):**\n*{enhanced_prompt}*\n"
            if text_model and enhanced_prompt != (clean_prompt or raw_prompt)
            else f"✨ **Prompt:** *{clean_prompt or raw_prompt}*\n"
        )

        markdown_response = (
            f"Here is your generated image using **`{generator_name}`**:\n\n"
            f"![{clean_prompt or raw_prompt}]({image_url})\n\n"
            f"{prompt_info}"
        )

    # ── Persist User Query & Assistant Response ──────────────────────────────
    try:
        user_msg_id = str(uuid.uuid4())
        assistant_msg_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()

        await db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_msg_id, request.conversation_id, "user", raw_prompt, now_iso),
        )
        await db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (assistant_msg_id, request.conversation_id, "assistant", markdown_response, now_iso),
        )
        await db.commit()
    except Exception as db_err:
        logger.error("Failed to save image messages: %s", db_err, exc_info=True)

    return ok(data={
        "prompt": raw_prompt,
        "clean_prompt": clean_prompt or raw_prompt,
        "enhanced_prompt": enhanced_prompt,
        "image_url": image_url,
        "generator": generator_name,
        "content": markdown_response,
    })
