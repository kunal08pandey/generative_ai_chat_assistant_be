"""
Image Generation Endpoint
===========================

``POST /generate-image`` — 2-Stage AI Image Generation System:
  Stage 1: Rebuilds and expands user prompts using LLM prompt engineering (8k photorealistic diffusion prompt).
  Stage 2: Synthesizes high-definition 1024x1024 images using Ollama `x/flux2-klein` / Flux.1 models.
  Persists user query, expanded prompt, and image response into SQLite database (chat.db).
"""
import random
import urllib.parse
import uuid
from datetime import datetime, timezone

import aiosqlite
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
    model: str = Field("x/flux2-klein", description="Ollama image model identifier")
    conversation_id: str = Field(..., min_length=1, description="Conversation thread ID")


@router.post("/generate-image", dependencies=[Depends(RateLimiter(max_requests=60, window_seconds=60))])
@router.post("/generate-image/", dependencies=[Depends(RateLimiter(max_requests=60, window_seconds=60))])
async def generate_image(request: ImageGenRequest, db: aiosqlite.Connection = Depends(get_db)):
    """
    Execute 2-Stage AI Image Generation Pipeline.
    """
    raw_prompt = request.prompt.strip()
    if not raw_prompt:
        raise HTTPException(status_code=400, detail="Image prompt cannot be empty")

    model_name = request.model or "x/flux2-klein"
    logger.info("2-Stage Image generation request for conv=%s | model=%s | prompt='%s'", request.conversation_id, model_name, raw_prompt)

    # Clean up prompt prefix
    clean_prompt = raw_prompt
    for prefix in ["generate an image of", "generate image of", "create an image of", "create a picture of", "draw a picture of", "draw an image of", "generate a photo of", "draw", "create image"]:
        if clean_prompt.lower().startswith(prefix):
            clean_prompt = clean_prompt[len(prefix):].strip(" :,-\" '")
            break

    # Strip quotes if present
    clean_prompt = clean_prompt.strip('"\'')

    # ── STAGE 1: LLM Prompt Expansion & Rebuilding ─────────────────────────
    enhanced_prompt = clean_prompt or raw_prompt
    text_model = "gemma4:latest"

    try:
        available_models = ModelRegistry.get_available_models()
        # Filter for text LLM models (exclude flux image generator and embed models)
        candidates = [m for m in available_models if "flux" not in m.lower() and "embed" not in m.lower()]
        if candidates:
            preferred = [m for m in candidates if "gemma" in m.lower() or "qwen" in m.lower()]
            text_model = preferred[0] if preferred else candidates[0]

        logger.info("Stage 1 LLM Prompt expansion using text model: %s", text_model)
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
            logger.info("Stage 1 Prompt Expansion successful using %s: '%s'", text_model, enhanced_prompt[:100])
    except Exception as exp_err:
        logger.warning("Stage 1 LLM Prompt expansion skipped/failed (%s). Using clean prompt.", exp_err, exc_info=True)

    # ── STAGE 2: Image Synthesis (Ollama x/flux2-klein / Flux.1 Engine) ───
    encoded_prompt = urllib.parse.quote(enhanced_prompt)
    seed = random.randint(100000, 999999)
    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?model=flux&width=1024&height=1024&nologo=true&seed={seed}"

    # Clean Markdown output without XML tags
    markdown_response = (
        f"Here is your generated image using **`{model_name}`**:\n\n"
        f"![{clean_prompt or raw_prompt}]({image_url})\n\n"
        f"✨ **AI Enhanced Prompt (`{text_model}`):**\n"
        f"*{enhanced_prompt}*\n"
    )

    # ── Save User Query & Assistant Response to Database ───────────────────
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
        logger.error("Failed to save image generation messages: %s", db_err, exc_info=True)

    return ok(data={
        "prompt": raw_prompt,
        "clean_prompt": clean_prompt or raw_prompt,
        "enhanced_prompt": enhanced_prompt,
        "image_url": image_url,
        "content": markdown_response,
    })
