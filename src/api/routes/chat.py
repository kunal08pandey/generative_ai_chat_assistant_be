"""
Chat Endpoint
==============

``POST /chat`` — Streaming chat interface via Server-Sent Events (SSE).

Supports:
  • Standard text chat with few-shot prompt formatting.
  • Vision models (base64-encoded images).
  • SSE streaming with a ``done`` event on completion.
  • Reliable assistant-message persistence (no DB connection leak).
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from src.api.schemas import ChatRequest
from src.db.database import get_db, get_db_path
from src.llm.client_factory import get_llm_client
from src.llm.registry import ModelRegistry
from src.prompt_engineering.few_shot import FewShotPromptTemplate
from src.utils.error_parser import parse_llm_error
from src.utils.logger import get_logger
from src.utils.rate_limiter import RateLimiter

logger = get_logger(__name__)
router = APIRouter()


@router.post("/", dependencies=[Depends(RateLimiter(max_requests=120, window_seconds=60))])
async def chat(request: ChatRequest, db: aiosqlite.Connection = Depends(get_db)):
    """
    Stream a chat response from the selected LLM.

    The user message is saved immediately. The complete assistant response
    is saved *after* the stream finishes using a dedicated DB connection
    scoped to the generator's ``finally`` block.
    """
    logger.info(
        "Chat request — conv=%s model=%s images=%d",
        request.conversation_id,
        request.model,
        len(request.images) if request.images else 0,
    )

    cursor = await db.execute(
        "SELECT id FROM conversations WHERE id = ?",
        (request.conversation_id,),
    )
    if not await cursor.fetchone():
        raise HTTPException(status_code=404, detail="Conversation not found")
    if not ModelRegistry.is_model_available(request.model):
        raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model}")

    # ── Resolve provider & create client before persisting the message ───
    provider = request.provider or ModelRegistry.get_provider_for_model(request.model) or "ollama"
    logger.info("Using provider=%s for model=%s", provider, request.model)

    # Detect image generation intent
    query_lower = request.query.strip().lower()
    is_gemini_model = "gemini" in request.model.lower() or "argon" in request.model.lower()
    is_flux_model = "flux" in request.model.lower()

    is_image_prompt = bool(re.search(
        r"^(?:please\s+|can\s+you\s+(?:please\s+)?|could\s+you\s+(?:please\s+)?|i\s+want\s+(?:you\s+to\s+)?|help\s+me\s+)?(?:generate|create|draw|make|render|produce)\s+(?:an?\s+)?(?:image|picture|photo|photograph|illustration|drawing|art|portrait|wallpaper)",
        query_lower,
    )) or bool(re.match(r"^(draw|paint|render)\s+", query_lower)) or any(query_lower.startswith(p) for p in [
        "generate an image", "generate image", "create an image", "create a picture",
        "draw a picture", "draw an image", "generate a photo", "make an image",
        "show me an image", "draw ", "create image", "paint ",
    ])

    # Route image generation requests:
    # If Gemini model is selected → generates with Gemini (Flash or Flash-Lite)
    # If non-Gemini model is selected → generates with x/flux2-klein
    if is_flux_model or is_image_prompt:
        from src.api.routes.image import generate_image, ImageGenRequest
        active_img_model = request.model if is_gemini_model else "x/flux2-klein"
        img_req = ImageGenRequest(prompt=request.query, model=active_img_model, conversation_id=request.conversation_id)
        img_res = await generate_image(img_req, db)
        content = img_res["data"]["content"]

        if is_gemini_model:
            model_lower = request.model.lower()
            if "lite" in model_lower:
                thinking_label = "Gemini Flash-Lite"
            elif "flash" in model_lower:
                thinking_label = "Gemini Flash"
            else:
                thinking_label = f"Gemini ({request.model})"
            thinking_provider = "gemini"
        else:
            thinking_label = "x/flux2-klein"
            thinking_provider = "ollama"

        async def stream_image_response():
            yield f"data: {json.dumps({'thinking': True, 'model': thinking_label, 'provider': thinking_provider})}\n\n"
            yield f"data: {json.dumps({'token': content})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"

        return StreamingResponse(stream_image_response(), media_type="text/event-stream")

    # ── Semantic Router Check (Bypass foundation models for Identity/Rules) ──
    if not (request.images and len(request.images) > 0):
        from src.routing.semantic_router import get_semantic_router
        semantic_router = get_semantic_router()
        route_match = await semantic_router.match(request.query)
        if route_match:
            matched_route, score = route_match
            logger.info(
                "Semantic Router: query directly handled by route='%s' (score=%.3f) — bypassing LLM",
                matched_route.name,
                score,
            )
            now_iso = datetime.now(timezone.utc).isoformat()
            await db.execute(
                "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), request.conversation_id, "user", request.query.strip(), now_iso),
            )
            await db.execute(
                "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), request.conversation_id, "assistant", matched_route.response, now_iso),
            )
            await db.commit()

            async def stream_routed_response():
                yield f"data: {json.dumps({'thinking': False, 'model': f'System ({matched_route.name})', 'provider': 'router'})}\n\n"
                yield f"data: {json.dumps({'token': matched_route.response})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"

            return StreamingResponse(
                stream_routed_response(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

    try:
        client = get_llm_client(provider=provider, model=request.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ── Fetch Past Conversation History for Context ──────────────────────
    cursor = await db.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
        (request.conversation_id,),
    )
    past_rows = await cursor.fetchall()
    
    # Format past turns (excluding current message) into conversation history string
    history_lines = []
    for role, content in past_rows[-10:]:
        role_label = "User" if role == "user" else "Assistant"
        history_lines.append(f"{role_label}: {content}")
    history_text = "\n".join(history_lines)

    # ── Save user message ────────────────────────────────────────────────
    stored_query = request.query.strip() or f"[Sent {len(request.images or [])} image(s)]"
    await db.execute(
        "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
        (
            str(uuid.uuid4()),
            request.conversation_id,
            "user",
            stored_query,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    await db.commit()

    # ── Build prompt ─────────────────────────────────────────────────────
    if request.images and len(request.images) > 0:
        prompt = request.query.strip() or "Describe the attached image or images."
    elif history_text:
        prompt = (
            "You are a helpful, expert AI assistant engaged in an ongoing conversation.\n"
            "Maintain full context from previous turns and respond directly to the user's latest message.\n"
            "When the user asks for an Excel file or spreadsheet (.xlsx), format the data cleanly as a markdown table or CSV block so it can be exported directly to Excel.\n\n"
            f"=== Conversation History ===\n{history_text}\n\n"
            f"User: {request.query}\nAssistant:"
        )
    else:
        chat_template = FewShotPromptTemplate(
            prefix=(
                "You are a helpful and expert AI assistant. Please provide concise, accurate, and structured answers. "
                "When the user asks for an Excel file or spreadsheet (.xlsx), format the data cleanly as a markdown table or CSV block so it can be exported directly to Excel."
            ),
            examples=[
                {"input": "Hi there!", "output": "Hello! How can I help you today?"},
                {"input": "What is the capital of France?", "output": "The capital of France is Paris."},
            ],
            suffix="User: {query}\nAssistant:",
        )
        prompt = chat_template.format(query=request.query)

    logger.debug("Formatted chat prompt (%d chars)", len(prompt))

    conv_id = request.conversation_id
    db_path = get_db_path()

    async def stream_and_save():
        """Generator that streams SSE tokens and saves the full response.
        
        Emits a 'thinking' event before tokens start so the frontend
        can display a Gemini-style "Thinking…" indicator.
        """
        full_response: list[str] = []
        save_db: aiosqlite.Connection | None = None
        try:
            # Send a thinking event so frontend shows collapsible thinking indicator
            yield f"data: {json.dumps({'thinking': True, 'model': request.model, 'provider': provider})}\n\n"

            async for token in client.generate_stream(prompt, images=request.images):
                full_response.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"

            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            logger.error("Chat streaming error: %s", e, exc_info=True)
            error_payload = parse_llm_error(e, model=request.model)
            yield f"data: {json.dumps({'error': error_payload})}\n\n"
            return
        finally:
            # ── Persist assistant response ───────────────────────────────
            complete_text = "".join(full_response)
            if complete_text:
                try:
                    save_db = await aiosqlite.connect(db_path, timeout=20.0)
                    await save_db.execute("PRAGMA journal_mode=WAL")
                    await save_db.execute("PRAGMA foreign_keys=ON")
                    await save_db.execute(
                        "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                        (
                            str(uuid.uuid4()),
                            conv_id,
                            "assistant",
                            complete_text,
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                    await save_db.commit()
                    logger.info("Saved assistant response for conv=%s (%d chars)", conv_id, len(complete_text))
                except Exception as save_err:
                    logger.error("Failed to save assistant message: %s", save_err, exc_info=True)
                finally:
                    if save_db:
                        await save_db.close()

    return StreamingResponse(
        stream_and_save(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
