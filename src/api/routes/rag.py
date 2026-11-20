"""
RAG Endpoint
=============

``POST /rag`` — Retrieval-Augmented Generation chat interface via Server-Sent Events (SSE).

Features:
  • Multi-step RAG chain: Query Check -> Context Retrieval -> Answer Generation.
  • Scoped strictly to authenticated owner's conversation and private document store (Phase 1).
  • Bypasses LLM query optimization for short queries (< 150 chars).
  • Proper resource cleanup in the streaming finally block.
"""
import json
import uuid
from datetime import datetime, timezone

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.api.dependencies.auth import get_current_user
from src.api.schemas import RagRequest
from src.db.database import get_db, get_db_path
from src.llm.client_factory import get_llm_client
from src.llm.registry import ModelRegistry
from src.prompt_engineering.chainer import LLMChain
from src.prompt_engineering.templates import PromptTemplate
from src.rag.retriever import retrieve_context
from src.utils.error_parser import parse_llm_error
from src.utils.host_resolver import is_localhost_request
from src.utils.logger import get_logger
from src.utils.rate_limiter import RateLimiter

logger = get_logger(__name__)
router = APIRouter(tags=["chat", "rag"])


@router.post("/", dependencies=[Depends(RateLimiter(max_requests=120, window_seconds=60))])
async def rag(
    request: RagRequest,
    http_request: Request,
    user: dict = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Execute a RAG query and stream status updates and final answer tokens.
    Scoped strictly to the authenticated user's conversations and indexed documents.
    """
    user_id = user["id"]
    logger.info(
        "RAG request — conv=%s user=%s model=%s provider=%s",
        request.conversation_id,
        user_id,
        request.model,
        request.provider,
    )

    # ── Verify conversation exists and belongs to current user ────────
    cursor = await db.execute(
        "SELECT id, owner_id FROM conversations WHERE id = ?",
        (request.conversation_id,),
    )
    conv_row = await cursor.fetchone()
    if not conv_row:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conv_row["owner_id"] and conv_row["owner_id"] != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    provider = request.provider or ModelRegistry.get_provider_for_model(request.model) or "ollama"
    if provider == "ollama" and not is_localhost_request(http_request):
        raise HTTPException(
            status_code=403,
            detail="Ollama models are only available on localhost",
        )

    # ── Semantic Router Check (Identity / Deterministic bypass) ──────────
    from src.routing.semantic_router import get_semantic_router
    semantic_router = get_semantic_router()
    route_match = await semantic_router.match(request.query)
    if route_match:
        matched_route, score = route_match
        logger.info(
            "Semantic Router (RAG): query handled by route='%s' (score=%.3f) — bypassing RAG/LLM",
            matched_route.name,
            score,
        )
        now_iso = datetime.now(timezone.utc).isoformat()
        await db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at, owner_id) VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), request.conversation_id, "user", request.query.strip(), now_iso, user_id),
        )
        await db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at, owner_id) VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), request.conversation_id, "assistant", matched_route.response, now_iso, user_id),
        )
        await db.commit()

        async def stream_routed_rag():
            yield f"data: {json.dumps({'status': f'Matched semantic route: {matched_route.name}'})}\n\n"
            yield f"data: {json.dumps({'token': matched_route.response})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"

        return StreamingResponse(
            stream_routed_rag(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Fetch Past Conversation History for Context ──────────────────────
    cursor = await db.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
        (request.conversation_id,),
    )
    past_rows = await cursor.fetchall()

    history_lines = []
    for role, content in past_rows[-10:]:
        role_label = "User" if role == "user" else "Assistant"
        history_lines.append(f"{role_label}: {content}")
    history_text = "\n".join(history_lines)

    # ── Save User Message ────────────────────────────────────────────────
    stored_query = request.query.strip() or "[Analyzed document]"
    await db.execute(
        "INSERT INTO messages (id, conversation_id, role, content, created_at, owner_id) VALUES (?, ?, ?, ?, ?, ?)",
        (
            str(uuid.uuid4()),
            request.conversation_id,
            "user",
            stored_query,
            datetime.now(timezone.utc).isoformat(),
            user_id,
        ),
    )
    await db.commit()

    # ── Resolve Provider & LLM Client ───────────────────────────────────
    provider = request.provider or ModelRegistry.get_provider_for_model(request.model) or "ollama"
    logger.info("Initializing LLM client (provider=%s) for model %s", provider, request.model)
    llm = get_llm_client(provider=provider, model=request.model)

    conv_id = request.conversation_id
    query = request.query.strip() or "Provide a comprehensive summary of the uploaded document(s)."
    db_path = get_db_path()

    async def stream_rag():
        full_response = []
        save_db = None
        try:
            # Step 1: Optimize query (Heuristically bypass for short prompts)
            if len(query) < 150:
                logger.info(
                    "Skipping query optimization: Query length (%d chars) < 150. Using original query.",
                    len(query),
                )
                optimized_query = query
            else:
                yield f"data: {json.dumps({'status': 'Optimizing search query...'})}\n\n"
                logger.info("Executing Chain Step 1: Optimizing the search query via LLM")
                optimization_template = PromptTemplate(
                    "Rewrite the following user query to be more search-engine friendly by extracting key terms. "
                    "Return only the optimized query.\n\nQuery: {query}\nOptimized Query:"
                )
                optimizer_chain = LLMChain(prompt_template=optimization_template, llm=llm)
                optimized_query = await optimizer_chain.run(query=query)
                logger.debug("Optimized Query: %s", optimized_query)

            # Step 2: Retrieve context (scoped strictly to user_id)
            yield f"data: {json.dumps({'status': 'Searching documents...'})}\n\n"
            logger.info("Retrieving context from vector store for user=%s", user_id)

            search_query = optimized_query
            if history_text:
                search_query = f"{history_text[-250:]} {optimized_query}"

            # Document isolation: ONLY retrieve documents belonging to this owner
            context = await retrieve_context(search_query, owner_id=user_id)
            citations = getattr(context, "citations", [])
            if citations:
                yield f"data: {json.dumps({'citations': citations})}\n\n"

            # Step 3: Stream final answer
            yield f"data: {json.dumps({'status': 'Generating answer...'})}\n\n"
            logger.info("Executing Chain Step 2: Streaming final RAG response based on history and context")

            if history_text:
                rag_prompt = (
                    "You are a helpful, expert AI assistant engaged in an ongoing conversation.\n"
                    "Use the Document Context and Conversation History below to accurately answer the user's latest request.\n"
                    "When referencing facts from the Document Context, cite sources clearly using references like [1], [2] matching the numbered sources.\n"
                    "Maintain full context from previous turns and answer directly in the user's requested language/tone.\n\n"
                    f"=== Document Context ===\n{context or 'No specific document snippet matched. Use conversation history.'}\n\n"
                    f"=== Conversation History ===\n{history_text}\n\n"
                    f"User: {query}\nAssistant:"
                )
            else:
                rag_prompt = (
                    "You are a helpful, expert AI assistant.\n"
                    "Answer the user's request based on the provided document context.\n"
                    "When referencing facts from the Document Context, cite sources clearly using references like [1], [2] matching the numbered sources.\n\n"
                    f"=== Document Context ===\n{context or 'No document snippet found.'}\n\n"
                    f"User: {query}\nAssistant:"
                )

            async for token in llm.generate_stream(rag_prompt):
                full_response.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"

            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            logger.error("RAG streaming error: %s", e, exc_info=True)
            error_payload = parse_llm_error(e, model=request.model)
            yield f"data: {json.dumps({'error': error_payload})}\n\n"
            return
        finally:
            complete_text = "".join(full_response)
            if complete_text:
                try:
                    save_db = await aiosqlite.connect(db_path, timeout=20.0)
                    await save_db.execute("PRAGMA journal_mode=WAL")
                    await save_db.execute("PRAGMA foreign_keys=ON")
                    await save_db.execute(
                        "INSERT INTO messages (id, conversation_id, role, content, created_at, owner_id) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            str(uuid.uuid4()),
                            conv_id,
                            "assistant",
                            complete_text,
                            datetime.now(timezone.utc).isoformat(),
                            user_id,
                        ),
                    )
                    await save_db.commit()
                    logger.info("Saved RAG assistant message for conv=%s (%d chars)", conv_id, len(complete_text))
                except Exception as save_err:
                    logger.error("Failed to save RAG assistant message: %s", save_err, exc_info=True)
                finally:
                    if save_db:
                        await save_db.close()

    return StreamingResponse(
        stream_rag(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
