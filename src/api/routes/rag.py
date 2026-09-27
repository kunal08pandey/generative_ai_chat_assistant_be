"""
RAG Endpoint
=============

``POST /rag`` — Retrieval-Augmented Generation chat interface via Server-Sent Events (SSE).

Features:
  • Multi-step RAG chain: Query Check -> Context Retrieval -> Answer Generation.
  • Bypasses LLM query optimization for short queries (< 150 chars), saving a full
    blocking LLM call and decreasing initial response latency by 3-10 seconds locally.
  • Proper resources cleanup in the streaming finally block (prevents DB lockups).
"""
import json
import uuid
from datetime import datetime, timezone

import aiosqlite
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.api.schemas import RagRequest
from src.db.database import get_db, get_db_path
from src.llm.client_factory import get_llm_client
from src.llm.registry import ModelRegistry
from src.prompt_engineering.chainer import LLMChain
from src.prompt_engineering.templates import PromptTemplate
from src.rag.retriever import retrieve_context
from src.utils.logger import get_logger
from src.utils.rate_limiter import RateLimiter

logger = get_logger(__name__)
router = APIRouter(tags=["chat", "rag"])


@router.post("/", dependencies=[Depends(RateLimiter(max_requests=120, window_seconds=60))])
async def rag(request: RagRequest, db: aiosqlite.Connection = Depends(get_db)):
    """
    Execute a RAG query and stream status updates and final answer tokens.

    Saves the user query immediately, then retrieves documents, formats
    the prompt, streams answer tokens via SSE, and persists the complete
    response in database afterward.
    """
    logger.info(
        "RAG request — conv=%s model=%s provider=%s",
        request.conversation_id,
        request.model,
        request.provider,
    )

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

    # ── Save User Message ────────────────────────────────────────────────
    stored_query = request.query.strip() or "[Analyzed document]"
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

    # ── Resolve Provider & LLM Client ───────────────────────────────────
    provider = request.provider or ModelRegistry.get_provider_for_model(request.model) or "ollama"
    logger.info("Initializing LLM client (provider=%s) for model %s", provider, request.model)
    llm = get_llm_client(provider=provider, model=request.model)

    conv_id = request.conversation_id
    query = request.query.strip() or "Provide a comprehensive summary of the uploaded document(s)."
    db_path = get_db_path()

    async def stream_rag():
        """
        Streams processing status events followed by the LLM answer tokens.
        """
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

            # Step 2: Retrieve context (Enrich search query with recent history topic)
            yield f"data: {json.dumps({'status': 'Searching documents...'})}\n\n"
            logger.info("Retrieving context from vector store for RAG")

            search_query = optimized_query
            if history_text:
                # Combine recent conversation context with query for high vector recall
                search_query = f"{history_text[-250:]} {optimized_query}"

            context = await retrieve_context(search_query)

            # Step 3: Stream final answer
            yield f"data: {json.dumps({'status': 'Generating answer...'})}\n\n"
            logger.info("Executing Chain Step 2: Streaming final RAG response based on history and context")

            if history_text:
                rag_prompt = (
                    "You are a helpful, expert AI assistant engaged in an ongoing conversation.\n"
                    "Use the Document Context and Conversation History below to accurately answer the user's latest request.\n"
                    "Maintain full context from previous turns (such as medical reports, diet requests, or prior instructions) and answer directly in the user's requested language/tone (e.g. Hinglish).\n\n"
                    f"=== Document Context ===\n{context or 'No specific document snippet matched. Use conversation history.'}\n\n"
                    f"=== Conversation History ===\n{history_text}\n\n"
                    f"User: {query}\nAssistant:"
                )
            else:
                rag_prompt = (
                    "You are a helpful, expert AI assistant.\n"
                    "Answer the user's request based on the provided document context.\n\n"
                    f"=== Document Context ===\n{context or 'No document snippet found.'}\n\n"
                    f"User: {query}\nAssistant:"
                )

            async for token in llm.generate_stream(rag_prompt):
                full_response.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"

            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            logger.error("RAG streaming error: %s", e, exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return
        finally:
            # ── Save Complete Assistant Message ───────────────────────────
            complete_text = "".join(full_response)
            if complete_text:
                try:
                    save_db = await aiosqlite.connect(db_path, timeout=20.0)
                    await save_db.execute("PRAGMA journal_mode=WAL")
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
