"""
Agent Workflow Endpoints
========================

Bounded Agent endpoints (Phase 6):
  • POST /agent/report — Document-to-Report Agent streaming workflow
"""
import json
import uuid
from datetime import datetime, timezone
from typing import Optional

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.agent.document_report_agent import DocumentReportAgent
from src.api.dependencies.auth import get_current_user
from src.db.database import get_db, get_db_path
from src.llm.client_factory import get_llm_client
from src.llm.registry import ModelRegistry
from src.utils.error_parser import parse_llm_error
from src.utils.logger import get_logger
from src.utils.rate_limiter import RateLimiter

logger = get_logger(__name__)
router = APIRouter(prefix="/agent", tags=["Agent Workflows"])


class AgentReportRequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=1000, description="Topic or goal of the report")
    conversation_id: str = Field(..., description="Target conversation ID")
    model: str = Field(default="gemini-3.5-flash", description="LLM model name")
    document_query: Optional[str] = Field(None, description="Optional search filter for document excerpts")


@router.post("/report", dependencies=[Depends(RateLimiter(max_requests=30, window_seconds=60))])
async def generate_document_report(
    request: AgentReportRequest,
    user: dict = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Executes the Document-to-Report bounded agent workflow.
    Streams intermediate reasoning (<think>...</think>) followed by the final report.
    """
    user_id = user["id"]
    conv_id = request.conversation_id

    # Verify conversation ownership (Phase 1)
    cursor = await db.execute(
        "SELECT id, owner_id FROM conversations WHERE id = ?",
        (conv_id,),
    )
    conv_row = await cursor.fetchone()
    if not conv_row:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conv_row[1] != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Record user message in DB
    now_iso = datetime.now(timezone.utc).isoformat()
    await db.execute(
        "INSERT INTO messages (id, conversation_id, role, content, created_at, owner_id) VALUES (?, ?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), conv_id, "user", f"Generate Report: {request.topic}", now_iso, user_id),
    )
    await db.commit()

    llm = get_llm_client(request.model)
    agent = DocumentReportAgent(llm=llm, owner_id=user_id)
    db_path = get_db_path()

    async def stream_agent():
        full_response = []
        try:
            yield f"data: {json.dumps({'status': 'Running Document-to-Report agent...'})}\n\n"
            async for token in agent.stream_workflow(
                topic=request.topic,
                document_query=request.document_query,
            ):
                full_response.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"

            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as exc:
            logger.error("Agent workflow error: %s", exc, exc_info=True)
            err_payload = parse_llm_error(exc, model=request.model)
            yield f"data: {json.dumps({'error': err_payload})}\n\n"
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
                except Exception as save_err:
                    logger.error("Failed to persist agent report message: %s", save_err)
                finally:
                    if save_db:
                        await save_db.close()

    return StreamingResponse(
        stream_agent(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
