"""
Messages Endpoints
===================

  • ``GET  /messages/{conv_id}`` — Paginated message history for a conversation.
  • ``POST /messages``           — Add a message to a conversation.
"""
import uuid
from datetime import datetime, timezone

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.schemas import MessageCreate, MessageOut, ok, paginated_ok
from src.db.database import get_db
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Messages"])


@router.get("/messages/{conv_id}")
async def get_messages(
    conv_id: str,
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(50, ge=1, le=200, description="Messages per page"),
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Retrieve paginated message history for a conversation, oldest first.

    Returns 404 if the conversation does not exist.
    """
    # Verify conversation exists
    cursor = await db.execute("SELECT id FROM conversations WHERE id = ?", (conv_id,))
    if not await cursor.fetchone():
        logger.warning("Messages requested for non-existent conversation — id=%s", conv_id)
        raise HTTPException(status_code=404, detail=f"Conversation {conv_id} not found")

    # Total count
    count_cursor = await db.execute(
        "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conv_id,)
    )
    (total_count,) = await count_cursor.fetchone()

    # Paginated rows
    offset = (page - 1) * page_size
    cursor = await db.execute(
        "SELECT id, conversation_id, role, content, created_at "
        "FROM messages WHERE conversation_id = ? ORDER BY created_at ASC LIMIT ? OFFSET ?",
        (conv_id, page_size, offset),
    )
    rows = await cursor.fetchall()
    items = [MessageOut(**dict(row)).model_dump() for row in rows]

    logger.debug("Served %d messages for conversation %s (page %d)", len(items), conv_id, page)
    return paginated_ok(items=items, total_count=total_count, page=page, page_size=page_size)


@router.post("/messages", status_code=201)
async def add_message(
    data: MessageCreate,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Add a message to a conversation.

    Validates that the conversation exists before inserting.
    """
    # Verify conversation exists
    cursor = await db.execute("SELECT id FROM conversations WHERE id = ?", (data.conversation_id,))
    if not await cursor.fetchone():
        logger.warning("Attempted to add message to non-existent conversation — id=%s", data.conversation_id)
        raise HTTPException(status_code=404, detail=f"Conversation {data.conversation_id} not found")

    msg_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    await db.execute(
        "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
        (msg_id, data.conversation_id, data.role, data.content, created_at),
    )
    await db.commit()

    message = MessageOut(
        id=msg_id,
        conversation_id=data.conversation_id,
        role=data.role,
        content=data.content,
        created_at=created_at,
    )
    logger.info("Added %s message to conversation %s", data.role, data.conversation_id)
    return ok(data=message.model_dump())