"""
Messages Endpoints
===================

  • ``GET  /messages/{conv_id}`` — Paginated message history (owner only).
  • ``POST /messages``           — Add a message (owner only).
  • ``DELETE /messages/{message_id}`` — Delete a message (owner only).

Phase 1: All endpoints require authenticated session and enforce ownership.
"""
import uuid
from datetime import datetime, timezone

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.schemas import MessageCreate, MessageOut, ok, paginated_ok
from src.api.dependencies.auth import get_current_user
from src.db.database import get_db
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Messages"])


async def _verify_conversation_owner(db: aiosqlite.Connection, conv_id: str, user_id: str):
    """Verify the conversation exists and belongs to the user."""
    cursor = await db.execute(
        "SELECT id, owner_id FROM conversations WHERE id = ?", (conv_id,)
    )
    row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Conversation {conv_id} not found")
    if row["owner_id"] and row["owner_id"] != user_id:
        raise HTTPException(status_code=404, detail=f"Conversation {conv_id} not found")
    return row


@router.get("/messages/{conv_id}")
async def get_messages(
    conv_id: str,
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(50, ge=1, le=200, description="Messages per page"),
    user: dict = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
):
    """Retrieve paginated message history for an owned conversation, oldest first."""
    await _verify_conversation_owner(db, conv_id, user["id"])

    count_cursor = await db.execute(
        "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conv_id,)
    )
    (total_count,) = await count_cursor.fetchone()

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
    user: dict = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
):
    """Add a message to an owned conversation."""
    await _verify_conversation_owner(db, data.conversation_id, user["id"])

    msg_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    await db.execute(
        "INSERT INTO messages (id, conversation_id, role, content, created_at, owner_id) VALUES (?, ?, ?, ?, ?, ?)",
        (msg_id, data.conversation_id, data.role, data.content, created_at, user["id"]),
    )
    await db.commit()

    message = MessageOut(
        id=msg_id,
        conversation_id=data.conversation_id,
        role=data.role,
        content=data.content,
        created_at=created_at,
    )
    logger.info("Added %s message to conversation %s by user=%s", data.role, data.conversation_id, user["id"])
    return ok(data=message.model_dump())


@router.delete("/messages/{message_id}", status_code=204)
async def delete_message(
    message_id: str,
    user: dict = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
):
    """Delete a specific message (owner only)."""
    cursor = await db.execute(
        "SELECT id, conversation_id FROM messages WHERE id = ?", (message_id,)
    )
    row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Message {message_id} not found")

    # Verify conversation ownership
    await _verify_conversation_owner(db, row["conversation_id"], user["id"])

    await db.execute("DELETE FROM messages WHERE id = ?", (message_id,))
    await db.commit()

    logger.info("Deleted message id=%s by user=%s", message_id, user["id"])
    return None


@router.delete("/conversations/{conv_id}/messages/{message_id}", status_code=204)
async def delete_conversation_message(
    conv_id: str,
    message_id: str,
    user: dict = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
):
    """Delete a specific message from a designated owned conversation."""
    await _verify_conversation_owner(db, conv_id, user["id"])

    cursor = await db.execute(
        "SELECT id FROM messages WHERE id = ? AND conversation_id = ?",
        (message_id, conv_id),
    )
    row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Message {message_id} not found in conversation {conv_id}")

    await db.execute("DELETE FROM messages WHERE id = ?", (message_id,))
    await db.commit()

    logger.info("Deleted message id=%s from conversation %s by user=%s", message_id, conv_id, user["id"])
    return None
