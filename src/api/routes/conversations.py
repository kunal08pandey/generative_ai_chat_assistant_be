"""
Conversations Endpoints
========================

CRUD operations for conversation threads.

  • ``GET    /conversations``           — List (paginated, purges empty abandoned threads).
  • ``POST   /conversations``           — Create.
  • ``PATCH  /conversations/{conv_id}`` — Update title.
  • ``DELETE /conversations/{conv_id}`` — Delete (cascade messages).
  • ``DELETE /conversations``           — Clear all history.
"""
import uuid
from datetime import datetime, timezone

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.schemas import (
    ConversationCreate,
    ConversationOut,
    ConversationUpdate,
    ok,
    paginated_ok,
)
from src.db.database import get_db
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Conversations"])


# ── LIST (paginated & auto-purging empty) ──────────────────────────────────
@router.get("/conversations")
async def list_conversations(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Return a paginated list of conversations ordered by most recent first.
    Automatically purges abandoned empty conversations (0 messages) except the latest active one.
    """
    logger.debug("Listing conversations — page=%d, page_size=%d", page, page_size)

    # Purge abandoned empty threads (conversations with 0 messages)
    try:
        await db.execute("""
            DELETE FROM conversations
            WHERE id NOT IN (SELECT DISTINCT conversation_id FROM messages)
              AND id != (SELECT id FROM conversations ORDER BY created_at DESC LIMIT 1)
        """)
        await db.commit()
    except Exception as exc:
        logger.warning("Could not purge empty conversations: %s", exc)

    # Total count
    count_cursor = await db.execute("SELECT COUNT(*) FROM conversations")
    (total_count,) = await count_cursor.fetchone()

    # Paginated rows
    offset = (page - 1) * page_size
    cursor = await db.execute(
        "SELECT id, title, created_at FROM conversations ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (page_size, offset),
    )
    rows = await cursor.fetchall()
    items = [ConversationOut(**dict(row)).model_dump() for row in rows]

    logger.info("Listed %d conversations (page %d/%d)", len(items), page, -(-total_count // page_size) if page_size else 1)
    return paginated_ok(items=items, total_count=total_count, page=page, page_size=page_size)


# ── CREATE ───────────────────────────────────────────────────────────────
@router.post("/conversations", status_code=201)
async def create_conversation(
    data: ConversationCreate,
    db: aiosqlite.Connection = Depends(get_db),
):
    """Create a new conversation thread."""
    conv_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    await db.execute(
        "INSERT INTO conversations (id, title, created_at) VALUES (?, ?, ?)",
        (conv_id, data.title, created_at),
    )
    await db.commit()

    conversation = ConversationOut(id=conv_id, title=data.title, created_at=created_at)
    logger.info("Created conversation id=%s title=%r", conv_id, data.title)
    return ok(data=conversation.model_dump())


# ── UPDATE ───────────────────────────────────────────────────────────────
@router.patch("/conversations/{conv_id}")
async def update_conversation(
    conv_id: str,
    data: ConversationUpdate,
    db: aiosqlite.Connection = Depends(get_db),
):
    """Update the title of an existing conversation."""
    cursor = await db.execute("SELECT id, title, created_at FROM conversations WHERE id = ?", (conv_id,))
    row = await cursor.fetchone()
    if not row:
        logger.warning("Conversation not found for title update — id=%s", conv_id)
        raise HTTPException(status_code=404, detail=f"Conversation {conv_id} not found")

    await db.execute(
        "UPDATE conversations SET title = ? WHERE id = ?",
        (data.title, conv_id),
    )
    await db.commit()

    updated = ConversationOut(id=conv_id, title=data.title, created_at=row["created_at"])
    logger.info("Updated title for conversation id=%s to %r", conv_id, data.title)
    return ok(data=updated.model_dump())


# ── CLEAR ALL CONVERSATIONS ──────────────────────────────────────────────
@router.delete("/conversations")
async def clear_all_conversations(db: aiosqlite.Connection = Depends(get_db)):
    """Delete all conversations and messages from the database."""
    await db.execute("DELETE FROM messages")
    await db.execute("DELETE FROM conversations")
    await db.commit()

    logger.info("Cleared all conversations and messages from database")
    return ok(data={"message": "All conversations cleared"})


# ── DELETE ONE ───────────────────────────────────────────────────────────
@router.delete("/conversations/{conv_id}")
async def delete_conversation(
    conv_id: str,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Delete a conversation and all its messages.
    """
    cursor = await db.execute("SELECT id FROM conversations WHERE id = ?", (conv_id,))
    row = await cursor.fetchone()
    if not row:
        logger.warning("Conversation not found for deletion — id=%s", conv_id)
        raise HTTPException(status_code=404, detail=f"Conversation {conv_id} not found")

    await db.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
    await db.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    await db.commit()

    logger.info("Deleted conversation id=%s and its messages", conv_id)
    return None


# ── ROLLBACK ─────────────────────────────────────────────────────────────
@router.post("/conversations/{conv_id}/rollback")
async def rollback_last_interaction(
    conv_id: str,
    db: aiosqlite.Connection = Depends(get_db)
):
    """
    Roll back the last conversation interaction:
    Deletes the last assistant message and the last user message.
    """
    cursor = await db.execute(
        "SELECT id, role FROM messages WHERE conversation_id = ? ORDER BY created_at DESC LIMIT 2",
        (conv_id,)
    )
    rows = await cursor.fetchall()
    
    if not rows:
        raise HTTPException(status_code=400, detail="No messages to roll back")
        
    ids_to_delete = [row["id"] for row in rows]
    placeholders = ", ".join("?" for _ in ids_to_delete)
    await db.execute(
        f"DELETE FROM messages WHERE id IN ({placeholders})",
        ids_to_delete
    )
    await db.commit()
    
    logger.info("Rolled back last interaction in conv=%s (deleted message IDs: %s)", conv_id, ids_to_delete)
    return ok(data={"deleted_count": len(ids_to_delete)})
