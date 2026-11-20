"""
Conversations Endpoints
========================

CRUD operations for conversation threads with owner-scoped data isolation.

  • ``GET    /conversations``           — List user's conversations (paginated).
  • ``POST   /conversations``           — Create (owned by authenticated user).
  • ``PATCH  /conversations/{conv_id}`` — Update title (owner only).
  • ``DELETE /conversations/{conv_id}`` — Delete (owner only, cascade messages).
  • ``DELETE /conversations``           — Clear all user's conversations (owner-scoped).

Phase 1 security improvements (audit S3):
  - Every endpoint requires a validated session (get_current_user dependency).
  - Conversations are scoped to owner_id — user A cannot see/modify user B's data.
  - "Clear all" only deletes the requesting user's conversations, not global.
  - Purge logic is scoped to the current user's empty threads.
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
from src.api.dependencies.auth import get_current_user
from src.db.database import get_db
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Conversations"])


# ── LIST (paginated, strictly read-only) ──────────────────────────────────
@router.get("/conversations")
async def list_conversations(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    user: dict = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Return a paginated list of the current user's conversations,
    ordered by most recent first. Strictly read-only (Phase 4).
    """
    user_id = user["id"]
    logger.debug("Listing conversations for user=%s — page=%d, page_size=%d", user_id, page, page_size)

    # Total count (owner-scoped)
    count_cursor = await db.execute(
        "SELECT COUNT(*) FROM conversations WHERE owner_id = ?", (user_id,)
    )
    (total_count,) = await count_cursor.fetchone()

    # Paginated rows (owner-scoped)
    offset = (page - 1) * page_size
    cursor = await db.execute(
        "SELECT id, title, created_at FROM conversations WHERE owner_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (user_id, page_size, offset),
    )
    rows = await cursor.fetchall()
    items = [ConversationOut(**dict(row)).model_dump() for row in rows]

    logger.info("Listed %d conversations for user=%s (page %d/%d)",
                len(items), user_id, page, -(-total_count // page_size) if page_size else 1)
    return paginated_ok(items=items, total_count=total_count, page=page, page_size=page_size)


# ── CREATE ───────────────────────────────────────────────────────────────
@router.post("/conversations", status_code=201)
async def create_conversation(
    data: ConversationCreate,
    user: dict = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
):
    """Create a new conversation thread owned by the authenticated user."""
    conv_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    user_id = user["id"]

    await db.execute(
        "INSERT INTO conversations (id, title, created_at, owner_id) VALUES (?, ?, ?, ?)",
        (conv_id, data.title, created_at, user_id),
    )
    await db.commit()

    conversation = ConversationOut(id=conv_id, title=data.title, created_at=created_at)
    logger.info("Created conversation id=%s title=%r owner=%s", conv_id, data.title, user_id)
    return ok(data=conversation.model_dump())


# ── UPDATE ───────────────────────────────────────────────────────────────
@router.patch("/conversations/{conv_id}")
async def update_conversation(
    conv_id: str,
    data: ConversationUpdate,
    user: dict = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
):
    """Update the title of a conversation owned by the authenticated user."""
    user_id = user["id"]

    cursor = await db.execute(
        "SELECT id, title, created_at, owner_id FROM conversations WHERE id = ?", (conv_id,)
    )
    row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Conversation {conv_id} not found")

    # Ownership check
    if row["owner_id"] and row["owner_id"] != user_id:
        raise HTTPException(status_code=404, detail=f"Conversation {conv_id} not found")

    await db.execute(
        "UPDATE conversations SET title = ? WHERE id = ? AND (owner_id = ? OR owner_id IS NULL)",
        (data.title, conv_id, user_id),
    )
    await db.commit()

    updated = ConversationOut(id=conv_id, title=data.title, created_at=row["created_at"])
    logger.info("Updated title for conversation id=%s to %r by user=%s", conv_id, data.title, user_id)
    return ok(data=updated.model_dump())


# ── CLEAR ALL CONVERSATIONS (owner-scoped) ───────────────────────────────
@router.delete("/conversations")
async def clear_all_conversations(
    user: dict = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Delete all of the current user's conversations and messages.
    This is owner-scoped — it does NOT delete other users' data (audit S3 fix).
    """
    user_id = user["id"]

    # Delete messages belonging to user's conversations
    await db.execute("""
        DELETE FROM messages WHERE conversation_id IN (
            SELECT id FROM conversations WHERE owner_id = ?
        )
    """, (user_id,))
    await db.execute("DELETE FROM conversations WHERE owner_id = ?", (user_id,))
    await db.commit()

    logger.info("Cleared all conversations for user=%s", user_id)
    return ok(data={"message": "All your conversations cleared"})


# ── DELETE ONE ───────────────────────────────────────────────────────────
@router.delete("/conversations/{conv_id}", status_code=204)
async def delete_conversation(
    conv_id: str,
    user: dict = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
):
    """Delete a conversation and all its messages (owner only)."""
    user_id = user["id"]

    cursor = await db.execute(
        "SELECT id, owner_id FROM conversations WHERE id = ?", (conv_id,)
    )
    row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Conversation {conv_id} not found")

    if row["owner_id"] and row["owner_id"] != user_id:
        raise HTTPException(status_code=404, detail=f"Conversation {conv_id} not found")

    await db.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
    await db.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    await db.commit()

    logger.info("Deleted conversation id=%s by user=%s", conv_id, user_id)
    return None


# ── ROLLBACK ─────────────────────────────────────────────────────────────
@router.post("/conversations/{conv_id}/rollback")
async def rollback_last_interaction(
    conv_id: str,
    user: dict = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
):
    """Roll back the last conversation interaction (owner only)."""
    user_id = user["id"]

    # Verify ownership
    cursor = await db.execute(
        "SELECT id, owner_id FROM conversations WHERE id = ?", (conv_id,)
    )
    conv = await cursor.fetchone()
    if not conv or (conv["owner_id"] and conv["owner_id"] != user_id):
        raise HTTPException(status_code=404, detail=f"Conversation {conv_id} not found")

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

    logger.info("Rolled back last interaction in conv=%s by user=%s (deleted %s)",
                conv_id, user_id, ids_to_delete)
    return ok(data={"deleted_count": len(ids_to_delete)})


@router.post("/conversations/{conv_id}/rollback-to/{message_id}")
async def rollback_to_message(
    conv_id: str,
    message_id: str,
    user: dict = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
):
    """Roll back conversation to a specific message (owner only)."""
    user_id = user["id"]

    # Verify ownership
    cursor = await db.execute(
        "SELECT id, owner_id FROM conversations WHERE id = ?", (conv_id,)
    )
    conv = await cursor.fetchone()
    if not conv or (conv["owner_id"] and conv["owner_id"] != user_id):
        raise HTTPException(status_code=404, detail=f"Conversation {conv_id} not found")

    cursor = await db.execute(
        "SELECT created_at FROM messages WHERE id = ? AND conversation_id = ?",
        (message_id, conv_id)
    )
    target = await cursor.fetchone()
    if not target:
        return await rollback_last_interaction(conv_id, user, db)

    await db.execute(
        "DELETE FROM messages WHERE conversation_id = ? AND created_at >= ?",
        (conv_id, target["created_at"])
    )
    await db.commit()
    logger.info("Rolled back conversation %s to message %s by user=%s", conv_id, message_id, user_id)
    return ok(data={"message": "Rolled back successfully"})
