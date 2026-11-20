"""
Documents Management Endpoints
==============================

Provides document management capabilities scoped strictly to the authenticated user:
  • GET /documents — List user's indexed documents
  • DELETE /documents/{doc_id} — Delete an indexed document and clean up vector store chunks
"""
from typing import Optional
import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies.auth import get_current_user
from src.api.schemas import ok
from src.db.database import get_db
from src.rag.vector_store import delete_document_by_id
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


@router.get("")
async def list_documents(
    user: dict = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=100),
):
    """
    List all documents indexed by the current user.
    """
    user_id = user["id"]
    offset = (page - 1) * page_size

    # Count total documents for pagination
    cursor = await db.execute(
        "SELECT COUNT(*) FROM documents WHERE owner_id = ?",
        (user_id,),
    )
    total_count = (await cursor.fetchone())[0]

    cursor = await db.execute(
        """
        SELECT id, filename, content_type, chunks_count, created_at
        FROM documents
        WHERE owner_id = ?
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
        """,
        (user_id, page_size, offset),
    )
    rows = await cursor.fetchall()

    documents = [
        {
            "id": row[0],
            "filename": row[1],
            "content_type": row[2],
            "chunks_count": row[3],
            "created_at": row[4],
        }
        for row in rows
    ]

    total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 1

    return ok(
        data=documents,
        meta={
            "page": page,
            "page_size": page_size,
            "total_count": total_count,
            "total_pages": total_pages,
        },
    )


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
    user: dict = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Delete an indexed document belonging to the authenticated user.
    Cleans up all associated chunk vectors from ChromaDB and the database.
    """
    user_id = user["id"]

    cursor = await db.execute(
        "SELECT id, filename FROM documents WHERE id = ? AND owner_id = ?",
        (doc_id, user_id),
    )
    doc = await cursor.fetchone()
    if not doc:
        logger.warning("Document deletion rejected: doc_id=%s not found or not owned by user=%s", doc_id, user_id)
        raise HTTPException(status_code=404, detail="Document not found")

    filename = doc[1]

    # 1. Clean up vectors from ChromaDB
    try:
        await delete_document_by_id(doc_id=doc_id, owner_id=user_id)
    except Exception as exc:
        logger.error("Failed to delete vector chunks for doc_id=%s: %s", doc_id, exc)

    # 2. Remove document from SQLite
    await db.execute(
        "DELETE FROM documents WHERE id = ? AND owner_id = ?",
        (doc_id, user_id),
    )
    await db.commit()

    logger.info("Deleted document %s (id=%s) for user=%s", filename, doc_id, user_id)
    return ok(
        data={"id": doc_id, "filename": filename, "message": f"Document '{filename}' deleted successfully"},
    )
