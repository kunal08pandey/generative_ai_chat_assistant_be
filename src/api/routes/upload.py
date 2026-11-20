"""
Upload Document Endpoint
=========================

``POST /upload`` — Upload documents (PDF, DOCX, TXT, XLSX) to index into ChromaDB vector store.

Phase 1 & Phase 2 Security & Isolation Improvements:
  • Requires authenticated session (`get_current_user`).
  • Enforces MAX_UPLOAD_BYTES limit upfront (HTTP 413) before expensive work (audit S6).
  • Rejects unsupported formats with HTTP 415 (Unsupported Media Type).
  • Generates stable UUID for every document to prevent filename chunk collisions (audit S4).
  • Attaches `owner_id` to vector chunks and persists record in SQLite `documents` table.
"""
import asyncio
import uuid
from datetime import datetime, timezone

import aiosqlite
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends

from src.api.dependencies.auth import get_current_user
from src.api.schemas import ok
from src.db.database import get_db
from src.rag.extractor import extract_text_from_file
from src.rag.text_splitter import RecursiveCharacterTextSplitter
from src.rag.vector_store import add_documents
from src.settings import MAX_UPLOAD_BYTES
from src.utils.logger import get_logger
from src.utils.rate_limiter import RateLimiter

logger = get_logger(__name__)
router = APIRouter(tags=["Upload"])


@router.post("/upload", dependencies=[Depends(RateLimiter(max_requests=60, window_seconds=60))])
async def upload_document(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Upload a document, extract text, split it into chunks, and add to the ChromaDB vector store.
    Strictly isolated to the authenticated user.
    """
    if file is None or not file.filename:
        logger.warning("Upload request rejected: No file provided")
        raise HTTPException(status_code=400, detail="No file uploaded")

    # ── 1. Enforce MAX_UPLOAD_BYTES limit upfront (audit S6) ───────────
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > MAX_UPLOAD_BYTES:
        logger.warning(
            "Upload rejected: %s (%d bytes) exceeds MAX_UPLOAD_BYTES (%d)",
            file.filename, file_size, MAX_UPLOAD_BYTES
        )
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed upload size of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )

    user_id = user["id"]
    doc_id = str(uuid.uuid4())
    logger.info("Upload request: user=%s filename=%s size=%d doc_id=%s",
                user_id, file.filename, file_size, doc_id)

    # ── 2. Extract text from file (running in thread pool) ─────────────
    try:
        logger.debug("Extracting text from %s (async thread pool)", file.filename)
        text = await asyncio.to_thread(extract_text_from_file, file)
        if not text or not text.strip():
            raise ValueError("No text could be extracted from the file")
    except Exception as e:
        logger.error("Failed to extract text from %s: %s", file.filename, e, exc_info=True)
        if "Unsupported file type" in str(e):
            raise HTTPException(status_code=415, detail=str(e))
        raise HTTPException(status_code=400, detail=f"Failed to extract text: {str(e)}")

    # ── 3. Split Text into Semantic Chunks ─────────────────────────────
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        if not chunks:
            chunks = [text]
    except Exception as split_err:
        logger.error("Failed to split text for %s: %s", file.filename, split_err)
        chunks = [text]

    ids = [f"{doc_id}_chunk_{idx}" for idx in range(len(chunks))]
    metadatas = [
        {
            "source": file.filename,
            "chunk": idx,
            "doc_id": doc_id,
            "owner_id": user_id,
        }
        for idx in range(len(chunks))
    ]

    # ── 4. Index into Vector Store (Chroma) with owner_id ──────────────
    logger.info("Ingesting %d chunks for doc_id=%s (owner=%s) into ChromaDB",
                len(chunks), doc_id, user_id)
    await add_documents(
        texts=chunks,
        ids=ids,
        metadatas=metadatas,
        owner_id=user_id,
    )

    # ── 5. Record document in SQLite database ──────────────────────────
    now_iso = datetime.now(timezone.utc).isoformat()
    await db.execute(
        """
        INSERT INTO documents (id, filename, content_type, chunks_count, owner_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (doc_id, file.filename, file.content_type or "application/octet-stream", len(chunks), user_id, now_iso),
    )
    await db.commit()

    logger.info("Document %s (id=%s) uploaded and indexed for user %s",
                file.filename, doc_id, user_id)
    return ok(data={
        "id": doc_id,
        "filename": file.filename,
        "content_type": file.content_type,
        "chunks_indexed": len(chunks),
        "status": "success",
    })
