"""
Upload Document Endpoint
=========================

``POST /upload`` — upload documents (PDF, DOCX, TXT) to index into ChromaDB vector store.

Optimizations:
  • Runs blocking synchronous extraction in an async thread pool.
  • Chunks extracted text into small overlapping paragraphs (standard RAG split)
    to reduce local context sizes, dramatically accelerating LLM processing times.
"""
import asyncio
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends

from src.api.schemas import ok
from src.rag.extractor import extract_text_from_file
from src.rag.text_splitter import RecursiveCharacterTextSplitter
from src.rag.vector_store import add_documents
from src.utils.logger import get_logger
from src.utils.rate_limiter import RateLimiter

logger = get_logger(__name__)
router = APIRouter(tags=["Upload"])


@router.post("/upload", dependencies=[Depends(RateLimiter(max_requests=60, window_seconds=60))])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document, extract text, split it into chunks, and add to the ChromaDB vector store.

    Runs extraction logic inside a background thread pool to ensure FastAPI concurrency,
    then splits the document into smaller semantic chunks before embedding them.
    """
    if file is None or not file.filename:
        logger.warning("Upload request rejected: No file provided")
        raise HTTPException(status_code=400, detail="No file uploaded")

    logger.info("Received file upload request. Filename: %s", file.filename)

    # ── Extract text from file (running in thread pool) ────────────────
    try:
        logger.debug("Extracting text from %s (async thread pool)", file.filename)
        text = await asyncio.to_thread(extract_text_from_file, file)
        if not text or not text.strip():
            raise ValueError("No text could be extracted from the file")
    except Exception as e:
        logger.error("Failed to extract text from %s: %s", file.filename, e, exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to extract text: {str(e)}")

    # ── Split Text into Semantic Chunks ──────────────────────────────────
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        if not chunks:
            chunks = [text]  # Fallback to single chunk if split returned nothing
    except Exception as split_err:
        logger.error("Failed to split text for %s: %s", file.filename, split_err)
        chunks = [text]

    # Generate unique IDs and source metadata for each chunk
    ids = [f"{file.filename}_chunk_{idx}" for idx in range(len(chunks))]
    metadatas = [{"source": file.filename, "chunk": idx} for idx in range(len(chunks))]

    # ── Index into Vector Store (Chroma) ─────────────────────────────────
    logger.info("Ingesting %d chunks for document %s to vector store", len(chunks), file.filename)
    await add_documents(
        texts=chunks,
        ids=ids,
        metadatas=metadatas,
    )

    logger.info("Document %s uploaded, chunked, and indexed successfully", file.filename)
    return ok(data={
        "filename": file.filename,
        "content_type": file.content_type,
        "chunks_indexed": len(chunks),
        "status": "success",
    })
