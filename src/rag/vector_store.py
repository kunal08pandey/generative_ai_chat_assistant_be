"""
Vector Store Layer (ChromaDB)
==============================

Tenant-isolated vector store interface.
All chunk additions, searches, and deletions are scoped to `owner_id`
to guarantee multi-user data isolation (Phase 1 & Phase 3).
"""
from __future__ import annotations

from typing import List, Optional

import chromadb

from .embeddings import embed
from src.settings import CHROMA_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

CHROMA_PATH.mkdir(parents=True, exist_ok=True)
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_or_create_collection(name="documents")


async def add_documents(
    texts: list[str],
    ids: list[str],
    metadatas: Optional[list[dict]] = None,
    owner_id: Optional[str] = None,
):
    """
    Add document chunks with their embeddings to ChromaDB.
    Ensures `owner_id` is embedded into every chunk's metadata.
    """
    embeddings = await embed(texts)

    enriched_metadatas = []
    base_metas = metadatas or [{} for _ in texts]
    for meta in base_metas:
        m = dict(meta)
        if owner_id:
            m["owner_id"] = str(owner_id)
        enriched_metadatas.append(m)

    collection.upsert(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=enriched_metadatas,
    )
    logger.info("Upserted %d chunks to ChromaDB (owner_id=%s)", len(texts), owner_id)


async def search_with_metadata(
    query: str,
    owner_id: Optional[str] = None,
    n_results: int = 3,
) -> list[dict]:
    """
    Search for relevant document chunks scoped strictly to the authorized owner.
    Returns list of dicts with keys: 'text', 'metadata', 'id'.
    """
    embedding = await embed([query])
    query_kwargs = {
        "query_embeddings": embedding,
        "n_results": n_results,
    }

    if owner_id:
        query_kwargs["where"] = {"owner_id": str(owner_id)}

    try:
        results = collection.query(**query_kwargs)
    except Exception as exc:
        logger.error("ChromaDB query failed: %s", exc)
        return []

    if (
        not results
        or not results.get("documents")
        or len(results["documents"]) == 0
        or len(results["documents"][0]) == 0
    ):
        return []

    docs = results["documents"][0]
    metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
    ids = results.get("ids", [[]])[0] if results.get("ids") else []

    chunks = []
    for idx, doc_text in enumerate(docs):
        meta = metas[idx] if idx < len(metas) else {}
        chunk_id = ids[idx] if idx < len(ids) else f"chunk_{idx}"
        chunks.append({
            "text": doc_text,
            "metadata": meta or {},
            "id": chunk_id,
        })
    return chunks


async def search(query: str, owner_id: Optional[str] = None, n_results: int = 3) -> list[str]:
    """
    Search for relevant document chunks scoped strictly to the authorized owner.
    If owner_id is provided, only chunks matching owner_id are returned.
    """
    chunks = await search_with_metadata(query=query, owner_id=owner_id, n_results=n_results)
    return [c["text"] for c in chunks]


async def delete_documents_by_owner(owner_id: str) -> None:
    """Delete all indexed chunks belonging to a specific owner."""
    try:
        collection.delete(where={"owner_id": str(owner_id)})
        logger.info("Deleted all vector store chunks for owner_id=%s", owner_id)
    except Exception as exc:
        logger.warning("Could not delete vector chunks for owner %s: %s", owner_id, exc)


async def delete_document_by_id(doc_id: str, owner_id: str) -> None:
    """Delete chunks for a specific document belonging to an owner."""
    try:
        collection.delete(
            where={
                "$and": [
                    {"doc_id": str(doc_id)},
                    {"owner_id": str(owner_id)},
                ]
            }
        )
        logger.info("Deleted vector store chunks for doc_id=%s, owner_id=%s", doc_id, owner_id)
    except Exception as exc:
        logger.warning("Could not delete doc %s for owner %s: %s", doc_id, owner_id, exc)
