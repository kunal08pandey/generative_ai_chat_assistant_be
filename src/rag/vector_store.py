from __future__ import annotations
import chromadb
from .embeddings import embed
from typing import Optional
from src.settings import CHROMA_PATH

CHROMA_PATH.mkdir(parents=True, exist_ok=True)
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_or_create_collection(name="documents")


async def add_documents(texts: list[str], ids: list[str], metadatas: Optional[list[dict]] = None):
    """Add documents with their embeddings to the ChromaDB collection."""
    embeddings = await embed(texts)

    collection.upsert(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas or [{} for _ in texts],
    )


async def search(query: str):
    """Search for relevant documents in the ChromaDB collection."""
    embedding = await embed([query])
    results = collection.query(
        query_embeddings=embedding,
        n_results=3,
    )
    if not results or not results.get("documents") or len(results["documents"]) == 0 or len(results["documents"][0]) == 0:
        return ""
    return results["documents"][0]
