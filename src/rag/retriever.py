"""
RAG Context Retriever
======================

Retrieves relevant document context filtered strictly by owner_id.
Preserves structured citations (filename, chunk offset, doc_id).
"""
from typing import Optional, Tuple, List, Dict, Any
from .vector_store import search, search_with_metadata


class ContextString(str):
    citations: List[Dict[str, Any]]

    def __new__(cls, content: str, citations: Optional[List[Dict[str, Any]]] = None):
        instance = super().__new__(cls, content)
        instance.citations = citations or []
        return instance


async def retrieve_context_and_citations(
    question: str,
    owner_id: Optional[str] = None,
    n_results: int = 4,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Retrieve relevant context and structured citations scoped strictly to owner_id.
    Returns (context_text, citations_list).
    """
    chunks = await search_with_metadata(question, owner_id=owner_id, n_results=n_results)
    if not chunks:
        return "", []

    context_parts = []
    citations = []
    seen = set()

    for idx, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "Document")
        chunk_num = meta.get("chunk", idx - 1)
        doc_id = meta.get("doc_id", "")
        text = chunk.get("text", "")

        context_parts.append(f"[{idx}] (Source: {source}, Chunk: {chunk_num})\n{text}")

        citation_key = (source, chunk_num)
        if citation_key not in seen:
            seen.add(citation_key)
            preview = text[:150] + "..." if len(text) > 150 else text
            citations.append({
                "index": idx,
                "source": source,
                "chunk": chunk_num,
                "doc_id": doc_id,
                "snippet": preview,
            })

    return "\n\n".join(context_parts), citations


async def retrieve_context(question: str, owner_id: Optional[str] = None) -> ContextString:
    """
    Retrieve relevant context from the vector store for a given question and owner.
    Returns a ContextString (subclass of str) that carries .citations metadata.
    """
    context_text, citations = await retrieve_context_and_citations(question, owner_id=owner_id)
    return ContextString(context_text, citations=citations)
