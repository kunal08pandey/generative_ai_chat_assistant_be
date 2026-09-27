from .vector_store import search


async def retrieve_context(question: str) -> str:
    """Retrieve relevant context from the vector store for a given question."""
    documents = await search(question)
    if not documents:
        return "No relevant context found"
    # Combine the top results
    return "\n".join(documents)