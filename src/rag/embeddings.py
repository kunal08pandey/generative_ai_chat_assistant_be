"""
Embeddings Generation
======================

Generates vector embeddings using Ollama's ``/api/embed`` endpoint.
Includes a deterministic dense vector fallback so RAG indexing and document search
never crash even if local Ollama lacks a dedicated embedding model.
"""
import hashlib
import math
from typing import List

import httpx

from src.utils.host_resolver import get_ollama_host
from src.utils.http_client import get_http_client
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_EMBED_MODEL = "nomic-embed-text"


def _fallback_dense_vector(text: str, dim: int = 384) -> List[float]:
    """Generate a deterministic 384-dim normalized float embedding vector from text."""
    vector = [0.0] * dim
    words = text.lower().split()
    if not words:
        return vector

    for word in words:
        h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
        idx1 = h % dim
        idx2 = (h >> 16) % dim
        vector[idx1] += 1.0
        vector[idx2] += 0.5

    magnitude = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / magnitude for v in vector]


async def embed(texts: List[str]) -> List[List[float]]:
    """
    Generate high-dimensional vector embeddings for a list of strings.

    Tries Ollama batch embedding first. Falls back to sequential calls or
    deterministic dense vectors if Ollama embedding models are unavailable.
    """
    if not texts:
        return []

    cleaned_texts = [t for t in texts if t and isinstance(t, str)]
    if not cleaned_texts:
        return []

    client = get_http_client()
    host = get_ollama_host()
    url = f"http://{host}:11434/api/embed"

    # ── 1. Check if Ollama has an embedding model ──────────────────────
    embed_model = _DEFAULT_EMBED_MODEL
    try:
        tags_res = await client.get(f"http://{host}:11434/api/tags", timeout=2.0)
        if tags_res.status_code == 200:
            models_data = tags_res.json().get("models", [])
            installed_names = [m.get("name", "") for m in models_data]
            # Check for common embedding models
            for preferred in ["nomic-embed-text", "all-minilm", "mxbai-embed-large", "bge-large"]:
                matching = [name for name in installed_names if preferred in name]
                if matching:
                    embed_model = matching[0]
                    break
    except Exception as exc:
        logger.debug("Could not query Ollama tags for embedding model: %s", exc)

    # ── 2. Attempt Ollama Batch Embedding ──────────────────────────────
    try:
        logger.info("Requesting batch embeddings (%d chunks) using model=%s", len(cleaned_texts), embed_model)
        response = await client.post(
            url,
            json={
                "model": embed_model,
                "input": cleaned_texts,
            },
            timeout=30.0,
        )
        if response.status_code == 200:
            data = response.json()
            embeddings = data.get("embeddings", [])
            if len(embeddings) == len(cleaned_texts):
                return embeddings
    except Exception as e:
        logger.warning("Ollama batch embedding failed (%s). Falling back...", e)

    # ── 3. Fallback: Sequential Ollama Embedding ────────────────────────
    embeddings = []
    use_fallback = False

    for text in cleaned_texts:
        try:
            response = await client.post(
                url,
                json={
                    "model": embed_model,
                    "input": text,
                },
                timeout=10.0,
            )
            if response.status_code == 200:
                res_data = response.json()
                embed_vector = res_data.get("embeddings", [None])[0] or res_data.get("embedding", [])
                if embed_vector:
                    embeddings.append(embed_vector)
                    continue
            use_fallback = True
            break
        except Exception:
            use_fallback = True
            break

    # ── 4. Fallback: Deterministic Dense Vectors ─────────────────────────
    if use_fallback or len(embeddings) != len(cleaned_texts):
        logger.info("Using deterministic dense vector generator for RAG (%d chunks)", len(cleaned_texts))
        return [_fallback_dense_vector(t) for t in cleaned_texts]

    return embeddings
