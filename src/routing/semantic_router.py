"""
Semantic Router
===============

Routes incoming queries based on semantic vector similarity against predefined trigger
clusters. Enables deterministic bypass of foundation models for critical queries (e.g.
Identity, Safety, Policies) without relying on unreliable system prompts.
"""
from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.rag.embeddings import embed
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0

    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


@dataclass
class Route:
    """
    Represents a target semantic route.

    :param name: Unique name for the route (e.g., 'identity').
    :param samples: Cluster of sample triggers representing semantic intent.
    :param response: Deterministic response string to return when routed.
    :param threshold: Cosine similarity cutoff (0.0 to 1.0) for semantic matching.
    :param patterns: Optional regex patterns for instant zero-latency matching.
    """
    name: str
    samples: List[str]
    response: str
    threshold: float = 0.68
    patterns: List[str] = field(default_factory=list)


class SemanticRouter:
    """
    Vector-based Semantic Router that classifies queries against sample clusters.
    """

    def __init__(self, routes: Optional[List[Route]] = None):
        self.routes: List[Route] = routes or []
        self._sample_embeddings: dict[str, List[List[float]]] = {}
        self._lock = asyncio.Lock()

    def add_route(self, route: Route) -> None:
        """Register a new route."""
        self.routes.append(route)
        self._sample_embeddings.pop(route.name, None)

    async def _ensure_route_embeddings(self, route: Route) -> List[List[float]]:
        """Compute and cache embeddings for a route's sample triggers."""
        if route.name in self._sample_embeddings:
            return self._sample_embeddings[route.name]

        async with self._lock:
            if route.name in self._sample_embeddings:
                return self._sample_embeddings[route.name]

            logger.info("Generating trigger embeddings for semantic route '%s' (%d samples)", route.name, len(route.samples))
            vectors = await embed(route.samples)
            self._sample_embeddings[route.name] = vectors
            return vectors

    async def match(self, query: str) -> Optional[Tuple[Route, float]]:
        """
        Evaluate user query against all registered routes.

        Returns:
            Tuple of (matched_route, confidence_score) if matched, else None.
        """
        clean_query = query.strip()
        if not clean_query:
            return None

        # ── 1. Fast Pattern Check (0ms pre-check) ──────────────────────
        clean_lower = clean_query.lower()
        for route in self.routes:
            for pattern in route.patterns:
                if re.search(pattern, clean_lower, re.IGNORECASE):
                    logger.info("Semantic Router: matched route '%s' via pattern '%s'", route.name, pattern)
                    return route, 1.0

        # ── 2. Vector Semantic Similarity Check ────────────────────────
        query_vectors = await embed([clean_query])
        if not query_vectors or not query_vectors[0]:
            return None

        query_vec = query_vectors[0]
        best_route: Optional[Route] = None
        best_score: float = -1.0

        for route in self.routes:
            trigger_vecs = await self._ensure_route_embeddings(route)
            if not trigger_vecs:
                continue

            # Calculate similarity against each sample in cluster
            similarities = [_cosine_similarity(query_vec, tv) for tv in trigger_vecs]
            max_sim = max(similarities) if similarities else 0.0

            if max_sim > best_score:
                best_score = max_sim
                best_route = route

        if best_route and best_score >= best_route.threshold:
            logger.info(
                "Semantic Router: matched route '%s' with similarity score %.3f (threshold=%.2f)",
                best_route.name,
                best_score,
                best_route.threshold,
            )
            return best_route, best_score

        logger.debug("Semantic Router: no route matched (best='%s', score=%.3f)", getattr(best_route, "name", "none"), best_score)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Default Identity Route Configuration
# ─────────────────────────────────────────────────────────────────────────────

IDENTITY_DETERMINISTIC_RESPONSE = (
    "I am an AI assistant built for Kshitiz AI Labs, developed by Kunal Pandey."
)

DEFAULT_IDENTITY_ROUTE = Route(
    name="identity",
    response=IDENTITY_DETERMINISTIC_RESPONSE,
    threshold=0.68,
    patterns=[
        r"\bwho (made|created|developed|built|programmed|authored) (you|this( platform| assistant| bot| app| website)?)\b",
        r"\b(owner|creator|developer|maker|founder|author) of (this|the) (model|platform|assistant|ai|bot|app|system|project)\b",
        r"\bare you (made|created|built|owned) by (google|meta|facebook|openai|microsoft|anthropic)\b",
        r"\bwho is kunal pandey\b",
        r"\bwho runs (kshitiz ai labs|gen labs)\b",
        r"\bwhat is kshitiz ai labs\b",
        r"\bwho owns you\b",
    ],
    samples=[
        "who made you?",
        "who developed this platform?",
        "owner of this model",
        "who created you?",
        "who is your creator?",
        "who built you?",
        "who developed this app?",
        "who is your developer?",
        "who built this assistant?",
        "who owns this ai?",
        "who made this project?",
        "are you made by google?",
        "are you made by meta?",
        "are you chatgpt or openai?",
        "which company created you?",
        "who is the author of this assistant?",
        "who designed this platform?",
        "tell me who created you",
        "who is the mastermind behind this?",
        "who is kunal pandey?",
    ],
)

_ROUTER_INSTANCE: Optional[SemanticRouter] = None


def get_semantic_router() -> SemanticRouter:
    """Singleton getter for the pre-configured application SemanticRouter."""
    global _ROUTER_INSTANCE
    if _ROUTER_INSTANCE is None:
        _ROUTER_INSTANCE = SemanticRouter([DEFAULT_IDENTITY_ROUTE])
    return _ROUTER_INSTANCE

