"""
Unit tests for the Semantic Router.
"""
import asyncio
import pytest
from src.routing.semantic_router import (
    Route,
    SemanticRouter,
    get_semantic_router,
    _cosine_similarity,
    IDENTITY_DETERMINISTIC_RESPONSE,
)


def test_cosine_similarity():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]
    assert _cosine_similarity(v1, v2) == pytest.approx(1.0)
    assert _cosine_similarity(v1, v3) == pytest.approx(0.0)
    assert _cosine_similarity([], v1) == 0.0


def test_identity_route_trigger_matches():
    async def run():
        router = get_semantic_router()
        triggers = [
            "who made you?",
            "who developed this platform?",
            "owner of this model",
            "who created you?",
            "are you made by google?",
            "are you made by meta?",
            "who is kunal pandey?",
            "who built this assistant?",
        ]

        for query in triggers:
            match = await router.match(query)
            assert match is not None, f"Failed to match trigger: '{query}'"
            route, score = match
            assert route.name == "identity"
            assert route.response == IDENTITY_DETERMINISTIC_RESPONSE
            assert score >= route.threshold

    asyncio.run(run())


def test_non_identity_fallthrough():
    async def run():
        router = get_semantic_router()
        unrelated_queries = [
            "what is the capital of France?",
            "write a python script to sort a list",
            "can you analyze this sales report?",
            "how does photosynthesis work?",
        ]

        for query in unrelated_queries:
            match = await router.match(query)
            assert match is None, f"Unrelated query '{query}' should not match identity route"

    asyncio.run(run())


def test_custom_route_registration():
    async def run():
        custom_route = Route(
            name="pricing",
            samples=["how much does this cost?", "what is the price?", "pricing plans"],
            response="This service is completely free and open-source.",
            threshold=0.65,
            patterns=[r"\b(price|pricing|cost)\b"],
        )

        router = SemanticRouter([custom_route])
        match = await router.match("what is the pricing?")
        assert match is not None
        route, _ = match
        assert route.name == "pricing"
        assert route.response == "This service is completely free and open-source."

    asyncio.run(run())


def test_chat_route_integration(client, conversation):
    """Verify that chatting with an identity question bypasses LLM and returns deterministic response."""
    response = client.post(
        "/chat/",
        json={
            "query": "Who developed this platform?",
            "model": "gen-labs-scout",
            "conversation_id": conversation["id"],
        },
    )
    assert response.status_code == 200
    assert IDENTITY_DETERMINISTIC_RESPONSE in response.text

    # Check that assistant message was saved to history
    history = client.get(f"/messages/{conversation['id']}").json()["data"]
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] == IDENTITY_DETERMINISTIC_RESPONSE
