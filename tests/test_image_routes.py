import pytest
from unittest.mock import AsyncMock, patch


def test_generate_image_with_gemini_flash_selected(client, conversation, monkeypatch):
    """When Gemini Flash is selected, image generation must use Gemini Flash."""
    fake_gemini_gen = AsyncMock(return_value=(
        "https://image.pollinations.ai/prompt/test_flash?width=1024&height=1024",
        "Gemini Flash",
        "A serene mountain lake at sunrise with golden light"
    ))
    monkeypatch.setattr("src.api.routes.image.generate_gemini_image", fake_gemini_gen)

    response = client.post(
        "/generate-image",
        json={
            "prompt": "A serene mountain lake at sunrise",
            "model": "gemini-3.5-flash",
            "conversation_id": conversation["id"],
        },
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["generator"] == "Gemini Flash"
    assert "Gemini Flash" in data["content"]
    assert "Gemini Scene Synthesis" in data["content"]
    fake_gemini_gen.assert_awaited_once()


def test_generate_image_with_gemini_flash_lite_selected(client, conversation, monkeypatch):
    """When Gemini Flash-Lite is selected, image generation must use Gemini Flash-Lite."""
    fake_gemini_gen = AsyncMock(return_value=(
        "https://image.pollinations.ai/prompt/test_lite?width=1024&height=1024",
        "Gemini Flash-Lite",
        "A quiet forest clearing with morning mist"
    ))
    monkeypatch.setattr("src.api.routes.image.generate_gemini_image", fake_gemini_gen)

    response = client.post(
        "/generate-image",
        json={
            "prompt": "A quiet forest clearing with morning mist",
            "model": "gemini-3.5-flash-lite",
            "conversation_id": conversation["id"],
        },
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["generator"] == "Gemini Flash-Lite"
    assert "Gemini Flash-Lite" in data["content"]
    fake_gemini_gen.assert_awaited_once()


def test_generate_image_with_non_gemini_selected(client, conversation, monkeypatch):
    """When Gemini is NOT selected (e.g. gemma4 or local model), image generation must use x/flux2-klein."""
    fake_flux_gen = AsyncMock(return_value=("https://image.pollinations.ai/prompt/test?model=flux", "x/flux2-klein"))
    monkeypatch.setattr("src.api.routes.image.generate_flux2_klein_image", fake_flux_gen)

    response = client.post(
        "/generate-image",
        json={
            "prompt": "A cyberpunk street with neon signs",
            "model": "gemma4:e4b",
            "conversation_id": conversation["id"],
        },
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["generator"] == "x/flux2-klein"
    assert "x/flux2-klein" in data["content"]
    fake_flux_gen.assert_awaited_once()


def test_generate_image_with_flux_selected(client, conversation, monkeypatch):
    """When x/flux2-klein is explicitly selected, image generation must use x/flux2-klein."""
    fake_flux_gen = AsyncMock(return_value=("data:image/png;base64,fakefluxb64", "x/flux2-klein"))
    monkeypatch.setattr("src.api.routes.image.generate_flux2_klein_image", fake_flux_gen)

    response = client.post(
        "/generate-image",
        json={
            "prompt": "A cozy cottage in autumn",
            "model": "x/flux2-klein",
            "conversation_id": conversation["id"],
        },
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["generator"] == "x/flux2-klein"
    assert "x/flux2-klein" in data["content"]
    fake_flux_gen.assert_awaited_once()


def test_chat_stream_routes_image_prompt_to_gemini_flash(client, conversation, monkeypatch):
    """When Gemini Flash is selected, /chat with an image prompt routes to Gemini Flash."""
    fake_gemini_gen = AsyncMock(return_value=(
        "https://image.pollinations.ai/prompt/test_flash?width=1024&height=1024",
        "Gemini Flash",
        "Golden retriever puppy playing"
    ))
    monkeypatch.setattr("src.api.routes.image.generate_gemini_image", fake_gemini_gen)

    response = client.post(
        "/chat/",
        json={
            "query": "Generate an image of a golden retriever puppy playing",
            "model": "gemini-3.5-flash",
            "conversation_id": conversation["id"],
        },
    )
    assert response.status_code == 200
    assert "Gemini Flash" in response.text
    fake_gemini_gen.assert_awaited_once()


def test_chat_stream_routes_image_prompt_to_gemini_flash_lite(client, conversation, monkeypatch):
    """When Gemini Flash-Lite is selected, /chat with an image prompt routes to Gemini Flash-Lite."""
    fake_gemini_gen = AsyncMock(return_value=(
        "https://image.pollinations.ai/prompt/test_lite?width=1024&height=1024",
        "Gemini Flash-Lite",
        "Golden retriever puppy playing"
    ))
    monkeypatch.setattr("src.api.routes.image.generate_gemini_image", fake_gemini_gen)

    response = client.post(
        "/chat/",
        json={
            "query": "Generate an image of a golden retriever puppy playing",
            "model": "gemini-3.5-flash-lite",
            "conversation_id": conversation["id"],
        },
    )
    assert response.status_code == 200
    assert "Gemini Flash-Lite" in response.text
    fake_gemini_gen.assert_awaited_once()


def test_chat_stream_routes_image_prompt_to_flux_when_non_gemini_selected(client, conversation, monkeypatch):
    """When Gemini is NOT selected, /chat with an image prompt routes to x/flux2-klein."""
    fake_flux_gen = AsyncMock(return_value=("https://image.pollinations.ai/prompt/test?model=flux", "x/flux2-klein"))
    monkeypatch.setattr("src.api.routes.image.generate_flux2_klein_image", fake_flux_gen)

    response = client.post(
        "/chat/",
        json={
            "query": "Create an image of an astronaut on Mars",
            "model": "gemma4:e4b",
            "conversation_id": conversation["id"],
        },
    )
    assert response.status_code == 200
    assert "x/flux2-klein" in response.text
    fake_flux_gen.assert_awaited_once()
