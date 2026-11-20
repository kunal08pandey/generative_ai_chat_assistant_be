"""
Audio Transcription Route Tests
===============================

Tests:
  - Audio upload and transcription using Groq Whisper.
  - Rejection of unauthenticated requests (401).
  - Validation of audio file presence and size limits (400, 413).
  - Clean error handling when Groq API key is missing (503).
"""
import pytest


def test_audio_unauthenticated_rejected(client):
    res = client.post(
        "/audio/transcribe",
        files={"file": ("test.webm", b"mock-audio-data" * 20, "audio/webm")},
    )
    assert res.status_code == 401


def test_audio_transcribe_success(auth_client, monkeypatch):
    async def mock_transcribe(audio_bytes, **kwargs):
        assert len(audio_bytes) > 0
        return "Hello world, this is a transcribed voice message."

    monkeypatch.setattr("src.llm.groq_client.GroqClient.transcribe_audio", mock_transcribe)

    res = auth_client.post(
        "/audio/transcribe",
        files={"file": ("recording.webm", b"RIFF" + b"0" * 200, "audio/webm")},
    )
    assert res.status_code == 200
    data = res.json()["data"]
    assert data["text"] == "Hello world, this is a transcribed voice message."
    assert "whisper" in data["model"]


def test_audio_empty_rejected(auth_client):
    res = auth_client.post(
        "/audio/transcribe",
        files={"file": ("empty.webm", b"short", "audio/webm")},
    )
    assert res.status_code == 400


def test_audio_oversized_rejected(auth_client, monkeypatch):
    monkeypatch.setattr("src.api.routes.audio.MAX_AUDIO_BYTES", 100)
    res = auth_client.post(
        "/audio/transcribe",
        files={"file": ("big.webm", b"0" * 150, "audio/webm")},
    )
    assert res.status_code == 413


def test_audio_missing_api_key_returns_503(auth_client, monkeypatch):
    async def mock_fail(*args, **kwargs):
        raise ValueError("No Groq API key found.")

    monkeypatch.setattr("src.llm.groq_client.GroqClient.transcribe_audio", mock_fail)

    res = auth_client.post(
        "/audio/transcribe",
        files={"file": ("recording.webm", b"RIFF" + b"0" * 200, "audio/webm")},
    )
    assert res.status_code == 503
