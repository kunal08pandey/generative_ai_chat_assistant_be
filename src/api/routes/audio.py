"""
Audio & Transcription Endpoints
===============================

Provides speech-to-text audio transcription powered by Groq Whisper models.
Bypasses external browser vendor endpoints (speech.googleapis.com) entirely.
"""
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from src.api.dependencies.auth import get_current_user
from src.api.schemas import ok
from src.llm.groq_client import GroqClient
from src.utils.logger import get_logger
from src.utils.rate_limiter import RateLimiter

logger = get_logger(__name__)
router = APIRouter(prefix="/audio", tags=["Audio"])

MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB limit


@router.post("/transcribe", dependencies=[Depends(RateLimiter(max_requests=60, window_seconds=60))])
async def transcribe_audio(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """
    Transcribe an uploaded audio recording using Groq's high-performance Whisper model.
    """
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")

    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file exceeds 25 MB size limit")

    if file_size < 100:
        raise HTTPException(status_code=400, detail="Audio recording is empty or too short")

    audio_bytes = await file.read()
    filename = file.filename or "recording.webm"
    content_type = file.content_type or "audio/webm"

    try:
        text = await GroqClient.transcribe_audio(
            audio_bytes=audio_bytes,
            filename=filename,
            content_type=content_type,
            model="whisper-large-v3-turbo",
        )
    except ValueError as val_err:
        logger.warning("Transcription configuration error: %s", val_err)
        raise HTTPException(status_code=503, detail=str(val_err))
    except Exception as exc:
        logger.error("Audio transcription failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Audio transcription failed: {str(exc)}")

    return ok(data={
        "text": text,
        "model": "whisper-large-v3-turbo",
    })
