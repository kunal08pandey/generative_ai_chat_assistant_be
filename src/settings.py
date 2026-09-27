"""Central application settings with stable project-relative defaults."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


BACKEND_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_ROOT / ".env")


def _project_path(env_name: str, default: str) -> Path:
    configured = Path(os.environ.get(env_name, default)).expanduser()
    return configured if configured.is_absolute() else BACKEND_ROOT / configured


def _ollama_base_url() -> str:
    explicit = os.environ.get("OLLAMA_BASE_URL")
    if explicit:
        return explicit.rstrip("/")

    default_host = "host.docker.internal" if Path("/.dockerenv").exists() else "127.0.0.1"
    host = os.environ.get("OLLAMA_HOST", default_host)
    if host.startswith(("http://", "https://")):
        return host.rstrip("/")
    port = os.environ.get("OLLAMA_PORT", "11434")
    return f"http://{host}:{port}"


DATABASE_PATH = _project_path("DATABASE_URL", "data/chat.db")
CHROMA_PATH = _project_path("CHROMA_PATH", "data/chroma")
LOG_PATH = _project_path("LOG_DIR", "logs")
OLLAMA_BASE_URL = _ollama_base_url()
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))


def allowed_origins() -> list[str]:
    value = os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in value.split(",") if origin.strip()]


def admin_api_key() -> str | None:
    return os.environ.get("ADMIN_API_KEY") or None
