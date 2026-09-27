"""
Generative AI API Main Entrypoint
=================================

Bootstraps the FastAPI application, registers routes, sets up global middleware,
configures logging, and initialises database connections.
"""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import (
    chat,
    conversations,
    health,
    image,
    logs,
    messages,
    models,
    rag,
    upload,
)
from src.db.database import init_db
from src.handlers.error_handler import setup_exception_handlers
from src.utils.logger import get_logger, setup_logging
from src.utils.request_context import set_request_id
from src.settings import allowed_origins

# ── 1. Initialize Logger Configuration ─────────────────────────────────────
setup_logging()
logger = get_logger(__name__)


# ── 2. Lifespan Event Handler ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle startup and shutdown sequences.
    """
    logger.info("==================================================")
    logger.info("   Starting Generative AI Assistant Backend v1.0   ")
    logger.info("==================================================")
    
    # Initialize the SQLite tables and indexes
    try:
        init_db()
    except Exception as e:
        logger.critical("Failed to initialize database: %s", e, exc_info=True)
        raise e

    yield

    logger.info("Closing HTTP client connections...")
    from src.utils.http_client import close_http_client
    await close_http_client()
    logger.info("Shutting down Generative AI Assistant Backend...")


# ── 3. FastAPI Application Setup ───────────────────────────────────────────
app = FastAPI(
    title="Generative AI API",
    description="Production-grade assistant backend with modular LLM/RAG integration",
    version="1.0.0",
    lifespan=lifespan,
)

# Register custom 500, 422 and 404 response handlers
setup_exception_handlers(app)


# ── 4. Global Middleware ───────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    HTTP middleware to:
      • Intercept incoming requests.
      • Inject or generate a request/correlation ID context.
      • Time execution latency.
      • Log request details.
      • Attach X-Request-ID response header.
    """
    # Try getting custom header or generate a new one
    request_id = request.headers.get("X-Request-ID")
    request_id = set_request_id(request_id)

    start_time = time.monotonic()
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(
            "Request failed: %s %s | Request ID: %s", 
            request.method, request.url.path, request_id, 
            exc_info=e
        )
        raise e
    
    latency_ms = (time.monotonic() - start_time) * 1000
    
    # Log request line with correlation metadata
    logger.info(
        "%s %s - Status: %d - Latency: %.2fms | Request ID: %s",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
        request_id,
    )
    
    # Attach correlation header to client response
    response.headers["X-Request-ID"] = request_id
    return response


# Configure Cross-Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


# ── 5. Router Registrations ────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(chat.router, prefix="/chat")
app.include_router(rag.router, prefix="/rag")
app.include_router(image.router)
app.include_router(models.router)
app.include_router(upload.router)
app.include_router(conversations.router)
app.include_router(messages.router)
app.include_router(logs.router)
