"""
Global Exception Handlers
===========================

Registers FastAPI exception handlers that return errors in the
standardised ``APIResponse`` envelope so consumers always get a
predictable JSON shape.
"""
from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.schemas import error_response
from src.utils.logger import get_logger
from src.utils.request_context import get_request_id

logger = get_logger(__name__)


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all for unhandled exceptions.

    Logs the full traceback and returns a clean ``500`` response
    with the request ID for correlation.
    """
    rid = get_request_id()
    logger.error(
        "Unhandled exception on %s %s  [request_id=%s]",
        request.method,
        request.url,
        rid,
        exc_info=exc,
    )
    body = error_response(
        code="INTERNAL_SERVER_ERROR",
        message="An unexpected error occurred. Please try again later.",
        details={"request_id": rid},
    )
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=jsonable_encoder(body))


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Catch Pydantic / FastAPI validation errors and return a ``422``
    response with machine-readable error details.
    """
    rid = get_request_id()
    logger.warning(
        "Validation error on %s %s  [request_id=%s]: %s",
        request.method,
        request.url,
        rid,
        exc.errors(),
    )
    body = error_response(
        code="VALIDATION_ERROR",
        message="Request validation failed.",
        details={"errors": exc.errors(), "request_id": rid},
    )
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content=jsonable_encoder(body))


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """Preserve the status and detail of every framework HTTP exception."""
    rid = get_request_id()
    is_not_found = exc.status_code == status.HTTP_404_NOT_FOUND
    detail = exc.detail
    message = detail if isinstance(detail, str) else "The request could not be completed."
    body = error_response(
        code="NOT_FOUND" if is_not_found else f"HTTP_{exc.status_code}",
        message=message,
        details={
            "request_id": rid,
            **({"detail": detail} if not isinstance(detail, str) else {}),
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=jsonable_encoder(body),
        headers=exc.headers,
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers on the FastAPI application."""
    app.add_exception_handler(Exception, global_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    logger.info("Global exception handlers registered (500, 422, HTTP status preserving)")
