from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.handlers.error_handler import setup_exception_handlers


def make_client():
    app = FastAPI()
    setup_exception_handlers(app)

    @app.get("/teapot")
    async def teapot():
        raise HTTPException(status_code=418, detail="Short and stout")

    @app.get("/crash")
    async def crash():
        raise ValueError("private failure")

    return TestClient(app, raise_server_exceptions=False)


def test_http_exception_status_is_preserved():
    response = make_client().get("/teapot")
    assert response.status_code == 418
    assert response.json()["error"]["message"] == "Short and stout"


def test_internal_error_is_sanitized():
    response = make_client().get("/crash")
    assert response.status_code == 500
    assert "private failure" not in response.text
