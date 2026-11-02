"""
Unit tests for Authentication Routes (/auth)
"""
import base64
import json
import pytest
from starlette.testclient import TestClient


def make_mock_jwt(payload: dict) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256", "typ": "JWT"}).encode()).decode().rstrip("=")
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    sig = "mock_signature"
    return f"{header}.{body}.{sig}"


def test_auth_google_direct_sign_in(client: TestClient):
    payload = {
        "email": "testuser@gmail.com",
        "name": "Test Google User",
        "picture": "https://lh3.googleusercontent.com/a/test",
        "google_id": "google-12345",
    }
    response = client.post("/auth/google", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "user" in data["data"]
    user = data["data"]["user"]
    assert user["email"] == "testuser@gmail.com"
    assert user["name"] == "Test Google User"
    assert user["picture"] == "https://lh3.googleusercontent.com/a/test"
    assert "token" in data["data"]


def test_auth_google_jwt_credential(client: TestClient):
    jwt_claims = {
        "sub": "google-999888",
        "email": "jwtuser@gmail.com",
        "name": "JWT Google User",
        "picture": "https://lh3.googleusercontent.com/a/jwt-avatar",
        "email_verified": True,
    }
    credential = make_mock_jwt(jwt_claims)
    response = client.post("/auth/google", json={"credential": credential})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    user = data["data"]["user"]
    assert user["email"] == "jwtuser@gmail.com"
    assert user["name"] == "JWT Google User"
    assert user["id"] == "google-999888"


def test_auth_google_missing_email_fails(client: TestClient):
    response = client.post("/auth/google", json={})
    assert response.status_code == 400


def test_auth_get_me(client: TestClient):
    # First sign in
    client.post(
        "/auth/google",
        json={"email": "profile@gmail.com", "name": "Profile User"},
    )
    # Fetch profile
    response = client.get("/auth/me?email=profile@gmail.com")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["user"]["email"] == "profile@gmail.com"


def test_auth_logout(client: TestClient):
    response = client.post("/auth/logout")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
