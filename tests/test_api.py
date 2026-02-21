"""Tests for the GitHub Secrets FastAPI module."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.config import Settings, get_settings
from src.api.github_client import GitHubClientError, GitHubSecretsClient, _encrypt_secret


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _override_settings() -> Settings:
    return Settings(github_token="test-token-123", github_api_url="https://api.github.com")


@pytest.fixture()
def client() -> TestClient:
    app.dependency_overrides[get_settings] = _override_settings
    yield TestClient(app)
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# List secrets
# ---------------------------------------------------------------------------


@patch.object(GitHubSecretsClient, "list_secrets", new_callable=AsyncMock)
def test_list_secrets(mock_list: AsyncMock, client: TestClient) -> None:
    mock_list.return_value = {
        "total_count": 1,
        "secrets": [
            {
                "name": "MY_SECRET",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-06-01T00:00:00Z",
            }
        ],
    }
    resp = client.get("/api/v1/repos/octocat/hello-world/secrets")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_count"] == 1
    assert body["secrets"][0]["name"] == "MY_SECRET"
    mock_list.assert_awaited_once_with("octocat", "hello-world", per_page=30, page=1)


@patch.object(GitHubSecretsClient, "list_secrets", new_callable=AsyncMock)
def test_list_secrets_pagination(mock_list: AsyncMock, client: TestClient) -> None:
    mock_list.return_value = {"total_count": 0, "secrets": []}
    resp = client.get("/api/v1/repos/octocat/hello-world/secrets?per_page=10&page=2")
    assert resp.status_code == 200
    mock_list.assert_awaited_once_with("octocat", "hello-world", per_page=10, page=2)


@patch.object(GitHubSecretsClient, "list_secrets", new_callable=AsyncMock)
def test_list_secrets_github_error(mock_list: AsyncMock, client: TestClient) -> None:
    mock_list.side_effect = GitHubClientError(404, "Not Found")
    resp = client.get("/api/v1/repos/octocat/missing/secrets")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Get single secret
# ---------------------------------------------------------------------------


@patch.object(GitHubSecretsClient, "get_secret", new_callable=AsyncMock)
def test_get_secret(mock_get: AsyncMock, client: TestClient) -> None:
    mock_get.return_value = {
        "name": "DB_PASS",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-06-01T00:00:00Z",
    }
    resp = client.get("/api/v1/repos/octocat/hello-world/secrets/DB_PASS")
    assert resp.status_code == 200
    assert resp.json()["name"] == "DB_PASS"


# ---------------------------------------------------------------------------
# Create / update secret
# ---------------------------------------------------------------------------


@patch.object(GitHubSecretsClient, "create_or_update_secret", new_callable=AsyncMock)
def test_create_secret(mock_create: AsyncMock, client: TestClient) -> None:
    mock_create.return_value = "created"
    resp = client.put(
        "/api/v1/repos/octocat/hello-world/secrets",
        json={"secret_name": "NEW_SECRET", "secret_value": "s3cret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "created"
    assert body["secret_name"] == "NEW_SECRET"


def test_create_secret_invalid_name(client: TestClient) -> None:
    resp = client.put(
        "/api/v1/repos/octocat/hello-world/secrets",
        json={"secret_name": "123-bad!", "secret_value": "val"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Delete secret
# ---------------------------------------------------------------------------


@patch.object(GitHubSecretsClient, "delete_secret", new_callable=AsyncMock)
def test_delete_secret(mock_del: AsyncMock, client: TestClient) -> None:
    mock_del.return_value = None
    resp = client.delete("/api/v1/repos/octocat/hello-world/secrets/OLD_SECRET")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"


# ---------------------------------------------------------------------------
# Missing token
# ---------------------------------------------------------------------------


def test_missing_token() -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(github_token="")
    c = TestClient(app)
    resp = c.get("/api/v1/repos/octocat/hello-world/secrets")
    assert resp.status_code == 500
    assert "GOLF_GITHUB_TOKEN" in resp.json()["detail"]
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Encryption helper
# ---------------------------------------------------------------------------


def test_encrypt_secret_produces_base64() -> None:
    """Verify that _encrypt_secret returns valid base64 without raising."""
    from nacl.public import PrivateKey
    import base64

    # Generate a real keypair so the sealed box can encrypt
    private_key = PrivateKey.generate()
    public_key_b64 = base64.b64encode(bytes(private_key.public_key)).decode("utf-8")

    encrypted = _encrypt_secret(public_key_b64, "hello-world")
    # Should be valid base64
    decoded = base64.b64decode(encrypted)
    assert len(decoded) > 0
