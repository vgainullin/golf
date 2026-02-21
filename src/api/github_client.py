from __future__ import annotations

import base64
from typing import Any

import httpx
from nacl import encoding, public


class GitHubClientError(Exception):
    """Raised when the GitHub API returns an error."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(message)


def _encrypt_secret(public_key: str, secret_value: str) -> str:
    """Encrypt a secret value using a repository's public key (libsodium sealed box)."""
    pk = public.PublicKey(public_key.encode("utf-8"), encoding.Base64Encoder())
    sealed = public.SealedBox(pk).encrypt(secret_value.encode("utf-8"))
    return base64.b64encode(sealed).decode("utf-8")


class GitHubSecretsClient:
    """Async client for the GitHub Actions repository secrets API."""

    def __init__(self, token: str, base_url: str = "https://api.github.com") -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def _request(
        self, method: str, path: str, **kwargs: Any
    ) -> httpx.Response:
        async with httpx.AsyncClient(
            base_url=self._base_url, headers=self._headers
        ) as client:
            resp = await client.request(method, path, **kwargs)
            if resp.status_code >= 400:
                body = resp.json() if resp.content else {}
                msg = body.get("message", resp.text)
                raise GitHubClientError(resp.status_code, msg)
            return resp

    async def list_secrets(
        self, owner: str, repo: str, per_page: int = 30, page: int = 1
    ) -> dict[str, Any]:
        """List all secrets for a repository (names and metadata only)."""
        resp = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/actions/secrets",
            params={"per_page": per_page, "page": page},
        )
        return resp.json()

    async def get_secret(self, owner: str, repo: str, secret_name: str) -> dict[str, Any]:
        """Get metadata for a single secret (name, created_at, updated_at)."""
        resp = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/actions/secrets/{secret_name}",
        )
        return resp.json()

    async def get_public_key(self, owner: str, repo: str) -> dict[str, Any]:
        """Get the repository's public key for encrypting secrets."""
        resp = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/actions/secrets/public-key",
        )
        return resp.json()

    async def create_or_update_secret(
        self, owner: str, repo: str, secret_name: str, secret_value: str
    ) -> str:
        """Create or update a repository secret.

        Returns 'created' or 'updated' depending on whether the secret existed.
        """
        pk_data = await self.get_public_key(owner, repo)
        encrypted = _encrypt_secret(pk_data["key"], secret_value)

        resp = await self._request(
            "PUT",
            f"/repos/{owner}/{repo}/actions/secrets/{secret_name}",
            json={"encrypted_value": encrypted, "key_id": pk_data["key_id"]},
        )
        return "created" if resp.status_code == 201 else "updated"

    async def delete_secret(self, owner: str, repo: str, secret_name: str) -> None:
        """Delete a repository secret."""
        await self._request(
            "DELETE",
            f"/repos/{owner}/{repo}/actions/secrets/{secret_name}",
        )
