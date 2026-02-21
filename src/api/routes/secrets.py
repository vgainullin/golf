from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.config import Settings, get_settings
from src.api.github_client import GitHubClientError, GitHubSecretsClient
from src.api.models import (
    CreateSecretRequest,
    CreateSecretResponse,
    DeleteSecretResponse,
    PublicKeyResponse,
    SecretListItem,
    SecretListResponse,
)

router = APIRouter(prefix="/repos/{owner}/{repo}/secrets", tags=["secrets"])


def _get_client(settings: Settings = Depends(get_settings)) -> GitHubSecretsClient:
    if not settings.github_token:
        raise HTTPException(
            status_code=500,
            detail="GOLF_GITHUB_TOKEN environment variable is not set",
        )
    return GitHubSecretsClient(settings.github_token, settings.github_api_url)


@router.get("", response_model=SecretListResponse)
async def list_secrets(
    owner: str,
    repo: str,
    per_page: int = Query(default=30, ge=1, le=100),
    page: int = Query(default=1, ge=1),
    client: GitHubSecretsClient = Depends(_get_client),
) -> SecretListResponse:
    """List all Actions secrets for a repository (names and metadata only).

    GitHub never exposes secret values through its API.
    """
    try:
        data = await client.list_secrets(owner, repo, per_page=per_page, page=page)
    except GitHubClientError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    return SecretListResponse(
        total_count=data["total_count"],
        secrets=[SecretListItem(**s) for s in data["secrets"]],
    )


@router.get("/{secret_name}", response_model=SecretListItem)
async def get_secret(
    owner: str,
    repo: str,
    secret_name: str,
    client: GitHubSecretsClient = Depends(_get_client),
) -> SecretListItem:
    """Get metadata for a single secret."""
    try:
        data = await client.get_secret(owner, repo, secret_name)
    except GitHubClientError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    return SecretListItem(**data)


@router.get("/public-key", response_model=PublicKeyResponse)
async def get_public_key(
    owner: str,
    repo: str,
    client: GitHubSecretsClient = Depends(_get_client),
) -> PublicKeyResponse:
    """Get the repository public key used to encrypt secrets."""
    try:
        data = await client.get_public_key(owner, repo)
    except GitHubClientError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    return PublicKeyResponse(key_id=data["key_id"], key=data["key"])


@router.put("", response_model=CreateSecretResponse)
async def create_or_update_secret(
    owner: str,
    repo: str,
    body: CreateSecretRequest,
    client: GitHubSecretsClient = Depends(_get_client),
) -> CreateSecretResponse:
    """Create or update a repository secret.

    The value is encrypted using the repo's public key before being sent to GitHub.
    """
    try:
        status = await client.create_or_update_secret(
            owner, repo, body.secret_name, body.secret_value
        )
    except GitHubClientError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    return CreateSecretResponse(status=status, secret_name=body.secret_name)


@router.delete("/{secret_name}", response_model=DeleteSecretResponse)
async def delete_secret(
    owner: str,
    repo: str,
    secret_name: str,
    client: GitHubSecretsClient = Depends(_get_client),
) -> DeleteSecretResponse:
    """Delete a repository secret."""
    try:
        await client.delete_secret(owner, repo, secret_name)
    except GitHubClientError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    return DeleteSecretResponse(status="deleted", secret_name=secret_name)
