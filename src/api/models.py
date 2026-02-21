from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SecretListItem(BaseModel):
    name: str
    created_at: datetime
    updated_at: datetime


class SecretListResponse(BaseModel):
    total_count: int
    secrets: list[SecretListItem]


class PublicKeyResponse(BaseModel):
    key_id: str
    key: str


class CreateSecretRequest(BaseModel):
    secret_name: str = Field(..., pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    secret_value: str = Field(..., min_length=1)


class CreateSecretResponse(BaseModel):
    status: str
    secret_name: str


class DeleteSecretResponse(BaseModel):
    status: str
    secret_name: str


class ErrorResponse(BaseModel):
    detail: str
