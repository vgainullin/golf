from fastapi import FastAPI

from src.api.config import get_settings
from src.api.routes import secrets

app = FastAPI(
    title="Golf – GitHub Secrets Manager",
    description=(
        "FastAPI interface for managing GitHub Actions repository secrets. "
        "Note: GitHub never exposes secret *values* via its API — only names "
        "and metadata (created_at / updated_at) are returned."
    ),
    version="0.2.0",
)

app.include_router(secrets.router, prefix="/api/v1")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )


if __name__ == "__main__":
    main()
