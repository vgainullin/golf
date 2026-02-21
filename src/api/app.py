from fastapi import FastAPI

from src.api.config import get_settings
from src.api.routes import game, secrets

app = FastAPI(
    title="Golf – Card Game & GitHub Secrets Manager",
    description=(
        "FastAPI interface for playing the Golf card game and managing "
        "GitHub Actions repository secrets."
    ),
    version="0.3.0",
)

app.include_router(secrets.router, prefix="/api/v1")
app.include_router(game.router, prefix="/api/v1")


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
