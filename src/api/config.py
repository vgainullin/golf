from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    github_token: str = ""
    github_api_url: str = "https://api.github.com"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = {"env_prefix": "GOLF_"}


def get_settings() -> Settings:
    return Settings()
