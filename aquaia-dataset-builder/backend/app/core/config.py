from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    app_name: str = "ADIAB — AquaIA Dataset Builder"
    app_version: str = "0.1.0"
    debug: bool = True

    database_url: str = f"sqlite+aiosqlite:///{BASE_DIR}/storage/adiab.db"
    secret_key: str  # Required — set SECRET_KEY env var; no insecure default

    storage_raw: Path = BASE_DIR / "storage" / "raw"
    storage_validated: Path = BASE_DIR / "storage" / "validated"
    storage_rejected: Path = BASE_DIR / "storage" / "rejected"
    storage_duplicates: Path = BASE_DIR / "storage" / "duplicates"
    storage_exports: Path = BASE_DIR / "storage" / "exports"
    storage_thumbnails: Path = BASE_DIR / "storage" / "thumbnails"

    thumbnail_size: tuple[int, int] = (256, 256)
    max_concurrent_downloads: int = 5
    request_timeout: int = 30

    cors_origins: list[str] = ["http://localhost:3000", "http://frontend:3000"]

    class Config:
        env_file = ".env"


settings = Settings()
