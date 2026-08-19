import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


class Settings:
    app_name: str = os.getenv("APP_NAME", "Aaraby Research Backend")
    environment: str = os.getenv("ENVIRONMENT", "development")
    model_dir: Path = Path(__file__).resolve().parents[1] / "models"
    frontend_dir: Path = Path(__file__).resolve().parents[3] / "frontend"
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    frontend_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv(
            "FRONTEND_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000",
        ).split(",")
        if origin.strip()
    )
    max_audio_bytes: int = 18 * 1024 * 1024
    disclaimer: str = (
        "This assessment is an informational screening aid based on speech "
        "features. It is not a medical diagnosis and should not replace "
        "evaluation by a qualified clinician."
    )


settings = Settings()
