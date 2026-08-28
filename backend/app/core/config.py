import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


class Settings:
    app_name: str = os.getenv("APP_NAME", "NeuroRisk Research Platform")
    environment: str = os.getenv("ENVIRONMENT", "development")
    model_dir: Path = Path(__file__).resolve().parents[1] / "models"
    frontend_dir: Path = Path(__file__).resolve().parents[3] / "frontend"
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
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
    biomedical_disclaimer: str = (
        "This neurological biomedical assessment is a research screening aid "
        "trained on a synthetic cohort. Its disease class is a pattern match and "
        "its risk score is an independent model estimate. It is not a medical "
        "diagnosis. A qualified clinician must interpret the output alongside "
        "symptoms, examination findings, assay methods, and the full clinical record."
    )

    # --- EEG risk module -------------------------------------------------
    eeg_model_dir: Path = Path(__file__).resolve().parents[1] / "models" / "eeg"
    # EEGLAB recordings are a .set header plus a .fdt data file; a 128-channel
    # 10-minute recording runs to roughly 90 MB across the pair.
    max_eeg_bytes: int = 220 * 1024 * 1024
    eeg_job_ttl_seconds: int = int(os.getenv("EEG_JOB_TTL_SECONDS", "1800"))
    eeg_max_active_jobs: int = int(os.getenv("EEG_MAX_ACTIVE_JOBS", "4"))
    eeg_disclaimer: str = (
        "This output is a research decision-support indicator produced by an EEG "
        "representation learning prototype. It is not a clinical diagnosis, has not "
        "been clinically validated, and must not be used as the basis of a medical "
        "decision."
    )


settings = Settings()
