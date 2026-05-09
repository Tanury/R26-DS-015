from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # dementia_speech_research/

MODELS_DIR = BASE_DIR / "notebooks" / "models"

TEXT_MODEL_PATH = MODELS_DIR / "text_dementia_model.joblib"
AUDIO_MODEL_PATH = MODELS_DIR / "audio_dementia_model.joblib"
AUDIO_FEATURE_COLUMNS_PATH = MODELS_DIR / "audio_feature_columns.json"

CLASS_NAMES = ["Control", "Dementia"]