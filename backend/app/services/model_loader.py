from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.exceptions import ModelLoadError


@dataclass(frozen=True)
class ModelAssets:
    model: Any
    scaler: Any
    label_encoder: Any
    feature_columns: list[str]


def _load_keras_model(path: Path) -> Any:
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        try:
            from keras.models import load_model
        except ImportError as exc:
            raise ModelLoadError("TensorFlow/Keras is required to load the classifier.") from exc
    return load_model(path)


@lru_cache(maxsize=1)
def load_model_assets() -> ModelAssets:
    try:
        import joblib
    except ImportError as exc:
        raise ModelLoadError("joblib is required to load model preprocessing assets.") from exc

    model_dir = settings.model_dir
    try:
        feature_columns = joblib.load(model_dir / "feature_columns.joblib")
        scaler = joblib.load(model_dir / "feature_scaler.joblib")
        label_encoder = joblib.load(model_dir / "label_encoder.joblib")
        model = _load_keras_model(model_dir / "speech_neuro_risk_classifier.keras")
    except Exception as exc:
        if isinstance(exc, ModelLoadError):
            raise
        raise ModelLoadError(f"Unable to load model assets: {exc}") from exc

    columns = [str(column) for column in list(feature_columns)]
    if not columns:
        raise ModelLoadError("feature_columns.joblib did not contain any feature names.")

    return ModelAssets(
        model=model,
        scaler=scaler,
        label_encoder=label_encoder,
        feature_columns=columns,
    )
