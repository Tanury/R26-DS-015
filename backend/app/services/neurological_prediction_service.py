from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings
from app.core.exceptions import ModelLoadError, PredictionError
from app.schemas.neurological_prediction_response import (
    NeurologicalPredictionResponse,
)
from app.schemas.neurological_risk_request import NeurologicalRiskRequest


MODEL_FILENAME = "neurological_risk_runtime_model.joblib"
SEX_FALLBACK = "Female"


@dataclass(frozen=True)
class NeurologicalModelAssets:
    pipeline: Any
    label_encoder: Any
    feature_columns: tuple[str, ...]


@lru_cache(maxsize=1)
def load_neurological_model() -> NeurologicalModelAssets:
    try:
        import joblib

        artifact = joblib.load(Path(settings.model_dir) / MODEL_FILENAME)
        pipeline = artifact["pipeline"]
        label_encoder = artifact["label_encoder"]
        feature_columns = tuple(str(name) for name in pipeline.feature_names_in_)
    except Exception as exc:
        raise ModelLoadError(
            "Unable to load the neurological risk model."
        ) from exc

    if not feature_columns:
        raise ModelLoadError(
            "The neurological risk model has no feature metadata."
        )
    if not hasattr(pipeline, "predict_proba"):
        raise ModelLoadError(
            "The neurological risk model does not support probabilities."
        )
    if not hasattr(label_encoder, "inverse_transform"):
        raise ModelLoadError(
            "The neurological risk model has an invalid label encoder."
        )

    return NeurologicalModelAssets(
        pipeline=pipeline,
        label_encoder=label_encoder,
        feature_columns=feature_columns,
    )


def predict_neurological_risk(
    payload: NeurologicalRiskRequest,
) -> NeurologicalPredictionResponse:
    return predict_neurological_risk_with_assets(
        payload,
        load_neurological_model(),
    )


def predict_neurological_risk_with_assets(
    payload: NeurologicalRiskRequest,
    assets: NeurologicalModelAssets,
) -> NeurologicalPredictionResponse:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ModelLoadError(
            "pandas is required to prepare neurological model features."
        ) from exc

    supplied_features = _request_values(payload)
    row = {column: np.nan for column in assets.feature_columns}
    row.update(supplied_features)

    if "sex" in row:
        row["sex"] = SEX_FALLBACK

    try:
        model_input = pd.DataFrame(
            [[row[column] for column in assets.feature_columns]],
            columns=assets.feature_columns,
        )
        raw_probabilities = assets.pipeline.predict_proba(model_input)
        probabilities = _normalize_probabilities(raw_probabilities)
        encoded_classes = np.asarray(assets.pipeline.classes_)
    except Exception as exc:
        raise PredictionError(
            "The neurological risk model could not make a prediction."
        ) from exc

    if encoded_classes.ndim != 1 or len(encoded_classes) != len(probabilities):
        raise PredictionError(
            "Model output size does not match the model classes."
        )

    try:
        class_names = [
            str(name)
            for name in assets.label_encoder.inverse_transform(encoded_classes)
        ]
    except Exception as exc:
        raise PredictionError(
            "The neurological risk model returned invalid class labels."
        ) from exc

    predicted_index = int(np.argmax(probabilities))
    return NeurologicalPredictionResponse(
        predicted_class=class_names[predicted_index],
        confidence=round(float(probabilities[predicted_index]), 6),
        class_probabilities={
            name: round(float(probability), 6)
            for name, probability in zip(class_names, probabilities)
        },
    )


def _request_values(payload: NeurologicalRiskRequest) -> dict[str, Any]:
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    return payload.dict()


def _normalize_probabilities(raw_probabilities: Any) -> np.ndarray:
    probabilities = np.asarray(raw_probabilities, dtype=np.float64)
    if probabilities.ndim == 2 and probabilities.shape[0] == 1:
        probabilities = probabilities[0]

    if probabilities.ndim != 1 or probabilities.size == 0:
        raise PredictionError(
            "Model prediction must be a one-dimensional class distribution."
        )
    if not np.all(np.isfinite(probabilities)):
        raise PredictionError(
            "Model prediction contains non-finite probabilities."
        )
    if np.any(probabilities < 0):
        raise PredictionError(
            "Model prediction contains negative probabilities."
        )

    total = float(probabilities.sum())
    if total <= 0.0:
        raise PredictionError("Model prediction probabilities sum to zero.")
    return probabilities / total
