import numpy as np

from app.core.config import settings
from app.core.exceptions import PredictionError
from app.schemas.prediction_request import PredictionRequest
from app.schemas.prediction_response import PredictionResponse
from app.services.model_loader import ModelAssets, load_model_assets
from app.services.recommendation_service import (
    build_observed_issues,
    build_recommendations,
)
from app.services.risk_service import calculate_risk_score, derive_risk_level
from app.utils.feature_validator import (
    build_model_feature_row,
    validate_requested_features,
)


def predict_risk(payload: PredictionRequest) -> PredictionResponse:
    validate_requested_features(payload.features)
    assets = load_model_assets()
    return predict_with_assets(payload, assets)


def predict_with_assets(
    payload: PredictionRequest,
    assets: ModelAssets,
) -> PredictionResponse:
    validate_requested_features(payload.features)

    ordered_values = build_model_feature_row(
        payload.features,
        assets.feature_columns,
        assets.scaler,
    )
    feature_array = np.asarray([ordered_values], dtype=np.float32)
    scaled_features = assets.scaler.transform(feature_array)

    raw_prediction = assets.model.predict(scaled_features, verbose=0)
    probabilities = _normalize_probabilities(raw_prediction)
    predicted_index = int(np.argmax(probabilities))
    predicted_class = str(
        assets.label_encoder.inverse_transform([predicted_index])[0]
    )
    class_labels = [str(label) for label in assets.label_encoder.classes_]

    if len(class_labels) != len(probabilities):
        raise PredictionError("Model output size does not match label encoder classes.")

    confidence = round(float(probabilities[predicted_index]), 6)
    risk_score = calculate_risk_score(predicted_class, confidence)
    risk_level = derive_risk_level(risk_score)

    return PredictionResponse(
        predicted_class=predicted_class,
        confidence_score=confidence,
        risk_score=risk_score,
        risk_level=risk_level,
        probabilities={
            label: round(float(probability), 6)
            for label, probability in zip(class_labels, probabilities)
        },
        observed_issues=build_observed_issues(predicted_class, risk_level),
        recommendations=build_recommendations(predicted_class, risk_level),
        disclaimer=settings.disclaimer,
    )


def _normalize_probabilities(raw_prediction: object) -> np.ndarray:
    probabilities = np.asarray(raw_prediction, dtype=np.float64)
    probabilities = np.squeeze(probabilities)

    if probabilities.ndim != 1 or probabilities.size == 0:
        raise PredictionError("Model prediction must be a one-dimensional class distribution.")
    if not np.all(np.isfinite(probabilities)):
        raise PredictionError("Model prediction contains non-finite values.")
    if np.any(probabilities < 0):
        raise PredictionError("Model prediction contains negative probabilities.")

    total = float(probabilities.sum())
    if total <= 0.0:
        raise PredictionError("Model prediction probabilities sum to zero.")

    return probabilities / total
