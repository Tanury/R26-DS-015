import json
import joblib
import logging
import numpy as np
import pandas as pd

from app.config import AUDIO_MODEL_PATH, AUDIO_FEATURE_COLUMNS_PATH, CLASS_NAMES

logger = logging.getLogger("app.audio_predictor")
_audio_model = None

with open(AUDIO_FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
    AUDIO_FEATURE_COLUMNS = json.load(f)


def get_audio_model():
    global _audio_model
    if _audio_model is None:
        try:
            _audio_model = joblib.load(str(AUDIO_MODEL_PATH))
        except Exception as exc:
            raise RuntimeError(f"Failed to load audio model from {AUDIO_MODEL_PATH}: {exc}") from exc
    return _audio_model


def risk_level_from_probability(dementia_prob_percent: float) -> str:
    if dementia_prob_percent >= 70:
        return "High"
    elif dementia_prob_percent >= 40:
        return "Medium"
    return "Low"


def predict_audio_features(
    features: dict,
    file_id: str = "audio_file",
    input_type: str = "uploaded audio file",
) -> dict:
    audio_model = get_audio_model()
    row = {}
    missing_features = []
    logger.debug("Starting predict_audio_features for file_id=%s", file_id)

    for col in AUDIO_FEATURE_COLUMNS:
        value = features.get(col, np.nan)
        if col not in features:
            missing_features.append(col)
        row[col] = value

    X = pd.DataFrame([row], columns=AUDIO_FEATURE_COLUMNS)

    # Log DataFrame diagnostics
    try:
        finite_counts = X.apply(lambda s: np.isfinite(s).sum())
        nan_counts = X.isna().sum()
        logger.debug("Feature coverage: %d/%d finite values", int(finite_counts.sum()), X.size)
        logger.debug("Top 10 NaN features: %s", nan_counts.sort_values(ascending=False).head(10).to_dict())
        logger.debug("Input feature sample (first 20): %s", X.iloc[0].head(20).to_dict())
    except Exception as exc:
        logger.exception("Failed to log DataFrame diagnostics: %s", exc)

    # If model is a pipeline with SelectKBest, log which features are being selected
    try:
        if hasattr(audio_model, "named_steps") and "select" in audio_model.named_steps:
            selector = audio_model.named_steps["select"]
            if hasattr(selector, "get_support"):
                support = selector.get_support()
                selected = [col for col, keep in zip(AUDIO_FEATURE_COLUMNS, support) if keep]
                logger.debug("SelectKBest selected %d features: %s", len(selected), selected[:20])
    except Exception:
        logger.exception("Failed to inspect selector in pipeline")

    prob = audio_model.predict_proba(X)[0]
    logger.debug("Raw model probabilities: %s", prob.tolist())

    control_prob = float(prob[0] * 100)
    dementia_prob = float(prob[1] * 100)

    pred_id = int(np.argmax(prob))
    predicted_class = CLASS_NAMES[pred_id]
    confidence = max(control_prob, dementia_prob)

    return {
        "input_metadata": {
            "file_id": file_id,
            "model_type": "audio-only",
            "input_type": input_type,
            "task": "Cookie Theft picture description / speech sample",
            "dataset": "DementiaBank Pitt Corpus"
        },
        "prediction": {
            "predicted_class": predicted_class,
            "risk_level": risk_level_from_probability(dementia_prob),
            "confidence": round(confidence, 2),
            "probabilities": {
                "Control": round(control_prob, 2),
                "Dementia": round(dementia_prob, 2)
            }
        },
        "audio_indicators": {
            "duration_sec": round(float(features.get("duration_sec", 0)), 3),
            "pause_ratio": round(float(features.get("pause_ratio", 0)), 3),
            "speech_ratio": round(float(features.get("speech_ratio", 0)), 3),
            "rms_energy_mean": round(float(features.get("rms_energy_mean", 0)), 6),
            "zero_crossing_rate_mean": round(float(features.get("zero_crossing_rate_mean", 0)), 6),
            "feature_coverage_pct": round(100.0 * (1.0 - len(missing_features) / max(1, len(AUDIO_FEATURE_COLUMNS))), 2),
            "missing_features_count": len(missing_features),
        },
        "explainability": {
            "method": "audio feature importance",
            "note": "The prediction is based on extracted acoustic features such as MFCC, pause ratio, speech ratio, energy, and spectral features."
        },
        "clinical_note": "This output is for research-based dementia risk screening only. It is not a clinical diagnosis."
    }
