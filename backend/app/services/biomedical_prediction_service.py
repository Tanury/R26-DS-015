import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.exceptions import ModelLoadError, PredictionError
from app.schemas.biomedical_prediction_request import BiomedicalPredictionRequest
from app.schemas.prediction_response import PredictionResponse


MODEL_FILENAME = "champion_SoftVoting_XGB_RF_HGB_.joblib"
PREPROCESSOR_FILENAME = "preprocessor.joblib"
LABEL_ENCODERS_FILENAME = "label_encoders.joblib"
CATEGORICAL_ENCODERS_FILENAME = "categorical_encoders.joblib"
FEATURE_NAMES_FILENAME = "feature_names.json"

EXPECTED_FEATURE_COUNT = 130
AVAILABILITY_SOURCES = {
    "csf_available": (
        "amyloid_beta_42_pg_ml",
        "t_tau_pg_ml",
        "p_tau181_pg_ml",
        "nfl_pg_ml",
        "gfap_pg_ml",
    ),
    "plasma_available": ("gdf15_pg_ml", "crp40_copy_number"),
    "serum_available": (),
    "saliva_available": (),
    "genetics_available": ("apoe_genotype",),
    "mirna_available": (),
    "proteomics_available": (),
    "lipidomics_available": (
        "kynurenic_acid_nm",
        "quinolinic_acid_nm",
        "dopamine_mrm_intensity",
    ),
    "rt_quic_available": ("alpha_synuclein_rt_quic_result",),
}
LOG_TARGETS = (
    "nfl_pg_ml",
    "gfap_pg_ml",
    "t_tau_pg_ml",
    "p_tau181_pg_ml",
    "amyloid_beta_42_pg_ml",
    "alpha_synuclein_pg_ml",
    "gdf15_pg_ml",
    "crp40_copy_number",
    "dopamine_mrm_intensity",
)


@dataclass(frozen=True)
class BiomedicalModelAssets:
    model: Any
    preprocessor: Any
    label_encoder: Any
    categorical_encoders: dict[str, Any]
    feature_names: list[str]


@lru_cache(maxsize=1)
def load_biomedical_model_assets() -> BiomedicalModelAssets:
    try:
        with (settings.model_dir / FEATURE_NAMES_FILENAME).open(encoding="utf-8") as file:
            feature_names = json.load(file)
        label_encoders = joblib.load(settings.model_dir / LABEL_ENCODERS_FILENAME)
        assets = BiomedicalModelAssets(
            model=joblib.load(settings.model_dir / MODEL_FILENAME),
            preprocessor=joblib.load(settings.model_dir / PREPROCESSOR_FILENAME),
            label_encoder=label_encoders["risk"],
            categorical_encoders=joblib.load(
                settings.model_dir / CATEGORICAL_ENCODERS_FILENAME
            ),
            feature_names=feature_names,
        )
    except Exception as exc:
        raise ModelLoadError("Biomedical model artifacts could not be loaded.") from exc

    if len(assets.feature_names) != EXPECTED_FEATURE_COUNT:
        raise ModelLoadError(
            f"Biomedical feature contract must contain {EXPECTED_FEATURE_COUNT} columns."
        )
    return assets


def predict_biomedical_risk(
    payload: BiomedicalPredictionRequest,
) -> PredictionResponse:
    return predict_biomedical_with_assets(payload, load_biomedical_model_assets())


def predict_biomedical_with_assets(
    payload: BiomedicalPredictionRequest,
    assets: BiomedicalModelAssets,
) -> PredictionResponse:
    try:
        feature_frame = build_biomedical_feature_frame(
            payload,
            assets.feature_names,
            assets.categorical_encoders,
        )
        prepared = assets.preprocessor.transform(feature_frame)
        probabilities = _normalize_probabilities(assets.model.predict_proba(prepared)[0])
        encoded_classes = np.asarray(assets.model.classes_)
        class_labels = [
            str(label)
            for label in assets.label_encoder.inverse_transform(encoded_classes)
        ]
    except PredictionError:
        raise
    except Exception as exc:
        raise PredictionError("Biomedical model inference failed.") from exc

    if len(class_labels) != len(probabilities):
        raise PredictionError("Model output size does not match risk label classes.")

    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_labels[predicted_index]
    probability_map = {
        label: round(float(probability), 6)
        for label, probability in zip(class_labels, probabilities)
    }
    confidence = round(float(probabilities[predicted_index]), 6)
    risk_score = round(
        float(
            probability_map.get("Medium", 0.0) * 0.5
            + probability_map.get("High", 0.0)
        ),
        6,
    )

    return PredictionResponse(
        predicted_class=predicted_class,
        confidence_score=confidence,
        risk_score=risk_score,
        risk_level=predicted_class.lower(),
        probabilities=probability_map,
        observed_issues=_build_observed_issues(predicted_class),
        recommendations=_build_recommendations(predicted_class),
        disclaimer=settings.biomedical_disclaimer,
    )


def build_biomedical_feature_frame(
    payload: BiomedicalPredictionRequest,
    feature_names: list[str],
    categorical_encoders: dict[str, Any],
) -> pd.DataFrame:
    if len(feature_names) != EXPECTED_FEATURE_COUNT:
        raise PredictionError(
            f"Expected {EXPECTED_FEATURE_COUNT} biomedical features, got {len(feature_names)}."
        )

    raw = _model_dump(payload)
    row: dict[str, Any] = {feature: np.nan for feature in feature_names}
    row.update(raw)

    # Derive all missingness indicators from the raw, pre-encoding values.
    for feature in feature_names:
        if feature.endswith("_missing") and feature != "total_missing":
            raw_feature = feature[: -len("_missing")]
            row[feature] = int(_is_missing(row.get(raw_feature)))

    # Derive acquisition modalities from the supplied public biomarkers.
    for availability_feature, source_features in AVAILABILITY_SOURCES.items():
        row[availability_feature] = int(
            any(not _is_missing(row.get(source)) for source in source_features)
        )

    for feature, encoder in categorical_encoders.items():
        raw_value = row.get(feature)
        if _is_missing(raw_value):
            raw_value = "nan"
        raw_value = str(raw_value)
        if raw_value not in encoder.classes_:
            raise PredictionError(f"Unsupported value for categorical feature '{feature}'.")
        row[feature] = int(encoder.transform([raw_value])[0])

    row["tau_amyloid_ratio"] = float(
        np.clip(
            row["t_tau_pg_ml"] / (row["amyloid_beta_42_pg_ml"] + 1e-6),
            0,
            1e4,
        )
    )
    row["ptau_ttau_ratio"] = float(
        np.clip(row["p_tau181_pg_ml"] / (row["t_tau_pg_ml"] + 1e-6), 0, 100)
    )
    row["neuroinflam_score"] = float(
        np.log1p(row["nfl_pg_ml"]) + np.log1p(row["gfap_pg_ml"])
    )
    row["kyn_ratio"] = float(
        np.clip(
            row["kynurenic_acid_nm"] / (row["quinolinic_acid_nm"] + 1e-6),
            0,
            1e3,
        )
    )

    availability_features = [
        feature for feature in feature_names if feature.endswith("_available")
    ]
    missingness_features = [
        feature
        for feature in feature_names
        if feature.endswith("_missing") and feature != "total_missing"
    ]
    row["modality_score"] = int(sum(row[feature] for feature in availability_features))
    row["total_missing"] = int(sum(row[feature] for feature in missingness_features))

    for feature in LOG_TARGETS:
        value = 0.0 if _is_missing(row.get(feature)) else float(row[feature])
        row[f"log_{feature}"] = float(np.log1p(value))

    frame = pd.DataFrame([[row[feature] for feature in feature_names]], columns=feature_names)
    if frame.shape != (1, EXPECTED_FEATURE_COUNT):
        raise PredictionError("Biomedical feature frame has an unexpected shape.")
    return frame.replace([np.inf, -np.inf], np.nan)


def _model_dump(payload: BiomedicalPredictionRequest) -> dict[str, Any]:
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    return payload.dict()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def _normalize_probabilities(values: Any) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64).reshape(-1)
    if probabilities.size == 0 or not np.all(np.isfinite(probabilities)):
        raise PredictionError("Model returned invalid probabilities.")
    if np.any(probabilities < 0):
        raise PredictionError("Model returned negative probabilities.")
    total = float(probabilities.sum())
    if total <= 0:
        raise PredictionError("Model probabilities sum to zero.")
    return probabilities / total


def _build_observed_issues(risk_class: str) -> list[str]:
    messages = {
        "Low": (
            "The ensemble found a biomarker pattern most consistent with its low-risk class."
        ),
        "Medium": (
            "The ensemble found a mixed biomarker pattern consistent with its medium-risk class."
        ),
        "High": (
            "The ensemble found a biomarker pattern consistent with its high-risk class."
        ),
    }
    return [messages.get(risk_class, "The ensemble produced a biomedical risk class.")]


def _build_recommendations(risk_class: str) -> list[str]:
    if risk_class == "High":
        return [
            "Arrange timely review by a qualified neurologist or the clinician who ordered these tests.",
            "Confirm unexpected or abnormal biomarker measurements with the responsible laboratory before acting on this result.",
            "Interpret the score together with symptoms, examination findings, medical history, imaging, and validated clinical tests.",
        ]
    if risk_class == "Medium":
        return [
            "Discuss the result with the clinician who ordered these biomarker tests.",
            "Review measurement quality and consider repeat or complementary testing if clinically appropriate.",
            "Monitor relevant symptoms and seek earlier review if they are new, persistent, or worsening.",
        ]
    return [
        "Continue routine clinical follow-up and follow the advice of the clinician who ordered the tests.",
        "Do not use a low model score to dismiss new, persistent, or worsening neurological symptoms.",
        "Retain the laboratory report so future measurements can be compared consistently.",
    ]
