import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings
from app.core.exceptions import ModelLoadError, PredictionError
from app.schemas.neurological_risk_request import NeurologicalRiskRequest
from app.schemas.prediction_response import PredictionResponse


MODEL_FILENAME = "neurological_risk_model.joblib"
METADATA_FILENAME = "neurological_risk_model_metadata.json"
EXPECTED_SCHEMA_VERSION = "neurological-biomedical-24-v1"
EXPECTED_SCHEMA_HASH = (
    "84a39236b4552164ea643cdc5103887b1be42982278ec3c6878675af0b71634b"
)
EXPECTED_FEATURE_COLUMNS = (
    "age", "sex", "education_years", "bmi", "family_history_pd",
    "systolic_bp", "diastolic_bp", "cognitive_screen_score_0_30",
    "rem_sleep_score", "updrs_part_i", "updrs_part_ii", "updrs_part_iii",
    "updrs_part_iv", "schwab_england_adl", "apoe_e4_count",
    "gba_variant_carrier", "amyloid_beta_42_40_ratio", "t_tau_pg_ml",
    "p_tau181_pg_ml", "nfl_pg_ml", "gfap_pg_ml",
    "alpha_synuclein_pg_ml", "gdf15_pg_ml", "crp40_copy_number",
)
EXPECTED_DISEASE_CLASSES = ("AD", "Healthy", "MS", "PD")
EXPECTED_RISK_CLASSES = ("High", "Low", "Medium")


@dataclass(frozen=True)
class NeurologicalModelAssets:
    disease_model: Any
    risk_model: Any
    feature_columns: tuple[str, ...]
    disease_classes: tuple[str, ...]
    risk_classes: tuple[str, ...]
    risk_score_weights: dict[str, float]
    run_id: str


@lru_cache(maxsize=1)
def load_neurological_model() -> NeurologicalModelAssets:
    try:
        import joblib
        import sklearn

        with (Path(settings.model_dir) / METADATA_FILENAME).open(
            encoding="utf-8"
        ) as file:
            metadata = json.load(file)
        artifact = joblib.load(Path(settings.model_dir) / MODEL_FILENAME)
    except Exception as exc:
        raise ModelLoadError("Unable to load the neurological risk model.") from exc

    _validate_runtime_version(sklearn.__version__, metadata)
    _validate_model_contract(artifact, metadata)

    weights = {
        str(label): float(weight)
        for label, weight in artifact["risk_score_weights"].items()
    }
    if max(weights.values(), default=0.0) > 1.0:
        weights = {label: weight / 100.0 for label, weight in weights.items()}

    return NeurologicalModelAssets(
        disease_model=artifact["disease_model"],
        risk_model=artifact["risk_model"],
        feature_columns=tuple(artifact["feature_columns"]),
        disease_classes=tuple(str(value) for value in artifact["disease_classes"]),
        risk_classes=tuple(str(value) for value in artifact["risk_classes"]),
        risk_score_weights=weights,
        run_id=str(artifact["run_id"]),
    )


def predict_neurological_risk(payload: NeurologicalRiskRequest) -> PredictionResponse:
    return predict_neurological_risk_with_assets(payload, load_neurological_model())


def predict_neurological_risk_with_assets(
    payload: NeurologicalRiskRequest,
    assets: NeurologicalModelAssets,
) -> PredictionResponse:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ModelLoadError(
            "pandas is required to prepare neurological model features."
        ) from exc

    values = _request_values(payload)
    try:
        model_input = pd.DataFrame(
            [[np.nan if values[column] is None else values[column] for column in assets.feature_columns]],
            columns=assets.feature_columns,
        )
        disease_probabilities = _normalize_probabilities(
            assets.disease_model.predict_proba(model_input)
        )
        risk_probabilities = _normalize_probabilities(
            assets.risk_model.predict_proba(model_input)
        )
    except PredictionError:
        raise
    except Exception as exc:
        raise PredictionError(
            "The neurological risk model could not make a prediction."
        ) from exc

    if len(disease_probabilities) != len(assets.disease_classes):
        raise PredictionError("Disease output size does not match its class metadata.")
    if len(risk_probabilities) != len(assets.risk_classes):
        raise PredictionError("Risk output size does not match its class metadata.")

    disease_index = int(np.argmax(disease_probabilities))
    risk_index = int(np.argmax(risk_probabilities))
    disease_class = assets.disease_classes[disease_index]
    risk_class = assets.risk_classes[risk_index]
    disease_map = {
        label: round(float(probability), 6)
        for label, probability in zip(
            assets.disease_classes, disease_probabilities, strict=True
        )
    }
    risk_score = float(
        sum(
            float(probability) * assets.risk_score_weights.get(label, 0.0)
            for label, probability in zip(
                assets.risk_classes, risk_probabilities, strict=True
            )
        )
    )

    return PredictionResponse(
        predicted_class=disease_class,
        confidence_score=round(float(disease_probabilities[disease_index]), 6),
        risk_score=round(float(np.clip(risk_score, 0.0, 1.0)), 6),
        risk_level=risk_class.lower(),
        probabilities=disease_map,
        observed_issues=_build_observed_issues(
            disease_class, risk_class, risk_score, values
        ),
        recommendations=_build_recommendations(disease_class, risk_class),
        disclaimer=settings.biomedical_disclaimer,
    )


def _validate_runtime_version(runtime_version: str, metadata: dict[str, Any]) -> None:
    trained_version = str(metadata.get("software", {}).get("scikit_learn", ""))
    if tuple(runtime_version.split(".")[:2]) != tuple(trained_version.split(".")[:2]):
        raise ModelLoadError(
            "Neurological model/runtime mismatch: the artifact requires "
            f"scikit-learn {trained_version}, but {runtime_version} is installed."
        )


def _validate_model_contract(artifact: Any, metadata: dict[str, Any]) -> None:
    required_keys = {
        "run_id", "schema_version", "schema_hash", "dataset_sha256",
        "feature_columns", "disease_model", "risk_model", "disease_classes",
        "risk_classes", "risk_score_weights",
    }
    if not isinstance(artifact, dict) or not required_keys.issubset(artifact):
        raise ModelLoadError("Neurological model bundle is incomplete.")

    feature_columns = tuple(str(value) for value in artifact["feature_columns"])
    metadata_features = tuple(str(value) for value in metadata.get("feature_columns", []))
    disease_classes = tuple(str(value) for value in artifact["disease_classes"])
    risk_classes = tuple(str(value) for value in artifact["risk_classes"])
    metadata_classes = metadata.get("classes", {})
    paired_values = (
        (artifact["run_id"], metadata.get("run_id")),
        (artifact["schema_version"], metadata.get("schema_version")),
        (artifact["schema_hash"], metadata.get("schema_hash")),
        (artifact["dataset_sha256"], metadata.get("dataset_sha256")),
    )

    if any(str(left) != str(right) for left, right in paired_values):
        raise ModelLoadError("Neurological model and metadata do not belong to one run.")
    if artifact["schema_version"] != EXPECTED_SCHEMA_VERSION:
        raise ModelLoadError("Unsupported neurological model schema version.")
    if artifact["schema_hash"] != EXPECTED_SCHEMA_HASH:
        raise ModelLoadError("Neurological model schema hash is not recognized.")
    if feature_columns != EXPECTED_FEATURE_COLUMNS or metadata_features != feature_columns:
        raise ModelLoadError("Neurological model does not expose the exact 24 features.")
    if disease_classes != EXPECTED_DISEASE_CLASSES:
        raise ModelLoadError("Neurological disease classes do not match the API contract.")
    if risk_classes != EXPECTED_RISK_CLASSES:
        raise ModelLoadError("Neurological risk classes do not match the API contract.")
    if tuple(metadata_classes.get("disease", [])) != disease_classes:
        raise ModelLoadError("Disease classes differ between model and metadata.")
    if tuple(metadata_classes.get("risk", [])) != risk_classes:
        raise ModelLoadError("Risk classes differ between model and metadata.")
    if tuple(str(value) for value in artifact["disease_model"].classes_) != disease_classes:
        raise ModelLoadError("Disease classifier class order differs from its metadata.")
    if tuple(str(value) for value in artifact["risk_model"].classes_) != risk_classes:
        raise ModelLoadError("Risk classifier class order differs from its metadata.")
    if set(artifact["risk_score_weights"]) != set(risk_classes):
        raise ModelLoadError("Risk score weights do not cover the risk classes.")
    if not hasattr(artifact["disease_model"], "predict_proba"):
        raise ModelLoadError("Neurological disease model has no probability output.")
    if not hasattr(artifact["risk_model"], "predict_proba"):
        raise ModelLoadError("Neurological risk model has no probability output.")


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
        raise PredictionError("Model prediction contains non-finite probabilities.")
    if np.any(probabilities < 0):
        raise PredictionError("Model prediction contains negative probabilities.")
    total = float(probabilities.sum())
    if total <= 0.0:
        raise PredictionError("Model prediction probabilities sum to zero.")
    return probabilities / total


def _build_observed_issues(
    disease_class: str,
    risk_class: str,
    risk_score: float,
    values: dict[str, Any],
) -> list[str]:
    disease_meanings = {
        "AD": "an Alzheimer-type pattern",
        "Healthy": "the trained healthy-reference pattern",
        "MS": "a multiple-sclerosis-type pattern",
        "PD": "a Parkinson-type pattern",
    }
    issues = [
        "Across the submitted clinical, genetic, and biomarker fields, the "
        f"disease classifier most closely matched {disease_meanings[disease_class]}. "
        "This is pattern similarity, not a diagnosis.",
        f"The independent risk pipeline selected {risk_class.lower()} risk; its "
        f"probability-weighted score is {round(risk_score * 100)}/100.",
    ]
    missing_count = sum(value is None for value in values.values())
    if missing_count:
        issues.append(
            f"{missing_count} of 24 values were not supplied and were imputed by "
            "the saved pipeline from training-set statistics, which reduces the "
            "patient-specific evidence in this estimate."
        )
    else:
        issues.append(
            "All 24 model inputs were supplied; no input value required model imputation."
        )

    cognitive = values.get("cognitive_screen_score_0_30")
    updrs_values = [
        values.get(f"updrs_part_{part}") for part in ("i", "ii", "iii", "iv")
    ]
    supplied_updrs = [float(value) for value in updrs_values if value is not None]
    context = []
    if cognitive is not None:
        context.append(f"cognitive screen {float(cognitive):g}/30")
    if supplied_updrs:
        context.append(f"submitted UPDRS total {sum(supplied_updrs):g}")
    if context:
        issues.append(
            "Clinical-scale context: " + " and ".join(context) + ". These values "
            "provide context but are not a feature-attribution explanation."
        )
    return issues


def _build_recommendations(disease_class: str, risk_class: str) -> list[str]:
    if risk_class == "High":
        recommendations = [
            "Arrange timely review with a neurologist or the clinician who ordered these measurements.",
            "Confirm unexpected biomarker results, units, specimen type, and assay method with the responsible laboratory before acting.",
        ]
    elif risk_class == "Medium":
        recommendations = [
            "Discuss the result with the clinician who knows the symptoms, examination findings, and testing history.",
            "Review missing or uncertain inputs and consider repeat or complementary testing only when clinically appropriate.",
        ]
    else:
        recommendations = [
            "Continue routine clinical follow-up; a low model score does not rule out neurological disease.",
            "Seek clinical review for new, persistent, or worsening cognitive, movement, sensory, visual, or balance symptoms.",
        ]

    disease_follow_up = {
        "AD": "If memory or cognitive concerns are present, ask about a validated cognitive assessment and an evidence-based dementia work-up.",
        "PD": "If tremor, slowness, stiffness, gait change, or REM-sleep symptoms are present, consider a movement-disorder examination.",
        "MS": "If relapsing sensory, visual, motor, or balance symptoms are present, ask whether neurological examination and imaging are appropriate.",
        "Healthy": "Do not interpret the healthy-reference match as clearance when symptoms or abnormal clinical findings are present.",
    }
    recommendations.append(disease_follow_up[disease_class])
    recommendations.append(
        "Use this research output only alongside clinical history, examination, validated tests, and qualified medical judgment."
    )
    return recommendations
