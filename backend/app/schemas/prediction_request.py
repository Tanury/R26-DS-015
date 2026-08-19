from math import isfinite
from typing import Any

from pydantic import BaseModel, Field

try:
    from pydantic import field_validator, model_validator
except ImportError:
    field_validator = None
    model_validator = None


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Speech feature values keyed by the supported public feature names.",
        json_schema_extra={
            "example": {
                "mfcc_1_mean": -245.2,
                "mfcc_2_mean": 82.1,
                "mfcc_3_mean": 14.7,
                "pitch_mean": 178.4,
                "pitch_std": 32.8,
                "jitter": 0.9,
                "shimmer": 3.1,
                "hnr": 18.6,
                "speech_rate": 2.4,
                "pause_count": 12,
                "mean_pause_duration": 0.42,
                "mean_energy": 61.3,
                "spectral_centroid_mean": 1840.5,
                "zero_crossing_rate_mean": 0.08,
            }
        },
    )

    if model_validator is not None:

        @model_validator(mode="before")
        @classmethod
        def accept_flat_feature_payload(cls, value: Any) -> Any:
            return _wrap_flat_feature_payload(value)

    else:
        from pydantic import root_validator

        @root_validator(pre=True)
        def accept_flat_feature_payload(cls, value: Any) -> Any:
            return _wrap_flat_feature_payload(value)

    if field_validator is not None:

        @field_validator("features")
        @classmethod
        def validate_feature_values(cls, value: dict[str, Any]) -> dict[str, float]:
            return _clean_features(value)

    else:
        from pydantic import validator

        @validator("features")
        def validate_feature_values(cls, value: dict[str, Any]) -> dict[str, float]:
            return _clean_features(value)


def _wrap_flat_feature_payload(value: Any) -> Any:
    if isinstance(value, dict) and "features" not in value:
        return {"features": value}
    return value


def _clean_features(value: dict[str, Any]) -> dict[str, float]:
    if not value:
        raise ValueError("features must not be empty")

    cleaned: dict[str, float] = {}
    for key, raw_value in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("all feature names must be non-empty strings")
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"feature '{key}' must be numeric") from exc
        if not isfinite(numeric_value):
            raise ValueError(f"feature '{key}' must be finite")
        cleaned[key] = numeric_value
    return cleaned
