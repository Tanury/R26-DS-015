from app.core.exceptions import FeatureValidationError


REQUEST_FEATURE_COLUMNS = [
    "mfcc_1_mean",
    "mfcc_2_mean",
    "mfcc_3_mean",
    "pitch_mean",
    "pitch_std",
    "jitter",
    "shimmer",
    "hnr",
    "speech_rate",
    "pause_count",
    "mean_pause_duration",
    "mean_energy",
    "spectral_centroid_mean",
    "zero_crossing_rate_mean",
]

MODEL_FEATURE_ALIASES = {
    "mfcc_1_mean": "mfcc1_mean",
    "mfcc_2_mean": "mfcc2_mean",
    "mfcc_3_mean": "mfcc3_mean",
    "pitch_mean": "pitch_mean_hz",
    "pitch_std": "pitch_std_hz",
    "jitter": "jitter_local_pct",
    "shimmer": "shimmer_local_pct",
    "hnr": "hnr_db",
    "speech_rate": "speech_rate_wps",
    "pause_count": "pause_count",
    "mean_pause_duration": "mean_pause_duration_sec",
    "mean_energy": "intensity_mean_db",
    "spectral_centroid_mean": "spectral_centroid_hz",
    "zero_crossing_rate_mean": "zero_crossing_rate",
}


def validate_exact_features(features: dict[str, float], required_columns: list[str]) -> None:
    """Ensure the incoming payload exactly matches the trained feature schema."""
    provided = set(features)
    required = set(required_columns)
    missing = sorted(required - provided)
    unexpected = sorted(provided - required)

    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing features: {', '.join(missing)}")
        if unexpected:
            parts.append(f"unexpected features: {', '.join(unexpected)}")
        raise FeatureValidationError("; ".join(parts))


def validate_requested_features(features: dict[str, float]) -> None:
    """Ensure the public API receives exactly the supported 14 speech features."""
    validate_exact_features(features, REQUEST_FEATURE_COLUMNS)


def build_model_feature_row(
    features: dict[str, float],
    model_columns: list[str],
    scaler: object,
) -> list[float]:
    """Map public feature names into the trained model feature vector.

    Columns not exposed by the first API version are filled with the scaler's
    training mean, so StandardScaler transforms them to neutral zero values.
    """
    model_values = {
        model_column: features[request_column]
        for request_column, model_column in MODEL_FEATURE_ALIASES.items()
    }
    fallback_values = _extract_scaler_means(model_columns, scaler)
    return [
        model_values.get(column, fallback_values.get(column, 0.0))
        for column in model_columns
    ]


def _extract_scaler_means(model_columns: list[str], scaler: object) -> dict[str, float]:
    means = getattr(scaler, "mean_", None)
    if means is None or len(means) != len(model_columns):
        return {}
    return {
        column: float(mean)
        for column, mean in zip(model_columns, means)
    }
