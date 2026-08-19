import numpy as np

from app.core.exceptions import FeatureValidationError
from app.schemas.prediction_request import PredictionRequest
from app.services.model_loader import ModelAssets
from app.services.prediction_service import predict_with_assets
from app.utils.feature_validator import REQUEST_FEATURE_COLUMNS


class FakeScaler:
    mean_ = np.zeros(91)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return values


class FakeModel:
    def predict(self, values: np.ndarray, verbose: int = 0) -> np.ndarray:
        return np.asarray([[0.05, 0.1, 0.8, 0.05]])


class FakeLabelEncoder:
    classes_ = np.asarray(["AD", "Healthy", "PD", "MS"])

    def inverse_transform(self, indexes: list[int]) -> np.ndarray:
        return self.classes_[indexes]


def test_prediction_pipeline_orders_features_and_returns_risk_payload() -> None:
    assets = ModelAssets(
        model=FakeModel(),
        scaler=FakeScaler(),
        label_encoder=FakeLabelEncoder(),
        feature_columns=[
            "age",
            "recording_duration_sec",
            "snr_db",
            "speech_rate_wps",
            "articulation_rate_sps",
            "pause_count",
            "mean_pause_duration_sec",
            "total_pause_duration_sec",
            "pause_ratio",
            "lexical_fluency_score",
            "word_finding_pause_score",
            "disfluency_rate",
            "pitch_mean_hz",
            "pitch_std_hz",
            "pitch_range_hz",
            "intensity_mean_db",
            "intensity_std_db",
            "jitter_local_pct",
            "shimmer_local_pct",
            "hnr_db",
            "nhr",
            "voice_tremor_score",
            "monotone_score",
            "phonation_time_sec",
            "rhythm_irregularity_score",
            "formant_f1_mean_hz",
            "formant_f2_mean_hz",
            "formant_f3_mean_hz",
            "formant_instability_score",
            "spectral_centroid_hz",
            "spectral_bandwidth_hz",
            "spectral_rolloff_hz",
            "zero_crossing_rate",
            "spectral_flatness",
            "mfcc1_mean",
            "mfcc2_mean",
            "mfcc3_mean",
            "mfcc4_mean",
            "mfcc5_mean",
            "mfcc6_mean",
            "mfcc7_mean",
            "mfcc8_mean",
            "mfcc9_mean",
            "mfcc10_mean",
            "mfcc11_mean",
            "mfcc12_mean",
            "mfcc13_mean",
            "mfcc1_std",
            "mfcc2_std",
            "mfcc3_std",
            "mfcc4_std",
            "mfcc5_std",
            "mfcc6_std",
            "mfcc7_std",
            "mfcc8_std",
            "mfcc9_std",
            "mfcc10_std",
            "mfcc11_std",
            "mfcc12_std",
            "mfcc13_std",
            "delta_mfcc1_mean",
            "delta_mfcc2_mean",
            "delta_mfcc3_mean",
            "delta_mfcc4_mean",
            "delta_mfcc5_mean",
            "delta_mfcc6_mean",
            "delta_mfcc7_mean",
            "delta_mfcc8_mean",
            "delta_mfcc9_mean",
            "delta_mfcc10_mean",
            "delta_mfcc11_mean",
            "delta_mfcc12_mean",
            "delta_mfcc13_mean",
            "delta2_mfcc1_mean",
            "delta2_mfcc2_mean",
            "delta2_mfcc3_mean",
            "delta2_mfcc4_mean",
            "delta2_mfcc5_mean",
            "delta2_mfcc6_mean",
            "delta2_mfcc7_mean",
            "delta2_mfcc8_mean",
            "delta2_mfcc9_mean",
            "delta2_mfcc10_mean",
            "delta2_mfcc11_mean",
            "delta2_mfcc12_mean",
            "delta2_mfcc13_mean",
            "sex_Male",
            "recording_task_monologue",
            "recording_task_picture_description",
            "recording_task_reading",
            "recording_task_sustained_vowel",
        ],
    )
    request = PredictionRequest(
        features={column: float(index) for index, column in enumerate(REQUEST_FEATURE_COLUMNS)}
    )

    result = predict_with_assets(request, assets)

    assert result.predicted_class == "PD"
    assert result.confidence_score == 0.8
    assert result.risk_score == 0.8
    assert result.risk_level == "high"
    assert result.probabilities == {
        "AD": 0.05,
        "Healthy": 0.1,
        "PD": 0.8,
        "MS": 0.05,
    }
    assert result.observed_issues
    assert result.recommendations
    assert "not a medical diagnosis" in result.disclaimer


def test_prediction_rejects_non_exact_feature_set() -> None:
    assets = ModelAssets(
        model=FakeModel(),
        scaler=FakeScaler(),
        label_encoder=FakeLabelEncoder(),
        feature_columns=["jitter_local_pct", "shimmer_local_pct"],
    )
    request = PredictionRequest(features={"jitter": 1.0, "extra": 3.0})

    try:
        predict_with_assets(request, assets)
    except FeatureValidationError as exc:
        assert "missing features:" in str(exc)
        assert "shimmer" in str(exc)
        assert "unexpected features: extra" in str(exc)
    else:
        raise AssertionError("FeatureValidationError was not raised")


def test_prediction_request_accepts_flat_feature_payload() -> None:
    payload = {
        column: float(index)
        for index, column in enumerate(REQUEST_FEATURE_COLUMNS)
    }

    if hasattr(PredictionRequest, "model_validate"):
        request = PredictionRequest.model_validate(payload)
    else:
        request = PredictionRequest.parse_obj(payload)

    assert request.features == payload
