import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.neurological_prediction_response import (
    NeurologicalPredictionResponse,
)


VALID_PAYLOAD = {
    "rhythm_irregularity_score": 0.42,
    "voice_tremor_score": 0.31,
    "monotone_score": 0.28,
    "word_finding_pause_score": 0.36,
    "formant_instability_score": 0.24,
    "shimmer_local_pct": 3.2,
    "age": 58,
    "lexical_fluency_score": 0.72,
    "jitter_local_pct": 0.84,
    "mean_pause_duration_sec": 0.41,
    "disfluency_rate": 0.09,
    "total_pause_duration_sec": 5.8,
    "pause_count": 14,
    "pause_ratio": 0.18,
    "intensity_std_db": 4.1,
    "pitch_range_hz": 118.0,
    "articulation_rate_sps": 4.3,
    "pitch_std_hz": 31.0,
    "recording_task": "reading",
    "mfcc1_mean": -241.5,
}


def test_neurological_prediction_endpoint(monkeypatch) -> None:
    expected = NeurologicalPredictionResponse(
        predicted_class="Healthy",
        confidence=0.91,
        class_probabilities={
            "AD": 0.02,
            "Healthy": 0.91,
            "MS": 0.03,
            "PD": 0.04,
        },
    )
    monkeypatch.setattr(
        "app.api.routes.neurological_prediction.predict_neurological_risk",
        lambda payload: expected,
    )

    with TestClient(app) as client:
        response = client.post("/neurological-risk/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    assert response.json() == expected.model_dump()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("age", 121),
        ("pause_count", -1),
        ("recording_task", "conversation"),
        ("voice_tremor_score", 1.1),
    ],
)
def test_neurological_prediction_validates_payload(field, value) -> None:
    payload = {**VALID_PAYLOAD, field: value}

    with TestClient(app) as client:
        response = client.post("/neurological-risk/predict", json=payload)

    assert response.status_code == 422


def test_probability_normalization() -> None:
    from app.services.neurological_prediction_service import (
        _normalize_probabilities,
    )

    probabilities = _normalize_probabilities([[2.0, 3.0, 4.0, 1.0]])

    assert np.allclose(probabilities, [0.2, 0.3, 0.4, 0.1])
