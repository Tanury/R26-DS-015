from fastapi.testclient import TestClient

from app.main import app
from app.schemas.prediction_response import PredictionResponse
from app.schemas.voice_assessment import GeminiAudioAnalysis


VALID_WAV = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 32


def extracted_analysis() -> GeminiAudioAnalysis:
    return GeminiAudioAnalysis(
        quality_status="usable",
        transcript="The grandfather clock ticked loudly.",
        quality_notes=["Clear single-speaker recording."],
        features={
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
        },
    )


def healthy_prediction() -> PredictionResponse:
    return PredictionResponse(
        predicted_class="Healthy",
        confidence_score=0.91,
        risk_score=0.09,
        risk_level="low",
        probabilities={"AD": 0.02, "Healthy": 0.91, "MS": 0.03, "PD": 0.04},
        observed_issues=["No elevated neurological speech-risk pattern was detected."],
        recommendations=["Continue routine monitoring."],
        disclaimer="This is not a medical diagnosis.",
    )


def test_voice_assessment_composes_extraction_and_prediction(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.api.routes.voice_assessment.extract_speech_features",
        lambda audio, mime: extracted_analysis(),
    )
    monkeypatch.setattr(
        "app.api.routes.voice_assessment.predict_risk",
        lambda payload: healthy_prediction(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/voice-assessments/",
            data={"patient_age": "58", "recording_task": "reading"},
            files={"file": ("sample.wav", VALID_WAV, "audio/wav")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"]["predicted_class"] == "Healthy"
    assert body["extracted_features"]["pitch_mean"] == 178.4
    assert body["transcript"].startswith("The grandfather")
    assert body["extraction_quality"] == "usable"


def test_voice_assessment_rejects_invalid_audio_signature() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/voice-assessments/",
            data={"patient_age": "58", "recording_task": "reading"},
            files={"file": ("sample.wav", b"not-wave-audio", "audio/wav")},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "The file content is not valid audio."


def test_voice_assessment_rejects_unsupported_format() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/voice-assessments/",
            data={"patient_age": "58", "recording_task": "reading"},
            files={"file": ("sample.txt", b"hello", "text/plain")},
        )

    assert response.status_code == 415
