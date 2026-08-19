import logging

from fastapi.testclient import TestClient

from app.core.exceptions import ModelLoadError
from app.main import app
from app.schemas.prediction_response import PredictionResponse
from app.utils.feature_validator import REQUEST_FEATURE_COLUMNS


def valid_features() -> dict[str, float]:
    return {
        column: float(index + 1)
        for index, column in enumerate(REQUEST_FEATURE_COLUMNS)
    }


def successful_response() -> PredictionResponse:
    return PredictionResponse(
        predicted_class="Healthy",
        confidence_score=0.9,
        risk_score=0.1,
        risk_level="low",
        probabilities={"AD": 0.02, "Healthy": 0.9, "MS": 0.03, "PD": 0.05},
        observed_issues=["No elevated neurological speech-risk pattern was detected."],
        recommendations=["Continue routine monitoring."],
        disclaimer="This is not a medical diagnosis.",
    )


def test_prediction_endpoint_returns_structured_response_and_request_id(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setattr(
        "app.api.routes.prediction.predict_risk",
        lambda payload: successful_response(),
    )
    caplog.set_level(logging.INFO)

    with TestClient(app) as client:
        response = client.post(
            "/predictions/",
            json=valid_features(),
            headers={"X-Request-ID": "api-test-123"},
        )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "api-test-123"
    assert response.json()["predicted_class"] == "Healthy"
    assert "request_complete request_id=api-test-123" in caplog.text


def test_prediction_endpoint_rejects_incomplete_feature_set() -> None:
    with TestClient(app) as client:
        response = client.post("/predictions/", json={"jitter": 0.5})

    assert response.status_code == 422
    assert "missing features:" in response.json()["detail"]


def test_prediction_endpoint_hides_model_load_details(monkeypatch) -> None:
    def unavailable(payload):
        raise ModelLoadError("C:/private/model/path is broken")

    monkeypatch.setattr("app.api.routes.prediction.predict_risk", unavailable)

    with TestClient(app) as client:
        response = client.post("/predictions/", json=valid_features())

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Prediction model is temporarily unavailable."
    }
    assert "private" not in response.text
