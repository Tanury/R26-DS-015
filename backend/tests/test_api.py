import logging

from fastapi.testclient import TestClient

from app.core.exceptions import ModelLoadError
from app.main import app
from app.schemas.prediction_response import PredictionResponse


def valid_features() -> dict[str, str | float]:
    return {
        "sex": "Female",
        "apoe_genotype": "3/3",
        "alpha_synuclein_rt_quic_result": "Negative",
        "amyloid_beta_42_pg_ml": 850.0,
        "t_tau_pg_ml": 250.0,
        "p_tau181_pg_ml": 35.0,
        "nfl_pg_ml": 650.0,
        "gfap_pg_ml": 180.0,
        "alpha_synuclein_pg_ml": 1200.0,
        "kynurenic_acid_nm": 42.0,
        "quinolinic_acid_nm": 310.0,
        "gdf15_pg_ml": 760.0,
        "crp40_copy_number": 2.0,
        "dopamine_mrm_intensity": 12500.0,
    }


def successful_response() -> PredictionResponse:
    return PredictionResponse(
        predicted_class="Low",
        confidence_score=0.9,
        risk_score=0.08,
        risk_level="low",
        probabilities={"High": 0.03, "Low": 0.9, "Medium": 0.07},
        observed_issues=["The biomarker pattern matched the low-risk class."],
        recommendations=["Continue routine clinical follow-up."],
        disclaimer="This is not a medical diagnosis.",
    )


def test_prediction_endpoint_returns_structured_response_and_request_id(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setattr(
        "app.api.routes.prediction.predict_biomedical_risk",
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
    assert response.json()["predicted_class"] == "Low"
    assert "request_complete request_id=api-test-123" in caplog.text


def test_prediction_endpoint_rejects_incomplete_feature_set() -> None:
    with TestClient(app) as client:
        response = client.post("/predictions/", json={"sex": "Female"})

    assert response.status_code == 422
    assert response.json()["detail"]


def test_prediction_endpoint_hides_model_load_details(monkeypatch) -> None:
    def unavailable(payload):
        raise ModelLoadError("C:/private/model/path is broken")

    monkeypatch.setattr(
        "app.api.routes.prediction.predict_biomedical_risk",
        unavailable,
    )

    with TestClient(app) as client:
        response = client.post("/predictions/", json=valid_features())

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Prediction model is temporarily unavailable."
    }
    assert "private" not in response.text
