import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.neurological_risk_request import NeurologicalRiskRequest
from app.schemas.prediction_response import PredictionResponse
from app.services.neurological_prediction_service import (
    EXPECTED_FEATURE_COLUMNS,
    NeurologicalModelAssets,
    predict_neurological_risk_with_assets,
)


VALID_PAYLOAD = {
    "age": 65,
    "sex": "Female",
    "education_years": 14,
    "bmi": 25.0,
    "family_history_pd": 0,
    "systolic_bp": 125,
    "diastolic_bp": 80,
    "cognitive_screen_score_0_30": 27,
    "rem_sleep_score": 4,
    "updrs_part_i": 4,
    "updrs_part_ii": 5,
    "updrs_part_iii": 12,
    "updrs_part_iv": 0,
    "schwab_england_adl": 90,
    "apoe_e4_count": 1,
    "gba_variant_carrier": 0,
    "amyloid_beta_42_40_ratio": 0.08,
    "t_tau_pg_ml": 300,
    "p_tau181_pg_ml": 50,
    "nfl_pg_ml": 800,
    "gfap_pg_ml": 200,
    "alpha_synuclein_pg_ml": 1200,
    "gdf15_pg_ml": 800,
    "crp40_copy_number": 2000,
}


class FakeProbabilityModel:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.received_columns = None

    def predict_proba(self, frame):
        self.received_columns = tuple(frame.columns)
        return [self.probabilities]


def test_neurological_prediction_endpoint(monkeypatch) -> None:
    expected = PredictionResponse(
        predicted_class="Healthy",
        confidence_score=0.91,
        risk_score=0.12,
        risk_level="low",
        probabilities={"AD": 0.02, "Healthy": 0.91, "MS": 0.03, "PD": 0.04},
        observed_issues=["Healthy-reference pattern selected."],
        recommendations=["Continue routine clinical follow-up."],
        disclaimer="Research use only.",
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
        ("sex", "Other"),
        ("cognitive_screen_score_0_30", 31),
        ("gba_variant_carrier", 2),
        ("amyloid_beta_42_40_ratio", -0.01),
    ],
)
def test_neurological_prediction_validates_payload(field, value) -> None:
    with TestClient(app) as client:
        response = client.post(
            "/neurological-risk/predict",
            json={**VALID_PAYLOAD, field: value},
        )

    assert response.status_code == 422


def test_all_keys_are_required_and_extra_keys_are_rejected() -> None:
    missing_age = dict(VALID_PAYLOAD)
    missing_age.pop("age")

    with TestClient(app) as client:
        missing_response = client.post("/neurological-risk/predict", json=missing_age)
        extra_response = client.post(
            "/neurological-risk/predict",
            json={**VALID_PAYLOAD, "unexpected": 1},
        )

    assert missing_response.status_code == 422
    assert extra_response.status_code == 422


def test_null_values_are_accepted_by_the_metadata_contract(monkeypatch) -> None:
    expected = PredictionResponse(
        predicted_class="Healthy",
        confidence_score=0.5,
        risk_score=0.5,
        risk_level="medium",
        probabilities={"AD": 0.1, "Healthy": 0.5, "MS": 0.2, "PD": 0.2},
        observed_issues=[],
        recommendations=[],
        disclaimer="Research use only.",
    )
    monkeypatch.setattr(
        "app.api.routes.neurological_prediction.predict_neurological_risk",
        lambda payload: expected,
    )

    with TestClient(app) as client:
        response = client.post(
            "/neurological-risk/predict",
            json={key: None for key in VALID_PAYLOAD},
        )

    assert response.status_code == 200


def test_dual_pipeline_output_preserves_general_response_format() -> None:
    disease_model = FakeProbabilityModel([0.1, 0.6, 0.2, 0.1])
    risk_model = FakeProbabilityModel([0.2, 0.3, 0.5])
    assets = NeurologicalModelAssets(
        disease_model=disease_model,
        risk_model=risk_model,
        feature_columns=EXPECTED_FEATURE_COLUMNS,
        disease_classes=("AD", "Healthy", "MS", "PD"),
        risk_classes=("High", "Low", "Medium"),
        risk_score_weights={"Low": 0.0, "Medium": 0.5, "High": 1.0},
        run_id="test-run",
    )

    result = predict_neurological_risk_with_assets(
        NeurologicalRiskRequest(**VALID_PAYLOAD),
        assets,
    )

    assert result.predicted_class == "Healthy"
    assert result.confidence_score == 0.6
    assert result.risk_level == "medium"
    assert result.risk_score == 0.45
    assert result.probabilities == {
        "AD": 0.1,
        "Healthy": 0.6,
        "MS": 0.2,
        "PD": 0.1,
    }
    assert disease_model.received_columns == EXPECTED_FEATURE_COLUMNS
    assert risk_model.received_columns == EXPECTED_FEATURE_COLUMNS
    assert any("All 24" in issue for issue in result.observed_issues)
    assert len(result.recommendations) == 4


def test_probability_normalization() -> None:
    from app.services.neurological_prediction_service import _normalize_probabilities

    probabilities = _normalize_probabilities([[2.0, 3.0, 4.0, 1.0]])

    assert np.allclose(probabilities, [0.2, 0.3, 0.4, 0.1])
