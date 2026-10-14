from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.neurological_prediction import router
from app.schemas.biomedical_record import GeminiBiomedicalRecordAnalysis
from app.services.gemini_biomedical_service import _gemini_response_schema


VALID_PDF = b"%PDF-1.4\n% biomedical test record"
app = FastAPI()
app.include_router(router, prefix="/neurological-risk")
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


def extracted_record() -> GeminiBiomedicalRecordAnalysis:
    return GeminiBiomedicalRecordAnalysis(
        extraction_quality="limited",
        quality_notes=["Eight supported fields were not present in the record."],
        features={
            **VALID_PAYLOAD,
            "rem_sleep_score": None,
            "updrs_part_i": None,
            "updrs_part_ii": None,
            "updrs_part_iii": None,
            "updrs_part_iv": None,
            "schwab_england_adl": None,
            "gba_variant_carrier": None,
            "crp40_copy_number": None,
        },
    )


def test_extract_record_returns_validated_form_values(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.api.routes.neurological_prediction.extract_biomedical_record",
        lambda document, mime: extracted_record(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/neurological-risk/extract-record",
            files={"file": ("lab-report.pdf", VALID_PDF, "application/pdf")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "lab-report.pdf"
    assert body["extraction_quality"] == "limited"
    assert body["extracted_count"] == 16
    assert body["features"]["age"] == 65
    assert body["features"]["rem_sleep_score"] is None


def test_extract_record_rejects_mismatched_file_content() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/neurological-risk/extract-record",
            files={"file": ("lab-report.pdf", b"not a PDF", "application/pdf")},
        )

    assert response.status_code == 422
    assert "does not match" in response.json()["detail"]


def test_extract_record_rejects_unsupported_format() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/neurological-risk/extract-record",
            files={"file": ("record.txt", b"age: 65", "text/plain")},
        )

    assert response.status_code == 415


def test_gemini_schema_contains_the_complete_no_extra_fields_contract() -> None:
    schema = _gemini_response_schema()
    features = schema["$defs"]["NeurologicalRiskRequest"]

    assert len(features["properties"]) == 24
    assert set(features["required"]) == set(VALID_PAYLOAD)

    def assert_no_additional_properties(node) -> None:
        if isinstance(node, dict):
            assert "additionalProperties" not in node
            for value in node.values():
                assert_no_additional_properties(value)
        elif isinstance(node, list):
            for value in node:
                assert_no_additional_properties(value)

    assert_no_additional_properties(schema)
