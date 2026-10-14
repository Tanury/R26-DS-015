import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import eeg_cohort_service as cohort


client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_caches():
    cohort.reset_caches()
    yield
    cohort.reset_caches()


def test_model_card_exposes_architecture_and_io() -> None:
    body = client.get("/eeg/model-card").json()
    assert body["architecture"]
    assert body["embedding_dim"] == 256
    assert body["risk_conditions"] == ["AD", "PD", "MS"]
    assert len(body["input_shape"]) == 3


def test_model_card_reports_per_condition_auc_with_intervals() -> None:
    performance = client.get("/eeg/model-card").json()["performance"]
    pooled = performance["pooled_per_condition"]
    for condition in ("AD", "PD", "MS"):
        assert pooled[condition]["auc"] is not None
        interval = pooled[condition]["auc_ci"]
        assert interval["ci_low"] < pooled[condition]["auc"] < interval["ci_high"], (
            "the point estimate must sit inside its own confidence interval"
        )


def test_model_card_carries_the_confound_probes() -> None:
    """Age evidence must be present in at least one form.

    The training run's embedding age probe can fail to run when demographics are
    missing — it did on this cohort. `finalize_eeg_model.py` then recovers ages
    and correlates the delivered risk scores against them instead. One of the two
    must be there, or the model is shipping without any age evidence at all.
    """
    disclosure = client.get("/eeg/model-card").json()["confound_disclosure"]
    assert disclosure["statement"], "a confound statement must always travel with the model"
    assert disclosure["site_probe_balanced_accuracy"] is not None

    correlations = disclosure["risk_score_age_correlation"]
    has_age_evidence = (
        disclosure["age_probe_mae_years"] is not None
        or any(v is not None for v in correlations.values())
    )
    assert has_age_evidence, (
        "neither the embedding age probe nor the risk-score age correlation is "
        "present; age is the dominant confound on this cohort and cannot go "
        "unmeasured"
    )


def test_intended_use_states_out_of_scope() -> None:
    intended = client.get("/eeg/model-card").json()["intended_use"]
    assert intended["out_of_scope"]
    joined = " ".join(intended["out_of_scope"]).lower()
    assert "diagnosis" in joined


def test_inference_availability_is_reported_honestly() -> None:
    """Tier 1 works without PyTorch; the card must say whether Tier 2 does."""
    body = client.get("/eeg/model-card").json()
    assert isinstance(body["inference_available"], bool)

    torch_present = True
    try:
        import torch  # noqa: F401
    except ImportError:
        torch_present = False
    if not torch_present:
        assert body["inference_available"] is False, (
            "inference_available must be False when PyTorch is absent"
        )


def test_upload_returns_503_when_inference_unavailable() -> None:
    pytest.importorskip
    if client.get("/eeg/model-card").json()["inference_available"]:
        pytest.skip("inference is available in this environment")
    response = client.post(
        "/eeg/assessments",
        files={"files": ("sub-001.set", b"MATLAB 5.0 MAT-file" + b"\x00" * 64, "application/octet-stream")},
    )
    assert response.status_code == 503
    assert "cohort browsing is unaffected" in response.json()["detail"].lower()
