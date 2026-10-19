"""Evidence must describe measured data, without manufacturing a match."""

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.eeg_assessment import SimilarEegRequest
from app.services import eeg_cohort_service as cohort
from app.services.eeg_preprocessing import waveform_preview


BANDS = cohort.COMPARISON_BANDS


def test_neighbours_are_ranked_across_labels_and_exclude_self_and_other_runs(monkeypatch):
    rows = [
        ("self", "AD", .2, "run"), ("ad", "AD", .3, "run"),
        ("control", "HC", .21, "run"), ("far-control", "HC", .4, "run"),
        ("stale", "MS", .2, "other"), ("partial", "PD", .2, "run"),
    ]
    subjects = [SimpleNamespace(subject_id=i, true_class=c, site="AR", signal_quality="Good")
                for i, c, _, _ in rows]
    reports = {i: SimpleNamespace(model_summary={"run_id": run}, dataset={},
                                  band_power_profile={b: value for b in BANDS})
               for i, _, value, run in rows}
    reports["partial"].band_power_profile.pop("theta")
    monkeypatch.setattr(cohort, "load_cohort_index", lambda: subjects)
    monkeypatch.setattr(cohort, "get_report", lambda i: reports[i])
    response = cohort.find_similar_samples(SimilarEegRequest(
        subject_id="self", run_id="run", band_power_profile={b: .2 for b in BANDS}))
    assert response.compared_subjects == 3
    assert [s.subject_id for s in response.samples] == ["control", "ad"]
    assert response.samples[0].distance == pytest.approx(.01)
    assert response.samples[1].distance == pytest.approx(.1)


def test_comparison_api_uses_real_cohort_and_rejects_incomplete_input():
    client = TestClient(app)
    report = cohort.get_report("AD-AR-sub-30001")
    payload = {"subject_id": report.subject_id, "run_id": report.model_summary["run_id"],
               "band_power_profile": {b: report.band_power_profile[b] for b in BANDS}}
    response = client.post("/eeg/similar-samples", json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert {s["true_class"] for s in result["samples"]} == {"HC", "AD", "PD", "MS"}
    assert all(s["subject_id"] != report.subject_id for s in result["samples"])
    assert result["compared_subjects"] > 4
    distances = [s["distance"] for s in result["samples"]]
    assert distances == sorted(distances)
    payload["band_power_profile"].pop("alpha")
    assert client.post("/eeg/similar-samples", json=payload).status_code == 422
    payload["band_power_profile"]["alpha"] = 1.1
    assert client.post("/eeg/similar-samples", json=payload).status_code == 422


def test_no_compatible_run_returns_no_matches():
    response = cohort.find_similar_samples(SimilarEegRequest(
        subject_id="upload", run_id="no-compatible-model", band_power_profile={b: .2 for b in BANDS}))
    assert response.samples == []
    assert response.compared_subjects == 0


def test_waveform_preserves_measured_samples_in_microvolts_without_mutating_input():
    epoch = np.array([[1e-6, -2e-6, 3e-6], [4e-6, 5e-6, 6e-6], [9e-6, 8e-6, 7e-6]])
    original = epoch.copy()
    result = waveform_preview(epoch, ["A1", "A2", "B1"], 256)
    assert result["channels"] == {"A1": [1., -2., 3.], "B1": [9., 8., 7.]}
    assert result["sampling_rate_hz"] == 256
    assert result["unit"] == "µV"
    np.testing.assert_array_equal(epoch, original)


def test_cleaned_waveform_survives_preprocessing_and_real_inference(monkeypatch, tmp_path):
    """Exercise the new report field through the real cleaner and deployed encoder."""
    import mne
    from app.services import eeg_preprocessing as preprocessing
    from app.services.eeg_inference_service import infer
    from app.services.eeg_model_loader import load_eeg_assets

    assets = load_eeg_assets()
    channels = list(assets.channel_order)
    sfreq = assets.sampling_rate_hz
    rng = np.random.default_rng(12)
    raw = mne.io.RawArray(rng.normal(0, 3e-6, (len(channels), int(sfreq * 8))),
                         mne.create_info(channels, sfreq, "eeg"), verbose=False)
    monkeypatch.setattr(preprocessing, "load_recording", lambda path: (raw, "continuous"))
    config = {"channel_order": channels, "sampling_rate_hz": sfreq, "epoch_length_seconds": 4,
              "min_clean_epochs": 1, "max_epochs": 2, "use_ica": False}
    processed = preprocessing.preprocess(tmp_path / "synthetic-test.set", config)
    result = infer(processed, "synthetic-waveform-test", explain=True)
    assert result.waveform_preview is not None
    assert set(result.waveform_preview.channels) == {"A1", "B1", "C1", "D1"}
    assert all(len(values) == int(sfreq * 4) for values in result.waveform_preview.channels.values())
    assert result.waveform_preview.model_dump() == processed["waveform_preview"]
    assert result.explainability.target_output == "independent_risk"
    assert result.explainability.target_condition == result.risk_assessment.highest_risk_condition
    assert result.optional_four_class_prediction is not None
