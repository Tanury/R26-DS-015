from __future__ import annotations

import pytest

from src.inference.report_generator import build_prediction_report


def _make_report(**kwargs):
    defaults = dict(
        subject_id="sub-001",
        input_metadata={"subject_id": "sub-001", "synthetic_data": False},
        preprocessing_summary={"clean_epochs_used": 10, "total_epochs_generated": 12, "signal_quality": "Good"},
        ad_probability=0.81,
        epoch_probs=[0.8, 0.82, 0.79, 0.83],
        z_eeg=[1.0] + [0.0] * 255,
        embedding_consistency=0.95,
        visual_outputs={},
    )
    defaults.update(kwargs)
    return build_prediction_report(**defaults)


def test_all_required_top_level_keys():
    report = _make_report()
    required = [
        "dataset", "input_metadata", "preprocessing_summary", "model_summary",
        "model_artifact", "evaluation_protocol", "subject_level_prediction", "epoch_probability_summary",
        "bar_chart_data", "embedding_output", "visual_outputs", "clinical_disclaimer",
    ]
    for key in required:
        assert key in report, f"Missing required top-level key: {key}"


def test_safe_clinical_wording():
    report = _make_report(ad_probability=0.81)
    prediction = report["subject_level_prediction"]["prediction"]
    assert prediction == "Alzheimer's EEG Pattern"
    assert "diagnosis" not in prediction.lower(), \
        "Prediction text must not contain 'diagnosis' — use 'EEG Pattern' wording"


def test_hc_prediction():
    report = _make_report(ad_probability=0.2)
    assert report["subject_level_prediction"]["prediction"] == "Healthy Control"
    assert report["subject_level_prediction"]["risk_level"] == "Low"


def test_near_threshold_prediction_is_uncertain():
    report = _make_report(ad_probability=0.495, decision_threshold=0.48)
    pred = report["subject_level_prediction"]
    assert pred["is_uncertain"] is True
    assert pred["prediction"] == "Uncertain - leaning Alzheimer's EEG Pattern"
    assert pred["decision_threshold"] == pytest.approx(0.48, abs=1e-4)


def test_risk_levels():
    assert _make_report(ad_probability=0.2)["subject_level_prediction"]["risk_level"] == "Low"
    assert _make_report(ad_probability=0.5)["subject_level_prediction"]["risk_level"] == "Medium"
    assert _make_report(ad_probability=0.75)["subject_level_prediction"]["risk_level"] == "High"


def test_embedding_output_fields():
    report = _make_report()
    emb = report["embedding_output"]
    assert emb["z_eeg_shape"] == [256]
    assert emb["l2_norm"] == pytest.approx(1.0, abs=1e-4)
    assert emb["availability_flag"] == 1
    assert "embedding_consistency" in emb, "embedding_consistency field must be present"
    assert 0.0 <= emb["embedding_consistency"] <= 1.0


def test_no_epoch_leakage_flag():
    report = _make_report()
    assert report["evaluation_protocol"]["no_epoch_leakage"] is True


def test_clinical_disclaimer_present():
    report = _make_report()
    assert "not a clinical diagnosis" in report["clinical_disclaimer"].lower()


def test_probabilities_consistent():
    report = _make_report(ad_probability=0.81)
    pred = report["subject_level_prediction"]
    assert pred["ad_eeg_pattern_probability"] == pytest.approx(0.81, abs=1e-4)
    assert pred["predicted_class_confidence"] == pytest.approx(0.81, abs=1e-4)
    bar = report["bar_chart_data"]
    ad_bar = next(b for b in bar if "Alzheimer" in b["class"])
    hc_bar = next(b for b in bar if b["class"] == "Healthy Control")
    assert abs(ad_bar["probability"] + hc_bar["probability"] - 1.0) < 1e-3


def test_synthetic_data_flag_propagated():
    report = _make_report(
        input_metadata={"subject_id": "sub-synth-001", "synthetic_data": True}
    )
    assert report["input_metadata"]["synthetic_data"] is True
    assert report["input_metadata"]["real_data"] is False
