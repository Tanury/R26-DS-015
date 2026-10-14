import json

import joblib
import numpy as np

from app.core.config import settings
from app.schemas.biomedical_prediction_request import BiomedicalPredictionRequest
from app.services.biomedical_prediction_service import (
    BiomedicalModelAssets,
    build_biomedical_feature_frame,
    predict_biomedical_with_assets,
)


def request_payload() -> BiomedicalPredictionRequest:
    return BiomedicalPredictionRequest(
        sex="Female",
        apoe_genotype="3/3",
        alpha_synuclein_rt_quic_result="Negative",
        amyloid_beta_42_pg_ml=850.0,
        t_tau_pg_ml=250.0,
        p_tau181_pg_ml=35.0,
        nfl_pg_ml=650.0,
        gfap_pg_ml=180.0,
        alpha_synuclein_pg_ml=1200.0,
        kynurenic_acid_nm=42.0,
        quinolinic_acid_nm=310.0,
        gdf15_pg_ml=760.0,
        crp40_copy_number=2.0,
        dopamine_mrm_intensity=12500.0,
    )


def saved_contract():
    with (settings.model_dir / "feature_names.json").open(encoding="utf-8") as file:
        feature_names = json.load(file)
    encoders = joblib.load(settings.model_dir / "categorical_encoders.joblib")
    labels = joblib.load(settings.model_dir / "label_encoders.joblib")
    return feature_names, encoders, labels["risk"]


def test_feature_builder_recreates_exact_130_column_contract() -> None:
    feature_names, encoders, _ = saved_contract()
    frame = build_biomedical_feature_frame(request_payload(), feature_names, encoders)

    assert frame.shape == (1, 130)
    assert frame.columns.tolist() == feature_names
    assert frame.at[0, "sex"] == 0
    assert frame.at[0, "apoe_genotype"] == 3
    assert frame.at[0, "alpha_synuclein_rt_quic_result"] == 0
    assert frame.at[0, "amyloid_beta_42_pg_ml_missing"] == 0
    assert frame.at[0, "klotho_pg_ml_missing"] == 1
    assert frame.at[0, "csf_available"] == 1
    assert frame.at[0, "plasma_available"] == 1
    assert frame.at[0, "genetics_available"] == 1
    assert frame.at[0, "lipidomics_available"] == 1
    assert frame.at[0, "rt_quic_available"] == 1
    assert frame.at[0, "modality_score"] == 5
    assert frame.at[0, "total_missing"] == 31
    assert frame.at[0, "tau_amyloid_ratio"] == np.clip(250 / (850 + 1e-6), 0, 1e4)


class PassThroughPreprocessor:
    def transform(self, frame):
        return frame.to_numpy()


class FakeModel:
    classes_ = np.asarray([0, 1, 2])

    def predict_proba(self, values):
        assert values.shape == (1, 130)
        return np.asarray([[0.7, 0.1, 0.2]])


def test_prediction_preserves_response_shape_and_ordinal_risk_score() -> None:
    feature_names, encoders, label_encoder = saved_contract()
    assets = BiomedicalModelAssets(
        model=FakeModel(),
        preprocessor=PassThroughPreprocessor(),
        label_encoder=label_encoder,
        categorical_encoders=encoders,
        feature_names=feature_names,
    )

    result = predict_biomedical_with_assets(request_payload(), assets)

    assert result.predicted_class == "High"
    assert result.confidence_score == 0.7
    assert result.risk_score == 0.8
    assert result.risk_level == "high"
    assert result.probabilities == {"High": 0.7, "Low": 0.1, "Medium": 0.2}
    assert result.observed_issues
    assert len(result.recommendations) == 3
    assert "not a medical diagnosis" in result.disclaimer
