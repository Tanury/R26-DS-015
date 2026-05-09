from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class NeurologicalInput(BaseModel):
    age: Optional[float] = Field(None, example=68)
    sex: Optional[str] = Field(None, example="Female")

    # Cognitive / clinical scores
    moca_total_score: Optional[float] = Field(None, example=21)
    updrs_part_i: Optional[float] = Field(None, example=8)
    updrs_part_ii: Optional[float] = Field(None, example=13)
    updrs_part_iii: Optional[float] = Field(None, example=28)
    updrs_part_iv: Optional[float] = Field(None, example=4)
    disease_duration_years: Optional[float] = Field(None, example=4.5)

    # Fluid biomarkers
    amyloid_beta_42_pg_ml: Optional[float] = Field(None, example=420.5)
    amyloid_beta_40_pg_ml: Optional[float] = Field(None, example=6200.0)
    p_tau181_pg_ml: Optional[float] = Field(None, example=3.2)
    t_tau_pg_ml: Optional[float] = Field(None, example=280.0)
    nfl_pg_ml: Optional[float] = Field(None, example=18.5)
    gfap_pg_ml: Optional[float] = Field(None, example=140.0)
    alpha_synuclein_pg_ml: Optional[float] = Field(None, example=920.0)

    # Derived / inflammation features
    neuroinflam_score: Optional[float] = Field(None, example=0.65)
    tau_amyloid_ratio: Optional[float] = Field(None, example=0.43)


class ImportantFeature(BaseModel):
    feature: str
    display_name: str
    feature_type: str
    importance: float


class PredictionResponse(BaseModel):
    predicted_disease: str
    disease_full_name: str
    disease_confidence: float

    predicted_risk: str
    risk_confidence: float

    disease_probabilities: Dict[str, float]
    risk_probabilities: Dict[str, float]

    conclusion: str
    clinical_recommendation: str

    important_features: List[ImportantFeature]

    embedding_info: Dict[str, Any]
    model_summary: Dict[str, Any]
    disclaimer: str