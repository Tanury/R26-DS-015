from pydantic import BaseModel, Field


class FactorContribution(BaseModel):
    """One submitted input's additive share of the predicted class probability."""

    feature: str
    value: float | str
    contribution: float


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: str
    probabilities: dict[str, float]
    observed_issues: list[str]
    recommendations: list[str]
    disclaimer: str
    # Populated only by models that support per-prediction attribution.
    top_factors: list[FactorContribution] = Field(default_factory=list)
