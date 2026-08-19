from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: str
    probabilities: dict[str, float]
    observed_issues: list[str]
    recommendations: list[str]
    disclaimer: str
