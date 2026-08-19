from pydantic import BaseModel, Field


class NeurologicalPredictionResponse(BaseModel):
    predicted_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_probabilities: dict[str, float]
