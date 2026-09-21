from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.neurological_risk_request import NeurologicalRiskRequest


class GeminiBiomedicalRecordAnalysis(BaseModel):
    """Strict structured response expected from Gemini for one uploaded record."""

    model_config = ConfigDict(extra="forbid")

    extraction_quality: Literal["usable", "limited", "unusable"]
    quality_notes: list[str] = Field(default_factory=list, max_length=8)
    features: NeurologicalRiskRequest


class BiomedicalRecordExtractionResponse(BaseModel):
    filename: str
    extraction_quality: Literal["usable", "limited"]
    quality_notes: list[str]
    extracted_count: int = Field(..., ge=1, le=24)
    features: NeurologicalRiskRequest
    extraction_disclaimer: str
