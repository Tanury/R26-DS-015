from pydantic import BaseModel, Field
from typing import Dict, Optional, List


class TranscriptRequest(BaseModel):
    file_id: Optional[str] = "manual_input"
    transcript: str = Field(..., min_length=3)


class AudioFeaturesRequest(BaseModel):
    file_id: Optional[str] = "manual_audio_features"
    features: Dict[str, float]


class PredictionResponse(BaseModel):
    input_metadata: Dict
    prediction: Dict
    explainability: Dict
    clinical_note: str